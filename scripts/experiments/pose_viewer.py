from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from g1_crawl.tasks.manager_based.g1_crawl.g1 import G1_CFG

import json
from typing import Dict, List
import time

import torch

# Optional keyboard input handling
import carb
import omni


def _assert_keys(data: dict, required_keys: List[str], context: str) -> None:
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys {missing} in {context}")


def create_scene_cfg():
    """Create a simple scene with ground, light, and the robot (free root)."""

    class SimpleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )
        Robot = G1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=G1_CFG.spawn.replace(
                rigid_props=G1_CFG.spawn.rigid_props.replace(disable_gravity=True),
                articulation_props=G1_CFG.spawn.articulation_props.replace(fix_root_link=False),
            ),
        )

    return SimpleSceneCfg


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Convert roll-pitch-yaw (XYZ intrinsic) to quaternion (w, x, y, z).

    Assumes inputs are radians.
    """
    half_r = 0.5 * roll
    half_p = 0.5 * pitch
    half_y = 0.5 * yaw

    cr = torch.cos(torch.tensor(half_r))
    sr = torch.sin(torch.tensor(half_r))
    cp = torch.cos(torch.tensor(half_p))
    sp = torch.sin(torch.tensor(half_p))
    cy = torch.cos(torch.tensor(half_y))
    sy = torch.sin(torch.tensor(half_y))

    # XYZ intrinsic -> wxyz
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = torch.stack([w, x, y, z], dim=0).to(dtype=torch.float32)
    # Normalize for safety
    quat = quat / torch.linalg.norm(quat)
    return quat


def load_pose_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _assert_keys(data, ["poses"], context="pose json root")
    poses = data["poses"]
    if not isinstance(poses, list) or len(poses) == 0:
        raise ValueError("'poses' must be a non-empty list")
    
    # Validate all poses
    validated_poses = []
    for i, pose in enumerate(poses):
        _assert_keys(pose, ["base_pos", "base_rpy", "joints"], context=f"pose entry {i}")
        base_pos = pose["base_pos"]
        base_rpy = pose["base_rpy"]
        joints = pose["joints"]
        if not (isinstance(base_pos, list) and len(base_pos) == 3):
            raise ValueError(f"Pose {i}: 'base_pos' must be a list of 3 floats: [x, y, z]")
        if not (isinstance(base_rpy, list) and len(base_rpy) == 3):
            raise ValueError(f"Pose {i}: 'base_rpy' must be a list of 3 floats: [roll, pitch, yaw]")
        if not isinstance(joints, dict) or len(joints) == 0:
            raise ValueError(f"Pose {i}: 'joints' must be a non-empty mapping of name->value")
        validated_poses.append({"base_pos": base_pos, "base_rpy": base_rpy, "joints": joints})
    
    return validated_poses


def build_joint_name_to_index_map(asset) -> Dict[str, int]:
    joint_names = [str(n) for n in asset.data.joint_names]
    name_to_index = {name: int(i) for i, name in enumerate(joint_names)}
    if len(name_to_index) == 0:
        raise RuntimeError("Robot asset reports zero joints; cannot apply pose")
    return name_to_index


def apply_pose(scene: InteractiveScene, pose: Dict, name_to_index: Dict[str, int]) -> None:
    device = scene["Robot"].device

    # Root pose: zero out x/y, only use z from pose
    base_x, base_y, base_z = pose["base_pos"]
    base_r, base_p, base_y = pose["base_rpy"]
    wxyz = rpy_to_quat_wxyz(float(base_r), float(base_p), float(base_y)).to(device=device)

    origin = scene.env_origins.to(device=device)[0] if hasattr(scene, "env_origins") else torch.zeros(3, device=device)
    new_root_state = torch.zeros(1, 13, device=device)
    # Force x=0, y=0, only use z from the pose
    base_pos_tensor = torch.tensor([0.0, 0.0, float(base_z)], device=device, dtype=torch.float32)
    new_root_state[0, :3] = base_pos_tensor + origin
    new_root_state[0, 3:7] = wxyz
    new_root_state[0, 7:13] = 0.0
    scene["Robot"].write_root_pose_to_sim(new_root_state[:, :7])
    scene["Robot"].write_root_velocity_to_sim(new_root_state[:, 7:])

    # Joint positions: start from defaults and overwrite named joints
    joint_positions = scene["Robot"].data.default_joint_pos.clone().to(device=device)
    joint_velocities = torch.zeros_like(joint_positions)

    for joint_name, value in pose["joints"].items():
        if joint_name not in name_to_index:
            raise KeyError(f"Joint '{joint_name}' not found in robot articulation. Available: {list(name_to_index.keys())}")
        j_idx = name_to_index[joint_name]
        joint_positions[0, j_idx] = float(value)

    scene["Robot"].write_joint_state_to_sim(joint_positions, joint_velocities)
    scene.write_data_to_sim()


def run_pose_viewer(sim: sim_utils.SimulationContext, scene: InteractiveScene, pose_path: str) -> None:
    poses = load_pose_json(pose_path)
    name_to_index = build_joint_name_to_index_map(scene["Robot"]) 
    
    # Track current pose index and auto-play state
    current_pose_idx = len(poses) - 1  # Start at last pose
    auto_play = False  # Start paused to view the last pose
    playback_direction = 1  # 1 for forward, -1 for backward
    
    # Frame number display tracking
    last_print_time = time.time()
    print_interval = 1.0  # seconds

    # Per-joint offsets for interactive tweaking (radians)
    adjustable_joints = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "left_elbow_joint",
        "right_elbow_joint",
    ]
    joint_offsets: Dict[str, float] = {j: 0.0 for j in adjustable_joints}
    OFFSET_STEP = 0.05  # radians per key press

    def apply_pose_with_offsets() -> None:
        # Build a temporary pose dict with offsets applied to selected joints
        pose = poses[current_pose_idx]
        p = {
            "base_pos": pose["base_pos"],
            "base_rpy": pose["base_rpy"],
            "joints": dict(pose["joints"]),
        }
        for jn, off in joint_offsets.items():
            if jn not in p["joints"]:
                raise KeyError(f"Joint '{jn}' not in pose json; cannot offset")
            p["joints"][jn] = float(p["joints"][jn]) + float(off)
        apply_pose(scene, p, name_to_index)

    def export_offsets() -> None:
        # Build changed-only mapping and full joints mapping with offsets
        pose = poses[current_pose_idx]
        changed = {}
        full = dict(pose["joints"])  # copy
        for jn, off in joint_offsets.items():
            if abs(off) > 0.0:
                new_val = float(pose["joints"][jn]) + float(off)
                changed[jn] = new_val
                full[jn] = new_val
        import json as _json
        print(f"\n=== Pose {current_pose_idx} - Changed joints (JSON snippet) ===")
        print(_json.dumps(changed, indent=2, sort_keys=True))
        print("=== Full joints block (JSON) ===")
        print(_json.dumps(full, indent=2, sort_keys=True))
        print("=== End export ===\n")

    # Keyboard setup for reset / exit
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()

    keys_pressed = {"R": False, "ESCAPE": False, "LEFT": False, "RIGHT": False, "SPACE": False, "B": False}

    def on_keyboard_event(event):
        nonlocal keys_pressed, current_pose_idx, auto_play, playback_direction
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key == "SPACE" and not keys_pressed["SPACE"]:
                keys_pressed["SPACE"] = True
                auto_play = not auto_play
                direction_str = "FORWARD" if playback_direction == 1 else "BACKWARD"
                print(f"Auto-play: {'ON' if auto_play else 'OFF'} ({direction_str})")
            elif key == "B" and not keys_pressed["B"]:
                keys_pressed["B"] = True
                playback_direction *= -1
                direction_str = "FORWARD" if playback_direction == 1 else "BACKWARD"
                print(f"Playback direction: {direction_str}")
            elif key == "R" and not keys_pressed["R"]:
                keys_pressed["R"] = True
                pose = poses[current_pose_idx]
                apply_pose(scene, pose, name_to_index)
                print(f"Pose {current_pose_idx}/{len(poses)-1} reapplied from JSON (no offsets)")
            elif key == "LEFT" and not keys_pressed["LEFT"]:
                keys_pressed["LEFT"] = True
                auto_play = False  # Pause auto-play when manually navigating
                current_pose_idx = (current_pose_idx - 1) % len(poses)
                # Clear offsets when changing poses
                for k in joint_offsets:
                    joint_offsets[k] = 0.0
                apply_pose_with_offsets()
                print(f"Previous pose: {current_pose_idx}/{len(poses)-1} [Auto-play paused]")
            elif key == "RIGHT" and not keys_pressed["RIGHT"]:
                keys_pressed["RIGHT"] = True
                auto_play = False  # Pause auto-play when manually navigating
                current_pose_idx = (current_pose_idx + 1) % len(poses)
                # Clear offsets when changing poses
                for k in joint_offsets:
                    joint_offsets[k] = 0.0
                apply_pose_with_offsets()
                print(f"Next pose: {current_pose_idx}/{len(poses)-1} [Auto-play paused]")
            elif key == "LEFT_BRACKET":
                # Skip backward by 10%
                skip_amount = max(1, len(poses) // 10)
                current_pose_idx = (current_pose_idx - skip_amount) % len(poses)
                for k in joint_offsets:
                    joint_offsets[k] = 0.0
                apply_pose_with_offsets()
                print(f"Skipped backward 10% to pose: {current_pose_idx}/{len(poses)-1}")
            elif key == "RIGHT_BRACKET":
                # Skip forward by 10%
                skip_amount = max(1, len(poses) // 10)
                current_pose_idx = (current_pose_idx + skip_amount) % len(poses)
                for k in joint_offsets:
                    joint_offsets[k] = 0.0
                apply_pose_with_offsets()
                print(f"Skipped forward 10% to pose: {current_pose_idx}/{len(poses)-1}")
            elif key == "HOME":
                # Jump to start
                current_pose_idx = 0
                for k in joint_offsets:
                    joint_offsets[k] = 0.0
                apply_pose_with_offsets()
                print(f"Jumped to start: {current_pose_idx}/{len(poses)-1}")
            elif key == "END":
                # Jump to end
                current_pose_idx = len(poses) - 1
                for k in joint_offsets:
                    joint_offsets[k] = 0.0
                apply_pose_with_offsets()
                print(f"Jumped to end: {current_pose_idx}/{len(poses)-1}")
            elif key == "P":
                export_offsets()
            elif key == "C":
                for k in joint_offsets:
                    joint_offsets[k] = 0.0
                apply_pose_with_offsets()
                print("Cleared shoulder offsets")
            # Left shoulder pitch: Q (+), A (-)
            elif key == "Q":
                joint_offsets["left_shoulder_pitch_joint"] += OFFSET_STEP
                apply_pose_with_offsets()
                print(f"left_shoulder_pitch += {OFFSET_STEP:.3f} -> {joint_offsets['left_shoulder_pitch_joint']:.3f}")
            elif key == "A":
                joint_offsets["left_shoulder_pitch_joint"] -= OFFSET_STEP
                apply_pose_with_offsets()
                print(f"left_shoulder_pitch -= {OFFSET_STEP:.3f} -> {joint_offsets['left_shoulder_pitch_joint']:.3f}")
            # Left shoulder roll: W (+), S (-)
            elif key == "W":
                joint_offsets["left_shoulder_roll_joint"] += OFFSET_STEP
                apply_pose_with_offsets()
                print(f"left_shoulder_roll += {OFFSET_STEP:.3f} -> {joint_offsets['left_shoulder_roll_joint']:.3f}")
            elif key == "S":
                joint_offsets["left_shoulder_roll_joint"] -= OFFSET_STEP
                apply_pose_with_offsets()
                print(f"left_shoulder_roll -= {OFFSET_STEP:.3f} -> {joint_offsets['left_shoulder_roll_joint']:.3f}")
            # Right shoulder pitch: E (+), D (-)
            elif key == "E":
                joint_offsets["right_shoulder_pitch_joint"] += OFFSET_STEP
                apply_pose_with_offsets()
                print(f"right_shoulder_pitch += {OFFSET_STEP:.3f} -> {joint_offsets['right_shoulder_pitch_joint']:.3f}")
            elif key == "D":
                joint_offsets["right_shoulder_pitch_joint"] -= OFFSET_STEP
                apply_pose_with_offsets()
                print(f"right_shoulder_pitch -= {OFFSET_STEP:.3f} -> {joint_offsets['right_shoulder_pitch_joint']:.3f}")
            # Right shoulder roll: T (+), G (-)
            elif key == "T":
                joint_offsets["right_shoulder_roll_joint"] += OFFSET_STEP
                apply_pose_with_offsets()
                print(f"right_shoulder_roll += {OFFSET_STEP:.3f} -> {joint_offsets['right_shoulder_roll_joint']:.3f}")
            elif key == "G":
                joint_offsets["right_shoulder_roll_joint"] -= OFFSET_STEP
                apply_pose_with_offsets()
                print(f"right_shoulder_roll -= {OFFSET_STEP:.3f} -> {joint_offsets['right_shoulder_roll_joint']:.3f}")
            # Left elbow: U (+), J (-)
            elif key == "U":
                joint_offsets["left_elbow_joint"] += OFFSET_STEP
                apply_pose_with_offsets()
                print(f"left_elbow += {OFFSET_STEP:.3f} -> {joint_offsets['left_elbow_joint']:.3f}")
            elif key == "J":
                joint_offsets["left_elbow_joint"] -= OFFSET_STEP
                apply_pose_with_offsets()
                print(f"left_elbow -= {OFFSET_STEP:.3f} -> {joint_offsets['left_elbow_joint']:.3f}")
            # Right elbow: I (+), K (-)
            elif key == "I":
                joint_offsets["right_elbow_joint"] += OFFSET_STEP
                apply_pose_with_offsets()
                print(f"right_elbow += {OFFSET_STEP:.3f} -> {joint_offsets['right_elbow_joint']:.3f}")
            elif key == "K":
                joint_offsets["right_elbow_joint"] -= OFFSET_STEP
                apply_pose_with_offsets()
                print(f"right_elbow -= {OFFSET_STEP:.3f} -> {joint_offsets['right_elbow_joint']:.3f}")
            elif event.input.name == "ESCAPE" and not keys_pressed["ESCAPE"]:
                keys_pressed["ESCAPE"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in keys_pressed:
                keys_pressed[event.input.name] = False

    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    # Initial apply
    apply_pose_with_offsets()
    
    # Step simulation to update body positions
    sim.step()
    scene.update(sim.get_physics_dt())
    
    # Print z-coordinates of pelvis and torso_link
    robot = scene["Robot"]
    body_names = [str(n) for n in robot.data.body_names]
    body_pos_w = robot.data.body_pos_w[0]  # [num_bodies, 3], get first env
    
    print("\n=== Body Heights (z-coordinates) ===")
    for body_name in ["pelvis", "torso_link"]:
        if body_name in body_names:
            body_idx = body_names.index(body_name)
            z_height = body_pos_w[body_idx, 2].item()
            print(f"{body_name}: z = {z_height:.4f} m")
        else:
            print(f"{body_name}: NOT FOUND in body names")
    print(f"Available body names: {body_names}")
    print("=====================================\n")

    print("Starting pose viewer...")
    print(f"Loaded {len(poses)} poses from {pose_path}")
    print(f"Playback rate: Full simulation rate (one pose per simulation step)")
    print("\nControls:")
    print("  SPACE - Toggle auto-play on/off")
    print("  B - Toggle playback direction (forward/backward)")
    print("  [ / ] - Skip backward/forward by 10%")
    print("  HOME / END - Jump to start/end")
    print("  LEFT/RIGHT ARROW - Navigate between poses (pauses auto-play)")
    print("  R - Reapply current pose from JSON (ignores offsets)")
    print("  C - Clear all shoulder/elbow offsets")
    print("  Q/A - Left shoulder pitch +/-")
    print("  W/S - Left shoulder roll +/-")
    print("  E/D - Right shoulder pitch +/-")
    print("  T/G - Right shoulder roll +/-")
    print("  U/J - Left elbow +/-")
    print("  I/K - Right elbow +/-")
    print("  P - Print changed and full joints JSON to console")
    print("  ESC - Exit")
    print(f"\nCurrent pose: {current_pose_idx}/{len(poses)-1}")
    direction_str = "FORWARD" if playback_direction == 1 else "BACKWARD"
    print(f"Auto-play: {'ON' if auto_play else 'OFF'} ({direction_str})")

    try:
        sim_dt = sim.get_physics_dt()
        while simulation_app.is_running():
            if keys_pressed["ESCAPE"]:
                print("Exiting pose viewer...")
                break
            
            # Auto-play logic - advance to next/previous pose based on direction
            if auto_play:
                current_pose_idx = (current_pose_idx + playback_direction) % len(poses)
                # Clear offsets when auto-playing
                for k in joint_offsets:
                    joint_offsets[k] = 0.0
            
            # Periodic frame number display
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                last_print_time = current_time
                print(f"Frame: {current_pose_idx}/{len(poses)-1}")
            
            # Apply pose each frame with current offsets
            apply_pose_with_offsets()
            sim.step()
            scene.update(sim_dt)
    finally:
        if keyboard_subscription:
            input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)


def main():
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # Scene
    scene_cfg_class = create_scene_cfg()
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Start simulation
    sim.reset()
    print("[INFO]: Setup complete...")

    # Hardcode the pose file path (adjust if needed)
    # pose_path = "assets/crawl-pose.json"
    # pose_path = "assets/stand-pose-rc2.json"
    pose_path = "assets/sorted-poses-rc3.json"

    print(f"[INFO]: Loading poses from '{pose_path}'...")
    run_pose_viewer(sim, scene, pose_path)


if __name__ == "__main__":
    main()
    simulation_app.close()


