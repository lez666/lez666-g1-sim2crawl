from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from g1_crawl.tasks.manager_based.g1_crawl.g1 import G1_CFG

import json
import torch
import math

# Optional keyboard input handling and UI
import carb
import omni
import omni.ui as ui


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
    """Convert roll-pitch-yaw (XYZ intrinsic) to quaternion (w, x, y, z)."""
    half_r = 0.5 * roll
    half_p = 0.5 * pitch
    half_y = 0.5 * yaw

    cr = torch.cos(torch.tensor(half_r))
    sr = torch.sin(torch.tensor(half_r))
    cp = torch.cos(torch.tensor(half_p))
    sp = torch.sin(torch.tensor(half_p))
    cy = torch.cos(torch.tensor(half_y))
    sy = torch.sin(torch.tensor(half_y))

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = torch.stack([w, x, y, z], dim=0).to(dtype=torch.float32)
    quat = quat / torch.linalg.norm(quat)
    return quat


def load_pose_json(path: str) -> dict:
    """Load pose from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = data["poses"]
    if not isinstance(poses, list) or len(poses) == 0:
        raise ValueError("'poses' must be a non-empty list")
    pose0 = poses[0]
    return {
        "base_pos": pose0["base_pos"],
        "base_rpy": pose0["base_rpy"],
        "joints": pose0["joints"]
    }


def apply_pose_with_orientation(scene: InteractiveScene, pose: dict, name_to_index: dict, 
                                 yaw_offset: float, pitch_offset: float) -> None:
    """Apply a pose to the robot with yaw and pitch offsets."""
    device = scene["Robot"].device

    # Root pose with orientation offsets
    base_x, base_y, base_z = pose["base_pos"]
    base_r, base_p_original, base_y_original = pose["base_rpy"]
    
    # Apply yaw and pitch offsets
    base_y_new = base_y_original + yaw_offset
    base_p_new = base_p_original + pitch_offset
    
    wxyz = rpy_to_quat_wxyz(float(base_r), float(base_p_new), float(base_y_new)).to(device=device)

    origin = scene.env_origins.to(device=device)[0] if hasattr(scene, "env_origins") else torch.zeros(3, device=device)
    new_root_state = torch.zeros(1, 13, device=device)
    base_pos_tensor = torch.tensor([float(base_x), float(base_y), float(base_z)], device=device, dtype=torch.float32)
    new_root_state[0, :3] = base_pos_tensor + origin
    new_root_state[0, 3:7] = wxyz
    new_root_state[0, 7:13] = 0.0
    scene["Robot"].write_root_pose_to_sim(new_root_state[:, :7])
    scene["Robot"].write_root_velocity_to_sim(new_root_state[:, 7:])

    # Joint positions
    joint_positions = scene["Robot"].data.default_joint_pos.clone().to(device=device)
    joint_velocities = torch.zeros_like(joint_positions)

    for joint_name, value in pose["joints"].items():
        if joint_name not in name_to_index:
            raise KeyError(f"Joint '{joint_name}' not found. Available: {list(name_to_index.keys())}")
        j_idx = name_to_index[joint_name]
        joint_positions[0, j_idx] = float(value)

    scene["Robot"].write_joint_state_to_sim(joint_positions, joint_velocities)
    scene.write_data_to_sim()


def compute_com_forward_reward(scene: InteractiveScene) -> dict:
    """Compute the COM forward of feet reward using the actual implementation logic.
    
    Uses gravity-aligned yaw frame to project onto ground plane.
    
    Returns dict with:
        - com_pos_yaw_x: COM X position in yaw frame (ground plane)
        - feet_pos_yaw_x_mean: Mean feet X position in yaw frame
        - forward_offset: Difference (reward value)
        - com_pos_w: COM position in world frame (for debugging)
        - feet_pos_w_mean: Mean feet position in world frame (for debugging)
        - root_yaw_deg: Robot yaw angle
        - root_pitch_deg: Robot pitch angle
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    
    device = scene["Robot"].device
    asset = scene["Robot"]
    
    # Get ankle roll link indices (our "feet")
    body_names = asset.body_names
    feet_indices = [i for i, name in enumerate(body_names) if "ankle_roll_link" in name]
    
    if len(feet_indices) == 0:
        raise RuntimeError("No ankle_roll_link bodies found!")
    
    # Get gravity-aligned yaw frame (removes pitch and roll, keeps only yaw)
    root_quat_w = asset.data.root_link_quat_w  # [1, 4]
    yaw_only_quat = yaw_quat(root_quat_w)  # [1, 4]
    
    # Transform COM to gravity-aligned yaw frame
    com_pos_w = asset.data.root_com_pos_w  # [1, 3]
    com_yaw_frame = quat_apply_inverse(yaw_only_quat, com_pos_w)  # [1, 3]
    com_x_yaw = com_yaw_frame[:, 0]  # [1] - X component in yaw frame
    
    # Transform feet to gravity-aligned yaw frame
    feet_pos_w = asset.data.body_pos_w[:, feet_indices, :]  # [1, num_feet, 3]
    
    # Transform each foot position to yaw frame
    N, num_feet, _ = feet_pos_w.shape
    feet_pos_w_flat = feet_pos_w.reshape(N * num_feet, 3)  # [num_feet, 3]
    yaw_quat_expanded = yaw_only_quat.unsqueeze(1).expand(-1, num_feet, -1).reshape(N * num_feet, 4)
    feet_yaw_frame_flat = quat_apply_inverse(yaw_quat_expanded, feet_pos_w_flat)  # [num_feet, 3]
    feet_yaw_frame = feet_yaw_frame_flat.reshape(N, num_feet, 3)  # [1, num_feet, 3]
    
    feet_x_yaw_mean = feet_yaw_frame[:, :, 0].mean(dim=1)  # [1] - average X in yaw frame
    
    # Reward = how far forward the COM is relative to feet on the ground plane
    forward_offset = com_x_yaw - feet_x_yaw_mean
    
    # Also get world frame positions for debugging
    feet_pos_w_mean = feet_pos_w.mean(dim=1)  # [1, 3]
    
    # Extract yaw and pitch angles for display
    # Yaw from quaternion
    yaw_rad = torch.atan2(
        2.0 * (root_quat_w[0, 0] * root_quat_w[0, 3] + root_quat_w[0, 1] * root_quat_w[0, 2]),
        1.0 - 2.0 * (root_quat_w[0, 2]**2 + root_quat_w[0, 3]**2)
    ).item()
    
    # Pitch from quaternion
    pitch_rad = torch.asin(
        2.0 * (root_quat_w[0, 0] * root_quat_w[0, 2] - root_quat_w[0, 3] * root_quat_w[0, 1])
    ).item()
    
    return {
        "com_pos_yaw_x": com_x_yaw.item(),
        "feet_pos_yaw_x_mean": feet_x_yaw_mean.item(),
        "forward_offset": forward_offset.item(),
        "com_pos_w": com_pos_w[0].cpu().numpy(),
        "feet_pos_w_mean": feet_pos_w_mean[0].cpu().numpy(),
        "root_yaw_deg": yaw_rad * 180.0 / math.pi,
        "root_pitch_deg": pitch_rad * 180.0 / math.pi,
    }


def build_joint_name_to_index_map(asset) -> dict:
    joint_names = [str(n) for n in asset.data.joint_names]
    name_to_index = {name: int(i) for i, name in enumerate(joint_names)}
    return name_to_index


def run_orientation_test(sim: sim_utils.SimulationContext,
                          scene: InteractiveScene,
                          pose_path: str) -> None:
    """Test the COM forward reward with different robot orientations (yaw and pitch)."""
    
    pose = load_pose_json(pose_path)
    name_to_index = build_joint_name_to_index_map(scene["Robot"])
    
    # Test state
    yaw_angle = 0.0  # Current yaw offset in radians
    pitch_angle = 0.0  # Current pitch offset in radians
    yaw_step = math.pi / 12  # 15 degrees
    pitch_step = math.pi / 12  # 15 degrees
    auto_rotate = False
    auto_rotate_speed = 0.02  # radians per frame
    
    # UI Window
    reward_window = ui.Window("COM Forward Reward Test", width=700, height=600)
    reward_labels = {}
    
    with reward_window.frame:
        with ui.VStack(spacing=10, style={"margin": 10}):
            ui.Label("COM Forward of Feet Reward Test", style={"font_size": 20, "color": 0xFFFFFFFF})
            ui.Label("Testing orientation-independence in gravity-aligned yaw frame", 
                    style={"font_size": 14, "color": 0xFFAAAAFF})
            ui.Spacer(height=10)
            
            # Orientation info
            reward_labels["yaw"] = ui.Label(f"Robot Yaw: {yaw_angle:.2f} rad (0.0°)", 
                                           style={"font_size": 16, "color": 0xFFFFFF00})
            reward_labels["pitch"] = ui.Label(f"Robot Pitch: {pitch_angle:.2f} rad (0.0°)", 
                                             style={"font_size": 16, "color": 0xFFFFFF00})
            ui.Spacer(height=10)
            
            # Reward values (yaw frame = ground plane projection)
            ui.Label("=== Yaw Frame / Ground Plane (Should Stay CONSTANT) ===", 
                    style={"font_size": 16, "color": 0xFF00FF00})
            reward_labels["com_yaw_x"] = ui.Label("COM X (yaw frame): --", style={"font_size": 14})
            reward_labels["feet_yaw_x"] = ui.Label("Feet X mean (yaw frame): --", style={"font_size": 14})
            reward_labels["forward_offset"] = ui.Label("Forward Offset (REWARD): --", 
                                                      style={"font_size": 16, "color": 0xFF00FFFF})
            ui.Spacer(height=10)
            
            # World frame values (for comparison)
            ui.Label("=== World Frame (Should Change with Rotation) ===", 
                    style={"font_size": 16, "color": 0xFFFF8800})
            reward_labels["com_w"] = ui.Label("COM pos (world): --", style={"font_size": 12})
            reward_labels["feet_w"] = ui.Label("Feet pos mean (world): --", style={"font_size": 12})
            ui.Spacer(height=15)
            
            ui.Label("Expected Behavior:", style={"font_size": 14, "color": 0xFFFFFF00})
            ui.Label("  • Yaw frame values should stay CONSTANT (both yaw & pitch)", 
                    style={"font_size": 12, "color": 0xFF00FF00})
            ui.Label("  • World frame values should CHANGE with rotation", 
                    style={"font_size": 12, "color": 0xFFFF8800})
            ui.Spacer(height=10)
            
            ui.Label("Controls:", style={"font_size": 14, "color": 0xFFFFFF00})
            ui.Label("  A/D: Yaw ±15°", style={"font_size": 12})
            ui.Label("  W/S: Pitch ±15°", style={"font_size": 12})
            ui.Label("  Q/E: Yaw ±90°", style={"font_size": 12})
            ui.Label("  SPACE: Toggle auto-rotation (yaw)", style={"font_size": 12})
            ui.Label("  R: Reset to 0°", style={"font_size": 12})
            ui.Label("  ESC: Exit", style={"font_size": 12})
    
    def update_ui(reward_data: dict, yaw_rad: float, pitch_rad: float) -> None:
        """Update UI labels with current values."""
        yaw_deg = yaw_rad * 180.0 / math.pi
        pitch_deg = pitch_rad * 180.0 / math.pi
        reward_labels["yaw"].text = f"Robot Yaw: {yaw_rad:.2f} rad ({yaw_deg:.1f}°)"
        reward_labels["pitch"].text = f"Robot Pitch: {pitch_rad:.2f} rad ({pitch_deg:.1f}°)"
        
        # Yaw frame / ground plane (should be constant)
        reward_labels["com_yaw_x"].text = f"COM X (yaw frame): {reward_data['com_pos_yaw_x']:.4f} m"
        reward_labels["feet_yaw_x"].text = f"Feet X mean (yaw frame): {reward_data['feet_pos_yaw_x_mean']:.4f} m"
        reward_labels["forward_offset"].text = f"Forward Offset (REWARD): {reward_data['forward_offset']:.4f} m"
        
        # World frame (should change)
        com_w = reward_data['com_pos_w']
        feet_w = reward_data['feet_pos_w_mean']
        reward_labels["com_w"].text = f"COM pos (world): [{com_w[0]:.3f}, {com_w[1]:.3f}, {com_w[2]:.3f}]"
        reward_labels["feet_w"].text = f"Feet pos mean (world): [{feet_w[0]:.3f}, {feet_w[1]:.3f}, {feet_w[2]:.3f}]"
    
    # Keyboard handling
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    
    keys_pressed = {"A": False, "D": False, "W": False, "S": False, "Q": False, "E": False,
                    "SPACE": False, "R": False, "ESCAPE": False}
    
    def on_keyboard_event(event):
        nonlocal yaw_angle, pitch_angle, auto_rotate, keys_pressed
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            
            # Yaw controls (A/D for small, Q/E for large)
            if key == "A" and not keys_pressed["A"]:
                keys_pressed["A"] = True
                yaw_angle -= yaw_step
                print(f"Yaw: {yaw_angle:.3f} rad ({yaw_angle * 180 / math.pi:.1f}°)")
            elif key == "D" and not keys_pressed["D"]:
                keys_pressed["D"] = True
                yaw_angle += yaw_step
                print(f"Yaw: {yaw_angle:.3f} rad ({yaw_angle * 180 / math.pi:.1f}°)")
            elif key == "Q" and not keys_pressed["Q"]:
                keys_pressed["Q"] = True
                yaw_angle -= math.pi / 2
                print(f"Yaw: {yaw_angle:.3f} rad ({yaw_angle * 180 / math.pi:.1f}°)")
            elif key == "E" and not keys_pressed["E"]:
                keys_pressed["E"] = True
                yaw_angle += math.pi / 2
                print(f"Yaw: {yaw_angle:.3f} rad ({yaw_angle * 180 / math.pi:.1f}°)")
            
            # Pitch controls (W/S)
            elif key == "W" and not keys_pressed["W"]:
                keys_pressed["W"] = True
                pitch_angle += pitch_step
                print(f"Pitch: {pitch_angle:.3f} rad ({pitch_angle * 180 / math.pi:.1f}°)")
            elif key == "S" and not keys_pressed["S"]:
                keys_pressed["S"] = True
                pitch_angle -= pitch_step
                print(f"Pitch: {pitch_angle:.3f} rad ({pitch_angle * 180 / math.pi:.1f}°)")
            
            # Other controls
            elif key == "SPACE" and not keys_pressed["SPACE"]:
                keys_pressed["SPACE"] = True
                auto_rotate = not auto_rotate
                print(f"Auto-rotate: {'ON' if auto_rotate else 'OFF'}")
            elif key == "R" and not keys_pressed["R"]:
                keys_pressed["R"] = True
                yaw_angle = 0.0
                pitch_angle = 0.0
                print(f"Reset to yaw=0.0, pitch=0.0")
            elif key == "ESCAPE" and not keys_pressed["ESCAPE"]:
                keys_pressed["ESCAPE"] = True
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in keys_pressed:
                keys_pressed[event.input.name] = False
    
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    print("\n" + "="*60)
    print("COM Forward of Feet Reward Test")
    print("="*60)
    print(f"Pose: {pose_path}")
    print("\nTesting that the reward is orientation-independent.")
    print("The yaw frame (ground plane) reward should stay constant")
    print("while rotating in both yaw and pitch.")
    print("\nControls:")
    print("  A/D: Yaw ±15°")
    print("  W/S: Pitch ±15°")
    print("  Q/E: Yaw ±90°")
    print("  SPACE: Toggle auto-rotation (yaw)")
    print("  R: Reset to 0°")
    print("  ESC: Exit")
    print("="*60 + "\n")
    
    # Track initial reward for comparison
    initial_reward = None
    
    try:
        sim_dt = sim.get_physics_dt()
        while simulation_app.is_running():
            if keys_pressed["ESCAPE"]:
                print("Exiting...")
                break
            
            # Auto-rotate
            if auto_rotate:
                yaw_angle += auto_rotate_speed
                # Wrap to [-pi, pi]
                if yaw_angle > math.pi:
                    yaw_angle -= 2 * math.pi
                elif yaw_angle < -math.pi:
                    yaw_angle += 2 * math.pi
            
            # Apply pose with current yaw and pitch
            apply_pose_with_orientation(scene, pose, name_to_index, yaw_angle, pitch_angle)
            
            # Compute reward
            reward_data = compute_com_forward_reward(scene)
            
            # Track initial reward
            if initial_reward is None:
                initial_reward = reward_data['forward_offset']
                print(f"\nInitial reward (yaw=0, pitch=0): {initial_reward:.4f}")
            else:
                # Check if reward is stable
                diff = abs(reward_data['forward_offset'] - initial_reward)
                if diff > 0.001:  # More than 1mm difference
                    print(f"⚠️  WARNING: Reward changed! Initial: {initial_reward:.4f}, Current: {reward_data['forward_offset']:.4f}, Diff: {diff:.4f}")
            
            # Update UI
            update_ui(reward_data, yaw_angle, pitch_angle)
            
            # Step simulation
            sim.step()
            scene.update(sim_dt)
            
    finally:
        if keyboard_subscription:
            input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)
        reward_window.visible = False
        
        if initial_reward is not None:
            print(f"\n" + "="*60)
            print(f"Test Summary:")
            print(f"  Initial reward: {initial_reward:.4f}")
            print(f"  Final reward: {reward_data['forward_offset']:.4f}")
            print(f"  Difference: {abs(reward_data['forward_offset'] - initial_reward):.6f}")
            print(f"  Final yaw: {yaw_angle:.2f} rad ({yaw_angle * 180 / math.pi:.1f}°)")
            print(f"  Final pitch: {pitch_angle:.2f} rad ({pitch_angle * 180 / math.pi:.1f}°)")
            if abs(reward_data['forward_offset'] - initial_reward) < 0.001:
                print(f"  ✓ PASS: Reward is orientation-independent in yaw frame!")
            else:
                print(f"  ✗ FAIL: Reward changed with orientation!")
            print("="*60)


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

    # Test with the crawl pose (where COM should be forward)
    pose_path = "assets/crawl-pose.json"

    print(f"[INFO]: Loading pose: {pose_path}")
    
    run_orientation_test(sim, scene, pose_path)


if __name__ == "__main__":
    main()
    simulation_app.close()

