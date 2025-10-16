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
import torch
import os

# Optional keyboard input handling and UI
import carb
import omni
import omni.ui as ui


# Reward computation parameters (adjust these to match your config)
REWARD_STD = 0.5
REWARD_WEIGHT_L1 = -0.5
REWARD_WEIGHT_EXP = 2.0


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


def load_pose_json(path: str) -> Dict:
    """Load pose from JSON file (same format as pose_viewer.py)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _assert_keys(data, ["poses"], context="pose json root")
    poses = data["poses"]
    if not isinstance(poses, list) or len(poses) == 0:
        raise ValueError("'poses' must be a non-empty list")
    pose0 = poses[0]
    _assert_keys(pose0, ["base_pos", "base_rpy", "joints"], context="pose entry")
    return {
        "base_pos": pose0["base_pos"], 
        "base_rpy": pose0["base_rpy"], 
        "joints": pose0["joints"]
    }


def interpolate_poses(pose_a: Dict, pose_b: Dict, alpha: float) -> Dict:
    """Linearly interpolate between two poses.
    
    Args:
        pose_a: First pose (alpha=0)
        pose_b: Second pose (alpha=1)
        alpha: Interpolation factor [0, 1]
    
    Returns:
        Interpolated pose
    """
    alpha = max(0.0, min(1.0, alpha))
    
    # Interpolate base position
    base_pos = [
        (1 - alpha) * pose_a["base_pos"][i] + alpha * pose_b["base_pos"][i]
        for i in range(3)
    ]
    
    # Interpolate base orientation (simple linear for RPY)
    base_rpy = [
        (1 - alpha) * pose_a["base_rpy"][i] + alpha * pose_b["base_rpy"][i]
        for i in range(3)
    ]
    
    # Interpolate joint positions
    joints = {}
    for joint_name in pose_a["joints"]:
        if joint_name not in pose_b["joints"]:
            raise KeyError(f"Joint '{joint_name}' not in pose_b")
        joints[joint_name] = (
            (1 - alpha) * pose_a["joints"][joint_name] + 
            alpha * pose_b["joints"][joint_name]
        )
    
    return {
        "base_pos": base_pos,
        "base_rpy": base_rpy,
        "joints": joints
    }


def build_joint_name_to_index_map(asset) -> Dict[str, int]:
    joint_names = [str(n) for n in asset.data.joint_names]
    name_to_index = {name: int(i) for i, name in enumerate(joint_names)}
    return name_to_index


def apply_pose(scene: InteractiveScene, pose: Dict, name_to_index: Dict[str, int]) -> None:
    """Apply a pose to the robot in the scene."""
    device = scene["Robot"].device

    # Root pose
    base_x, base_y, base_z = pose["base_pos"]
    base_r, base_p, base_y = pose["base_rpy"]
    wxyz = rpy_to_quat_wxyz(float(base_r), float(base_p), float(base_y)).to(device=device)

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


def compute_rewards(scene: InteractiveScene, 
                     pose_crawl: Dict, 
                     pose_stand: Dict,
                     name_to_index: Dict[str, int]) -> Dict[str, float]:
    """Compute the reward terms for the current robot state.
    
    Returns dict with:
        - l1_dev_cmd0: L1 deviation when command=0 (target=crawl)
        - l1_dev_cmd1: L1 deviation when command=1 (target=stand)
        - exp_prox_cmd0: Exponential proximity when command=0 (target=crawl)
        - exp_prox_cmd1: Exponential proximity when command=1 (target=stand)
        - weighted_l1_cmd0: L1 with weight -0.5 for cmd=0
        - weighted_l1_cmd1: L1 with weight -0.5 for cmd=1
        - weighted_exp_cmd0: Exponential with weight 2.0 for cmd=0
        - weighted_exp_cmd1: Exponential with weight 2.0 for cmd=1
        - total_cmd0: Total reward if command=0
        - total_cmd1: Total reward if command=1
    """
    device = scene["Robot"].device
    
    # Get current joint positions [1, num_joints]
    current_joint_pos = scene["Robot"].data.joint_pos
    
    # Build target vectors for both poses
    def pose_to_tensor(pose: Dict) -> torch.Tensor:
        """Convert pose dict to joint position tensor in correct order."""
        values = []
        for name in scene["Robot"].data.joint_names:
            name_str = str(name)
            if name_str not in pose["joints"]:
                raise KeyError(f"Joint '{name_str}' not in pose")
            values.append(pose["joints"][name_str])
        return torch.tensor(values, dtype=current_joint_pos.dtype, device=device).unsqueeze(0)
    
    target_crawl = pose_to_tensor(pose_crawl)
    target_stand = pose_to_tensor(pose_stand)
    
    # L1 deviations (sum of absolute differences)
    l1_dev_crawl = torch.sum(torch.abs(current_joint_pos - target_crawl)).item()
    l1_dev_stand = torch.sum(torch.abs(current_joint_pos - target_stand)).item()
    
    # Exponential proximity
    err_sq_crawl = torch.sum(torch.square(current_joint_pos - target_crawl)).item()
    err_sq_stand = torch.sum(torch.square(current_joint_pos - target_stand)).item()
    exp_prox_crawl = float(torch.exp(torch.tensor(-err_sq_crawl / (REWARD_STD ** 2))).item())
    exp_prox_stand = float(torch.exp(torch.tensor(-err_sq_stand / (REWARD_STD ** 2))).item())
    
    # Apply weights
    weighted_l1_crawl = REWARD_WEIGHT_L1 * l1_dev_crawl
    weighted_l1_stand = REWARD_WEIGHT_L1 * l1_dev_stand
    weighted_exp_crawl = REWARD_WEIGHT_EXP * exp_prox_crawl
    weighted_exp_stand = REWARD_WEIGHT_EXP * exp_prox_stand
    
    total_crawl = weighted_l1_crawl + weighted_exp_crawl
    total_stand = weighted_l1_stand + weighted_exp_stand
    
    return {
        "l1_dev_cmd0": l1_dev_crawl,
        "l1_dev_cmd1": l1_dev_stand,
        "exp_prox_cmd0": exp_prox_crawl,
        "exp_prox_cmd1": exp_prox_stand,
        "weighted_l1_cmd0": weighted_l1_crawl,
        "weighted_l1_cmd1": weighted_l1_stand,
        "weighted_exp_cmd0": weighted_exp_crawl,
        "weighted_exp_cmd1": weighted_exp_stand,
        "total_cmd0": total_crawl,
        "total_cmd1": total_stand,
    }


def run_interpolation_viewer(sim: sim_utils.SimulationContext, 
                              scene: InteractiveScene, 
                              pose_a_path: str,
                              pose_b_path: str) -> None:
    """Main viewer loop with interpolation control."""
    
    pose_a = load_pose_json(pose_a_path)
    pose_b = load_pose_json(pose_b_path)
    name_to_index = build_joint_name_to_index_map(scene["Robot"])
    
    # Interpolation state
    alpha = 0.0  # 0 = pose_a (crawl), 1 = pose_b (stand)
    alpha_step = 0.01
    auto_play = False
    auto_play_speed = 0.005
    
    # UI Window for displaying rewards
    reward_window = ui.Window("Reward Analysis", width=600, height=500)
    reward_labels = {}
    
    with reward_window.frame:
        with ui.VStack(spacing=10, style={"margin": 10}):
            ui.Label("Pose Interpolation Viewer", style={"font_size": 20, "color": 0xFFFFFFFF})
            ui.Spacer(height=5)
            
            # Alpha display
            reward_labels["alpha"] = ui.Label(f"Alpha (interpolation): {alpha:.3f}", 
                                              style={"font_size": 16, "color": 0xFFFFFF00})
            ui.Label("  0.0 = Crawl pose | 1.0 = Stand pose", 
                    style={"font_size": 12, "color": 0xFFAAAAAA})
            ui.Spacer(height=10)
            
            ui.Label("=== Command 0 (Crawling) ===", style={"font_size": 16, "color": 0xFFFF8800})
            reward_labels["l1_cmd0"] = ui.Label("L1 Deviation: --", style={"font_size": 14})
            reward_labels["l1_weighted_cmd0"] = ui.Label("  Weighted (×-0.5): --", style={"font_size": 12, "color": 0xFFAAAAFF})
            reward_labels["exp_cmd0"] = ui.Label("Exp Proximity: --", style={"font_size": 14})
            reward_labels["exp_weighted_cmd0"] = ui.Label("  Weighted (×2.0): --", style={"font_size": 12, "color": 0xFFAAAAFF})
            reward_labels["total_cmd0"] = ui.Label("TOTAL: --", style={"font_size": 16, "color": 0xFF00FF00})
            
            ui.Spacer(height=15)
            
            ui.Label("=== Command 1 (Standing) ===", style={"font_size": 16, "color": 0xFF0088FF})
            reward_labels["l1_cmd1"] = ui.Label("L1 Deviation: --", style={"font_size": 14})
            reward_labels["l1_weighted_cmd1"] = ui.Label("  Weighted (×-0.5): --", style={"font_size": 12, "color": 0xFFAAAAFF})
            reward_labels["exp_cmd1"] = ui.Label("Exp Proximity: --", style={"font_size": 14})
            reward_labels["exp_weighted_cmd1"] = ui.Label("  Weighted (×2.0): --", style={"font_size": 12, "color": 0xFFAAAAFF})
            reward_labels["total_cmd1"] = ui.Label("TOTAL: --", style={"font_size": 16, "color": 0xFF00FF00})
            
            ui.Spacer(height=15)
            
            ui.Label("Controls:", style={"font_size": 14, "color": 0xFFFFFF00})
            ui.Label("  LEFT/RIGHT: Adjust alpha (±0.01)", style={"font_size": 12})
            ui.Label("  UP/DOWN: Adjust alpha (±0.1)", style={"font_size": 12})
            ui.Label("  SPACE: Toggle auto-play", style={"font_size": 12})
            ui.Label("  R: Reset to alpha=0", style={"font_size": 12})
            ui.Label("  ESC: Exit", style={"font_size": 12})
    
    def update_ui(rewards: Dict[str, float], alpha_val: float) -> None:
        """Update UI labels with current reward values."""
        reward_labels["alpha"].text = f"Alpha (interpolation): {alpha_val:.3f}"
        
        # Command 0
        reward_labels["l1_cmd0"].text = f"L1 Deviation: {rewards['l1_dev_cmd0']:.4f}"
        reward_labels["l1_weighted_cmd0"].text = f"  Weighted (×-0.5): {rewards['weighted_l1_cmd0']:.4f}"
        reward_labels["exp_cmd0"].text = f"Exp Proximity: {rewards['exp_prox_cmd0']:.4f}"
        reward_labels["exp_weighted_cmd0"].text = f"  Weighted (×2.0): {rewards['weighted_exp_cmd0']:.4f}"
        reward_labels["total_cmd0"].text = f"TOTAL: {rewards['total_cmd0']:.4f}"
        
        # Command 1
        reward_labels["l1_cmd1"].text = f"L1 Deviation: {rewards['l1_dev_cmd1']:.4f}"
        reward_labels["l1_weighted_cmd1"].text = f"  Weighted (×-0.5): {rewards['weighted_l1_cmd1']:.4f}"
        reward_labels["exp_cmd1"].text = f"Exp Proximity: {rewards['exp_prox_cmd1']:.4f}"
        reward_labels["exp_weighted_cmd1"].text = f"  Weighted (×2.0): {rewards['weighted_exp_cmd1']:.4f}"
        reward_labels["total_cmd1"].text = f"TOTAL: {rewards['total_cmd1']:.4f}"
    
    # Keyboard handling
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    
    keys_pressed = {"LEFT": False, "RIGHT": False, "UP": False, "DOWN": False, 
                    "SPACE": False, "R": False, "ESCAPE": False}
    
    def on_keyboard_event(event):
        nonlocal alpha, auto_play, keys_pressed
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            
            if key == "LEFT" and not keys_pressed["LEFT"]:
                keys_pressed["LEFT"] = True
                alpha = max(0.0, alpha - alpha_step)
                print(f"Alpha: {alpha:.3f}")
            elif key == "RIGHT" and not keys_pressed["RIGHT"]:
                keys_pressed["RIGHT"] = True
                alpha = min(1.0, alpha + alpha_step)
                print(f"Alpha: {alpha:.3f}")
            elif key == "UP" and not keys_pressed["UP"]:
                keys_pressed["UP"] = True
                alpha = min(1.0, alpha + 0.1)
                print(f"Alpha: {alpha:.3f}")
            elif key == "DOWN" and not keys_pressed["DOWN"]:
                keys_pressed["DOWN"] = True
                alpha = max(0.0, alpha - 0.1)
                print(f"Alpha: {alpha:.3f}")
            elif key == "SPACE" and not keys_pressed["SPACE"]:
                keys_pressed["SPACE"] = True
                auto_play = not auto_play
                print(f"Auto-play: {'ON' if auto_play else 'OFF'}")
            elif key == "R" and not keys_pressed["R"]:
                keys_pressed["R"] = True
                alpha = 0.0
                print(f"Reset to alpha=0.0")
            elif key == "ESCAPE" and not keys_pressed["ESCAPE"]:
                keys_pressed["ESCAPE"] = True
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in keys_pressed:
                keys_pressed[event.input.name] = False
    
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    print("\n" + "="*60)
    print("Reward Interpolation Viewer")
    print("="*60)
    print(f"Pose A (alpha=0): {pose_a_path}")
    print(f"Pose B (alpha=1): {pose_b_path}")
    print("\nControls:")
    print("  LEFT/RIGHT: Adjust alpha ±0.01")
    print("  UP/DOWN: Adjust alpha ±0.1")
    print("  SPACE: Toggle auto-play")
    print("  R: Reset to alpha=0")
    print("  ESC: Exit")
    print("="*60 + "\n")
    
    try:
        sim_dt = sim.get_physics_dt()
        while simulation_app.is_running():
            if keys_pressed["ESCAPE"]:
                print("Exiting...")
                break
            
            # Auto-play
            if auto_play:
                alpha += auto_play_speed
                if alpha >= 1.0:
                    alpha = 0.0
            
            # Clamp alpha
            alpha = max(0.0, min(1.0, alpha))
            
            # Interpolate and apply pose
            interpolated_pose = interpolate_poses(pose_a, pose_b, alpha)
            apply_pose(scene, interpolated_pose, name_to_index)
            
            # Compute rewards
            rewards = compute_rewards(scene, pose_a, pose_b, name_to_index)
            
            # Update UI
            update_ui(rewards, alpha)
            
            # Step simulation
            sim.step()
            scene.update(sim_dt)
            
    finally:
        if keyboard_subscription:
            input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)
        reward_window.visible = False


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

    # Pose paths (from config)
    pose_a_path = "assets/crawl-pose.json"  # Command 0
    pose_b_path = "assets/default-pose.json"  # Command 1

    print(f"[INFO]: Loading poses...")
    print(f"  Pose A (Crawl): {pose_a_path}")
    print(f"  Pose B (Stand): {pose_b_path}")
    
    run_interpolation_viewer(sim, scene, pose_a_path, pose_b_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
