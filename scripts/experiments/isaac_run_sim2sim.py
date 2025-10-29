from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.devices import Se2Gamepad, Se2GamepadCfg

from g1_crawl.tasks.manager_based.g1_crawl.g1 import G1_STAND_CFG

import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import carb
import omni.appwindow


# ==== Policy list configuration ====
# Add your policy paths here to cycle through them with N/P keys
POLICY_PATHS = [
    "/home/logan/Projects/g1_crawl/logs/rsl_rl/g1-locomotion-sweep_z4AI/2025-10-29_09-14-56_rRPh/exported/policy.pt",
    "/home/logan/Projects/g1_crawl/logs/rsl_rl/g1-locomotion-sweep_z4AI/2025-10-29_09-45-41_9q-J/exported/policy.pt",
    "/home/logan/Projects/g1_crawl/logs/rsl_rl/g1-locomotion-sweep_z4AI/2025-10-29_10-16-35_mSMn/exported/policy.pt",
    "/home/logan/Projects/g1_crawl/logs/rsl_rl/g1-locomotion-sweep_z4AI/2025-10-29_10-48-16_Yk3B/exported/policy.pt",
    # "/home/logan/Projects/g1_crawl/logs/rsl_rl/g1-locomotion-sweep_NC2K/2025-10-28_22-57-00_qJdS/exported/policy.pt",
    # "/home/logan/Projects/g1_crawl/logs/rsl_rl/g1-locomotion-sweep_NC2K/2025-10-28_20-52-53_n-Km/exported/policy.pt",
    # "deployment/policies/policy_crawl.pt",
    # "deployment/policies/policy_crawl_start.pt",
    # "deployment/policies/policy_shamble.pt",
]

# ==== Command range configuration (easy to tweak) ====
# Raw SE2 gamepad outputs are clamped to these ranges before mapping to the policy inputs.
CMD_MIN_VX = -1.0
CMD_MAX_VX = 1.0
CMD_MIN_VY = -1.0
CMD_MAX_VY = 1.0
CMD_MIN_OMEGA_Z = -1.0
CMD_MAX_OMEGA_Z = 1.0


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    half_r = 0.5 * roll
    half_p = 0.5 * pitch
    half_y = 0.5 * yaw

    cr = math.cos(half_r)
    sr = math.sin(half_r)
    cp = math.cos(half_p)
    sp = math.sin(half_p)
    cy = math.cos(half_y)
    sy = math.sin(half_y)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = torch.tensor([w, x, y, z], dtype=torch.float32)
    quat = quat / torch.linalg.norm(quat)
    return quat


def load_pose_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "poses" not in data or not data["poses"]:
        raise ValueError(f"Invalid pose JSON: expected key 'poses' with at least one entry: {path}")
    pose = data["poses"][0]
    # Basic validation
    for key in ("base_pos", "base_rpy", "joints"):
        if key not in pose:
            raise KeyError(f"Pose JSON missing '{key}' field: {path}")
    return pose


def build_joint_name_to_index_map(asset) -> Dict[str, int]:
    name_to_index: Dict[str, int] = {}
    for i, n in enumerate(asset.data.joint_names):
        name_to_index[str(n)] = int(i)
    return name_to_index


def apply_pose(scene: InteractiveScene, pose: Dict, name_to_index: Dict[str, int], zero_velocities: bool = True) -> None:
    device = scene["Robot"].device

    base_x, base_y, base_z = pose["base_pos"]
    base_r, base_p, base_yaw = pose["base_rpy"]
    wxyz = rpy_to_quat_wxyz(float(base_r), float(base_p), float(base_yaw)).to(device=device)

    origin = scene.env_origins.to(device=device)[0] if hasattr(scene, "env_origins") else torch.zeros(3, device=device)
    new_root_state = torch.zeros(1, 13, device=device)
    base_pos_tensor = torch.tensor([float(base_x), float(base_y), float(base_z)], device=device, dtype=torch.float32)
    new_root_state[0, :3] = base_pos_tensor + origin
    new_root_state[0, 3:7] = wxyz
    if zero_velocities:
        new_root_state[0, 7:13] = 0.0
    else:
        new_root_state[0, 7:13] = scene["Robot"].data.root_state_w[0, 7:13]
    scene["Robot"].write_root_pose_to_sim(new_root_state[:, :7])
    scene["Robot"].write_root_velocity_to_sim(new_root_state[:, 7:])

    joint_positions = scene["Robot"].data.default_joint_pos.clone().to(device=device)
    joint_velocities = torch.zeros_like(joint_positions) if zero_velocities else scene["Robot"].data.joint_vel.clone().to(device=device)

    for joint_name, value in pose["joints"].items():
        if joint_name not in name_to_index:
            raise KeyError(f"Joint '{joint_name}' not found in robot articulation. Available: {list(name_to_index.keys())}")
        j_idx = name_to_index[joint_name]
        joint_positions[0, j_idx] = float(value)

    scene["Robot"].write_joint_state_to_sim(joint_positions, joint_velocities)
    scene.write_data_to_sim()


def create_scene_cfg():
    class SimpleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )
        Robot = G1_STAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=G1_STAND_CFG.spawn.replace(
                rigid_props=G1_STAND_CFG.spawn.rigid_props.replace(disable_gravity=False),
                articulation_props=G1_STAND_CFG.spawn.articulation_props.replace(fix_root_link=False),
            ),
        )

    return SimpleSceneCfg


class IsaacPolicyController:
    """Minimal Isaac policy controller with joint remapping and clamping.

    Expects policy to take a flat observation tensor and output 23 joint targets
    in the PyTorch training order. We remap to Isaac articulation order.
    """

    PYTORCH_JOINT_ORDER: List[str] = [
        'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
        'left_hip_roll_joint', 'right_hip_roll_joint',
        'left_hip_yaw_joint', 'right_hip_yaw_joint',
        'left_knee_joint', 'right_knee_joint',
        'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
        'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
        'left_ankle_roll_joint', 'right_ankle_roll_joint',
        'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
        'left_elbow_joint', 'right_elbow_joint',
        'left_wrist_roll_joint', 'right_wrist_roll_joint',
    ]

    def __init__(
        self,
        scene: InteractiveScene,
        policy_path: Path,
        device: str = "cuda",
        action_scale: float = 0.5,
        control_dt: float = 0.02,
        suppress_clamp_warnings: bool = False,
    ) -> None:
        self.scene = scene
        self.device = torch.device(device)
        self.action_scale = float(action_scale)
        self._control_dt = float(control_dt)
        self._suppress_clamp_warnings = bool(suppress_clamp_warnings)

        # Isaac articulation joint list (string names) in Isaac order
        self.joint_names_isaac: List[str] = [str(n) for n in scene["Robot"].data.joint_names]
        self.num_robot_joints: int = len(self.joint_names_isaac)

        # Build remap indices: pytorch index -> isaac index (or -1 if missing)
        self.pytorch_to_isaac_idx: List[int] = []
        for jt in self.PYTORCH_JOINT_ORDER:
            if jt in self.joint_names_isaac:
                self.pytorch_to_isaac_idx.append(self.joint_names_isaac.index(jt))
            else:
                self.pytorch_to_isaac_idx.append(-1)

        # Defaults and limits from Isaac
        self.default_joint_pos_isaac = scene["Robot"].data.default_joint_pos[0].to(self.device)
        self.joint_limits_lower = scene["Robot"].data.joint_pos_limits[0, :, 0].to(self.device)
        self.joint_limits_upper = scene["Robot"].data.joint_pos_limits[0, :, 1].to(self.device)

        # Create default in pytorch order for observation relative encoding
        default_pos_pytorch = torch.zeros(len(self.PYTORCH_JOINT_ORDER), device=self.device, dtype=torch.float32)
        for p_idx, isa_idx in enumerate(self.pytorch_to_isaac_idx):
            if 0 <= isa_idx < len(self.default_joint_pos_isaac):
                default_pos_pytorch[p_idx] = self.default_joint_pos_isaac[isa_idx]
        self.default_joint_pos_pytorch = default_pos_pytorch

        # Load policy
        print(f"[INFO] Loading policy: {policy_path}")
        self._policy = torch.load(policy_path, map_location=self.device, weights_only=False)
        self._policy.eval()

        # Last actions for observation (zeros to start)
        self.last_actions_pytorch = torch.zeros(len(self.PYTORCH_JOINT_ORDER), device=self.device)

        # Safety monitoring state and thresholds (match standalone runner)
        self._prev_joint_pos = None
        self._prev_joint_vel = None
        self._safety_initialized = False
        self._max_position_jump = 0.3
        self._max_velocity = 25.0
        self._max_acceleration = 1500.0
        self._max_action_magnitude = 5.0

    def _isaac_to_pytorch(self, isaac_vec: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(len(self.PYTORCH_JOINT_ORDER), device=self.device)
        for p_idx, isa_idx in enumerate(self.pytorch_to_isaac_idx):
            if 0 <= isa_idx < len(isaac_vec):
                out[p_idx] = isaac_vec[isa_idx]
        return out

    def _pytorch_to_isaac(self, pytorch_vec: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.num_robot_joints, device=self.device)
        for p_idx, isa_idx in enumerate(self.pytorch_to_isaac_idx):
            if 0 <= isa_idx < self.num_robot_joints:
                out[isa_idx] = pytorch_vec[p_idx]
        return out

    def get_observation(self, lin_vel_z: float, lin_vel_y: float, ang_vel_x: float) -> torch.Tensor:
        robot = self.scene["Robot"]
        # Isaac provides current joint pos/vel in Isaac order
        joint_pos_isaac = robot.data.joint_pos[0].to(self.device)
        joint_vel_isaac = robot.data.joint_vel[0].to(self.device)

        # Projected gravity like training used (compute from root quat)
        wxyz = robot.data.root_quat_w[0].to(self.device)
        qw, qx, qy, qz = wxyz
        gravity_x = 2 * (-qz * qx + qw * qy)
        gravity_y = -2 * (qz * qy + qw * qx)
        gravity_z = 1 - 2 * (qw * qw + qz * qz)
        proj_grav = torch.stack([gravity_x, gravity_y, gravity_z], dim=0)

        # Command vector
        vel_cmd = torch.tensor([lin_vel_z, lin_vel_y, ang_vel_x], device=self.device, dtype=torch.float32)

        # Remap to pytorch order
        joint_pos_pytorch = self._isaac_to_pytorch(joint_pos_isaac)
        joint_vel_pytorch = self._isaac_to_pytorch(joint_vel_isaac)

        # Relative positions
        joint_pos_rel = joint_pos_pytorch - self.default_joint_pos_pytorch

        obs = torch.cat([
            proj_grav,
            vel_cmd,
            joint_pos_rel,
            joint_vel_pytorch,
            self.last_actions_pytorch,
        ], dim=-1)
        return obs

    def get_targets_isaac(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions_pytorch = self._policy(obs)

        self.last_actions_pytorch = actions_pytorch.clone()

        # Safety check against current measured state
        robot = self.scene["Robot"]
        joint_pos_isaac = robot.data.joint_pos[0].to(self.device)
        joint_vel_isaac = robot.data.joint_vel[0].to(self.device)
        self._check_safety_limits(joint_pos_isaac, joint_vel_isaac, actions_pytorch)

        # Convert to isaac order and scale relative to defaults
        actions_isaac = self._pytorch_to_isaac(actions_pytorch)
        targets_unclamped = actions_isaac * self.action_scale + self.default_joint_pos_isaac
        targets = torch.clamp(targets_unclamped, self.joint_limits_lower, self.joint_limits_upper)

        # Warn on clamping
        clamped_mask = targets != targets_unclamped
        if (not self._suppress_clamp_warnings) and clamped_mask.any():
            clamped_indices = torch.where(clamped_mask)[0].detach().cpu().numpy()
            print("WARNING: Clamping motor targets for joints:")
            for idx in clamped_indices:
                joint_name = self.joint_names_isaac[int(idx)]
                unclamped_val = targets_unclamped[int(idx)].item()
                clamped_val = targets[int(idx)].item()
                lower_limit = self.joint_limits_lower[int(idx)].item()
                upper_limit = self.joint_limits_upper[int(idx)].item()
                print(f"  {joint_name}: {unclamped_val:.3f} -> {clamped_val:.3f} (limits: [{lower_limit:.3f}, {upper_limit:.3f}])")
        return targets

    def _check_safety_limits(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, actions_pytorch: torch.Tensor) -> bool:
        """Print LOUD warnings on position/velocity/acceleration spikes and action magnitude."""
        # Action magnitude check (in pytorch order)
        if torch.any(torch.abs(actions_pytorch) > self._max_action_magnitude):
            max_mag = torch.max(torch.abs(actions_pytorch)).item()
            print("\n" + "=" * 60)
            print("⚠️  SAFETY WARNING: LARGE ACTION MAGNITUDE! ⚠️")
            print("=" * 60)
            print(f"Max action magnitude: {max_mag:.3f} (threshold: {self._max_action_magnitude:.3f})")
            print("=" * 60 + "\n")

        # Initialize previous state on first call
        jp = joint_pos.detach().cpu().numpy()
        jv = joint_vel.detach().cpu().numpy()
        if not self._safety_initialized:
            self._prev_joint_pos = jp.copy()
            self._prev_joint_vel = jv.copy()
            self._safety_initialized = True
            return True

        # Position jump check
        position_delta = abs(jp - self._prev_joint_pos)
        max_position_delta = position_delta.max()
        position_jump_joint = int(position_delta.argmax())
        if max_position_delta > self._max_position_jump:
            joint_name = self.joint_names_isaac[position_jump_joint]
            print("\n" + "=" * 60)
            print("⚠️  SAFETY VIOLATION: POSITION JUMP DETECTED! ⚠️")
            print("=" * 60)
            print(f"{joint_name} (joint {position_jump_joint}) jumped {max_position_delta:.3f} rad in one timestep")
            print(f"(threshold: {self._max_position_jump:.3f} rad)")
            print(f"Current pos: {jp[position_jump_joint]:.3f}, Previous: {self._prev_joint_pos[position_jump_joint]:.3f}")
            print("⚠️  THIS WOULD TRIGGER DAMPED MODE ON REAL ROBOT! ⚠️")
            print("=" * 60 + "\n")

        # Velocity spike check
        velocity_magnitude = abs(jv)
        max_velocity = velocity_magnitude.max()
        velocity_spike_joint = int(velocity_magnitude.argmax())
        if max_velocity > self._max_velocity:
            joint_name = self.joint_names_isaac[velocity_spike_joint]
            print("\n" + "=" * 60)
            print("⚠️  SAFETY VIOLATION: VELOCITY SPIKE DETECTED! ⚠️")
            print("=" * 60)
            print(f"{joint_name} (joint {velocity_spike_joint}) velocity: {jv[velocity_spike_joint]:.3f} rad/s")
            print(f"(threshold: {self._max_velocity:.3f} rad/s)")
            print("⚠️  THIS WOULD TRIGGER DAMPED MODE ON REAL ROBOT! ⚠️")
            print("=" * 60 + "\n")

        # Acceleration spike check
        acceleration = (jv - self._prev_joint_vel) / max(self._control_dt, 1e-6)
        acc_abs = abs(acceleration)
        max_acceleration = acc_abs.max()
        acceleration_spike_joint = int(acc_abs.argmax())
        if max_acceleration > self._max_acceleration:
            joint_name = self.joint_names_isaac[acceleration_spike_joint]
            print("\n" + "=" * 60)
            print("⚠️  SAFETY VIOLATION: ACCELERATION SPIKE DETECTED! ⚠️")
            print("=" * 60)
            print(f"{joint_name} (joint {acceleration_spike_joint}) acceleration: {acceleration[acceleration_spike_joint]:.1f} rad/s^2")
            print(f"(threshold: {self._max_acceleration:.1f} rad/s^2)")
            print(f"Velocity changed from {self._prev_joint_vel[acceleration_spike_joint]:.3f} to {jv[acceleration_spike_joint]:.3f} rad/s")
            print("⚠️  THIS WOULD TRIGGER DAMPED MODE ON REAL ROBOT! ⚠️")
            print("=" * 60 + "\n")

        # Update previous state
        self._prev_joint_pos = jp.copy()
        self._prev_joint_vel = jv.copy()
        return True


def run_multi_policy(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    policy_paths: List[str],
    use_gamepad: bool = True,
    pose_path: Optional[str] = None,
    debug_gamepad: bool = False,
    suppress_clamp_warnings: bool = False,
) -> None:
    """Run multiple policies with keyboard cycling support.
    
    Keyboard controls:
    - N: Next policy
    - P: Previous policy
    - R: Reset to initial pose
    """
    
    # Load initial pose if provided
    initial_pose: Optional[dict] = None
    name_to_index: Optional[Dict[str, int]] = None
    if pose_path is not None:
        try:
            initial_pose = load_pose_json(pose_path)
            name_to_index = build_joint_name_to_index_map(scene["Robot"])
            apply_pose(scene, initial_pose, name_to_index, zero_velocities=True)
            print(f"[INFO] Applied initial pose from: {pose_path}")
        except Exception as e:
            print(f"[WARN] Failed to apply initial pose '{pose_path}': {e}")
    
    # Initialize gamepad
    sim_dt = sim.get_physics_dt()
    lin_vel_z = 0.0
    lin_vel_y = 0.0
    ang_vel_x = 0.0
    
    gamepad = None
    if use_gamepad:
        try:
            gamepad = Se2Gamepad(
                Se2GamepadCfg(
                    v_x_sensitivity=1.5,
                    v_y_sensitivity=1.5,
                    omega_z_sensitivity=1.5,
                    dead_zone=0.05,
                    sim_device=scene["Robot"].device,
                )
            )
            print("[INFO] SE2 Gamepad initialized (left stick: x/y, right stick: yaw)")
        except Exception as e:
            print(f"[WARN] Gamepad init failed: {e}. Continuing without gamepad...")
            gamepad = None
    elif debug_gamepad:
        print("[WARN] --debug-gamepad specified but gamepad disabled with --no-gamepad")
    
    # Policy cycling state
    current_policy_idx = 0
    controller = None
    switch_policy_requested = True  # Load initial policy
    
    def load_policy(idx: int) -> IsaacPolicyController:
        """Load and initialize a policy controller."""
        policy_path = policy_paths[idx]
        print("\n" + "=" * 80)
        print(f"🔄 LOADING POLICY {idx + 1}/{len(policy_paths)}")
        print("=" * 80)
        print(f"Policy: {policy_path}")
        print("=" * 80 + "\n")
        
        ctrl = IsaacPolicyController(
            scene=scene,
            policy_path=Path(policy_path),
            device="cuda",
            action_scale=0.5,
            control_dt=sim.get_physics_dt(),
            suppress_clamp_warnings=suppress_clamp_warnings,
        )
        
        # Reset to initial pose when switching
        if initial_pose is not None and name_to_index is not None:
            apply_pose(scene, initial_pose, name_to_index, zero_velocities=True)
        
        return ctrl
    
    # Keyboard setup
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keys_pressed = {"R": False, "N": False, "P": False}
    
    def on_keyboard_event(event):
        nonlocal keys_pressed, current_policy_idx, switch_policy_requested, initial_pose, name_to_index
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            
            # Reset pose
            if key == "R" and not keys_pressed["R"]:
                keys_pressed["R"] = True
                if initial_pose is not None and name_to_index is not None:
                    print("[INFO] Resetting robot to initial pose (zero velocities)...")
                    apply_pose(scene, initial_pose, name_to_index, zero_velocities=True)
                else:
                    print("[WARN] No initial pose loaded; provide --pose to enable reset.")
            
            # Next policy
            elif key == "N" and not keys_pressed["N"] and len(policy_paths) > 1:
                keys_pressed["N"] = True
                current_policy_idx = (current_policy_idx + 1) % len(policy_paths)
                switch_policy_requested = True
                print(f"[INFO] Switching to next policy ({current_policy_idx + 1}/{len(policy_paths)})...")
            
            # Previous policy
            elif key == "P" and not keys_pressed["P"] and len(policy_paths) > 1:
                keys_pressed["P"] = True
                current_policy_idx = (current_policy_idx - 1) % len(policy_paths)
                switch_policy_requested = True
                print(f"[INFO] Switching to previous policy ({current_policy_idx + 1}/{len(policy_paths)})...")
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in keys_pressed:
                keys_pressed[event.input.name] = False
    
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    # Print initial info
    print("[INFO] Running multi-policy Isaac controller")
    print(f"[INFO] Loaded {len(policy_paths)} policies:")
    for i, p in enumerate(policy_paths, 1):
        print(f"  {i}. {p}")
    print("\n" + "=" * 60)
    print("CONTROLS")
    print("=" * 60)
    print("  N - Next policy")
    print("  P - Previous policy")
    print("  R - Reset to initial pose")
    if use_gamepad:
        print("  Left stick - Forward/lateral movement")
        print("  Right stick - Yaw rotation")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("SAFETY MONITORING ENABLED")
    print("=" * 60)
    print("Real robot safety thresholds are active:")
    print(f"  - Position jump: Max 0.30 rad/timestep")
    print(f"  - Velocity spike: Max 25.0 rad/s")
    print(f"  - Acceleration spike: Max 1500.0 rad/s²")
    print("\nLOUD warnings will appear if policy would trigger safety shutoff!")
    print("=" * 60 + "\n")
    
    if debug_gamepad:
        print("[DEBUG] Gamepad debug enabled: printing inputs every 20 steps")
    
    step = 0
    while simulation_app.is_running():
        # Switch policy if requested
        if switch_policy_requested:
            controller = load_policy(current_policy_idx)
            switch_policy_requested = False
        
        # Update velocity commands from gamepad if available
        if gamepad is not None:
            cmd = gamepad.advance()
            raw_vx = float(cmd[0].item())
            raw_vy = float(cmd[1].item())
            raw_omega = float(cmd[2].item())
            vx = _clamp(raw_vx, CMD_MIN_VX, CMD_MAX_VX)
            vy = _clamp(raw_vy, CMD_MIN_VY, CMD_MAX_VY)
            omega = _clamp(raw_omega, CMD_MIN_OMEGA_Z, CMD_MAX_OMEGA_Z)
            lin_vel_z = vx
            lin_vel_y = vy
            ang_vel_x = omega
            if debug_gamepad and (step % 20 == 0):
                print(
                    f"[DEBUG] Gamepad raw: v_x={raw_vx:.3f}, v_y={raw_vy:.3f}, omega_z={raw_omega:.3f} -> "
                    f"clamped: v_x={vx:.3f}, v_y={vy:.3f}, omega_z={omega:.3f} -> "
                    f"mapped: lin_vel_z={lin_vel_z:.3f}, lin_vel_y={lin_vel_y:.3f}, ang_vel_x={ang_vel_x:.3f}"
                )
        
        obs = controller.get_observation(lin_vel_z, lin_vel_y, ang_vel_x)
        targets = controller.get_targets_isaac(obs)
        
        scene["Robot"].set_joint_position_target(targets.unsqueeze(0))
        scene.write_data_to_sim()
        
        sim.step()
        scene.update(sim_dt)
        step += 1
    
    # Cleanup
    try:
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run policies in Isaac with policy cycling support. Use N/P keys to cycle through policies. "
                    "Edit POLICY_PATHS at the top of this file to configure which policies to test."
    )
    parser.add_argument(
        "--policy", 
        type=str, 
        action="append",
        help="(Optional) Override POLICY_PATHS - specify policy .pt file(s) via CLI instead"
    )
    parser.add_argument("--pose", type=str, default="assets/default-pose.json", help="Path to pose JSON for initial/reset pose")
    parser.add_argument("--no-gamepad", action="store_true", help="Disable SE2 gamepad")
    parser.add_argument("--debug-gamepad", action="store_true", help="Print gamepad inputs and mapped commands (throttled)")
    parser.add_argument("--no-clamp-warnings", action="store_true", help="Silence clamping warnings for motor targets")
    args = parser.parse_args()

    # Use POLICY_PATHS from top of file by default, or CLI override if provided
    if args.policy is None or len(args.policy) == 0:
        policy_paths = POLICY_PATHS
    else:
        policy_paths = args.policy

    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg_class = create_scene_cfg()
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    pose_path = args.pose
    print(f"[INFO]: Using pose: {pose_path}")
    run_multi_policy(
        sim,
        scene,
        policy_paths,
        use_gamepad=(not args.no_gamepad),
        pose_path=pose_path,
        debug_gamepad=args.debug_gamepad,
        suppress_clamp_warnings=args.no_clamp_warnings,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()


