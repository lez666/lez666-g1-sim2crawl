#!/usr/bin/env python3
"""Standalone policy deployment using only standard MuJoCo (no mjlab).

This script runs trained policies using exported MJCF, requiring only:
- mujoco
- torch
- numpy
- glfw (for gamepad support)

Run export_mjcf.py first to generate the scene XML file.

USAGE:
1. Edit the CONFIG section below with your desired settings
2. Run: python deploy_standalone.py
"""

import json
import math
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch


# ============================================================================
# CONFIGURATION - Edit these variables to change behavior
# ============================================================================

CONFIG = {
    # Path to exported MJCF model
    "model_xml": "scene.xml",
    
    # Initial policy file (will be overridden by gamepad if connected)
    "policy_path": "policies/policy_shamble.pt",
    
    # Device to run on (cpu only for standalone)
    "device": "cpu",
    
    # Global action scale multiplier
    "action_scale": 0.5,
    
    # Number of simulation substeps per policy step
    "n_substeps": 4,
    
    # Path to JSON file with initial pose (or None to use default)
    "init_pose_json": "poses/default-pose.json",
    
    # === GAMEPAD SETTINGS ===
    
    # Enable gamepad control (requires glfw)
    "use_gamepad": True,
    
    # Max forward/backward velocity (m/s) - scaled by left stick Y
    "max_lin_vel": 2.0,
    
    # Max angular velocity (rad/s) - scaled by right stick X  
    "max_ang_vel": 1.5,
    
    # Gamepad face button to policy mapping
    # Buttons: 0=A (bottom), 1=B (right), 2=X (left), 3=Y (top)
    "button_policy_map": {
        0: "policies/policy_crawl.pt",      # A button
        1: "policies/policy_shamble.pt",    # B button
        2: "policies/policy_crawl_start.pt", # X button
        3: "policies/policy_shamble_start.pt", # Y button
    },
    
    # === CAMERA SETTINGS ===
    
    # Camera distance from robot (meters)
    "camera_distance": 3.0,
    
    # Camera azimuth angle (degrees, 0=front, 90=side)
    "camera_azimuth": 90,
    
    # Camera elevation angle (degrees, negative=look down)
    "camera_elevation": -20,
    
    # Track robot with camera (follows robot position)
    "camera_tracking": True,
}

# ============================================================================
# END CONFIGURATION
# ============================================================================


# PyTorch model expects joints in this specific order
PYTORCH_JOINT_ORDER = [
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
    'left_wrist_roll_joint', 'right_wrist_roll_joint'
]

# Training defaults (in MuJoCo joint order)
TRAINING_DEFAULT_ANGLES = {
    "left_hip_pitch_joint": -1.6796101123595506,
    "left_hip_roll_joint": 2.4180011235955052,
    "left_hip_yaw_joint": 1.2083865168539325,
    "left_knee_joint": 2.1130298764044944,
    "left_ankle_pitch_joint": 0.194143033707865,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -1.6796101123595506,
    "right_hip_roll_joint": -2.4180011235955052,
    "right_hip_yaw_joint": -1.2083865168539325,
    "right_knee_joint": 2.1130298764044944,
    "right_ankle_pitch_joint": 0.194143033707865,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 1.4578526315789473,
    "left_shoulder_roll_joint": 1.5778684210526317,
    "left_shoulder_yaw_joint": 1.4238245614035088,
    "left_elbow_joint": -0.3124709677419355,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 1.4578526315789473,
    "right_shoulder_roll_joint": -1.5778684210526317,
    "right_shoulder_yaw_joint": -1.4238245614035088,
    "right_elbow_joint": -0.3124709677419355,
    "right_wrist_roll_joint": 0.0,
}


class GamepadController:
    """Gamepad input handler using GLFW."""
    
    def __init__(self, button_policy_map: dict[int, str], max_lin_vel: float, max_ang_vel: float):
        self.button_policy_map = button_policy_map
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.joystick_id = glfw.JOYSTICK_1
        self.last_button_state = [0] * 16  # Track button states for edge detection
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Check for gamepad
        if not glfw.joystick_present(self.joystick_id):
            glfw.terminate()
            raise RuntimeError("No gamepad found")
        
        gamepad_name = glfw.get_joystick_name(self.joystick_id)
        print(f"[GAMEPAD] Connected: {gamepad_name}")
        
        if glfw.joystick_is_gamepad(self.joystick_id):
            print(f"[GAMEPAD] Using standard mapping: {glfw.get_gamepad_name(self.joystick_id)}")
        else:
            print("[GAMEPAD] Warning: Not a standard gamepad mapping")
        
        # Print button mapping
        print("\n[GAMEPAD] Face Button -> Policy Mapping:")
        button_names = {0: "A (bottom)", 1: "B (right)", 2: "X (left)", 3: "Y (top)"}
        for btn_id, policy_path in sorted(self.button_policy_map.items()):
            btn_name = button_names.get(btn_id, f"Button {btn_id}")
            policy_name = Path(policy_path).stem
            print(f"  {btn_name:12} -> {policy_name}")
        print()
    
    def get_velocity_commands(self) -> tuple[float, float]:
        """Get velocity commands from analog sticks.
        
        Returns:
            (lin_vel_z, ang_vel_x): Forward velocity and angular velocity
        """
        state = glfw.get_gamepad_state(self.joystick_id)
        if not state:
            return 0.0, 0.0
        
        # Left stick Y (inverted) for forward/backward
        # Stick axes are -1 (up) to 1 (down), so negate for intuitive control
        left_y = -state.axes[1] if len(state.axes) > 1 else 0.0
        lin_vel_z = left_y * self.max_lin_vel
        
        # Right stick X for rotation
        right_x = state.axes[2] if len(state.axes) > 2 else 0.0
        ang_vel_x = right_x * self.max_ang_vel
        
        # Apply deadzone
        deadzone = 0.1
        if abs(lin_vel_z) < deadzone:
            lin_vel_z = 0.0
        if abs(ang_vel_x) < deadzone:
            ang_vel_x = 0.0
        
        return lin_vel_z, ang_vel_x
    
    def check_policy_switch(self) -> str | None:
        """Check if a face button was pressed to switch policy.
        
        Returns:
            Policy path if button pressed, None otherwise
        """
        state = glfw.get_gamepad_state(self.joystick_id)
        if not state:
            return None
        
        # Check each mapped button for rising edge (0->1)
        for btn_id, policy_path in self.button_policy_map.items():
            if btn_id < len(state.buttons):
                current = state.buttons[btn_id]
                previous = self.last_button_state[btn_id]
                
                if current == 1 and previous == 0:  # Button just pressed
                    self.last_button_state[btn_id] = current
                    return policy_path
                
                self.last_button_state[btn_id] = current
        
        return None
    
    def cleanup(self):
        """Clean up GLFW resources."""
        glfw.terminate()


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll-pitch-yaw to quaternion (w, x, y, z)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def compute_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Project gravity into body frame from quaternion."""
    qw, qx, qy, qz = quat
    
    # Exact formula from training
    gravity_x = 2 * (-qz * qx + qw * qy)
    gravity_y = -2 * (qz * qy + qw * qx)
    gravity_z = 1 - 2 * (qw * qw + qz * qz)
    
    return np.array([gravity_x, gravity_y, gravity_z], dtype=np.float32)


class StandalonePolicyController:
    """Standalone policy controller using only numpy/torch (no mjlab)."""
    
    def __init__(
        self,
        policy_path: Path,
        mj_model: mujoco.MjModel,
        device: str = "cpu",
        action_scale: float = 0.5,
        training_defaults: dict[str, float] | None = None,
    ):
        self.mj_model = mj_model
        self.device = device
        self.action_scale = action_scale
        
        # Get joint names from MuJoCo model (skip free joint)
        self.joint_names = []
        for i in range(mj_model.njnt):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            # Skip free joint (floating base)
            if name and name not in ["floating_base_joint", "root", "rootJoint"]:
                self.joint_names.append(name)
        
        self.num_robot_joints = len(self.joint_names)
        self.num_policy_joints = 23
        
        print(f"[INFO] Found {self.num_robot_joints} actuated joints")
        
        # Build remapping indices
        self.mujoco_to_pytorch_idx = []
        for pytorch_joint in PYTORCH_JOINT_ORDER:
            if pytorch_joint in self.joint_names:
                mujoco_idx = self.joint_names.index(pytorch_joint)
                self.mujoco_to_pytorch_idx.append(mujoco_idx)
            else:
                self.mujoco_to_pytorch_idx.append(-1)
        
        # Get default joint positions
        if training_defaults is None:
            training_defaults = TRAINING_DEFAULT_ANGLES
        
        default_pos_mujoco = []
        for joint_name in self.joint_names:
            if joint_name in training_defaults:
                default_pos_mujoco.append(training_defaults[joint_name])
            else:
                print(f"[WARN] Joint {joint_name} not in training defaults, using 0.0")
                default_pos_mujoco.append(0.0)
        
        self.default_joint_pos_mujoco = torch.tensor(default_pos_mujoco, device=device, dtype=torch.float32)
        self.default_joint_pos_pytorch = self._remap_mujoco_to_pytorch(self.default_joint_pos_mujoco)
        
        # Initialize last actions
        self.last_actions_pytorch = torch.zeros(self.num_policy_joints, device=device, dtype=torch.float32)
        
        # Load policy
        print(f"[INFO] Loading policy from: {policy_path}")
        self._policy = torch.load(policy_path, map_location=device, weights_only=False)
        self._policy.eval()
        self._current_policy_path = policy_path
        print(f"[INFO] Policy loaded successfully")
    
    def load_policy(self, policy_path: Path) -> None:
        """Hot-swap to a different policy."""
        if policy_path == self._current_policy_path:
            return  # Already loaded
        
        print(f"[POLICY] Switching to: {policy_path.stem}")
        self._policy = torch.load(policy_path, map_location=self.device, weights_only=False)
        self._policy.eval()
        self._current_policy_path = policy_path
        
        # Reset last actions to avoid discontinuities
        self.last_actions_pytorch = torch.zeros(self.num_policy_joints, device=self.device, dtype=torch.float32)
    
    def _remap_mujoco_to_pytorch(self, mujoco_data: torch.Tensor) -> torch.Tensor:
        """Remap joint data from MuJoCo order to PyTorch order."""
        pytorch_data = torch.zeros(self.num_policy_joints, device=self.device, dtype=torch.float32)
        for pytorch_idx, mujoco_idx in enumerate(self.mujoco_to_pytorch_idx):
            if mujoco_idx >= 0 and mujoco_idx < len(mujoco_data):
                pytorch_data[pytorch_idx] = mujoco_data[mujoco_idx]
        return pytorch_data
    
    def _remap_pytorch_to_mujoco(self, pytorch_data: torch.Tensor) -> torch.Tensor:
        """Remap joint data from PyTorch order to MuJoCo order."""
        mujoco_data = torch.zeros(self.num_robot_joints, device=self.device, dtype=torch.float32)
        for pytorch_idx in range(min(len(pytorch_data), len(self.mujoco_to_pytorch_idx))):
            mujoco_idx = self.mujoco_to_pytorch_idx[pytorch_idx]
            if mujoco_idx >= 0 and mujoco_idx < self.num_robot_joints:
                mujoco_data[mujoco_idx] = pytorch_data[pytorch_idx]
        return mujoco_data
    
    def get_observation(
        self,
        projected_gravity: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        lin_vel_z: float,
        ang_vel_x: float,
    ) -> torch.Tensor:
        """Build observation tensor matching training format."""
        # Convert to torch
        proj_grav_t = torch.from_numpy(projected_gravity).to(self.device)
        joint_pos_t = torch.from_numpy(joint_pos).to(self.device)
        joint_vel_t = torch.from_numpy(joint_vel).to(self.device)
        
        velocity_commands = torch.tensor(
            [lin_vel_z, 0.0, ang_vel_x],
            device=self.device,
            dtype=torch.float32
        )
        
        # Remap to PyTorch order
        joint_pos_pytorch = self._remap_mujoco_to_pytorch(joint_pos_t)
        joint_vel_pytorch = self._remap_mujoco_to_pytorch(joint_vel_t)
        
        # Joint positions relative to default
        joint_pos_rel_pytorch = joint_pos_pytorch - self.default_joint_pos_pytorch
        
        # Stack observation
        obs = torch.cat([
            proj_grav_t,                    # (3,)
            velocity_commands,              # (3,)
            joint_pos_rel_pytorch,          # (23,)
            joint_vel_pytorch,              # (23,)
            self.last_actions_pytorch,      # (23,)
        ], dim=-1)
        
        return obs  # (75,)
    
    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        """Run policy to get actions."""
        with torch.no_grad():
            actions_pytorch = self._policy(obs)
        
        self.last_actions_pytorch = actions_pytorch.clone()
        
        # Convert to MuJoCo order
        actions_mujoco = self._remap_pytorch_to_mujoco(actions_pytorch)
        
        # Apply to joint targets
        targets = actions_mujoco * self.action_scale + self.default_joint_pos_mujoco
        
        return targets.cpu().numpy()


def load_initial_pose(json_path: Path, mj_model: mujoco.MjModel, data: mujoco.MjData):
    """Load initial pose from JSON file."""
    with open(json_path, 'r') as f:
        pose_data = json.load(f)
    
    pose = pose_data['poses'][0]
    base_pos = pose['base_pos']
    base_rpy = pose['base_rpy']
    joints = pose['joints']
    
    # Set base position
    data.qpos[0] = base_pos[0]
    data.qpos[1] = base_pos[1]
    data.qpos[2] = base_pos[2]
    
    # Set base orientation (quaternion)
    quat = rpy_to_quat(base_rpy[0], base_rpy[1], base_rpy[2])
    data.qpos[3:7] = quat
    
    # Set joint positions
    for i in range(mj_model.njnt):
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name and joint_name in joints:
            # Find joint position index (skip free joint positions)
            if i > 0:  # Free joint is index 0
                qpos_idx = 7 + (i - 1)  # 7 for free joint (3 pos + 4 quat)
                if qpos_idx < len(data.qpos):
                    data.qpos[qpos_idx] = joints[joint_name]
    
    mujoco.mj_forward(mj_model, data)


def main():
    """Main entry point for standalone deployment."""
    
    # Parse config
    model_xml = Path(CONFIG["model_xml"])
    policy_path = Path(CONFIG["policy_path"])
    device = CONFIG["device"]
    action_scale = CONFIG["action_scale"]
    n_substeps = CONFIG["n_substeps"]
    init_pose_json = Path(CONFIG["init_pose_json"]) if CONFIG["init_pose_json"] else None
    
    use_gamepad = CONFIG["use_gamepad"]
    max_lin_vel = CONFIG.get("max_lin_vel", 2.0)
    max_ang_vel = CONFIG.get("max_ang_vel", 1.5)
    button_policy_map = CONFIG.get("button_policy_map", {})
    
    # Camera settings
    camera_distance = CONFIG.get("camera_distance", 3.0)
    camera_azimuth = CONFIG.get("camera_azimuth", 90)
    camera_elevation = CONFIG.get("camera_elevation", -20)
    camera_tracking = CONFIG.get("camera_tracking", True)
    
    print("[INFO] Starting standalone policy deployment")
    print(f"[INFO] Model XML: {model_xml}")
    print(f"[INFO] Policy: {policy_path}")
    
    # Load MuJoCo model
    if not model_xml.exists():
        print(f"[ERROR] Model XML not found: {model_xml}")
        print("[INFO] Run export_mjcf.py first to generate the scene XML")
        return
    
    print("[INFO] Loading MuJoCo model...")
    # Use absolute path to ensure mesh files are found relative to XML location
    model_xml_abs = model_xml.absolute()
    mj_model = mujoco.MjModel.from_xml_path(str(model_xml_abs))
    data = mujoco.MjData(mj_model)
    
    print(f"[INFO] Model loaded: nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
    
    # Set initial pose
    if init_pose_json and init_pose_json.exists():
        print(f"[INFO] Loading initial pose from: {init_pose_json}")
        load_initial_pose(init_pose_json, mj_model, data)
    else:
        mujoco.mj_forward(mj_model, data)
    
    # Initialize gamepad if enabled
    gamepad = None
    if use_gamepad:
        try:
            gamepad = GamepadController(
                button_policy_map=button_policy_map,
                max_lin_vel=max_lin_vel,
                max_ang_vel=max_ang_vel
            )
            print("[INFO] Gamepad initialized - using analog sticks for control")
        except RuntimeError as e:
            print(f"[WARN] Gamepad initialization failed: {e}")
            print("[INFO] Falling back to keyboard control (not implemented)")
            use_gamepad = False
    
    # Create controller
    print("[INFO] Creating policy controller...")
    controller = StandalonePolicyController(
        policy_path=policy_path,
        mj_model=mj_model,
        device=device,
        action_scale=action_scale,
        training_defaults=TRAINING_DEFAULT_ANGLES,
    )
    
    # Launch viewer
    print("[INFO] Launching viewer...")
    try:
        with viewer.launch_passive(
            model=mj_model,
            data=data,
            show_left_ui=False,
            show_right_ui=False
        ) as v:
            # Set camera from config
            v.cam.distance = camera_distance
            v.cam.azimuth = camera_azimuth
            v.cam.elevation = camera_elevation
            
            print("[INFO] Viewer launched. Running policy...")
            print(f"[INFO] Camera: distance={camera_distance}m, azimuth={camera_azimuth}°, tracking={camera_tracking}")
            if not use_gamepad:
                print("[INFO] No gamepad - robot will stand still")
            
            step = 0
            lin_vel_z = 0.0
            ang_vel_x = 0.0
            
            while v.is_running():
                # Get velocity commands from gamepad
                if gamepad:
                    lin_vel_z, ang_vel_x = gamepad.get_velocity_commands()
                    
                    # Check for policy switch
                    new_policy_path = gamepad.check_policy_switch()
                    if new_policy_path:
                        controller.load_policy(Path(new_policy_path))
                
                # Get robot state
                root_quat = data.qpos[3:7]
                joint_pos = data.qpos[7:7+controller.num_robot_joints]
                joint_vel = data.qvel[6:6+controller.num_robot_joints]  # Skip free joint velocities
                
                # Compute projected gravity
                proj_gravity = compute_projected_gravity(root_quat)
                
                # Get observation
                obs = controller.get_observation(
                    projected_gravity=proj_gravity,
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    lin_vel_z=lin_vel_z,
                    ang_vel_x=ang_vel_x,
                )
                
                # Get action from policy (every n_substeps)
                if step % n_substeps == 0:
                    joint_targets = controller.get_action(obs)
                    data.ctrl[:controller.num_robot_joints] = joint_targets
                
                # Step simulation
                mujoco.mj_step(mj_model, data)
                step += 1
                
                # Update camera to track robot (if enabled)
                if camera_tracking and step % 10 == 0:  # Update every 10 steps for smoothness
                    root_pos = data.qpos[0:3]
                    v.cam.lookat[:] = root_pos  # Camera follows robot position
                
                # Sync viewer
                if step % 2 == 0:
                    v.sync()
                
                # Print status
                if step % 500 == 0:
                    root_pos = data.qpos[0:3]
                    vel_str = f"vel=[{lin_vel_z:.2f}, {ang_vel_x:.2f}]" if use_gamepad else ""
                    print(f"Step {step}: pos=[{root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}] {vel_str}")
    finally:
        # Cleanup gamepad
        if gamepad:
            gamepad.cleanup()
    
    print("[INFO] Deployment finished.")


if __name__ == "__main__":
    main()

