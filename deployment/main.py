from utils import (
    init_cmd_hg,
    create_damping_cmd,
    create_zero_cmd,
    MotorMode,
    NonBlockingInput,
    joint2motor_idx,
    Kp,
    Kd,
    get_kp_kd_scaled,
    G1_NUM_MOTOR,
    default_pos,
    crawl_angles,
    default_angles_config,
    stand_angles_config,
    get_gravity_orientation,
    action_scale,
    RESTRICTED_JOINT_RANGE,
    G1MjxJointIndex,
    G1PyTorchJointIndex,
    pytorch2mujoco_idx,
    mujoco2pytorch_idx,
    init_joint_mappings,
    remap_pytorch_to_mujoco,
    remap_mujoco_to_pytorch,
    MUJOCO_JOINT_NAMES,
)
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
import time
import sys
import struct
import argparse
import select
import numpy as np
import torch

NETWORK_CARD_NAME = 'eth0'

# ============================================================================
# CONFIGURATION
# ============================================================================

# List of policies to cycle through with A button
# All policies are loaded at startup for seamless switching
POLICY_PATHS = [
    "policies/policy_shamble.pt",
    "policies/policy_crawl_start.pt",
    "policies/policy_crawl.pt",
    # "policies/policy_shamble_start.pt",
]

# Explicit mapping of policy filenames to default joint set names
# Valid values: "stand", "crawl"
POLICY_DEFAULTS = {
    "policies/policy_shamble.pt": "stand",
    "policies/policy_crawl_start.pt": "stand",
    "policies/policy_crawl.pt": "crawl",
    # "policies/policy_shamble_start.pt": "stand",
}

# Velocity command limits per policy (m/s for linear, rad/s for angular)
# Each axis is specified as (min, max) to support asymmetric ranges
POLICY_VELOCITY_LIMITS = {
    "policies/policy_shamble.pt": {"forward": (-0.3, 0.3), "lateral": (-0.3, 0.3), "yaw": (-1.0, 1.0)},
    "policies/policy_crawl_start.pt": {"forward": (0.0, 0.0), "lateral": (0.0, 0.0), "yaw": (0.0, 0.0)},
    "policies/policy_crawl.pt": {"forward": (0.0, 1.9), "lateral": (-0.1, 0.1), "yaw": (-1.0, 1.0)},
    # "policies/policy_shamble_start.pt": {"forward": (-0.3, 0.3), "lateral": (-0.3, 0.3), "yaw": (-0.3, 0.3)},
}

# Gain multipliers per policy (single value applied to both KP and KD)
POLICY_GAIN_MULTIPLIERS = {
    "policies/policy_shamble.pt": 1.4,
    "policies/policy_crawl_start.pt": 1.2,
    "policies/policy_crawl.pt": 1.4,
    # "policies/policy_shamble_start.pt": 1.2,
}

# Safety flag: set to False to test state logic without moving motors
ENABLE_MOTOR_COMMANDS = True

# Safety monitoring: set to False to disable auto-damping on position/velocity/acceleration violations
# Note: Remote heartbeat monitoring is always enabled regardless of this setting
ENABLE_SAFETY_MONITORING = False

# Policy cycling behavior: if False, stops at last policy instead of looping back to first
# Useful to prevent accidental return to first policy before transition is smooth
LOOP_POLICIES = False

# NOTE: Velocity command limits are now policy-dependent (see POLICY_VELOCITY_LIMITS above)
# These old global constants are no longer used
# MAX_LIN_VEL_FORWARD = 1.  # m/s - Max forward/backward velocity (left stick Y)
# MAX_LIN_VEL_LATERAL = 1.  # m/s - Max lateral velocity (left stick X)
# MAX_ANG_VEL_YAW = 1.      # rad/s - Max angular velocity (right stick X)

# Transition timing (seconds) - smooth lerping between modes
TRANSITION_TIME_HOLD_POSES = 2.0  # For default_pos and crawl_pos modes
TRANSITION_TIME_POLICY = 0.5      # For policy mode

# NOTE: Gain multipliers are now policy-dependent (see POLICY_GAIN_MULTIPLIERS above)
# These old global constants are no longer used
# KP_MULTIPLIER_LEGS = 1.2
# KD_MULTIPLIER_LEGS = 1.2
# KP_MULTIPLIER_UPPER = 1.2
# KD_MULTIPLIER_UPPER = 1.2

GAIN_STEP = 0.1

# Control modes (switched via D-pad):
#   UP:    default_pos - Hold default standing position
#   DOWN:  crawl_pos   - Hold crawl position
#   LEFT:  damped      - Damped/compliant mode
#   RIGHT: policy      - Active neural network control
#
# Velocity commands (in policy mode):
#   Left stick Y:  Forward/backward velocity (policy-dependent limits)
#   Left stick X:  Lateral velocity/strafe (policy-dependent limits)
#   Right stick X: Angular velocity/yaw rotation (policy-dependent limits)
# ============================================================================



class TorchPolicy:
    """PyTorch controller for the G1 robot."""

    def __init__(self, policy_paths: list):
        """Load multiple policies for seamless switching."""
        self._policies = []
        print(f"Loading {len(policy_paths)} policies...")
        for i, path in enumerate(policy_paths):
            print(f"  [{i}] Loading {path}")
            policy = torch.load(path, weights_only=False)
            policy.eval()  # Set to evaluation mode
            self._policies.append(policy)
        
        self._current_policy_idx = 0
        print(f"Active policy: [{self._current_policy_idx}] {policy_paths[self._current_policy_idx]}")
        
        # Initialize joint mappings
        init_joint_mappings()

    def set_policy_index(self, idx: int):
        """Switch to a different policy by index."""
        if 0 <= idx < len(self._policies):
            self._current_policy_idx = idx

    def get_num_policies(self) -> int:
        """Return the number of loaded policies."""
        return len(self._policies)

    def get_control(self, obs: np.ndarray) -> np.ndarray:
        """Get control actions from currently active PyTorch policy."""
        # Convert to torch tensor and run inference
        obs_tensor = torch.from_numpy(obs).float()
        
        with torch.no_grad():
            action_tensor = self._policies[self._current_policy_idx](obs_tensor)
            pytorch_pred = action_tensor.numpy()  # Actions in PyTorch model joint order

        return pytorch_pred


class unitreeRemoteController:
    def __init__(self):
        self.Lx = 0.0
        self.Rx = 0.0
        self.Ry = 0.0
        self.Ly = 0.0
        self.L1 = 0
        self.L2 = 0
        self.R1 = 0
        self.R2 = 0
        self.A = 0
        self.B = 0
        self.X = 0
        self.Y = 0
        self.Up = 0
        self.Down = 0
        self.Left = 0
        self.Right = 0
        self.Select = 0
        self.F1 = 0
        self.F3 = 0
        self.Start = 0

    def parse_botton(self, data1: int, data2: int):
        self.R1 = (data1 >> 0) & 1
        self.L1 = (data1 >> 1) & 1
        self.Start = (data1 >> 2) & 1
        self.Select = (data1 >> 3) & 1
        self.R2 = (data1 >> 4) & 1
        self.L2 = (data1 >> 5) & 1
        self.F1 = (data1 >> 6) & 1
        self.F3 = (data1 >> 7) & 1
        self.A = (data2 >> 0) & 1
        self.B = (data2 >> 1) & 1
        self.X = (data2 >> 2) & 1
        self.Y = (data2 >> 3) & 1
        self.Up = (data2 >> 4) & 1
        self.Right = (data2 >> 5) & 1
        self.Down = (data2 >> 6) & 1
        self.Left = (data2 >> 7) & 1

    def parse_key(self, data: bytes):
        lx_offset = 4
        self.Lx = struct.unpack('<f', data[lx_offset:lx_offset + 4])[0]
        rx_offset = 8
        self.Rx = struct.unpack('<f', data[rx_offset:rx_offset + 4])[0]
        ry_offset = 12
        self.Ry = struct.unpack('<f', data[ry_offset:ry_offset + 4])[0]
        ly_offset = 20
        self.Ly = struct.unpack('<f', data[ly_offset:ly_offset + 4])[0]

    def parse(self, remoteData: bytes):
        if remoteData is None:
            return
        try:
            # Expecting a bytes-like object
            self.parse_key(remoteData)
            self.parse_botton(remoteData[2], remoteData[3])
        except Exception:
            # Fail quietly, keep previous state
            pass


class Controller:
    def __init__(self, policy: TorchPolicy) -> None:
        self.policy = policy

        self.qj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.dqj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.action = np.zeros(G1_NUM_MOTOR, dtype=np.float32)  # In MuJoCo order
        self.last_action_pytorch = np.zeros(G1_NUM_MOTOR, dtype=np.float32)  # In PyTorch order
        self.counter = 0

        # Convert joint range tuples to numpy arrays for efficient clamping
        joint_limits = np.array(RESTRICTED_JOINT_RANGE, dtype=np.float32)
        self._joint_lower_bounds = joint_limits[:, 0]
        self._joint_upper_bounds = joint_limits[:, 1]
        
        # Safety monitoring for neural network control
        self._prev_qj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self._prev_dqj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self._safety_initialized = False
        
        # Safety thresholds
        self._max_position_jump = 0.4  # rad per timestep (15 rad/s at 20ms)
        self._max_velocity = 25.0  # rad/s
        self._max_acceleration = 1500.0  # rad/s^2
        self._max_action_magnitude = 5.0  # Action output limit

        # Static velocity command placeholder; keyboard controls removed

        self.control_dt = 0.02
        
        # Remote heartbeat tracking
        self._last_remote_update_time = time.time()
        self._remote_timeout_seconds = 0.5  # Switch to damped if no update for 0.5s
        self._remote_connected = True
        self._last_remote_tick = None
        
        # Debug logging
        self._last_debug_print_time = time.time()
        self._debug_print_interval = 1.0  # Print debug info every second

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        self.default_pos_array = np.array(default_pos)
        self.crawl_angles_array = np.array(crawl_angles)
        self.default_angles_array = np.array(default_angles_config)

        # Runtime-tunable gain multiplier (single value for both KP and KD)
        self.gain_multiplier = 1.0  # Will be set by _apply_policy_config

        # Compute initial scaled KP/KD arrays (will be updated by _apply_policy_config)
        self._recompute_gains()

        # Remote controller state
        self.remote = unitreeRemoteController()

        # Audio client for TTS feedback
        self.audio_client = AudioClient()
        self.audio_client.SetTimeout(10.0)
        self.audio_client.Init()

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

        # Unified control state - 4 modes mapped to D-pad
        self.control_mode = "damped"  # one of {"default_pos", "crawl_pos", "damped", "policy"}
        self._prev_buttons = {"A": 0, "Up": 0, "Down": 0, "Left": 0, "Right": 0, "Select": 0, "F1": 0, "Start": 0}
        
        # Policy cycling
        self._current_policy_idx = 0
        self._policy_paths = POLICY_PATHS  # Store for display
        # Validate default mapping and apply for initial policy
        self._policy_defaults = dict(POLICY_DEFAULTS)
        self._policy_velocity_limits = dict(POLICY_VELOCITY_LIMITS)
        self._policy_gain_multipliers = dict(POLICY_GAIN_MULTIPLIERS)
        init_policy = self._policy_paths[self._current_policy_idx]
        if init_policy not in self._policy_defaults:
            available = ", ".join(sorted(self._policy_defaults.keys())) if self._policy_defaults else "(none)"
            raise KeyError(f"Initial policy '{init_policy}' missing from POLICY_DEFAULTS. Available: {available}")
        if init_policy not in self._policy_velocity_limits:
            available = ", ".join(sorted(self._policy_velocity_limits.keys())) if self._policy_velocity_limits else "(none)"
            raise KeyError(f"Initial policy '{init_policy}' missing from POLICY_VELOCITY_LIMITS. Available: {available}")
        if init_policy not in self._policy_gain_multipliers:
            available = ", ".join(sorted(self._policy_gain_multipliers.keys())) if self._policy_gain_multipliers else "(none)"
            raise KeyError(f"Initial policy '{init_policy}' missing from POLICY_GAIN_MULTIPLIERS. Available: {available}")
        for p in self._policy_paths:
            if p not in self._policy_defaults:
                raise KeyError(f"Policy '{p}' missing from POLICY_DEFAULTS")
            if p not in self._policy_velocity_limits:
                raise KeyError(f"Policy '{p}' missing from POLICY_VELOCITY_LIMITS")
            if p not in self._policy_gain_multipliers:
                raise KeyError(f"Policy '{p}' missing from POLICY_GAIN_MULTIPLIERS")
        
        # Current velocity limits (will be set by _apply_policy_config)
        # Each is a tuple of (min, max) to support asymmetric ranges
        self.vel_forward_range = (0.0, 0.0)
        self.vel_lateral_range = (0.0, 0.0)
        self.vel_yaw_range = (0.0, 0.0)
        
        # Apply initial policy configuration
        self._apply_policy_config(init_policy)
        
        # Transition state for smooth lerping between modes
        self._in_transition = False
        self._transition_start_pos = None
        self._transition_target_pos = None
        self._transition_start_time = None
        self._transition_duration = 0.0
        self._transition_target_mode = None
        
        # Safety gate: require Start button press before allowing controls
        self._system_started = False

    def _recompute_gains(self):
        self._kp_scaled, self._kd_scaled = get_kp_kd_scaled(
            kp_multiplier_legs=self.gain_multiplier,
            kd_multiplier_legs=self.gain_multiplier,
            kp_multiplier_upper=self.gain_multiplier,
            kd_multiplier_upper=self.gain_multiplier,
        )

    def _apply_policy_config(self, policy_path: str):
        """Apply all policy-specific configuration: default positions, velocity limits, and gain multiplier."""
        # Apply default position set
        set_name = self._policy_defaults.get(policy_path)
        if set_name == "stand":
            self.default_angles_array = np.array(stand_angles_config)
        elif set_name == "crawl":
            self.default_angles_array = np.array(default_angles_config)
        else:
            valid = ", ".join(["stand", "crawl"]) 
            raise KeyError(f"Invalid default set '{set_name}' for policy '{policy_path}'. Valid: {valid}")
        
        # Apply velocity limits (as ranges: (min, max))
        vel_limits = self._policy_velocity_limits.get(policy_path)
        if vel_limits is None:
            raise KeyError(f"No velocity limits defined for policy '{policy_path}'")
        self.vel_forward_range = tuple(vel_limits["forward"])
        self.vel_lateral_range = tuple(vel_limits["lateral"])
        self.vel_yaw_range = tuple(vel_limits["yaw"])
        
        # Apply gain multiplier
        gain_mult = self._policy_gain_multipliers.get(policy_path)
        if gain_mult is None:
            raise KeyError(f"No gain multiplier defined for policy '{policy_path}'")
        self.gain_multiplier = float(gain_mult)
        self._recompute_gains()

    def adjust_gains(self, delta: float):
        """Adjust the unified gain multiplier (applies to both KP and KD)."""
        self.gain_multiplier = max(0.0, self.gain_multiplier + delta)
        self._recompute_gains()
        print(f"Gain multiplier updated: {self.gain_multiplier:.2f}")


    def _rising(self, name: str, current: int) -> bool:
        was = self._prev_buttons.get(name, 0)
        self._prev_buttons[name] = current
        return current == 1 and was == 0

    def _start_transition(self, target_mode: str, target_pos: np.ndarray, duration: float):
        """Start a smooth transition to a new mode with target position."""
        # Capture current joint positions
        current_pos = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        for i in range(G1_NUM_MOTOR):
            current_pos[i] = self.low_state.motor_state[joint2motor_idx[i]].q
        
        self._in_transition = True
        self._transition_start_pos = current_pos.copy()
        self._transition_target_pos = target_pos.copy()
        self._transition_start_time = time.time()
        self._transition_duration = duration
        self._transition_target_mode = target_mode
        self._safety_initialized = False  # Reset safety checks for new mode

    def send_cmd(self, cmd: LowCmd_):
        """Centralized command sending with safety flag check."""
        cmd.crc = CRC().Crc(cmd)
        if ENABLE_MOTOR_COMMANDS:
            self.lowcmd_publisher_.Write(cmd)
        else:
            # In test mode, just silently skip sending
            pass

    def _execute_transition_step(self):
        """Execute one step of the transition. Returns True if transition complete."""
        if not self._in_transition:
            return True
        
        elapsed = time.time() - self._transition_start_time
        alpha = min(1.0, elapsed / self._transition_duration)
        
        # Lerp from start to target
        for i in range(G1_NUM_MOTOR):
            motor_idx = joint2motor_idx[i]
            current_q = self._transition_start_pos[i] * (1 - alpha) + self._transition_target_pos[i] * alpha
            self.low_cmd.motor_cmd[motor_idx].q = current_q
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self._kp_scaled[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self._kd_scaled[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        
        self.send_cmd(self.low_cmd)
        
        # Check if transition complete
        if alpha >= 1.0:
            self._in_transition = False
            self.control_mode = self._transition_target_mode
            print(f"Transition complete: Now in {self.control_mode} mode")
            return True
        
        return False

    def send_default_pos_command(self):
        """Send command to hold default position."""
        for i in range(len(joint2motor_idx)):
            motor_idx = joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = float(self.default_pos_array[i])
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self._kp_scaled[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self._kd_scaled[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)

    def send_crawl_pos_command(self):
        """Send command to hold crawl position."""
        for i in range(len(joint2motor_idx)):
            motor_idx = joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = float(self.crawl_angles_array[i])
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self._kp_scaled[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self._kd_scaled[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)

    # def print_debug_state(self):
    #     """Print debug information about low_state periodically."""
    #     current_time = time.time()
    #     if current_time - self._last_debug_print_time >= self._debug_print_interval:
    #         self._last_debug_print_time = current_time
            
    #         # Print information about wireless_remote field
    #         wr = self.low_state.wireless_remote
    #         wr_len = len(wr) if wr is not None else 0
    #         wr_bytes = wr[:8] if wr is not None and len(wr) >= 8 else []
            
    #         print(f"[DEBUG] tick={self.low_state.tick} | "
    #               f"wireless_remote: len={wr_len}, first_8_bytes={list(wr_bytes)} | "
    #               f"Lx={self.remote.Lx:.3f} Ly={self.remote.Ly:.3f} | "
    #               f"A={self.remote.A} B={self.remote.B} X={self.remote.X} Y={self.remote.Y} | "
    #               f"remote_connected={self._remote_connected}")

    def check_remote_heartbeat(self):
        """Check if remote is still connected; switch to damped mode if disconnected."""
        current_time = time.time()
        time_since_update = current_time - self._last_remote_update_time
        
        if time_since_update > self._remote_timeout_seconds:
            if self._remote_connected:
                print("=" * 60)
                print("WARNING: REMOTE CONTROL DISCONNECTED!")
                print(f"No heartbeat for {time_since_update:.2f}s (timeout: {self._remote_timeout_seconds}s)")
                print("Automatically switching to DAMPED mode for safety.")
                print("=" * 60)
                self._remote_connected = False
                # Force damped mode
                if self.control_mode != "damped":
                    self.control_mode = "damped"
            return False
        return True

    def check_safety_limits(self, pytorch_actions: np.ndarray) -> bool:
        """
        Check if current state violates safety limits (position jumps, velocity spikes, etc.)
        Returns True if safe, False if safety violation detected.
        """
        if not self._safety_initialized:
            # First iteration - just store current state
            self._prev_qj = self.qj.copy()
            self._prev_dqj = self.dqj.copy()
            self._safety_initialized = True
            return True
        
        # Check 1: Position jumps
        position_delta = np.abs(self.qj - self._prev_qj)
        max_position_delta = np.max(position_delta)
        position_jump_joint = np.argmax(position_delta)
        
        if max_position_delta > self._max_position_jump:
            joint_name = MUJOCO_JOINT_NAMES[position_jump_joint]
            print("=" * 60)
            print("SAFETY VIOLATION: POSITION JUMP DETECTED!")
            print(f"{joint_name} (joint {position_jump_joint}) jumped {max_position_delta:.3f} rad in one timestep")
            print(f"(threshold: {self._max_position_jump:.3f} rad)")
            print(f"Current pos: {self.qj[position_jump_joint]:.3f}, Previous: {self._prev_qj[position_jump_joint]:.3f}")
            print("Switching to DAMPED mode for safety.")
            print("=" * 60)
            self.control_mode = "damped"
            return False
        
        # Check 2: Velocity spikes
        velocity_magnitude = np.abs(self.dqj)
        max_velocity = np.max(velocity_magnitude)
        velocity_spike_joint = np.argmax(velocity_magnitude)
        
        if max_velocity > self._max_velocity:
            joint_name = MUJOCO_JOINT_NAMES[velocity_spike_joint]
            print("=" * 60)
            print("SAFETY VIOLATION: VELOCITY SPIKE DETECTED!")
            print(f"{joint_name} (joint {velocity_spike_joint}) velocity: {self.dqj[velocity_spike_joint]:.3f} rad/s")
            print(f"(threshold: {self._max_velocity:.3f} rad/s)")
            print("Switching to DAMPED mode for safety.")
            print("=" * 60)
            self.control_mode = "damped"
            return False
        
        # Check 3: Acceleration spikes
        acceleration = (self.dqj - self._prev_dqj) / self.control_dt
        max_acceleration = np.max(np.abs(acceleration))
        acceleration_spike_joint = np.argmax(np.abs(acceleration))
        
        if max_acceleration > self._max_acceleration:
            joint_name = MUJOCO_JOINT_NAMES[acceleration_spike_joint]
            print("=" * 60)
            print("SAFETY VIOLATION: ACCELERATION SPIKE DETECTED!")
            print(f"{joint_name} (joint {acceleration_spike_joint}) acceleration: {acceleration[acceleration_spike_joint]:.1f} rad/s^2")
            print(f"(threshold: {self._max_acceleration:.1f} rad/s^2)")
            print(f"Velocity changed from {self._prev_dqj[acceleration_spike_joint]:.3f} to {self.dqj[acceleration_spike_joint]:.3f} rad/s")
            print("Switching to DAMPED mode for safety.")
            print("=" * 60)
            self.control_mode = "damped"
            return False
        
        # Update previous state for next iteration
        self._prev_qj = self.qj.copy()
        self._prev_dqj = self.dqj.copy()
        
        return True

    def process_global_buttons(self):
        # Select: immediate quit (ALWAYS ACTIVE)
        if self._rising("Select", self.remote.Select):
            print("Select pressed: Quitting...")
            sys.exit(0)
        
        # Start: Enable system (ALWAYS ACTIVE)
        if self._rising("Start", self.remote.Start):
            if not self._system_started:
                print("=" * 60)
                print("START PRESSED: System is now ACTIVE!")
                print("All controls are now enabled.")
                print("=" * 60)
                self.audio_client.LedControl(0, 255, 0)  # Green LED to indicate system is active
                self._system_started = True
            else:
                print("Start pressed again (system already active).")
            return
        
        # All other buttons require system to be started
        if not self._system_started:
            return
        
        # D-pad: Mode switching with smooth transitions
        if self._rising("Up", self.remote.Up):
            print(f"D-PAD UP: Transitioning to DEFAULT POSITION mode ({TRANSITION_TIME_HOLD_POSES}s)")
            self._start_transition("default_pos", self.default_pos_array, TRANSITION_TIME_HOLD_POSES)
            
        if self._rising("Down", self.remote.Down):
            print(f"D-PAD DOWN: Transitioning to CRAWL POSITION mode ({TRANSITION_TIME_HOLD_POSES}s)")
            self._start_transition("crawl_pos", self.crawl_angles_array, TRANSITION_TIME_HOLD_POSES)
            
        if self._rising("Left", self.remote.Left):
            # Damped mode: no transition needed, instant switch
            self.control_mode = "damped"
            self._in_transition = False
            self._safety_initialized = False
            print(f"D-PAD LEFT: Switched to DAMPED mode")
            
        if self._rising("Right", self.remote.Right):
            # Policy mode: instant switch, no transition
            self.control_mode = "policy"
            self._in_transition = False
            self._safety_initialized = False
            print(f"D-PAD RIGHT: Switched to NEURAL NETWORK mode")
            print(f"  Active policy: [{self._current_policy_idx}] {self._policy_paths[self._current_policy_idx]}")
            print("  Safety monitoring: position jumps, velocity spikes, and acceleration")
        
        # A: Cycle through policies
        if self._rising("A", self.remote.A):
            num_policies = self.policy.get_num_policies()
            next_idx = self._current_policy_idx + 1
            
            if LOOP_POLICIES:
                # Loop back to first policy after last
                self._current_policy_idx = next_idx % num_policies
                self.policy.set_policy_index(self._current_policy_idx)
                new_policy = self._policy_paths[self._current_policy_idx]
                print(f"A PRESSED: Cycled to policy [{self._current_policy_idx}] {new_policy}")
                # Apply policy configuration (default positions, velocity limits, and gain multiplier)
                self._apply_policy_config(new_policy)
                print(f"  Velocity limits: forward=[{self.vel_forward_range[0]}, {self.vel_forward_range[1]}] m/s, "
                      f"lateral=[{self.vel_lateral_range[0]}, {self.vel_lateral_range[1]}] m/s, "
                      f"yaw=[{self.vel_yaw_range[0]}, {self.vel_yaw_range[1]}] rad/s")
                print(f"  Gain multiplier: {self.gain_multiplier:.2f}")
                self._safety_initialized = False
            else:
                # Stop at last policy, don't loop back
                if next_idx < num_policies:
                    self._current_policy_idx = next_idx
                    self.policy.set_policy_index(self._current_policy_idx)
                    new_policy = self._policy_paths[self._current_policy_idx]
                    print(f"A PRESSED: Cycled to policy [{self._current_policy_idx}] {new_policy}")
                    # Apply policy configuration (default positions, velocity limits, and gain multiplier)
                    self._apply_policy_config(new_policy)
                    print(f"  Velocity limits: forward=[{self.vel_forward_range[0]}, {self.vel_forward_range[1]}] m/s, "
                          f"lateral=[{self.vel_lateral_range[0]}, {self.vel_lateral_range[1]}] m/s, "
                          f"yaw=[{self.vel_yaw_range[0]}, {self.vel_yaw_range[1]}] rad/s")
                    print(f"  Gain multiplier: {self.gain_multiplier:.2f}")
                    self._safety_initialized = False
                else:
                    print(f"A PRESSED: Already at last policy [{self._current_policy_idx}] {self._policy_paths[self._current_policy_idx]}")
                    print("  (Policy looping disabled - set LOOP_POLICIES=True to enable)")
            
        # F1: LED control
        if self._rising("F1", self.remote.F1):
            print("F1 pressed: Switching LED.")
            self.audio_client.LedControl(255, 0, 0)

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Press A on remote to continue; B for EMERGENCY STOP (damped, then A to quit); Y to return to default position.")
        while True:
            if self.remote.B == 1:
                self.emergency_damped_and_confirm_quit()
            if self.remote.Y == 1:
                print("Returning to default position (Y pressed)...")
                self.move_to_default_pos()
            if self.remote.A == 1:
                print("Zero torque state confirmed. Proceeding...")
                break
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def emergency_damped_and_confirm_quit(self):
        print("EMERGENCY: Damped stop engaged.")
        print("Press A on remote to CONFIRM QUIT. Robot will remain damped until confirmation. Press Y to return to default position.")
        while True:
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            # Optional: go back to default position while in damped stop
            if self.remote.Y == 1:
                print("Returning to default position (Y pressed) while damped...")
                self.move_to_default_pos()
            if self.remote.A == 1:
                break
            time.sleep(self.control_dt)
        print("Confirmed. Exiting now.")
        sys.exit(0)

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        # Update remote controller state and heartbeat
        try:
            wr = self.low_state.wireless_remote
            if wr is not None and len(wr) >= 2:
                # Check for remote activity in first two bytes
                # When remote is ON: bytes 0-1 are non-zero (e.g., [85, 81])
                # When remote is OFF: bytes 0-1 are [0, 0]
                if not (wr[0] == 0 and wr[1] == 0):
                    # Remote is active - update heartbeat
                    self._last_remote_update_time = time.time()
                    if not self._remote_connected:
                        print("REMOTE RECONNECTED: Remote control is back online.")
                        self._remote_connected = True
                self.remote.parse(wr)
        except Exception:
            pass

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.control_dt)

        init_dof_pos = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        for i in range(G1_NUM_MOTOR):
            init_dof_pos[i] = self.low_state.motor_state[joint2motor_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(G1_NUM_MOTOR):
                motor_idx = joint2motor_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * \
                    (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kp_scaled[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kd_scaled[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def move_to_pose(self, target_pose: np.ndarray, total_time: float = 2.0):
        print("Moving to target pose.")
        num_step = int(total_time / self.control_dt)

        init_dof_pos = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        for i in range(G1_NUM_MOTOR):
            init_dof_pos[i] = self.low_state.motor_state[joint2motor_idx[i]].q

        for i in range(num_step):
            # Allow global button mapping to interrupt motion (avoid recursive A/B during move)
            if self.remote.Select == 1:
                print("Select pressed during move: Quitting...")
                sys.exit(0)
            if self.remote.Left == 1:
                print("D-pad Left pressed during move: Switching to damped mode.")
                self.control_mode = "damped"
                self._in_transition = False
                return
            if self.remote.Right == 1:
                print("D-pad Right pressed during move: Switching to neural network mode.")
                self.control_mode = "policy"
                self._in_transition = False
                return

            alpha = i / num_step
            for j in range(G1_NUM_MOTOR):
                motor_idx = joint2motor_idx[j]
                target_pos = float(target_pose[j])
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kp_scaled[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kd_scaled[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def transition_to_crawl_pose(self, total_time: float = 2.0):
        print("Moving to crawl pose.")
        self.move_to_pose(self.crawl_angles_array, total_time=total_time)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Press A to start; B for EMERGENCY STOP (damped, then A to quit); Y to return to default position.")
        while True:
            if self.remote.B == 1:
                self.emergency_damped_and_confirm_quit()
            if self.remote.Y == 1:
                print("Returning to default position (Y pressed)...")
                self.move_to_default_pos()
            if self.remote.A == 1:
                break
            for i in range(len(joint2motor_idx)):
                motor_idx = joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = default_pos[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kp_scaled[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kd_scaled[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)
        print("Default position state confirmed. Starting controller...")

    def hold_crawl_pose_until_remote_A(self):
        print("Holding crawl pose. Press A to start; B for EMERGENCY STOP (damped, then A to quit); Y to return to default position.")
        while True:
            if self.remote.B == 1:
                self.emergency_damped_and_confirm_quit()
            if self.remote.Y == 1:
                print("Returning to default position (Y pressed)...")
                self.move_to_default_pos()
            if self.remote.A == 1:
                break
            for i in range(len(joint2motor_idx)):
                motor_idx = joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.crawl_angles_array[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kp_scaled[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kd_scaled[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)
        print("Confirmed: starting controller from crawl pose...")

    def reach_and_hold_crawl_pose(self, total_time: float = 2.0):
        """Transition to crawl pose, then hold until remote A is pressed."""
        self.transition_to_crawl_pose(total_time=total_time)
        self.hold_crawl_pose_until_remote_A()

    def get_obs(self) -> np.ndarray:
        """Construct observation in the format expected by the PyTorch model."""
        # Get projected gravity (3 dimensions)
        quat = self.low_state.imu_state.quaternion
        gravity = get_gravity_orientation(quat)
        
        # Get velocity commands from remote sticks (lin_vel_z, lin_vel_y, ang_vel_x)
        # Left stick Y: forward/backward velocity (inverted for intuitive control)
        # Left stick X: lateral velocity (strafe left/right, inverted for intuitive control)
        # Right stick X: angular velocity (yaw rotation, inverted for intuitive control)
        ly = float(self.remote.Ly)  # Forward/backward
        lx = float(self.remote.Lx)  # Lateral (strafe)
        rx = float(self.remote.Rx)  # Yaw rotation
        
        # Apply deadzone
        deadzone = 0.1
        ly = 0.0 if abs(ly) < deadzone else ly
        lx = 0.0 if abs(lx) < deadzone else lx
        rx = 0.0 if abs(rx) < deadzone else rx
        
        # Clip to [-1, 1] and scale by policy-specific velocity ranges (supports asymmetric ranges)
        # Map stick value: stick=0 always maps to velocity=0
        # Negative stick values map from -1→min_vel to 0→0
        # Positive stick values map from 0→0 to 1→max_vel
        ly_clipped = np.clip(ly, -1.0, 1.0)
        lx_clipped = -np.clip(lx, -1.0, 1.0)  # Inverted for intuitive control
        rx_clipped = -np.clip(rx, -1.0, 1.0)  # Inverted for intuitive control
        
        # Map stick values to velocity: stick=0 always gives velocity=0
        # For negative stick: velocity = stick * abs(min_vel)
        # For positive stick: velocity = stick * max_vel
        lin_vel_z = ly_clipped * self.vel_forward_range[1] if ly_clipped >= 0 else ly_clipped * abs(self.vel_forward_range[0])
        lin_vel_y = lx_clipped * self.vel_lateral_range[1] if lx_clipped >= 0 else lx_clipped * abs(self.vel_lateral_range[0])
        ang_vel_x = rx_clipped * self.vel_yaw_range[1] if rx_clipped >= 0 else rx_clipped * abs(self.vel_yaw_range[0])
        
        velocity_commands = np.array([lin_vel_z, lin_vel_y, ang_vel_x], dtype=np.float32)
        
        # Get joint positions and velocities in MuJoCo order, then convert to PyTorch order
        joint_pos_mujoco = self.qj.copy()  # Already relative to default_angles_config
        joint_vel_mujoco = self.dqj.copy()
        
        # Convert to PyTorch model joint order for the observation
        joint_pos_pytorch = remap_mujoco_to_pytorch(joint_pos_mujoco)
        joint_vel_pytorch = remap_mujoco_to_pytorch(joint_vel_mujoco)
        
        # Concatenate all observations: 3 + 3 + 23 + 23 + 23 = 75
        obs = np.hstack([
            gravity,
            velocity_commands, 
            joint_pos_pytorch,
            joint_vel_pytorch,
            self.last_action_pytorch,
        ])
        
        return obs.astype(np.float32)

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(G1_NUM_MOTOR):
            self.qj[i] = self.low_state.motor_state[joint2motor_idx[i]].q - self.default_angles_array[i]
            self.dqj[i] = self.low_state.motor_state[joint2motor_idx[i]].dq

        # Create observation
        obs = self.get_obs()

        # Get actions from policy (in PyTorch order)
        pytorch_actions = self.policy.get_control(obs)
        
        # SAFETY CHECK: Verify state and actions are safe before proceeding (if enabled)
        if ENABLE_SAFETY_MONITORING and not self.check_safety_limits(pytorch_actions):
            # Safety violation detected - check_safety_limits already switched to damped mode
            # Send damping command and return early
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)
            return
        
        # Store last action in PyTorch order for next observation
        self.last_action_pytorch = pytorch_actions.copy()
        
        # Convert actions from PyTorch order to MuJoCo order
        mujoco_actions = remap_pytorch_to_mujoco(pytorch_actions)
        
        # Apply action scaling
        action_effect = mujoco_actions * action_scale
        
        # Calculate motor targets
        motor_targets_unclamped = self.default_angles_array + action_effect

        # Clamp motor targets to joint limits and check for clamping
        motor_targets = np.clip(
            motor_targets_unclamped, self._joint_lower_bounds, self._joint_upper_bounds
        )
        clamped_indices = np.where(motor_targets != motor_targets_unclamped)[0]
        if clamped_indices.size > 0:
            print("WARNING: Clamping motor targets for joints:")
            for idx in clamped_indices:
                joint_name = MUJOCO_JOINT_NAMES[idx]
                print(f"  {joint_name}: {motor_targets_unclamped[idx]:.3f} -> {motor_targets[idx]:.3f} (limits: [{self._joint_lower_bounds[idx]:.3f}, {self._joint_upper_bounds[idx]:.3f}])")

        # print(f"Kp: {Kp}")
        # print(f"Kd: {Kd}")
        # Build low cmd
        for i in range(G1_NUM_MOTOR):
            motor_idx = joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = motor_targets[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self._kp_scaled[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self._kd_scaled[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)
        time.sleep(self.control_dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G1 real deployment controller")
    parser.add_argument("--print-sticks", action="store_true", help="Print remote stick/button values and exit on Select")
    args = parser.parse_args()

    if args.print_sticks:
        print("Initializing remote monitor (stick printer)...")
        ChannelFactoryInitialize(0, NETWORK_CARD_NAME)
        remote = unitreeRemoteController()

        def _ls_handler(msg: LowState_):
            try:
                remote.parse(msg.wireless_remote)
            except Exception:
                pass

        ls_sub = ChannelSubscriber("rt/lowstate", LowState_)
        ls_sub.Init(_ls_handler, 10)
        print("Printing at 10 Hz. Wiggle sticks and press Select to quit.")
        try:
            while True:
                print(f"Lx={remote.Lx:.3f} Ly={remote.Ly:.3f}  Rx={remote.Rx:.3f} Ry={remote.Ry:.3f}  "
                      f"A={remote.A} B={remote.B} X={remote.X} Y={remote.Y} L1={remote.L1} R1={remote.R1} L2={remote.L2} R2={remote.R2} Start={remote.Start} Select={remote.Select}")
                if remote.Select == 1:
                    print("Select pressed. Exiting.")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        sys.exit(0)

    print("Setting up policies...")
    policy = TorchPolicy(POLICY_PATHS)
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    print("\n" + "=" * 60)
    print("SAFETY: Press START button to enable all controls!")
    print("=" * 60)
    print("\nControls (active ONLY after pressing Start):")
    print("  D-PAD UP: Default position mode")
    print("  D-PAD DOWN: Crawl position mode")
    print("  D-PAD LEFT: Damped mode")
    print("  D-PAD RIGHT: Neural network control mode")
    print(f"  A: Cycle through loaded policies ({'loops' if LOOP_POLICIES else 'stops at last'})")
    print("  F1: RED LED")
    print("\nVelocity commands (in neural network mode, policy-dependent limits):")
    print(f"  Left stick Y: Forward/backward")
    print(f"  Left stick X: Strafe left/right")
    print(f"  Right stick X: Rotate/yaw")
    print("\nKeyboard tuning (non-blocking, no Enter needed):")
    print(f"  q/a: Gain multiplier +/- {GAIN_STEP}")
    print("  h  : Print current gain multiplier")
    print("\nAlways active:")
    print("  Start: ENABLE system (press this first!)")
    print("  Select: QUIT immediately")
    print("\nSafety features:")
    print("  - Remote heartbeat: Auto-damped if remote disconnects (timeout: 0.5s) [ALWAYS ON]")
    if ENABLE_SAFETY_MONITORING:
        print("  - Position jump detection: Max 0.4 rad/timestep (20 rad/s) [ENABLED]")
        print("  - Velocity spike detection: Max 25.0 rad/s [ENABLED]")
        print("  - Acceleration spike detection: Max 1500 rad/s^2 [ENABLED]")
        print("  - Auto-damped mode on any safety violation")
    else:
        print("  - Position/velocity/acceleration monitoring: [DISABLED]")
    print(f"\nMotor commands: {'ENABLED' if ENABLE_MOTOR_COMMANDS else 'DISABLED (test mode)'}")

    ChannelFactoryInitialize(0, NETWORK_CARD_NAME)

    controller = Controller(policy)

    print("\n" + "=" * 60)
    print("Controller ready. Press START to begin!")
    print("=" * 60)
    print(f"\nActive policy configuration:")
    print(f"  Velocity limits:")
    print(f"    Forward: [{controller.vel_forward_range[0]}, {controller.vel_forward_range[1]}] m/s")
    print(f"    Lateral: [{controller.vel_lateral_range[0]}, {controller.vel_lateral_range[1]}] m/s")
    print(f"    Yaw: [{controller.vel_yaw_range[0]}, {controller.vel_yaw_range[1]}] rad/s")
    print(f"  Gain multiplier: {controller.gain_multiplier:.2f}")
    with NonBlockingInput() as kb:
        while True:
            # Handle keyboard (single key, non-blocking)
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                ch = sys.stdin.read(1)
                if ch == 'q':
                    controller.adjust_gains(+GAIN_STEP)
                elif ch == 'a':
                    controller.adjust_gains(-GAIN_STEP)
                elif ch == 'h':
                    print(f"Gain multiplier: {controller.gain_multiplier:.2f}")

            # Check remote heartbeat first (safety check)
            controller.check_remote_heartbeat()
            
            controller.process_global_buttons()

            # Handle transitions first - they override normal mode behavior
            if controller._in_transition:
                controller._execute_transition_step()
                time.sleep(controller.control_dt)
            elif controller.control_mode == "policy":
                controller.run()
            elif controller.control_mode == "default_pos":
                controller.send_default_pos_command()
                time.sleep(controller.control_dt)
            elif controller.control_mode == "crawl_pos":
                controller.send_crawl_pos_command()
                time.sleep(controller.control_dt)
            else:  # "damped" or any unknown -> default to damped
                create_damping_cmd(controller.low_cmd)
                controller.send_cmd(controller.low_cmd)
                time.sleep(controller.control_dt)

    # Final cleanup is handled in emergency quit path; normal exit can just end.
    print("Exit")