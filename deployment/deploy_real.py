from utils import (
    init_cmd_hg,
    create_damping_cmd,
    create_zero_cmd,
    MotorMode,
    NonBlockingInput,
    joint2motor_idx,
    Kp,
    Kd,
    G1_NUM_MOTOR,
    default_pos,
    crawl_angles,
    default_angles_config,
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
import numpy as np
import torch

NETWORK_CARD_NAME = 'enxc8a362b43bfd'
POLICY_PATH = "/home/logan/Projects/g1_crawl/deployment/policy.pt"



class TorchPolicy:
    """PyTorch controller for the G1 robot."""

    def __init__(self, policy_path: str):
        self._policy = torch.load(policy_path, weights_only=False)
        self._policy.eval()  # Set to evaluation mode
        
        # Initialize joint mappings
        init_joint_mappings()

    def get_control(self, obs: np.ndarray) -> np.ndarray:
        """Get control actions from PyTorch policy."""
        # Convert to torch tensor and run inference
        obs_tensor = torch.from_numpy(obs).float()
        
        with torch.no_grad():
            action_tensor = self._policy(obs_tensor)
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

        # Unified control state
        self.control_mode = "damped"  # one of {"damped", "policy", "hold"}
        self.hold_target_pose = None  # type: np.ndarray | None
        self._prev_buttons = {"A": 0, "B": 0, "X": 0, "Y": 0, "Select": 0, "F1": 0, "Start": 0}
        self._max_forward_speed = 1.5  # Ly -> [0, 1.5]
        
        # Safety gate: require Start button press before allowing controls
        self._system_started = False

    def _rising(self, name: str, current: int) -> bool:
        was = self._prev_buttons.get(name, 0)
        self._prev_buttons[name] = current
        return current == 1 and was == 0

    def _send_hold_command(self):
        if self.hold_target_pose is None:
            return
        for i in range(len(joint2motor_idx)):
            motor_idx = joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = float(self.hold_target_pose[i])
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
            self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)

    def set_hold_pose(self, target_pose: np.ndarray, total_time: float = 2.0):
        self.move_to_pose(target_pose, total_time=total_time)
        self.hold_target_pose = target_pose.copy()
        self.control_mode = "hold"

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
                    self.hold_target_pose = None
            return False
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
        
        # Y: Damped mode
        if self._rising("Y", self.remote.Y):
            print("Y pressed: Entering damped mode.")
            self.control_mode = "damped"
            self.hold_target_pose = None
        # X: Neural network control
        if self._rising("X", self.remote.X):
            print("X pressed: Entering neural network control mode.")
            self.control_mode = "policy"
            self.hold_target_pose = None
        # A: Lerp to crawl pose and hold
        if self._rising("A", self.remote.A):
            print("A pressed: Moving to set pose and holding.")
            self.set_hold_pose(self.crawl_angles_array, total_time=2.0)
        # B: Lerp to default pose and hold
        if self._rising("B", self.remote.B):
            print("B pressed: Moving to default position and holding.")
            self.set_hold_pose(self.default_pos_array, total_time=2.0)
        # F1: LED control
        if self._rising("F1", self.remote.F1):
            print("F1 pressed: Switching LED.")
            self.audio_client.LedControl(255,0,0)

    def send_cmd(self, cmd: LowCmd_):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

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
                self.low_cmd.motor_cmd[motor_idx].kp = Kp[j]
                self.low_cmd.motor_cmd[motor_idx].kd = Kd[j]
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
            if self.remote.Y == 1:
                print("Y pressed during move: Switching to damped mode.")
                self.control_mode = "damped"
                self.hold_target_pose = None
                return
            if self.remote.X == 1:
                print("X pressed during move: Switching to neural network mode.")
                self.control_mode = "policy"
                self.hold_target_pose = None
                return

            alpha = i / num_step
            for j in range(G1_NUM_MOTOR):
                motor_idx = joint2motor_idx[j]
                target_pos = float(target_pose[j])
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = Kp[j]
                self.low_cmd.motor_cmd[motor_idx].kd = Kd[j]
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
                self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
                self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
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
                self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
                self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
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
        
        # Get velocity commands from remote sticks (vx, vy, yaw_rate)
        # Ly: forward speed in [0, 1.5], negatives mapped to 0
        # Lx: yaw rate in [-1, 1]
        ly = float(self.remote.Ly)
        lx = float(self.remote.Lx)
        ly_norm = np.clip(ly, -1.0, 1.0)
        forward_speed = max(0.0, ly_norm) * self._max_forward_speed
        yaw_rate = float(np.clip(lx, -1.0, 1.0))
        velocity_commands = np.array([forward_speed, 0.0, yaw_rate], dtype=np.float32)
        
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
                print(f"  Joint {idx}: {motor_targets_unclamped[idx]:.3f} -> {motor_targets[idx]:.3f} (limits: [{self._joint_lower_bounds[idx]:.3f}, {self._joint_upper_bounds[idx]:.3f}])")

        # Build low cmd
        for i in range(G1_NUM_MOTOR):
            motor_idx = joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = motor_targets[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
            self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
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

    print("Setting up policy...")
    policy = TorchPolicy(POLICY_PATH)
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    print("\n" + "=" * 60)
    print("SAFETY: Press START button to enable all controls!")
    print("=" * 60)
    print("\nControls (active ONLY after pressing Start):")
    print("  A: Move to crawl pose and HOLD")
    print("  B: Move to DEFAULT pose and HOLD")
    print("  X: Neural network CONTROL (policy)")
    print("  Y: DAMPED mode")
    print("  F1: RED LED")
    print("\nAlways active:")
    print("  Start: ENABLE system (press this first!)")
    print("  Select: QUIT immediately")
    print("\nSafety features:")
    print("  - Remote heartbeat: Auto-damped if remote disconnects (timeout: 0.5s)")

    ChannelFactoryInitialize(0, NETWORK_CARD_NAME)

    controller = Controller(policy)

    print("\n" + "=" * 60)
    print("Controller ready. Press START to begin!")
    print("=" * 60)
    print("\n[DEBUG MODE ENABLED] Printing low_state info every second...")
    print("Watch for changes when you turn off the remote controller.\n")
    while True:
        # Print debug state periodically
        # controller.print_debug_state()
        
        # Check remote heartbeat first (safety check)
        controller.check_remote_heartbeat()
        
        controller.process_global_buttons()

        if controller.control_mode == "policy":
            controller.run()
        elif controller.control_mode == "hold":
            controller._send_hold_command()
            time.sleep(controller.control_dt)
        else:  # "damped" or any unknown -> default to damped
            create_damping_cmd(controller.low_cmd)
            controller.send_cmd(controller.low_cmd)
            time.sleep(controller.control_dt)

    # Final cleanup is handled in emergency quit path; normal exit can just end.
    print("Exit")