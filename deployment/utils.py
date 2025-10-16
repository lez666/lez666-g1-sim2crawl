from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
import numpy as np
import select
import tty
import termios
import sys

G1_NUM_MOTOR = 23

# Actuator constants from g1.py (calculated based on armature, natural frequency, damping ratio)
STIFFNESS_5020 = 14.25062309787429
STIFFNESS_7520_14 = 40.17923847137318
STIFFNESS_7520_22 = 99.09842777666113
DAMPING_5020 = 0.907222843292423
DAMPING_7520_14 = 2.5578897650279457
DAMPING_7520_22 = 6.3088018534966395

# Kp/Kd values based on motor types for each joint
# Joint mapping:
#   hip_pitch, hip_yaw: 7520_14 motors
#   hip_roll, knee: 7520_22 motors
#   ankles: 5020 motors (2x for feet)
#   waist_yaw: 7520_14 motor
#   arms (shoulder/elbow/wrist): 5020 motors
Kp = [
    STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22, 2*STIFFNESS_5020, 2*STIFFNESS_5020,     # legs (L: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22, 2*STIFFNESS_5020, 2*STIFFNESS_5020,     # legs (R: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    STIFFNESS_7520_14,                                                                                                     # waist_yaw
    STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,                                      # arms (L: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll)
    STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,                                      # arms (R: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll)
]

Kd = [
    DAMPING_7520_14, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22, 2*DAMPING_5020, 2*DAMPING_5020,    # legs (L)
    DAMPING_7520_14, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22, 2*DAMPING_5020, 2*DAMPING_5020,    # legs (R)
    DAMPING_7520_14,                                                                                          # waist_yaw
    DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,                                   # arms (L)
    DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,                                   # arms (R)
]

default_pos = [
    -0.1, 0, 0, 0.3, -0.2, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    0,
    0.2, 0.2, 0, 1.28, 0,
    0.2, -0.2, 0, 1.28, 0,
]

# Default angles based on env.yaml configuration (in MuJoCo joint order)
# These are the values used by the PyTorch model training
default_angles_config = [
    -1.6796101123595506,   # LeftHipPitch
    2.4180011235955052,    # LeftHipRoll
    1.2083865168539325,    # LeftHipYaw
    2.1130298764044944,    # LeftKnee
    0.194143033707865,     # LeftAnklePitch
    0.0,                   # LeftAnkleRoll
    -1.6796101123595506,   # RightHipPitch
    -2.4180011235955052,   # RightHipRoll
    -1.2083865168539325,   # RightHipYaw
    2.1130298764044944,    # RightKnee
    0.194143033707865,     # RightAnklePitch
    0.0,                   # RightAnkleRoll
    0.0,                   # WaistYaw
    1.4578526315789473,    # LeftShoulderPitch
    1.5778684210526317,    # LeftShoulderRoll
    1.4238245614035088,    # LeftShoulderYaw
    -0.3124709677419355,   # LeftElbow
    0.0,                   # LeftWristRoll
    1.4578526315789473,    # RightShoulderPitch
    -1.5778684210526317,   # RightShoulderRoll
    -1.4238245614035088,   # RightShoulderYaw
    -0.3124709677419355,   # RightElbow
    0.0,                   # RightWristRoll
]

crawl_angles = [
    -1.5814894736842104,
    2.293456140350877,
    1.2094736842105265,
    2.098992894736842,
    0.0,
    -0.0,
    -1.5814894736842104,
    -2.293456140350877,
    -1.2094736842105265,
    2.098992894736842,
    0.0,
    0.0,
    0.0,
    1.1547157894736841,
    1.9146842105263158,
    1.1482456140350878,
    0.0,
    -0.0,
    1.1547157894736841,
    -1.9146842105263158,
    -1.1482456140350878,
    0.0,
    0.0
 ]

crawl_pose_delta = [
    0.09812063867534015,
    -0.12454498324462815,
    0.0010871673565939766, 
    -0.014036981667652437, 
    -0.194143033707865, 
    -0.0, 0.09812063867534015, 
    0.12454498324462815, 
    -0.0010871673565939766, 
    -0.014036981667652437, 
    -0.194143033707865, 
    0.0, 
    0.0, 
    -0.30313684210526315, 
    0.33681578947368407, 
    -0.2755789473684209, 
    0.3124709677419355, 
    -0.0, 
    -0.30313684210526315, 
    -0.33681578947368407, 
    0.2755789473684209, 
    0.3124709677419355, 
    0.0
]

action_scale = 0.5


class G1MjxJointIndex:
    """Joint indices based on the order in g1_mjx_alt.xml (23 DoF model)."""
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    WaistYaw = 12
    LeftShoulderPitch = 13
    LeftShoulderRoll = 14
    LeftShoulderYaw = 15
    LeftElbow = 16
    LeftWristRoll = 17
    RightShoulderPitch = 18
    RightShoulderRoll = 19
    RightShoulderYaw = 20
    RightElbow = 21
    RightWristRoll = 22

    # Note: This model has 23 degrees of freedom (indices 0-22).
    # It lacks WaistRoll, WaistPitch, LeftWristPitch, LeftWristYaw,
    # RightWristPitch, and RightWristYaw compared to the original G1JointIndex.


class G1PyTorchJointIndex:
    """Joint indices based on the order in your PyTorch model."""
    # Actual joint order from PyTorch training:
    # ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint',
    # 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    # 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
    # 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    # 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint']

    LeftHipPitch = 0       # left_hip_pitch_joint
    RightHipPitch = 1      # right_hip_pitch_joint
    WaistYaw = 2           # waist_yaw_joint
    LeftHipRoll = 3        # left_hip_roll_joint
    RightHipRoll = 4       # right_hip_roll_joint
    LeftHipYaw = 5         # left_hip_yaw_joint
    RightHipYaw = 6        # right_hip_yaw_joint
    LeftKnee = 7           # left_knee_joint
    RightKnee = 8          # right_knee_joint
    LeftShoulderPitch = 9  # left_shoulder_pitch_joint
    RightShoulderPitch = 10  # right_shoulder_pitch_joint
    LeftAnklePitch = 11    # left_ankle_pitch_joint
    RightAnklePitch = 12   # right_ankle_pitch_joint
    LeftShoulderRoll = 13  # left_shoulder_roll_joint
    RightShoulderRoll = 14  # right_shoulder_roll_joint
    LeftAnkleRoll = 15     # left_ankle_roll_joint
    RightAnkleRoll = 16    # right_ankle_roll_joint
    LeftShoulderYaw = 17   # left_shoulder_yaw_joint
    RightShoulderYaw = 18  # right_shoulder_yaw_joint
    LeftElbow = 19         # left_elbow_joint
    RightElbow = 20        # right_elbow_joint
    LeftWristRoll = 21     # left_wrist_roll_joint
    RightWristRoll = 22    # right_wrist_roll_joint


# Mapping from PyTorch model joint order to MuJoCo joint order
pytorch2mujoco_idx = [
    # PyTorch idx -> MuJoCo idx
    G1MjxJointIndex.LeftHipPitch,      # 0: left_hip_pitch_joint -> LeftHipPitch (0)
    G1MjxJointIndex.RightHipPitch,     # 1: right_hip_pitch_joint -> RightHipPitch (6)
    G1MjxJointIndex.WaistYaw,          # 2: waist_yaw_joint -> WaistYaw (12)
    G1MjxJointIndex.LeftHipRoll,       # 3: left_hip_roll_joint -> LeftHipRoll (1)
    G1MjxJointIndex.RightHipRoll,      # 4: right_hip_roll_joint -> RightHipRoll (7)
    G1MjxJointIndex.LeftHipYaw,        # 5: left_hip_yaw_joint -> LeftHipYaw (2)
    G1MjxJointIndex.RightHipYaw,       # 6: right_hip_yaw_joint -> RightHipYaw (8)
    G1MjxJointIndex.LeftKnee,          # 7: left_knee_joint -> LeftKnee (3)
    G1MjxJointIndex.RightKnee,         # 8: right_knee_joint -> RightKnee (9)
    G1MjxJointIndex.LeftShoulderPitch,  # 9: left_shoulder_pitch_joint -> LeftShoulderPitch (13)
    G1MjxJointIndex.RightShoulderPitch,  # 10: right_shoulder_pitch_joint -> RightShoulderPitch (18)
    G1MjxJointIndex.LeftAnklePitch,    # 11: left_ankle_pitch_joint -> LeftAnklePitch (4)
    G1MjxJointIndex.RightAnklePitch,   # 12: right_ankle_pitch_joint -> RightAnklePitch (10)
    G1MjxJointIndex.LeftShoulderRoll,  # 13: left_shoulder_roll_joint -> LeftShoulderRoll (14)
    G1MjxJointIndex.RightShoulderRoll,  # 14: right_shoulder_roll_joint -> RightShoulderRoll (19)
    G1MjxJointIndex.LeftAnkleRoll,     # 15: left_ankle_roll_joint -> LeftAnkleRoll (5)
    G1MjxJointIndex.RightAnkleRoll,    # 16: right_ankle_roll_joint -> RightAnkleRoll (11)
    G1MjxJointIndex.LeftShoulderYaw,   # 17: left_shoulder_yaw_joint -> LeftShoulderYaw (15)
    G1MjxJointIndex.RightShoulderYaw,  # 18: right_shoulder_yaw_joint -> RightShoulderYaw (20)
    G1MjxJointIndex.LeftElbow,         # 19: left_elbow_joint -> LeftElbow (16)
    G1MjxJointIndex.RightElbow,        # 20: right_elbow_joint -> RightElbow (21)
    G1MjxJointIndex.LeftWristRoll,     # 21: left_wrist_roll_joint -> LeftWristRoll (17)
    G1MjxJointIndex.RightWristRoll,    # 22: right_wrist_roll_joint -> RightWristRoll (22)
]

# Inverse mapping from MuJoCo joint order to PyTorch model order
mujoco2pytorch_idx = [0] * 23


def init_joint_mappings():
    """Initialize the inverse mapping from MuJoCo to PyTorch indices."""
    global mujoco2pytorch_idx
    for pytorch_idx, mujoco_idx in enumerate(pytorch2mujoco_idx):
        mujoco2pytorch_idx[mujoco_idx] = pytorch_idx


def remap_pytorch_to_mujoco(pytorch_actions: np.ndarray) -> np.ndarray:
    """Remap actions from PyTorch model joint order to MuJoCo joint order."""
    mujoco_actions = np.zeros_like(pytorch_actions)
    for pytorch_idx, mujoco_idx in enumerate(pytorch2mujoco_idx):
        mujoco_actions[mujoco_idx] = pytorch_actions[pytorch_idx]
    return mujoco_actions


def remap_mujoco_to_pytorch(mujoco_data: np.ndarray) -> np.ndarray:
    """Remap data from MuJoCo joint order to PyTorch model joint order."""
    pytorch_data = np.zeros_like(mujoco_data)
    for pytorch_idx, mujoco_idx in enumerate(pytorch2mujoco_idx):
        pytorch_data[pytorch_idx] = mujoco_data[mujoco_idx]
    return pytorch_data


# Mapping from G1MjxJointIndex (0-22) to G1JointIndex (0-28)
joint2motor_idx = [
    0,  # LeftHipPitch
    1,  # LeftHipRoll
    2,  # LeftHipYaw
    3,  # LeftKnee
    4,  # LeftAnklePitch
    5,  # LeftAnkleRoll
    6,  # RightHipPitch
    7,  # RightHipRoll
    8,  # RightHipYaw
    9,  # RightKnee
    10,  # RightAnklePitch
    11,  # RightAnkleRoll
    12,  # WaistYaw
    15,  # LeftShoulderPitch (skips WaistRoll=13, WaistPitch=14)
    16,  # LeftShoulderRoll
    17,  # LeftShoulderYaw
    18,  # LeftElbow
    19,  # LeftWristRoll (skips LeftWristPitch=20, LeftWristYaw=21)
    22,  # RightShoulderPitch
    23,  # RightShoulderRoll
    24,  # RightShoulderYaw
    25,  # RightElbow
    26,  # RightWristRoll (skips RightWristPitch=27, RightWristYaw=28)
]


class MotorMode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


def init_cmd_hg(cmd: LowCmd_, mode_machine: int, mode_pr: int):
    cmd.mode_machine = mode_machine
    cmd.mode_pr = mode_pr
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0


def create_damping_cmd(cmd: LowCmd_):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 8
        cmd.motor_cmd[i].tau = 0


def create_zero_cmd(cmd: LowCmd_):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


# --- Non-blocking Keyboard Input Context Manager ---
class NonBlockingInput:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        try:
            tty.setraw(sys.stdin.fileno())
        except termios.error as e:
            # Fallback if not a tty (e.g., running in certain IDEs/environments)
            print(f"Warning: Could not set raw mode: {e}. Key detection might not work.", file=sys.stderr)
            self.old_settings = None  # Indicate failure
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_settings:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        print("\nRestored terminal settings.")  # Optional: provide feedback

    def check_key(self, key='\n'):
        """Check if a specific key is pressed without blocking."""
        if not self.old_settings:  # If raw mode failed, don't check
            return False
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            ch = sys.stdin.read(1)
            # In raw mode, Enter is often '\r' (carriage return)
            return ch == (key if key != '\n' else '\r')
        return False
# -----------------------------------------------------


RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Right leg. 6
    (-2.5307, 2.8798),
    (-2.9671, 0.5236),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Waist.
    (-2.618, 2.618),
    # Left shoulder.
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    # Right shoulder.
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
)
