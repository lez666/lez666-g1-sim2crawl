import json
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch
from utils import (
    default_angles_config,
    init_joint_mappings,
    remap_pytorch_to_mujoco,
    remap_mujoco_to_pytorch,
)


# Hardcoded poses JSON path (no CLI argument)
POSES_JSON_PATH: Path = Path("/home/logan/Projects/g1_crawl/deployment/_REF/output-example.json")
# SCENE_XML_PATH: Path = Path("/home/logan/Projects/g1_crawl/deployment/g1_description/scene_torso_collision_test.xml")
SCENE_XML_PATH: Path = Path("/home/logan/Projects/g1_crawl/deployment/g1_description/scene_mjx_alt.xml")

START_FACE_DOWN: bool = True
DROP_HEIGHT_M: float = 0.5

# Optional: start from a specific pose in the JSON (1-based index). Set to None to disable.
START_POSE_INDEX: int | None = 2  # e.g., set to 1 to start at "Pose 1"

# NN control constants (simple, no args/params)
NN_SWITCH_SEC: float = 2
NN_POLICY_PATH: Path = Path("/home/logan/Projects/g1_crawl/deployment/policy.pt")
NN_ACTION_SCALE: float = 0.5
NN_N_SUBSTEPS: int = 4

def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
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

    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


def quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


def load_poses(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text())
    poses: list[dict] = []
    for item in data.get("poses", []):
        base_rpy = item.get("base_rpy")
        joints_raw = item.get("joints", {})
        # Support new format with joint-name keys; also accept legacy index keys
        joints: dict[int | str, float] = {}
        for k, v in joints_raw.items():
            try:
                # If key is numeric (e.g., "12"), keep as int for backward compatibility
                k_int = int(k)  # type: ignore[arg-type]
                joints[k_int] = float(v)
            except (ValueError, TypeError):
                joints[str(k)] = float(v)
        poses.append({"base_rpy": base_rpy, "joints": joints})
    return poses



def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict, free_qpos_addr: int | None, face_down_fallback: bool = True) -> None:
    # Orientation is not adjusted when applying poses (simulate real-test behavior).
    # Hinge/slide joints from pose (ids are MuJoCo joint ids)
    for j_key, val in pose.get("joints", {}).items():
        # Resolve joint id from either int id or joint name
        if isinstance(j_key, int):
            j_id = j_key
        else:
            try:
                j_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(j_key)))
            except Exception:
                continue
        # Skip FREE/BALL joints; only SLIDE/HINGE have nq=1
        if model.jnt_type[int(j_id)] not in (2, 3):
            continue
        adr = model.jnt_qposadr[int(j_id)]
        data.qpos[adr] = float(val)


class TorchController:
    """Minimal PyTorch controller: fixed commands, no keyboard."""

    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
    ) -> None:
        self._policy = torch.load(policy_path, weights_only=False)
        self._policy.eval()

        self._default_angles = default_angles.astype(np.float32)
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)
        self._action_scale = float(action_scale)
        print("ACTION_SCALE", self._action_scale)
        self._n_substeps = int(n_substeps)
        self._counter = 0

        # Initialize joint mappings once
        init_joint_mappings()

    def _get_obs(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        # Project gravity into IMU frame
        world_gravity = model.opt.gravity
        world_gravity = world_gravity / np.linalg.norm(world_gravity)
        imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
        projected_gravity = imu_xmat.T @ world_gravity

        # Fixed velocity commands [lin_vel_z, lin_vel_y, ang_vel_x]
        velocity_commands = np.array([0.4, 0.0, 0.0], dtype=np.float32)

        # Joint states (MuJoCo order) → PyTorch order
        joint_pos_mujoco = data.qpos[7:] - self._default_angles
        joint_vel_mujoco = data.qvel[6:]
        joint_pos_pytorch = remap_mujoco_to_pytorch(joint_pos_mujoco)
        joint_vel_pytorch = remap_mujoco_to_pytorch(joint_vel_mujoco)
        actions_pytorch = remap_mujoco_to_pytorch(self._last_action)

        obs = np.hstack([
            projected_gravity,
            velocity_commands,
            joint_pos_pytorch,
            joint_vel_pytorch,
            actions_pytorch,
        ])
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps != 0:
            return

        obs = self._get_obs(model, data)
        obs_tensor = torch.from_numpy(obs).float()
        with torch.no_grad():
            action_tensor = self._policy(obs_tensor)
            pytorch_pred = action_tensor.numpy()

        mujoco_pred = remap_pytorch_to_mujoco(pytorch_pred)
        self._last_action = mujoco_pred.copy()
        data.ctrl[:] = mujoco_pred * self._action_scale + self._default_angles
        # data.ctrl[:] = self._default_angles



class PosesUI:
    pass


def main() -> None:
    poses_path = POSES_JSON_PATH

    poses = load_poses(poses_path)
    if not poses:
        raise SystemExit(f"No poses found in {poses_path}")

    model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH.as_posix())
    data = mujoco.MjData(model)

    # Find a FREE joint if present
    free_qpos_addr: int | None = None
    for j in range(model.njnt):
        if model.jnt_type[j] == 0:
            free_qpos_addr = int(model.jnt_qposadr[j])
            break

    # Start face-down with a small drop if a FREE joint is present
    if START_FACE_DOWN and free_qpos_addr is not None:
        data.qpos[free_qpos_addr + 0] = 0.0
        data.qpos[free_qpos_addr + 1] = 0.0
        data.qpos[free_qpos_addr + 2] = float(max(0.0, DROP_HEIGHT_M))
        qw, qx, qy, qz = rpy_to_quat(0.0, math.pi / 2.0, 0.0)
        data.qpos[free_qpos_addr + 3] = qw
        data.qpos[free_qpos_addr + 4] = qx
        data.qpos[free_qpos_addr + 5] = qy
        data.qpos[free_qpos_addr + 6] = qz

    # Apply the selected pose to set hinge/slide joints before first forward
    if START_POSE_INDEX is not None and len(poses) > 0:
        idx0 = max(0, min(len(poses) - 1, int(START_POSE_INDEX) - 1))
        apply_pose(model, data, poses[idx0], free_qpos_addr, face_down_fallback=False)

    mujoco.mj_forward(model, data)

    # Build mapping from joint id -> actuator id (names are aligned in XML)
    joint_to_actuator: dict[int, int] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] not in (2, 3):
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jname is None:
            continue
        try:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        except Exception:
            aid = -1
        if aid != -1:
            joint_to_actuator[int(j)] = int(aid)

    controlled_joint_ids = sorted(joint_to_actuator.keys())

    # Hold targets at current qpos (which reflect the applied pose) so nothing moves
    hold_targets: dict[int, float] = {}
    for j in controlled_joint_ids:
        adr = model.jnt_qposadr[j]
        hold_targets[j] = float(data.qpos[adr])

    # Prepare NN controller and switching timer
    controller = TorchController(
        policy_path=NN_POLICY_PATH.as_posix(),
        default_angles=np.array(default_angles_config, dtype=np.float32),
        n_substeps=NN_N_SUBSTEPS,
        action_scale=NN_ACTION_SCALE,
    )
    t0 = time.time()

    with viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as v:
        while v.is_running():
            elapsed = time.time() - t0
            if elapsed < NN_SWITCH_SEC:
                # Hold pose by setting actuator targets to initial qpos
                for j in controlled_joint_ids:
                    aid = joint_to_actuator[j]
                    data.ctrl[aid] = hold_targets[j]
            else:
                # NN takes over control
                controller.get_control(model, data)

            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep * 0.5))


if __name__ == "__main__":
    main()


