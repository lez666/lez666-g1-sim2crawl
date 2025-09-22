# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#modified by Logan
"""Deploy a PyTorch policy to C MuJoCo and play with it."""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch
import json
import math
from pathlib import Path
import os
from keyboard_reader import KeyboardController
from utils import (
    default_angles_config,

    init_joint_mappings,
    remap_pytorch_to_mujoco,
    remap_mujoco_to_pytorch,
)

POLICY_PATH = "deployment/crawl_policy.pt"
POSES_JSON_PATH = (Path(__file__).parent / "_REF" / "output-example.json").as_posix()
START_POSE_INDEX = 2  # 1-based index, consistent with sim-poses.py
# Control mode: "nn" to use neural net, "hold" to freeze at initialized pose
CONTROL_MODE = os.environ.get("G1_CONTROL_MODE", "nn").lower()
HOLD_BEFORE_NN_SEC = float(os.environ.get("G1_HOLD_BEFORE_NN_SEC", "1.0"))
ALWAYS_HOLD_AT_START = os.environ.get("G1_ALWAYS_HOLD_AT_START", "1").strip().lower() in {"1", "true", "yes", "y"}


class TorchController:
  """PyTorch controller for the Go-1 robot."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      n_substeps: int,
      action_scale: float = 0.5,
      vel_scale_x: float = 1.0,
      vel_scale_y: float = 1.0,
      vel_scale_rot: float = 1.0,
  ):
    self._policy = torch.load(policy_path, weights_only=False)
    self._policy.eval()  # Set to evaluation mode

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)  # In MuJoCo order

    self._counter = 0
    self._n_substeps = n_substeps

    self._controller = KeyboardController(
        vel_scale_x=vel_scale_x,
        vel_scale_y=vel_scale_y,
        vel_scale_rot=vel_scale_rot,
    )

    # Initialize joint mappings
    init_joint_mappings()


  def get_obs(self, model, data) -> np.ndarray:
    # Simplified observation: 75 dimensions total
    # projected_gravity (3) + velocity_commands (3) + joint_pos (23) + joint_vel (23) + actions (23)
    
    # Get projected gravity (3 dimensions)
    # imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
    # projected_gravity = imu_xmat.T @ np.array([0, 0, -1])
    world_gravity = model.opt.gravity
    world_gravity = world_gravity / np.linalg.norm(world_gravity)  # Normalize
    imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
    projected_gravity = imu_xmat.T @ world_gravity
    # Get velocity commands and map to env's expected order [lin_vel_z, lin_vel_y, ang_vel_x]
    kb_vx, kb_vy, kb_wz = self._controller.get_command()
    lin_vel_z = 2.0#np.clip(kb_vx, 0.0, 2.0)
    lin_vel_y = 0.0  # training used zero lateral velocity
    ang_vel_x = 0.0 #np.clip(kb_wz, -1.0, 1.0)
    velocity_commands = np.array([lin_vel_z, lin_vel_y, ang_vel_x], dtype=np.float32)
    # velocity_commands = np.array([0.25, 0.0, 0.0])  # Forward velocity command
    
    # Get joint positions and velocities in MuJoCo order, then convert to PyTorch order
    joint_pos_mujoco = data.qpos[7:] - self._default_angles
    joint_vel_mujoco = data.qvel[6:]
    
    # Convert to PyTorch model joint order for the observation
    joint_pos_pytorch = remap_mujoco_to_pytorch(joint_pos_mujoco)
    joint_vel_pytorch = remap_mujoco_to_pytorch(joint_vel_mujoco)
    
    # Last action should also be in PyTorch order for the observation
    # Convert the MuJoCo-ordered last action to PyTorch order
    actions_pytorch = remap_mujoco_to_pytorch(self._last_action)
    
    # Concatenate all observations: 3 + 3 + 23 + 23 + 23 = 75
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
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      
      # Convert to torch tensor and run inference
      obs_tensor = torch.from_numpy(obs).float()
      
      with torch.no_grad():
        action_tensor = self._policy(obs_tensor)
        pytorch_pred = action_tensor.numpy()  # Actions in PyTorch model joint order


      # Convert actions from PyTorch order to MuJoCo order
      mujoco_pred = remap_pytorch_to_mujoco(pytorch_pred)

      self._last_action = mujoco_pred.copy()  # Store in MuJoCo order
      # data.ctrl[:] =  self._default_angles
      # print(mujoco_pred)

      data.ctrl[:] = mujoco_pred * self._action_scale + self._default_angles

def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
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


def _find_free_qpos_addr(model: mujoco.MjModel) -> int | None:
  for j in range(model.njnt):
    if int(model.jnt_type[j]) == 0:  # mjJNT_FREE
      return int(model.jnt_qposadr[j])
  return None


def _load_poses_json(json_path: str) -> list[dict]:
  data = json.loads(Path(json_path).read_text())
  poses = []
  for item in data.get("poses", []):
    base_rpy = item.get("base_rpy")
    joints_raw = item.get("joints", {})
    joints: dict[int | str, float] = {}
    for k, v in joints_raw.items():
      try:
        k_int = int(k)  # type: ignore[arg-type]
        joints[k_int] = float(v)
      except (ValueError, TypeError):
        joints[str(k)] = float(v)
    poses.append({"base_rpy": base_rpy, "joints": joints})
  return poses


def _apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict, drop_height_m: float = 0.5) -> None:
  # Set base orientation if FREE joint exists
  free_adr = _find_free_qpos_addr(model)
  base_rpy = pose.get("base_rpy") if isinstance(pose, dict) else None
  if free_adr is not None and isinstance(base_rpy, (list, tuple)) and len(base_rpy) == 3:
    roll, pitch, yaw = float(base_rpy[0]), float(base_rpy[1]), float(base_rpy[2])
    qw, qx, qy, qz = _rpy_to_quat(roll, pitch, yaw)
    data.qpos[free_adr + 0] = 0.0
    data.qpos[free_adr + 1] = 0.0
    data.qpos[free_adr + 2] = float(max(0.0, drop_height_m))
    data.qpos[free_adr + 3] = qw
    data.qpos[free_adr + 4] = qx
    data.qpos[free_adr + 5] = qy
    data.qpos[free_adr + 6] = qz

  # Apply hinge/slide joints by id or name
  joints = pose.get("joints", {}) if isinstance(pose, dict) else {}
  for j_key, val in joints.items():
    if isinstance(j_key, int):
      j_id = int(j_key)
    else:
      try:
        j_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(j_key)))
      except Exception:
        continue
    # Only SLIDE(2) and HINGE(3) have nq=1
    if int(model.jnt_type[j_id]) not in (2, 3):
      continue
    adr = int(model.jnt_qposadr[j_id])
    data.qpos[adr] = float(val)


def _build_joint_to_actuator_map(model: mujoco.MjModel) -> dict[int, int]:
  mapping: dict[int, int] = {}
  for j in range(model.njnt):
    if int(model.jnt_type[j]) not in (2, 3):
      continue
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if jname is None:
      continue
    try:
      aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
    except Exception:
      aid = -1
    if int(aid) != -1:
      mapping[int(j)] = int(aid)
  return mapping


class HoldPoseController:
  def __init__(self, joint_to_actuator: dict[int, int], hold_targets: dict[int, float]):
    self._joint_to_actuator = joint_to_actuator
    self._hold_targets = hold_targets

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for j, aid in self._joint_to_actuator.items():
      data.ctrl[aid] = self._hold_targets[j]


# class HoldThenNNController:
#   def __init__(self, hold_controller: HoldPoseController, nn_controller: TorchController, switch_time_s: float):
#     self._hold_controller = hold_controller
#     self._nn_controller = nn_controller
#     self._switch_time_s = float(switch_time_s)
#     self._switched = False
#     self._logged_first = False

#   def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
#     if (not self._switched) and data.time >= self._switch_time_s:
#       self._switched = True
#       print(f"[HoldThenNN] Switching to NN at t={data.time:.3f}s (threshold {self._switch_time_s:.3f}s)")
#     if self._switched:
#       self._nn_controller.get_control(model, data)
#     else:
#       if not self._logged_first:
#         print(f"[HoldThenNN] Holding pose until t={self._switch_time_s:.3f}s; current t={data.time:.3f}s")
#         self._logged_first = True
#       self._hold_controller.get_control(model, data)


def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)


  model = mujoco.MjModel.from_xml_path('deployment/g1_description/scene_mjx_alt.xml')
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 1)

  # Initialize to crawl orientation (body and joints) using poses JSON
  poses = _load_poses_json(POSES_JSON_PATH)
  if not poses:
    raise RuntimeError(f"No poses found in {POSES_JSON_PATH}")
  idx = max(0, min(len(poses) - 1, int(START_POSE_INDEX) - 1))
  _apply_pose(model, data, poses[idx], drop_height_m=0.5)
  mujoco.mj_forward(model, data)

  print(model.opt.timestep)
  sim_dt = 0.005
  n_substeps = 4
  model.opt.timestep = sim_dt
  
  joint_to_actuator = _build_joint_to_actuator_map(model)
  if not joint_to_actuator:
    raise RuntimeError("No joint->actuator mapping found; cannot hold pose")
  hold_targets: dict[int, float] = {}
  for j in sorted(joint_to_actuator.keys()):
    adr = int(model.jnt_qposadr[int(j)])
    hold_targets[int(j)] = float(data.qpos[adr])
  hold = HoldPoseController(joint_to_actuator, hold_targets)
  for j, aid in joint_to_actuator.items():
      data.ctrl[aid] = hold_targets[j]

  mujoco.set_mjcb_control(hold.get_control)
  # if CONTROL_MODE == "nn":
  #   policy = TorchController(
  #       policy_path=POLICY_PATH,
  #       default_angles=np.array(default_angles_config),
  #       n_substeps=n_substeps,
  #       action_scale=0.5,
  #       vel_scale_x=2.0,
  #       vel_scale_y=0.0,
  #       vel_scale_rot=1.0,
  #   )
  #   if ALWAYS_HOLD_AT_START and HOLD_BEFORE_NN_SEC > 0.0:
  #     # Build a hold from current qpos, then switch to NN after delay
  #     joint_to_actuator = _build_joint_to_actuator_map(model)
  #     if not joint_to_actuator:
  #       raise RuntimeError("No joint->actuator mapping found; cannot hold pose before NN")
  #     hold_targets: dict[int, float] = {}
  #     for j in sorted(joint_to_actuator.keys()):
  #       adr = int(model.jnt_qposadr[int(j)])
  #       hold_targets[int(j)] = float(data.qpos[adr])
  #     hold = HoldPoseController(joint_to_actuator, hold_targets)
  #     switching = HoldThenNNController(hold, policy, HOLD_BEFORE_NN_SEC)
  #     mujoco.set_mjcb_control(switching.get_control)
  #   else:
  #     mujoco.set_mjcb_control(policy.get_control)
  # else:
  #   # Freeze joints at initialized pose by holding actuator targets to current qpos
  #   joint_to_actuator = _build_joint_to_actuator_map(model)
  #   if not joint_to_actuator:
  #     raise RuntimeError("No joint->actuator mapping found; cannot hold pose")
  #   hold_targets: dict[int, float] = {}
  #   for j in sorted(joint_to_actuator.keys()):
  #     adr = int(model.jnt_qposadr[int(j)])
  #     hold_targets[int(j)] = float(data.qpos[adr])
  #   hold = HoldPoseController(joint_to_actuator, hold_targets)
  #   # Prepare neural controller and switch after a delay
  #   policy = TorchController(
  #       policy_path=POLICY_PATH,
  #       default_angles=np.array(default_angles_config),
  #       n_substeps=n_substeps,
  #       action_scale=0.5,
  #       vel_scale_x=2.0,
  #       vel_scale_y=0.0,
  #       vel_scale_rot=1.0,
  #   )
  #   switching = HoldThenNNController(hold, policy, HOLD_BEFORE_NN_SEC)
  #   mujoco.set_mjcb_control(switching.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
