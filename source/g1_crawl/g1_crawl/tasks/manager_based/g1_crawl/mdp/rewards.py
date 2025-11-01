# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
import json
import os
from typing import TYPE_CHECKING

from isaaclab.assets import  RigidObject
from isaaclab.assets import Articulation

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Reuse animation helpers
from ..g1 import get_animation, build_joint_index_map
from .observations import compute_animation_phase_and_frame

def both_feet_on_ground(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward when both feet are in contact with the ground.

    This function rewards the agent when both feet are in contact with the ground, 
    encouraging stable bipedal stance. Use with feet_slide penalty to discourage sliding.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.

    Returns:
        1 if both feet are in contact, 0 otherwise.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Check if feet are in contact using contact time
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    # Count feet in contact
    both_feet_in_contact = torch.sum(in_contact.int(), dim=1) == 2
    return both_feet_in_contact.float()


def either_foot_off_ground(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty when either foot is not in contact with the ground.

    Returns 1.0 if fewer than two selected feet are in contact; 0.0 otherwise.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    both_feet_in_contact = torch.sum(in_contact.int(), dim=1) == 2
    return (1.0 - both_feet_in_contact.float())


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def both_feet_on_ground_when_stationary(
    env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.1
) -> torch.Tensor:
    """Penalize when both feet are NOT in contact with the ground during stationary commands.

    This function penalizes the agent when both feet are not in contact with the ground
    during stationary commands (velocity near zero). This encourages stable bipedal 
    stance when not moving.
    
    Args:
        env: The environment instance.
        command_name: Name of the velocity command to check.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Velocity command threshold below which penalty applies (default: 0.1).

    Returns:
        1 if both feet are NOT in contact and commands are near zero, 0 otherwise.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Check if feet are in contact using contact time
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    # Count feet in contact
    both_feet_in_contact = torch.sum(in_contact.int(), dim=1) == 2
    # Invert: penalty when feet are NOT both on ground
    penalty = 1.0 - both_feet_in_contact.float()
    # Only apply penalty when velocity commands are near zero (stationary)
    cmd = env.command_manager.get_command(command_name)
    # Check both linear (xy) and angular (z) velocity commands
    cmd_magnitude = torch.sqrt(cmd[:, 0]**2 + cmd[:, 1]**2 + cmd[:, 2]**2)
    is_stationary = cmd_magnitude < threshold
    return penalty * is_stationary.float()

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_proximity_bonus_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Exponential bonus reward for being close to default joint positions.
    
    reward = exp(-sum(|joint_pos - default_pos|^2) / std^2)
    
    Args:
        env: RL environment.
        std: Standard deviation parameter controlling reward falloff (smaller = sharper falloff).
        asset_cfg: Scene entity for the robot asset.
    
    Returns:
        Tensor of shape (num_envs,) with proximity bonuses in [0, 1].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Compute joint position deviations
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviations = joint_pos - default_pos
    
    # Compute squared L2 norm of deviations
    err_sq = torch.sum(torch.square(deviations), dim=1)
    
    # Exponential bonus: exp(-err_sq / std^2)
    # When err_sq = 0 (perfect match): reward = 1.0
    # As err_sq increases: reward approaches 0
    reward = torch.exp(-err_sq / (std ** 2))
    
    return reward


# Global cache for pose data to avoid repeated file loading
_pose_cache = {}
_pose_full_cache = {}

def clear_pose_cache():
    """Clear the pose data cache. Useful for testing or when pose files change."""
    global _pose_cache
    _pose_cache.clear()
    _pose_full_cache.clear()

def _load_pose_from_json(json_path: str) -> dict:
    """Load joint pose data from JSON file with caching.
    
    Args:
        json_path: Path to the JSON file containing pose data.
        
    Returns:
        Dictionary containing joint positions.
    """
    # Check cache first
    if json_path in _pose_cache:
        return _pose_cache[json_path]
    
    # Get the project root directory (assuming this is called from the project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
    full_path = os.path.join(project_root, json_path)
    
    with open(full_path, 'r') as f:
        data = json.load(f)
    
    # Extract the first pose from the poses array
    if "poses" in data and len(data["poses"]) > 0:
        pose_data = data["poses"][0]["joints"]
        # Cache the result
        _pose_cache[json_path] = pose_data
        return pose_data
    else:
        raise ValueError(f"No poses found in {json_path}")


def _build_target_vector_from_pose_dict(asset: Articulation, pose: dict) -> torch.Tensor:
    """Build a target joint position vector in the exact order of asset.joint_names.

    Fails loudly if any joint name is missing in the provided pose dict.
    """
    names = asset.joint_names
    missing = [n for n in names if n not in pose]
    if len(missing) > 0:
        raise KeyError(f"Pose dict missing joint keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    values = [pose[n] for n in names]
    return torch.tensor(values, dtype=asset.data.joint_pos.dtype, device=asset.data.joint_pos.device)


def _get_full_target_vector(json_path: str, asset: Articulation) -> torch.Tensor:
    """Return a cached full-length target vector for the given pose json and asset.

    Keyed by (json_path, joint_names tuple, dtype, device). Fails loudly if pose is missing joints.
    """
    key = (
        json_path,
        tuple(asset.joint_names),
        str(asset.data.joint_pos.dtype),
        str(asset.data.joint_pos.device),
    )
    if key in _pose_full_cache:
        return _pose_full_cache[key]

    pose_dict = _load_pose_from_json(json_path)
    vec = _build_target_vector_from_pose_dict(asset, pose_dict)
    _pose_full_cache[key] = vec
    return vec


def command_based_pose_proximity_bonus_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    command_0_pose_path: str = "assets/crawl-pose.json",
    command_1_pose_path: str = "assets/stand-pose.json",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Exponential bonus reward for being close to command-based goal joint positions.
    
    When command is 0: uses command_0_pose_path as target
    When command is 1: uses command_1_pose_path as target
    
    Note: Pose data is cached after first load to avoid repeated file I/O.
    
    Args:
        env: RL environment.
        command_name: Name of the boolean command to use for pose selection.
        std: Standard deviation parameter controlling reward falloff.
        command_0_pose_path: Path to pose JSON file for command=0.
        command_1_pose_path: Path to pose JSON file for command=1.
        asset_cfg: Scene entity for the robot asset.
    
    Returns:
        Tensor of shape (num_envs,) with proximity bonuses in [0, 1].
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Command mask (num_envs,)
    command = env.command_manager.get_command(command_name)
    command_values = command[:, 0]

    # Validate std
    std_val = float(std)
    if std_val <= 0.0:
        raise RuntimeError("command_based_pose_proximity_bonus_exp requires std > 0")

    # Load and build full target vectors in joint order
    pose_0_full = _get_full_target_vector(command_0_pose_path, asset)
    pose_1_full = _get_full_target_vector(command_1_pose_path, asset)

    num_envs = asset.data.joint_pos.shape[0]
    # Build per-env full target matrix based on command (broadcast where)
    mask = (command_values > 0.5).view(num_envs, 1)
    target_full = torch.where(
        mask,
        pose_1_full.view(1, -1).expand(num_envs, -1),
        pose_0_full.view(1, -1).expand(num_envs, -1),
    )

    # Select only configured joints
    joint_pos_sel = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target_sel = target_full[:, asset_cfg.joint_ids]

    if not torch.isfinite(joint_pos_sel).all():
        raise RuntimeError("Non-finite joint positions in command_based_pose_proximity_bonus_exp")

    deviations = joint_pos_sel - target_sel
    err_sq = torch.sum(torch.square(deviations), dim=1)
    reward = torch.exp(-err_sq / (std_val ** 2))
    return reward


def command_based_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_0_pose_path: str = "assets/crawl-pose.json",
    command_1_pose_path: str = "assets/stand-pose.json",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """L1 penalty for joint positions that deviate from command-based target poses.
    
    When command is 0: uses command_0_pose_path as target
    When command is 1: uses command_1_pose_path as target
    
    Note: Pose data is cached after first load to avoid repeated file I/O.
    
    Args:
        env: RL environment.
        command_name: Name of the boolean command to use for pose selection.
        command_0_pose_path: Path to pose JSON file for command=0.
        command_1_pose_path: Path to pose JSON file for command=1.
        asset_cfg: Scene entity for the robot asset.
    
    Returns:
        Tensor of shape (num_envs,) with L1 deviation penalties.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Command mask (num_envs,)
    command = env.command_manager.get_command(command_name)
    command_values = command[:, 0]

    # Load and build full target vectors in joint order
    pose_0_full = _get_full_target_vector(command_0_pose_path, asset)
    pose_1_full = _get_full_target_vector(command_1_pose_path, asset)

    num_envs = asset.data.joint_pos.shape[0]
    mask = (command_values > 0.5).view(num_envs, 1)
    target_full = torch.where(
        mask,
        pose_1_full.view(1, -1).expand(num_envs, -1),
        pose_0_full.view(1, -1).expand(num_envs, -1),
    )

    joint_pos_sel = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target_sel = target_full[:, asset_cfg.joint_ids]

    if not torch.isfinite(joint_pos_sel).all():
        raise RuntimeError("Non-finite joint positions in command_based_joint_deviation_l1")

    deviations = torch.abs(joint_pos_sel - target_sel)
    total_deviation = torch.sum(deviations, dim=1)
    return total_deviation


def pose_json_deviation_l1(
    env: ManagerBasedRLEnv,
    pose_path: str = "assets/default-pose.json",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """L1 penalty for joint positions that deviate from a target pose defined in a JSON file.
    
    Simple single-pose version without command logic.
    Note: Pose data is cached after first load to avoid repeated file I/O.
    
    Args:
        env: RL environment.
        pose_path: Path to pose JSON file.
        asset_cfg: Scene entity for the robot asset.
    
    Returns:
        Tensor of shape (num_envs,) with L1 deviation penalties.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Load and build full target vector in joint order
    target_full = _get_full_target_vector(pose_path, asset)

    # Select the joints specified in asset_cfg
    joint_pos_sel = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target_sel = target_full[asset_cfg.joint_ids]

    if not torch.isfinite(joint_pos_sel).all():
        raise RuntimeError("Non-finite joint positions in pose_json_deviation_l1")

    # Compute L1 deviation
    deviations = torch.abs(joint_pos_sel - target_sel)
    total_deviation = torch.sum(deviations, dim=1)
    return total_deviation

def animation_pose_similarity_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 pose error between current joint positions and animation frame joints.

    - Uses a per-env advancing animation frame counter stored on the env (initialized on reset).
    - Advances the frame counter each call by step_dt / anim_dt frames.
    - Excludes floating base by relying on the joint index map built from animation metadata.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    anim = get_animation()
    qpos: torch.Tensor = anim["qpos"]  # [T, nq] on GPU
    T = int(anim["num_frames"])

    # Build and cache joint index map on the env
    if not hasattr(env, "_anim_joint_index_map"):
        index_map_list = build_joint_index_map(asset, anim.get("joints_meta"), anim.get("qpos_labels"))
        index_map = torch.tensor(index_map_list, dtype=torch.long, device=torch.device("cpu"))
        setattr(env, "_anim_joint_index_map", index_map)
    index_map_cpu: torch.Tensor = env._anim_joint_index_map  # type: ignore[attr-defined]

    # One-time debug print about mapping coverage
    if not hasattr(env, "_anim_debug_checked"):
        num_robot_dofs = int(asset.data.joint_pos.shape[1])
        num_missing = int((index_map_cpu < 0).sum().item())
        print(f"[anim debug] joint index map built: length={index_map_cpu.shape[0]} robot_dofs={num_robot_dofs} missing={num_missing}")
        if num_missing > 0:
            missing_idxs = torch.nonzero(index_map_cpu < 0, as_tuple=False).squeeze(-1).tolist()
            if not isinstance(missing_idxs, list):
                missing_idxs = [int(missing_idxs)]
            sample = missing_idxs[:5]
            sample_names = [str(asset.data.joint_names[i]) for i in sample]
            print(f"[anim debug] sample missing joints: {sample_names}")
        setattr(env, "_anim_debug_checked", True)

    # Validate index map shape and range (allow -1 for missing)
    num_robot_dofs = int(asset.data.joint_pos.shape[1])
    if int(index_map_cpu.shape[0]) != num_robot_dofs:
        raise RuntimeError(f"Animation index map length {int(index_map_cpu.shape[0])} != robot dofs {num_robot_dofs}")
    nq_anim = int(qpos.shape[1])
    if torch.any(index_map_cpu >= nq_anim) or torch.any(index_map_cpu < -1):
        raise RuntimeError("Animation index map contains out-of-range entries (valid are [-1, nq-1])")

    # Ensure all required joints in asset_cfg.joint_ids exist in the animation mapping (fail loudly)
    joint_ids = asset_cfg.joint_ids
    if not isinstance(joint_ids, slice):
        raise TypeError(f"asset_cfg.joint_ids must be a slice; got {type(joint_ids).__name__}")
    # Expand slice into explicit indices using robot DoF count for validation
    joint_ids_list = list(range(num_robot_dofs))[joint_ids]
    required_map = index_map_cpu[joint_ids_list]
    missing_mask = required_map < 0
    if torch.any(missing_mask):
        missing_pos = torch.nonzero(missing_mask, as_tuple=False).squeeze(-1).tolist()
        if not isinstance(missing_pos, list):
            missing_pos = [int(missing_pos)]
        missing_robot_joint_indices = [joint_ids_list[i] for i in missing_pos]
        missing_joint_names = [str(asset.data.joint_names[i]) for i in missing_robot_joint_indices]
        raise RuntimeError(
            f"Animation is missing qpos indices for required joints: {missing_joint_names}"
        )

    # Ensure phase offsets were initialized
    if not hasattr(env, "_anim_phase_offset"):
        raise RuntimeError("Missing _anim_phase_offset on env. Ensure reset_from_animation/init_animation_phase_offsets ran.")

    # Derive frame from episode time + phase offset (single source of truth)
    # Get frame indices on device and index GPU qpos directly
    _, frame_idx = compute_animation_phase_and_frame(env)


    # Compute integer frame indices and gather target joints
    target_qpos = qpos[frame_idx]  # [N, nq]
    # Map animation qpos to robot joint order
    target_joint_full = target_qpos.index_select(dim=1, index=index_map_cpu.to(device))  # [N, num_robot_dofs]
    target_joint_full = target_joint_full.to(dtype=asset.data.joint_pos.dtype)

    # Mirror joint_deviation_l1 style: operate directly on cfg.joint_ids using L1
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - target_joint_full[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(angle), dim=1)


    return reward


def animation_forward_velocity_similarity_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential tracking of base-frame YZ linear velocity to animation target.

    Uses metadata key 'base_forward_velocity_mps' to set target vz, with vy target fixed to 0.
    reward = exp(- ( (vz_b - v_target)^2 + (vy_b - 0)^2 ) / std^2 )
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    anim = get_animation()
    meta = anim.get("metadata", {}) or {}
    if "base_forward_velocity_mps" not in meta or meta["base_forward_velocity_mps"] is None:
        raise RuntimeError("Animation metadata is missing required key 'base_forward_velocity_mps'")
    if meta["base_forward_velocity_mps"] <= 0.0:
        raise RuntimeError("Animation metadata key 'base_forward_velocity_mps' is less than or equal to 0.0")
    base_target = float(meta["base_forward_velocity_mps"])
    # Per-env playback speed scaling
    if not hasattr(env, "_anim_playback_speed"):
        raise RuntimeError("Missing _anim_playback_speed on env. Ensure init_animation_phase_offsets ran.")
    speed = env._anim_playback_speed  # type: ignore[attr-defined]
    if speed.dim() == 0:
        speed = speed.view(1).expand(asset.data.root_lin_vel_b.shape[0])
    v_target = base_target * speed.to(device=asset.device, dtype=asset.data.root_lin_vel_b.dtype)

    # Measured base linear velocity in YZ order: [vz, vy]
    vel_b = asset.data.root_lin_vel_b[:, :3]
    if not torch.isfinite(vel_b).all():
        num_nan = torch.isnan(vel_b).sum().item()
        num_posinf = (vel_b == float("inf")).sum().item()
        num_neginf = (vel_b == float("-inf")).sum().item()
        print(f"[reward debug] animation_forward_velocity_similarity_exp: non-finite vel_b: NaN={num_nan} +Inf={num_posinf} -Inf={num_neginf} shape={tuple(vel_b.shape)}")
        raise RuntimeError("animation_forward_velocity_similarity_exp encountered non-finite vel_b")
    meas_yz = vel_b[:, [2, 1]]
    # Broadcast per-env target [vz, vy] = [v_target, 0]
    target_yz = torch.stack([v_target, torch.zeros_like(v_target)], dim=1)
    # Validate std
    try:
        std_val = float(std)
    except Exception:
        print(f"[reward debug] animation_forward_velocity_similarity_exp: invalid std type: {type(std)} value={std}")
        raise
    if not torch.isfinite(torch.tensor(std_val, device=meas_yz.device, dtype=meas_yz.dtype)):
        print(f"[reward debug] animation_forward_velocity_similarity_exp: non-finite std: {std_val}")
        raise RuntimeError("animation_forward_velocity_similarity_exp received non-finite std")
    if std_val <= 0.0:
        print(f"[reward debug] animation_forward_velocity_similarity_exp: non-positive std: {std_val}")
        raise RuntimeError("animation_forward_velocity_similarity_exp requires std > 0")
    err_sq = torch.sum(torch.square(meas_yz - target_yz), dim=1)
    out = torch.exp(-err_sq / (std_val ** 2))
    if not torch.isfinite(out).all():
        num_nan = torch.isnan(out).sum().item()
        num_posinf = (out == float("inf")).sum().item()
        num_neginf = (out == float("-inf")).sum().item()
        print(f"[reward debug] animation_forward_velocity_similarity_exp: non-finite reward: NaN={num_nan} +Inf={num_posinf} -Inf={num_neginf} err_sq_min={float(torch.nanmin(err_sq).item()) if err_sq.numel() > 0 else 'n/a'} err_sq_max={float(torch.nanmax(err_sq).item()) if err_sq.numel() > 0 else 'n/a'} std={std_val}")
        raise RuntimeError("animation_forward_velocity_similarity_exp produced non-finite reward")
    return out


def animation_forward_velocity_similarity_world_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential tracking of world-frame XY linear velocity to animation target.

    Uses metadata key 'base_forward_velocity_mps' to set target vx (world +x), with vy target fixed to 0.
    reward = exp(- ( (vx_w - v_target)^2 + (vy_w - 0)^2 ) / std^2 )
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    anim = get_animation()
    meta = anim.get("metadata", {}) or {}
    if "base_forward_velocity_mps" not in meta or meta["base_forward_velocity_mps"] is None:
        raise RuntimeError("Animation metadata is missing required key 'base_forward_velocity_mps'")
    if meta["base_forward_velocity_mps"] <= 0.0:
        raise RuntimeError("Animation metadata key 'base_forward_velocity_mps' is less than or equal to 0.0")
    base_target = float(meta["base_forward_velocity_mps"])
    # Per-env playback speed scaling
    if not hasattr(env, "_anim_playback_speed"):
        raise RuntimeError("Missing _anim_playback_speed on env. Ensure init_animation_phase_offsets ran.")
    speed = env._anim_playback_speed  # type: ignore[attr-defined]
    if speed.dim() == 0:
        speed = speed.view(1).expand(asset.data.root_lin_vel_w.shape[0])
    v_target = base_target * speed.to(device=asset.device, dtype=asset.data.root_lin_vel_w.dtype)

    # Measured base linear velocity in world frame XY order: [vx, vy]
    vel_w = asset.data.root_lin_vel_w[:, :3]
    if not torch.isfinite(vel_w).all():
        num_nan = torch.isnan(vel_w).sum().item()
        num_posinf = (vel_w == float("inf")).sum().item()
        num_neginf = (vel_w == float("-inf")).sum().item()
        print(f"[reward debug] animation_forward_velocity_similarity_world_exp: non-finite vel_w: NaN={num_nan} +Inf={num_posinf} -Inf={num_neginf} shape={tuple(vel_w.shape)}")
        raise RuntimeError("animation_forward_velocity_similarity_world_exp encountered non-finite vel_w")
    meas_xy = vel_w[:, [0, 1]]

    # Broadcast per-env target [vx, vy] = [v_target, 0]
    target_xy = torch.stack([v_target, torch.zeros_like(v_target)], dim=1)
    
    err_sq = torch.sum(torch.square(meas_xy - target_xy), dim=1)
    out = torch.exp(-err_sq / (std ** 2))
    return out


def heading_xy_alignment_world_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential alignment of base +Z axis projected to world XY with world +X.

    Steps:
    - Apply base quaternion to +Z to obtain heading in world frame.
    - Project to XY plane and normalize.
    - Target is +X world (1, 0) in XY.
    - reward = exp(- ||h_xy - [1,0]||^2 / std^2)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_quat_w = asset.data.root_quat_w

    # Rotate base +Z into world frame
    z_b = torch.tensor([0.0, 0.0, 1.0], device=base_quat_w.device, dtype=base_quat_w.dtype)
    z_b = z_b.unsqueeze(0).expand(base_quat_w.shape[0], -1)
    z_w = quat_apply(base_quat_w, z_b)
    

    # Project to XY and normalize
    hx = z_w[:, 0]
    hy = z_w[:, 1]
    norm_xy = torch.sqrt(hx * hx + hy * hy)
    hx_n = hx / norm_xy
    hy_n = hy / norm_xy
    h_xy = torch.stack([hx_n, hy_n], dim=1)

    # Target +X
    target = torch.tensor([1.0, 0.0], dtype=h_xy.dtype, device=h_xy.device)


    std_val = float(std)

    err_sq = torch.sum(torch.square(h_xy - target), dim=1)
    return torch.exp(-err_sq / (std_val ** 2))

def track_lin_vel_yz_base_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands in base YZ plane using exponential kernel.

    Compares the commanded [vz, vy] (base frame) to measured base linear velocity [vz, vy].
    """
    asset = env.scene[asset_cfg.name]
    vel_b = asset.data.root_lin_vel_b[:, :3]
    cmd_yz = env.command_manager.get_command(command_name)[:, :2]  # [vz, vy]
    meas_yz = vel_b[:, [2, 1]]  # [vz, vy]
    lin_vel_error = torch.sum(torch.square(cmd_yz - meas_yz), dim=1)
    return torch.exp(-lin_vel_error / (std ** 2))


# def track_ang_vel_x_world_exp(
#     env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward tracking of roll angular velocity command (ωx) in world frame using exponential kernel."""
#     asset = env.scene[asset_cfg.name]
#     ang_vel_error = torch.square(
#         env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 0]
#     )
#     return torch.exp(-ang_vel_error / (std ** 2))


def track_lin_vel_xy_yaw_frame_exp_shamble(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)



def both_feet_air(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize when both feet are in the air.

    This function penalizes the agent when both feet are off the ground simultaneously, which helps
    promote stable bipedal locomotion by encouraging at least one foot to maintain ground contact.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.

    Returns:
        1 if both feet are airborne, 0 otherwise.
    """
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # # Check if feet are in contact (contact force > threshold)
    # in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # # Count feet in contact
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    feet_in_air = air_time > 0.0
    both_feet_air = torch.sum(feet_in_air.int(), dim=1) == 2
    return both_feet_air.float()


def lin_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize all base linear velocity using L2 squared kernel.

    This penalizes any movement of the robot's base in x, y, or z directions.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :3]), dim=1)


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize all base angular velocity using L2 squared kernel.

    This penalizes any rotation of the robot's base about x, y, or z axes.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :3]), dim=1)


def align_projected_gravity_to_target_l2(
    env: ManagerBasedRLEnv,
    target: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alignment of projected gravity with an arbitrary target direction in base frame.

    This uses the projected gravity in base frame ``g_b = asset.data.projected_gravity_b`` and
    a target direction (3,) or per-env targets (num_envs, 3). The target is normalized internally.

    reward = 1 - 0.5 * || g_b - \hat{target} ||^2, yielding values in [-1, 1].

    Args:
        env: RL environment.
        target: Desired gravity direction in base frame. Shape (3,) or (num_envs, 3).
        asset_cfg: Scene entity for the robot asset.

    Returns:
        Tensor of shape (num_envs,) with alignment rewards in [-1, 1].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # (num_envs, 3)

    # Prepare target on correct device/dtype and broadcast if needed
    if target.dim() == 1:
        target_b = target.unsqueeze(0).expand(g_b.shape[0], -1)
    elif target.dim() == 2 and target.shape[0] == g_b.shape[0] and target.shape[1] == 3:
        target_b = target
    else:
        raise ValueError("target must have shape (3,) or (num_envs, 3)")
    target_b = target_b.to(dtype=g_b.dtype, device=g_b.device)

    # Normalize target direction per env to avoid scale effects
    target_norm = torch.clamp(torch.norm(target_b, dim=1, keepdim=True), min=1e-6)
    target_b = target_b / target_norm

    dist_sq = torch.sum(torch.square(g_b - target_b), dim=1)  # in [0, 4]
    return 1.0 - 0.5 * dist_sq


def align_projected_gravity_plus_x_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alignment of projected gravity with +X axis in base frame using L2 mapping.

    reward = 1 - 0.5 * || g_b - [1, 0, 0] ||^2, in [-1, 1].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # (num_envs, 3)
    target = torch.tensor([1.0, 0.0, 0.0], dtype=g_b.dtype, device=g_b.device)
    dist_sq = torch.sum(torch.square(g_b - target), dim=1)
    return 1.0 - 0.5 * dist_sq


def misalign_projected_gravity_plus_x_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for misalignment of projected gravity with +X axis in base frame using L2 mapping.

    penalty = 0.5 * || g_b - [1, 0, 0] ||^2, in [0, 2].
    Larger penalty -> more misaligned.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # (num_envs, 3)
    target = torch.tensor([1.0, 0.0, 0.0], dtype=g_b.dtype, device=g_b.device)
    dist_sq = torch.sum(torch.square(g_b - target), dim=1)
    return 0.5 * dist_sq


def flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def flat_orientation_bonus_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential bonus for flat base orientation using projected gravity xy.

    reward = exp(- ||g_b.xy||^2 / std^2), bounded in (0, 1].

    Args:
        env: RL environment.
        std: Standard deviation controlling falloff (must be > 0).
        asset_cfg: Scene entity for the robot asset.

    Returns:
        Tensor of shape (num_envs,) with bonuses in [0, 1].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_xy = asset.data.projected_gravity_b[:, :2]
    std_val = float(std)
    if std_val <= 0.0:
        raise RuntimeError("flat_orientation_bonus_exp requires std > 0")
    dist_sq = torch.sum(torch.square(g_xy), dim=1)
    return torch.exp(-dist_sq / (std_val ** 2))


def flat_orientation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward-style mapping for flat orientation.

    Converts the flat-orientation penalty into a bounded reward similar to
    :func:`align_projected_gravity_plus_x_l2` by mapping
    reward = 1 - 0.5 * || g_b.xy ||^2.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_xy = asset.data.projected_gravity_b[:, :2]
    dist_sq = torch.sum(torch.square(g_xy), dim=1)
    return 1.0 - 0.5 * dist_sq


def command_based_orientation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale_crawl: float = 1.0,
    scale_stand: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Command-based orientation reward with consistent positive semantics.

    - If command == 0 (crawl): use align_projected_gravity_plus_x_l2 (reward style in [-1, 1]).
    - If command == 1 (stand): use flat_orientation_reward (reward style similar bounds).

    The output is a reward-like value that can be scaled by the outer RewTerm ``weight``.
    Optional ``scale_crawl``/``scale_stand`` let you rebalance branches without changing sign.
    """
    # Command mask (num_envs,)
    command = env.command_manager.get_command(command_name)
    command_values = command[:, 0]

    crawl_term = align_projected_gravity_plus_x_l2(env, asset_cfg=asset_cfg)
    stand_term = flat_orientation_reward(env, asset_cfg=asset_cfg)

    mask = (command_values > 0.5)
    # Select weighted terms per env
    out = torch.where(mask, scale_stand * stand_term, scale_crawl * crawl_term)
    return out


def command_based_orientation_l2_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Command-aware L2 penalty for orientation.

    - Crawl (command==0): penalty = || g_b - [1,0,0] ||^2
    - Stand (command==1): penalty = || g_b.xy ||^2
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # (N, 3)

    target_crawl = torch.tensor([1.0, 0.0, 0.0], dtype=g_b.dtype, device=g_b.device)
    dist_sq_crawl = torch.sum(torch.square(g_b - target_crawl), dim=1)

    g_xy = g_b[:, :2]
    dist_sq_stand = torch.sum(torch.square(g_xy), dim=1)

    command = env.command_manager.get_command(command_name)
    command_values = command[:, 0]
    mask = (command_values > 0.5)
    return torch.where(mask, dist_sq_stand, dist_sq_crawl)


def command_based_orientation_bonus_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_crawl: float,
    std_stand: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential bonus close to command-specific orientation target.

    Thin wrapper around :func:`command_based_orientation_proximity_exp` for naming consistency.
    """
    return command_based_orientation_proximity_exp(
        env,
        command_name=command_name,
        std_crawl=std_crawl,
        std_stand=std_stand,
        asset_cfg=asset_cfg,
    )

def command_based_orientation_proximity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_crawl: float,
    std_stand: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Smooth proximity bonus to the command-specific orientation target using an exponential kernel.

    - Crawl (command==0): proximity to +X target in base frame: exp(-||g_b - [1,0,0]||^2 / std_crawl^2)
    - Stand (command==1): proximity to flat: exp(-||g_b.xy||^2 / std_stand^2)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # (N, 3)

    # Validate stds
    if float(std_crawl) <= 0.0 or float(std_stand) <= 0.0:
        raise RuntimeError("command_based_orientation_proximity_exp requires std_crawl>0 and std_stand>0")

    # Compute both branch rewards
    target_crawl = torch.tensor([1.0, 0.0, 0.0], dtype=g_b.dtype, device=g_b.device)
    dist_sq_crawl = torch.sum(torch.square(g_b - target_crawl), dim=1)
    prox_crawl = torch.exp(-dist_sq_crawl / (float(std_crawl) ** 2))

    g_xy = g_b[:, :2]
    dist_sq_stand = torch.sum(torch.square(g_xy), dim=1)
    prox_stand = torch.exp(-dist_sq_stand / (float(std_stand) ** 2))

    command = env.command_manager.get_command(command_name)
    command_values = command[:, 0]
    mask = (command_values > 0.5)
    return torch.where(mask, prox_stand, prox_crawl)


def command_based_orientation_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    tol_crawl: float,
    tol_stand: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sharp success bonus when orientation is within a tolerance of the command target.

    - Crawl (command==0): success if ||g_b - [1,0,0]|| <= tol_crawl
    - Stand (command==1): success if ||g_b.xy|| <= tol_stand
    Returns 1.0 on success else 0.0, to be scaled by the RewTerm weight.
    """
    if float(tol_crawl) <= 0.0 or float(tol_stand) <= 0.0:
        raise RuntimeError("command_based_orientation_success_bonus requires tol_crawl>0 and tol_stand>0")

    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # (N, 3)

    target_crawl = torch.tensor([1.0, 0.0, 0.0], dtype=g_b.dtype, device=g_b.device)
    err_crawl = torch.linalg.norm(g_b - target_crawl, dim=1)
    ok_crawl = (err_crawl <= float(tol_crawl)).to(dtype=g_b.dtype)

    err_stand = torch.linalg.norm(g_b[:, :2], dim=1)
    ok_stand = (err_stand <= float(tol_stand)).to(dtype=g_b.dtype)

    command = env.command_manager.get_command(command_name)
    command_values = command[:, 0]
    mask = (command_values > 0.5)
    return torch.where(mask, ok_stand, ok_crawl)


def animation_contact_flags_mismatch_l1(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    label_groups: list[list[int]],
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """L1 mismatch between animation contact flags and measured contacts.

    - Expects the animation JSON to contain per-frame contact flags under key ``contact_flags``
      shaped [T, K], with ordering described by ``metadata.contact.order``.
    - ``label_groups`` must contain K groups, each being a list of indices into ``sensor_cfg.body_ids``
      corresponding to the bodies that realize contact for that label (logical OR across the group).
    - Measured contacts are computed from the contact sensor net forces with a force threshold.

    Returns a per-env penalty equal to the sum over labels of |expected - measured| in [0, K].
    """
    # Retrieve expected contact flags and ordering from animation
    anim = get_animation()
    contact_flags = anim.get("contact_flags", None)
    if contact_flags is None:
        raise RuntimeError("Animation JSON is missing 'contact_flags'; cannot compute contact match reward")
    contact_order = anim.get("contact_order", None)
    if contact_order is None:
        raise RuntimeError("Animation JSON is missing 'metadata.contact.order'; cannot align contact labels")

    # Validate label groups length matches flags dimension
    K = int(contact_flags.shape[1])
    if len(label_groups) != K:
        raise RuntimeError(
            f"label_groups length {len(label_groups)} must match number of contact labels {K} from animation"
        )

    # Current frame index per env
    _, frame_idx = compute_animation_phase_and_frame(env)

    # Expected flags for each env at current frame
    expected: torch.Tensor = contact_flags[frame_idx]  # [N, K], on GPU
    if not torch.isfinite(expected).all():
        num_nan = torch.isnan(expected).sum().item()
        num_posinf = (expected == float("inf")).sum().item()
        num_neginf = (expected == float("-inf")).sum().item()
        print(f"[reward debug] animation_contact_flags_mismatch_l1: non-finite expected flags: NaN={num_nan} +Inf={num_posinf} -Inf={num_neginf} shape={tuple(expected.shape)}")
        raise RuntimeError("animation_contact_flags_mismatch_l1 encountered non-finite expected flags")

    # Measured contacts per body from sensor; produce [N, B] boolean
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history: [N, H, B, 3] -> norm over xyz, max over history -> [N, B]
    forces_hist = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    forces_mag = forces_hist.norm(dim=-1)
    if not torch.isfinite(forces_mag).all():
        num_nan = torch.isnan(forces_mag).sum().item()
        num_posinf = (forces_mag == float("inf")).sum().item()
        num_neginf = (forces_mag == float("-inf")).sum().item()
        print(f"[reward debug] animation_contact_flags_mismatch_l1: non-finite forces_mag: NaN={num_nan} +Inf={num_posinf} -Inf={num_neginf} shape={tuple(forces_mag.shape)}")
        raise RuntimeError("animation_contact_flags_mismatch_l1 encountered non-finite forces magnitude from sensor")
    contacts_per_body = (forces_mag.max(dim=1)[0] > float(force_threshold))  # [N, B]

    # Aggregate per label group via OR across group members
    N = contacts_per_body.shape[0]
    measured = torch.zeros((N, K), dtype=expected.dtype, device=expected.device)
    for k in range(K):
        idxs = label_groups[k]
        if not isinstance(idxs, (list, tuple)) or len(idxs) == 0:
            raise RuntimeError(f"label_groups[{k}] must be a non-empty list of indices")
        # Validate indices bounds against sensor_cfg.body_ids length
        B = int(contacts_per_body.shape[1])
        if any((int(i) < 0 or int(i) >= B) for i in idxs):
            raise RuntimeError(f"label_groups[{k}] contains indices out of range for sensor body set of size {B}")
        measured[:, k] = contacts_per_body[:, idxs].any(dim=1).to(dtype=expected.dtype)
    if not torch.isfinite(measured).all():
        num_nan = torch.isnan(measured).sum().item()
        num_posinf = (measured == float("inf")).sum().item()
        num_neginf = (measured == float("-inf")).sum().item()
        print(f"[reward debug] animation_contact_flags_mismatch_l1: non-finite measured flags: NaN={num_nan} +Inf={num_posinf} -Inf={num_neginf} shape={tuple(measured.shape)}")
        raise RuntimeError("animation_contact_flags_mismatch_l1 encountered non-finite measured flags")

    # L1 mismatch per env
    penalty = torch.sum(torch.abs(expected - measured), dim=1)
    if not torch.isfinite(penalty).all():
        num_nan = torch.isnan(penalty).sum().item()
        num_posinf = (penalty == float("inf")).sum().item()
        num_neginf = (penalty == float("-inf")).sum().item()
        print(f"[reward debug] animation_contact_flags_mismatch_l1: non-finite penalty: NaN={num_nan} +Inf={num_posinf} -Inf={num_neginf} shape={tuple(penalty.shape)}")
        raise RuntimeError("animation_contact_flags_mismatch_l1 produced non-finite penalty")
    return penalty


def animation_contact_flags_mismatch_feet_l1(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Strict feet contact mismatch penalty using FL/FR/RL/RR ordering.

    Requirements (fail loudly if violated):
    - Animation must define contact_flags and metadata.contact.order == ["FL","FR","RL","RR"].
    - Sensor entity must select exactly 4 bodies in the order [FL, FR, RL, RR].
    """
    anim = get_animation()
    contact_order = anim.get("contact_order", None)
    if contact_order is None:
        raise RuntimeError("Animation missing contact_order; ensure metadata.contact.order is present")
    expected_order = ["FL", "FR", "RL", "RR"]
    if list(contact_order) != expected_order:
        raise RuntimeError(
            f"metadata.contact.order must be {expected_order}, got {list(contact_order)}"
        )

    # Validate sensor body selection count
    # Note: We can't access names here, so rely on env config to pass bodies in required order.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # sensor_cfg.body_ids is resolved by managers; ensure 4 ids were selected
    num_bodies = int(len(sensor_cfg.body_ids)) if hasattr(sensor_cfg, "body_ids") else -1
    if num_bodies != 4:
        raise RuntimeError(
            f"sensor_cfg must select exactly 4 bodies in FL,FR,RL,RR order; got {num_bodies}"
        )

    label_groups = [[0], [1], [2], [3]]
    return animation_contact_flags_mismatch_l1(env, sensor_cfg, label_groups, force_threshold=force_threshold)

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.01
    return reward


# ============================================
# Command-conditional locomotion rewards
# ============================================

def track_lin_vel_xy_yaw_frame_exp_when_standing(
    env, std: float, command_name: str, boolean_command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands only when standing (boolean_command==1).
    
    When crawling (boolean_command==0), returns 0 to encourage staying in place.
    
    Args:
        env: Environment instance
        std: Standard deviation for exponential kernel
        command_name: Name of velocity command
        boolean_command_name: Name of boolean command (0=crawl, 1=stand)
        asset_cfg: Robot asset configuration
    
    Returns:
        Velocity tracking reward when standing, 0 when crawling
    """
    # Get boolean command
    boolean_cmd = env.command_manager.get_command(boolean_command_name)[:, 0]
    standing_mask = (boolean_cmd > 0.5)
    
    # Compute base reward
    base_reward = track_lin_vel_xy_yaw_frame_exp_shamble(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
    
    # Zero out when crawling
    return base_reward * standing_mask.float()


def track_ang_vel_z_world_exp_when_standing(
    env, command_name: str, boolean_command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands only when standing (boolean_command==1).
    
    When crawling (boolean_command==0), returns 0 to encourage staying in place.
    
    Args:
        env: Environment instance
        command_name: Name of velocity command
        boolean_command_name: Name of boolean command (0=crawl, 1=stand)
        std: Standard deviation for exponential kernel
        asset_cfg: Robot asset configuration
    
    Returns:
        Angular velocity tracking reward when standing, 0 when crawling
    """
    # Get boolean command
    boolean_cmd = env.command_manager.get_command(boolean_command_name)[:, 0]
    standing_mask = (boolean_cmd > 0.5)
    
    # Compute base reward
    base_reward = track_ang_vel_z_world_exp(env, command_name=command_name, std=std, asset_cfg=asset_cfg)
    
    # Zero out when crawling
    return base_reward * standing_mask.float()


def feet_air_time_positive_biped_when_standing(
    env, velocity_command_name: str, boolean_command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps/gait quality only when standing (boolean_command==1).
    
    When crawling (boolean_command==0), returns 0.
    
    Args:
        env: Environment instance
        velocity_command_name: Name of velocity command for gating
        boolean_command_name: Name of boolean command (0=crawl, 1=stand)
        threshold: Air time threshold
        sensor_cfg: Contact sensor configuration
    
    Returns:
        Feet air time reward when standing, 0 when crawling
    """
    # Get boolean command
    boolean_cmd = env.command_manager.get_command(boolean_command_name)[:, 0]
    standing_mask = (boolean_cmd > 0.5)
    
    # Compute base reward
    base_reward = feet_air_time_positive_biped(env, command_name=velocity_command_name, threshold=threshold, sensor_cfg=sensor_cfg)
    
    # Zero out when crawling
    return base_reward * standing_mask.float()


def both_feet_air_when_standing(env, boolean_command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize both feet in air only when standing (boolean_command==1).
    
    When crawling (boolean_command==0), returns 0 (no penalty).
    
    Args:
        env: Environment instance
        boolean_command_name: Name of boolean command (0=crawl, 1=stand)
        sensor_cfg: Contact sensor configuration
    
    Returns:
        Both feet air penalty when standing, 0 when crawling
    """
    # Get boolean command
    boolean_cmd = env.command_manager.get_command(boolean_command_name)[:, 0]
    standing_mask = (boolean_cmd > 0.5)
    
    # Compute base penalty
    base_penalty = both_feet_air(env, sensor_cfg=sensor_cfg)
    
    # Zero out when crawling
    return base_penalty * standing_mask.float()


def feet_slide_when_standing(
    env, boolean_command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding only when standing (boolean_command==1).
    
    When crawling (boolean_command==0), returns 0 (no penalty).
    
    Args:
        env: Environment instance
        boolean_command_name: Name of boolean command (0=crawl, 1=stand)
        sensor_cfg: Contact sensor configuration
        asset_cfg: Robot asset configuration
    
    Returns:
        Feet slide penalty when standing, 0 when crawling
    """
    # Get boolean command
    boolean_cmd = env.command_manager.get_command(boolean_command_name)[:, 0]
    standing_mask = (boolean_cmd > 0.5)
    
    # Compute base penalty
    base_penalty = feet_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)
    
    # Zero out when crawling
    return base_penalty * standing_mask.float()


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty for base height deviation from a fixed target height.
    
    Args:
        env: Environment instance
        target_height: Target height (meters)
        asset_cfg: Robot asset configuration with body_names specified
    
    Returns:
        L2 squared height error penalty
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # Z position
    height_error = body_pos_w - target_height
    return torch.sum(torch.square(height_error), dim=1)


def base_height_l2_sensor(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def base_height_l2_command_based(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height_crawl: float,
    target_height_stand: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty for base height deviation from command-based target.
    
    When crawling (command==0): target is target_height_crawl
    When standing (command==1): target is target_height_stand
    
    Args:
        env: Environment instance
        command_name: Name of boolean command (0=crawl, 1=stand)
        target_height_crawl: Target height when crawling (meters)
        target_height_stand: Target height when standing (meters)
        asset_cfg: Robot asset configuration
    
    Returns:
        L2 squared height error penalty
    """
    # Get boolean command
    command = env.command_manager.get_command(command_name)[:, 0]
    standing_mask = (command > 0.5)
    
    # Get asset and body position
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # Z position
    
    # Select target height based on command
    target_height = torch.where(
        standing_mask,
        torch.full_like(body_pos_w, target_height_stand),
        torch.full_like(body_pos_w, target_height_crawl)
    )
    
    # Compute L2 squared error
    height_error = body_pos_w - target_height
    return torch.sum(torch.square(height_error), dim=1)


def com_forward_of_feet(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Reward when the center of mass is positioned forward relative to the feet on the ground plane.
    
    This encourages the robot to lean forward during standing-to-crawling transitions.
    Without forward weight shift, the robot will fall backward.
    
    Uses a gravity-aligned yaw frame to project everything onto the ground plane.
    This makes the reward work correctly in both standing (upright) and crawling (pitched forward) poses.
    
    The reward is computed as:
    reward = com_x_yaw - mean_feet_x_yaw
    
    where positions are in a gravity-aligned coordinate frame (yaw rotation only, no pitch/roll).
    
    Positive values indicate the COM is forward of the feet on the ground plane (stable).
    Negative values indicate the COM is behind the feet (will fall backward).
    
    Args:
        env: Environment instance
        asset_cfg: Configuration for the robot asset
        feet_cfg: Configuration for the feet bodies (should select ankle/foot links)
    
    Returns:
        Tensor of shape (num_envs,) with forward offset in meters (ground plane projection)
    """
    from isaaclab.utils.math import yaw_quat
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get gravity-aligned yaw frame (removes pitch and roll, keeps only yaw)
    # This gives us the robot's heading direction projected onto the ground plane
    yaw_only_quat = yaw_quat(asset.data.root_link_quat_w)  # [N, 4]
    
    # Transform COM to gravity-aligned yaw frame
    com_pos_w = asset.data.root_com_pos_w  # [N, 3]
    com_yaw_frame = quat_apply_inverse(yaw_only_quat, com_pos_w)  # [N, 3]
    com_x_yaw = com_yaw_frame[:, 0]  # [N] - X component in yaw frame (forward on ground)
    
    # Transform feet to gravity-aligned yaw frame
    feet_pos_w = asset.data.body_pos_w[:, feet_cfg.body_ids, :]  # [N, num_feet, 3]
    
    # Transform each foot position to yaw frame
    N, num_feet, _ = feet_pos_w.shape
    feet_pos_w_flat = feet_pos_w.reshape(N * num_feet, 3)  # [N*num_feet, 3]
    yaw_quat_expanded = yaw_only_quat.unsqueeze(1).expand(-1, num_feet, -1).reshape(N * num_feet, 4)
    feet_yaw_frame_flat = quat_apply_inverse(yaw_quat_expanded, feet_pos_w_flat)  # [N*num_feet, 3]
    feet_yaw_frame = feet_yaw_frame_flat.reshape(N, num_feet, 3)  # [N, num_feet, 3]
    
    feet_x_yaw_mean = feet_yaw_frame[:, :, 0].mean(dim=1)  # [N] - average X in yaw frame
    
    # Reward = how far forward the COM is relative to feet on the ground plane
    # Positive values mean COM is ahead of feet (stable, leaning forward)
    forward_offset = com_x_yaw - feet_x_yaw_mean
    
    return forward_offset


def com_forward_of_feet_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Exponential bonus for having the center of mass at a target forward offset relative to feet.
    
    This encourages the robot to maintain a specific forward lean, important for stable
    crawling transitions. Uses an exponential kernel centered at target_offset.
    
    Uses the actual center of mass position (root_com_pos_w) rather than a proxy body link.
    
    reward = exp(- (offset - target)^2 / std^2)
    
    Args:
        env: Environment instance
        std: Standard deviation controlling reward falloff (smaller = sharper)
        target_offset: Target forward offset in meters (positive = forward)
        asset_cfg: Configuration for the robot asset
        feet_cfg: Configuration for the feet bodies (should select ankle/foot links)
    
    Returns:
        Tensor of shape (num_envs,) with exponential bonus in [0, 1]
    """
    if float(std) <= 0.0:
        raise RuntimeError("com_forward_of_feet_exp requires std > 0")
    
    # Get current forward offset using actual COM
    forward_offset = com_forward_of_feet(env, asset_cfg, feet_cfg)
    
    # Exponential bonus around target
    error_sq = torch.square(forward_offset - float(target_offset))
    return torch.exp(-error_sq / (float(std) ** 2))


def com_forward_of_feet_linear(
    env: ManagerBasedRLEnv,
    tolerance: float,
    target_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Linear bonus for having the center of mass at a target forward offset relative to feet.
    
    This encourages the robot to maintain a specific forward/backward position of COM
    relative to feet on the ground plane. Uses linear kernel centered at target_offset.
    
    Uses the actual center of mass position (root_com_pos_w) and projects positions onto
    the ground plane using a gravity-aligned yaw frame (same logic as com_forward_of_feet).
    
    reward = max(0, 1 - |offset - target| / tolerance)
    
    When offset = target: reward = 1.0
    When |offset - target| >= tolerance: reward = 0.0
    Linear falloff in between.
    
    Args:
        env: Environment instance
        tolerance: Distance tolerance controlling reward falloff (smaller = sharper)
        target_offset: Target forward offset in meters (positive = forward, negative = backward)
        asset_cfg: Configuration for the robot asset
        feet_cfg: Configuration for the feet bodies (should select ankle/foot links)
    
    Returns:
        Tensor of shape (num_envs,) with linear bonus in [0, 1]
    """
    if float(tolerance) <= 0.0:
        raise RuntimeError("com_forward_of_feet_linear requires tolerance > 0")
    
    # Get current forward offset using actual COM (ground plane projection)
    forward_offset = com_forward_of_feet(env, asset_cfg, feet_cfg)
    
    # Linear bonus around target
    error = torch.abs(forward_offset - float(target_offset))
    reward = torch.clamp(1.0 - error / float(tolerance), min=0.0, max=1.0)
    return reward


def com_aligned_with_velocity_command_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Reward when CoM is shifted in the direction of velocity command (leaning into movement).
    
    This encourages the robot to shift its center of mass in the direction it's walking,
    which is essential for stable dynamic locomotion. The robot should lean forward when
    walking forward, sideways when walking sideways, etc.
    
    Computes the CoM position relative to feet in the gravity-aligned yaw frame (XY plane),
    then rewards alignment with the velocity command direction using an exponential kernel.
    
    reward = exp(- ||com_offset_xy - command_xy * scale||^2 / std^2)
    
    Args:
        env: Environment instance
        command_name: Name of the velocity command (expects [vx, vy, vz] format)
        std: Standard deviation controlling reward falloff (smaller = sharper)
        asset_cfg: Configuration for the robot asset
        feet_cfg: Configuration for the feet bodies (should select ankle/foot links)
    
    Returns:
        Tensor of shape (num_envs,) with exponential reward in [0, 1]
    """
    if float(std) <= 0.0:
        raise RuntimeError("com_aligned_with_velocity_command_exp requires std > 0")
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get gravity-aligned yaw frame (removes pitch and roll, keeps only yaw)
    yaw_only_quat = yaw_quat(asset.data.root_link_quat_w)  # [N, 4]
    
    # Get CoM position relative to feet in yaw frame (ground plane projection)
    com_pos_w = asset.data.root_com_pos_w  # [N, 3]
    com_yaw_frame = quat_apply_inverse(yaw_only_quat, com_pos_w)  # [N, 3]
    
    feet_pos_w = asset.data.body_pos_w[:, feet_cfg.body_ids, :]  # [N, num_feet, 3]
    N, num_feet, _ = feet_pos_w.shape
    feet_pos_w_flat = feet_pos_w.reshape(N * num_feet, 3)  # [N*num_feet, 3]
    yaw_quat_expanded = yaw_only_quat.unsqueeze(1).expand(-1, num_feet, -1).reshape(N * num_feet, 4)
    feet_yaw_frame_flat = quat_apply_inverse(yaw_quat_expanded, feet_pos_w_flat)  # [N*num_feet, 3]
    feet_yaw_frame = feet_yaw_frame_flat.reshape(N, num_feet, 3)  # [N, num_feet, 3]
    
    feet_center_yaw = feet_yaw_frame.mean(dim=1)  # [N, 3] - center of feet in yaw frame
    
    # CoM offset from feet center in XY (yaw frame ground plane)
    com_offset_xy = com_yaw_frame[:, :2] - feet_center_yaw[:, :2]  # [N, 2]
    
    # Get velocity command in yaw frame (XY only)
    cmd = env.command_manager.get_command(command_name)[:, :2]  # [N, 2] - [vx, vy]
    
    # Scale the command to a reasonable CoM offset target
    # Typical CoM offset for leaning might be ~0.05-0.15m per 1 m/s of velocity
    scale = 0.1  # meters of CoM shift per m/s of velocity command
    target_com_offset = cmd * scale  # [N, 2]
    
    # Exponential reward for CoM offset matching scaled velocity command
    error_sq = torch.sum(torch.square(com_offset_xy - target_com_offset), dim=1)
    reward = torch.exp(-error_sq / (float(std) ** 2))
    
    return reward


def com_centered_between_feet_and_hands_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    hands_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_wrist_link"),
) -> torch.Tensor:
    """Exponential reward for keeping CoM centered between feet and hands contact points.
    
    This encourages the robot to maintain balance during crawling by keeping the center
    of mass positioned at the geometric center of all contact points (feet + hands).
    This prevents falling forward or backward during quadruped locomotion.
    
    Uses gravity-aligned yaw frame to project all positions onto the ground plane.
    
    reward = exp(- ||com_xy - contact_center_xy||^2 / std^2)
    
    Args:
        env: Environment instance
        std: Standard deviation controlling reward falloff (smaller = sharper)
        asset_cfg: Configuration for the robot asset
        feet_cfg: Configuration for the feet bodies (should select ankle/foot links)
        hands_cfg: Configuration for the hand bodies (should select wrist links)
    
    Returns:
        Tensor of shape (num_envs,) with exponential reward in [0, 1]
    """
    if float(std) <= 0.0:
        raise RuntimeError("com_centered_between_feet_and_hands_exp requires std > 0")
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get gravity-aligned yaw frame (removes pitch and roll, keeps only yaw)
    yaw_only_quat = yaw_quat(asset.data.root_link_quat_w)  # [N, 4]
    
    # Transform CoM to yaw frame (ground plane projection)
    com_pos_w = asset.data.root_com_pos_w  # [N, 3]
    com_yaw_frame = quat_apply_inverse(yaw_only_quat, com_pos_w)  # [N, 3]
    com_xy = com_yaw_frame[:, :2]  # [N, 2] - CoM in ground plane
    
    # Transform feet positions to yaw frame
    feet_pos_w = asset.data.body_pos_w[:, feet_cfg.body_ids, :]  # [N, num_feet, 3]
    N, num_feet, _ = feet_pos_w.shape
    feet_pos_w_flat = feet_pos_w.reshape(N * num_feet, 3)  # [N*num_feet, 3]
    yaw_quat_feet = yaw_only_quat.unsqueeze(1).expand(-1, num_feet, -1).reshape(N * num_feet, 4)
    feet_yaw_frame_flat = quat_apply_inverse(yaw_quat_feet, feet_pos_w_flat)  # [N*num_feet, 3]
    feet_yaw_frame = feet_yaw_frame_flat.reshape(N, num_feet, 3)  # [N, num_feet, 3]
    feet_xy = feet_yaw_frame[:, :, :2]  # [N, num_feet, 2]
    
    # Transform hands positions to yaw frame
    hands_pos_w = asset.data.body_pos_w[:, hands_cfg.body_ids, :]  # [N, num_hands, 3]
    _, num_hands, _ = hands_pos_w.shape
    hands_pos_w_flat = hands_pos_w.reshape(N * num_hands, 3)  # [N*num_hands, 3]
    yaw_quat_hands = yaw_only_quat.unsqueeze(1).expand(-1, num_hands, -1).reshape(N * num_hands, 4)
    hands_yaw_frame_flat = quat_apply_inverse(yaw_quat_hands, hands_pos_w_flat)  # [N*num_hands, 3]
    hands_yaw_frame = hands_yaw_frame_flat.reshape(N, num_hands, 3)  # [N, num_hands, 3]
    hands_xy = hands_yaw_frame[:, :, :2]  # [N, num_hands, 2]
    
    # Compute center of all contact points (feet + hands)
    # Concatenate along the contact point dimension
    all_contacts_xy = torch.cat([feet_xy, hands_xy], dim=1)  # [N, num_feet + num_hands, 2]
    contact_center_xy = all_contacts_xy.mean(dim=1)  # [N, 2] - geometric center of contacts
    
    # Compute error between CoM and contact center
    error_sq = torch.sum(torch.square(com_xy - contact_center_xy), dim=1)
    
    # Exponential reward - maximum when CoM is centered
    reward = torch.exp(-error_sq / (float(std) ** 2))
    
    return reward

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def small_command_responsiveness(
    env,
    command_name: str,
    small_threshold: float = 0.2,
    min_threshold: float = 0.01,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward responsiveness to small non-zero velocity commands.
    
    This term activates when commanded velocity magnitude is between min_threshold 
    and small_threshold, encouraging the robot to move rather than remain stationary.
    It rewards having non-zero base velocity in the commanded direction.
    
    This prevents the robot from ignoring small commands and helps it respond to 
    fine-grained velocity control.
    
    Args:
        env: Environment instance.
        command_name: Name of velocity command (expects [vx, vy, vz] format).
        small_threshold: Upper threshold for "small" commands (default: 0.2 m/s).
        min_threshold: Lower threshold to ignore truly zero commands (default: 0.01 m/s).
        std: Standard deviation for exponential reward kernel (default: 0.1).
        asset_cfg: Robot asset configuration.
    
    Returns:
        Reward in [0, 1] when command is small, 0 otherwise.
    """
    asset = env.scene[asset_cfg.name]
    
    # Get commanded velocity magnitude (xy plane)
    cmd = env.command_manager.get_command(command_name)[:, :2]  # [N, 2]
    cmd_mag = torch.norm(cmd, dim=1)  # [N]
    
    # Only activate for small non-zero commands
    is_small = (cmd_mag >= min_threshold) & (cmd_mag <= small_threshold)
    
    # Get actual velocity in yaw frame (xy plane)
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    vel_xy = vel_yaw[:, :2]  # [N, 2]
    vel_mag = torch.norm(vel_xy, dim=1)  # [N]
    
    # Reward having velocity in roughly the commanded direction
    # Use exponential kernel centered on matching the command magnitude
    vel_error = torch.square(vel_mag - cmd_mag)
    reward = torch.exp(-vel_error / (std ** 2))
    
    # Only apply reward when in small command regime
    return reward * is_small.float()


# def track_ang_vel_z_world_exp(
#     env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
#     return torch.exp(-ang_vel_error / std**2)
def _steps_since_reset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-env steps since episode reset as a tensor on the sim device."""
    # Isaac Lab commonly exposes one of these names:
    for name in ("episode_step_counter", "episode_step_count", "episode_length_buf"):
        if hasattr(env, name):
            t = getattr(env, name)
            # ensure it's a 1-D tensor [num_envs] on the right device/dtype
            return t.to(device=env.device)
    raise AttributeError(
        "Could not find a per-env steps-since-reset counter on env. "
        "Try exposing `episode_step_counter` (int32/64, shape [num_envs])."
    )

def _step_dt(env: ManagerBasedRLEnv) -> float:
    # Isaac Lab usually exposes env.step_dt
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    # fallback if your build differs
    if hasattr(env, "sim") and hasattr(env.sim, "dt"):
        return float(env.sim.dt)
    raise AttributeError("Could not determine step dt (looked for env.step_dt and env.sim.dt).")


def joint_position_rate_violation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rate_limit: float = 15.0,  # rad/s equivalent from absolute position jump check
) -> torch.Tensor:
    """Penalty when absolute joint position rate |Δq|/dt exceeds rate_limit.

    Mirrors the deploy-time "absolute position jump" check by computing finite
    differences of joint positions using the simulation dt.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qpos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    if not torch.isfinite(qpos).all():
        raise RuntimeError("joint_position_rate_violation_penalty received non-finite joint positions")

    dt = _step_dt(env)
    if not (dt > 0.0):
        raise RuntimeError(f"joint_position_rate_violation_penalty invalid dt: {dt}")

    prev_key = "_safety_prev_joint_pos"
    if not hasattr(env, prev_key):
        setattr(env, prev_key, asset.data.joint_pos.detach().clone())
    prev_full: torch.Tensor = getattr(env, prev_key)
    if prev_full.shape != asset.data.joint_pos.shape or prev_full.device != asset.data.joint_pos.device:
        prev_full = asset.data.joint_pos.detach().clone()
        setattr(env, prev_key, prev_full)

    prev_sel = prev_full[:, asset_cfg.joint_ids]
    rate = torch.abs(qpos - prev_sel) / float(dt)
    if not torch.isfinite(rate).all():
        raise RuntimeError("joint_position_rate_violation_penalty produced non-finite position rates")

    excess = torch.clamp(rate - float(rate_limit), min=0.0)
    penalty = torch.sum(excess * excess, dim=1)

    setattr(env, prev_key, asset.data.joint_pos.detach().clone())
    return penalty


def joint_velocity_violation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_limit: float = 25.0,  # rad/s
) -> torch.Tensor:
    """Penalty when |qdot| exceeds vel_limit (velocity spike)."""
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if not torch.isfinite(qvel).all():
        raise RuntimeError("joint_velocity_violation_penalty received non-finite joint velocities")

    excess = torch.clamp(torch.abs(qvel) - float(vel_limit), min=0.0)
    return torch.sum(excess * excess, dim=1)


def joint_acceleration_violation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    acc_limit: float = 1500.0,  # rad/s^2
) -> torch.Tensor:
    """Penalty when |qddot| exceeds acc_limit (acceleration spike).

    Uses finite differences of joint velocities with simulation dt.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if not torch.isfinite(qvel).all():
        raise RuntimeError("joint_acceleration_violation_penalty received non-finite joint velocities")

    dt = _step_dt(env)
    if not (dt > 0.0):
        raise RuntimeError(f"joint_acceleration_violation_penalty invalid dt: {dt}")

    prev_key = "_safety_prev_joint_vel"
    if not hasattr(env, prev_key):
        setattr(env, prev_key, env.scene[asset_cfg.name].data.joint_vel.detach().clone())
    prev_full: torch.Tensor = getattr(env, prev_key)
    if prev_full.shape != env.scene[asset_cfg.name].data.joint_vel.shape or prev_full.device != env.scene[asset_cfg.name].data.joint_vel.device:
        prev_full = env.scene[asset_cfg.name].data.joint_vel.detach().clone()
        setattr(env, prev_key, prev_full)

    prev_sel = prev_full[:, asset_cfg.joint_ids]
    qacc = (qvel - prev_sel) / float(dt)
    if not torch.isfinite(qacc).all():
        raise RuntimeError("joint_acceleration_violation_penalty produced non-finite accelerations")

    excess = torch.clamp(torch.abs(qacc) - float(acc_limit), min=0.0)
    penalty = torch.sum(excess * excess, dim=1)

    setattr(env, prev_key, env.scene[asset_cfg.name].data.joint_vel.detach().clone())
    return penalty

def pose_json_deviation_l1_after_delay(
    env: ManagerBasedRLEnv,
    pose_path: str = "assets/default-pose.json",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delay_s: float = 1.0,           # time before this term activates
    ramp_s: float = 0.0,            # optional: smoothly ramp in over this many seconds
) -> torch.Tensor:
    """L1 penalty for joint deviations from a JSON pose, activated after a time delay.

    Before `delay_s`, this returns 0. Between `delay_s` and `delay_s + ramp_s`, the penalty
    linearly ramps in (if ramp_s > 0). After that, full strength.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Build target vector in joint order (cache inside your helper if desired)
    target_full = _get_full_target_vector(pose_path, asset)

    # Gather joint positions in the selected order
    joint_pos_sel = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # ensure target is on same device/dtype
    target_sel = target_full[asset_cfg.joint_ids].to(
        device=joint_pos_sel.device, dtype=joint_pos_sel.dtype
    )

    if not torch.isfinite(joint_pos_sel).all():
        raise RuntimeError("Non-finite joint positions in pose_json_deviation_l1_after_delay")

    # Base penalty (L1 over selected joints)
    deviations = torch.abs(joint_pos_sel - target_sel)
    base_penalty = torch.sum(deviations, dim=1)  # [num_envs]

    # ---- Time gating (per env) ----
    steps = _steps_since_reset(env)  # [num_envs], int
    dt = _step_dt(env)
    delay_steps = int(max(0, round(delay_s / dt)))
    ramp_steps  = int(max(0, round(ramp_s  / dt)))

    # piecewise weight: 0 until delay, then (0..1) ramp, then 1
    steps_f = steps.to(dtype=base_penalty.dtype)
    if ramp_steps > 0:
        w = torch.clamp((steps_f - delay_steps) / max(1, ramp_steps), min=0.0, max=1.0)
    else:
        w = (steps >= delay_steps).to(dtype=base_penalty.dtype)

    return w * base_penalty


def pose_json_deviation_l1_two_stage(
    env: ManagerBasedRLEnv,
    pose_path_before: str = "assets/crawl-pose.json",
    pose_path_after: str = "assets/stand-pose.json",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delay_s: float = 1.0,           # time before transitioning from before->after pose
    ramp_s: float = 0.5,            # time to smoothly blend between poses
) -> torch.Tensor:
    """L1 penalty that transitions between two target poses over time.

    Before `delay_s`: penalizes deviation from pose_path_before (full strength).
    Between `delay_s` and `delay_s + ramp_s`: smoothly blends penalty from pose_before to pose_after.
    After `delay_s + ramp_s`: penalizes deviation from pose_path_after (full strength).

    Args:
        env: RL environment.
        pose_path_before: Path to JSON pose file for early episode.
        pose_path_after: Path to JSON pose file for late episode.
        asset_cfg: Scene entity for the robot asset.
        delay_s: Time in seconds before transitioning starts.
        ramp_s: Time in seconds to smoothly blend between poses (0 = instant switch).

    Returns:
        Tensor of shape (num_envs,) with L1 deviation penalty from time-dependent target pose.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Build target vectors for both poses in joint order
    target_before_full = _get_full_target_vector(pose_path_before, asset)
    target_after_full = _get_full_target_vector(pose_path_after, asset)

    # Gather joint positions in the selected order
    joint_pos_sel = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # Ensure targets are on same device/dtype
    target_before_sel = target_before_full[asset_cfg.joint_ids].to(
        device=joint_pos_sel.device, dtype=joint_pos_sel.dtype
    )
    target_after_sel = target_after_full[asset_cfg.joint_ids].to(
        device=joint_pos_sel.device, dtype=joint_pos_sel.dtype
    )

    if not torch.isfinite(joint_pos_sel).all():
        raise RuntimeError("Non-finite joint positions in pose_json_deviation_l1_two_stage")

    # Compute L1 penalties for both target poses
    penalty_before = torch.sum(torch.abs(joint_pos_sel - target_before_sel), dim=1)  # [num_envs]
    penalty_after = torch.sum(torch.abs(joint_pos_sel - target_after_sel), dim=1)    # [num_envs]

    # ---- Time-based blending (per env) ----
    steps = _steps_since_reset(env)  # [num_envs], int
    dt = _step_dt(env)
    delay_steps = int(max(0, round(delay_s / dt)))
    ramp_steps  = int(max(0, round(ramp_s  / dt)))

    # Blend weight: 0 before delay (use pose_before), 1 after delay+ramp (use pose_after)
    steps_f = steps.to(dtype=penalty_before.dtype)
    if ramp_steps > 0:
        # Smooth transition from 0 to 1 during ramp period
        blend = torch.clamp((steps_f - delay_steps) / max(1, ramp_steps), min=0.0, max=1.0)
    else:
        # Instant switch at delay_steps
        blend = (steps >= delay_steps).to(dtype=penalty_before.dtype)

    # Linear interpolation: (1-blend)*penalty_before + blend*penalty_after
    return (1.0 - blend) * penalty_before + blend * penalty_after


def pose_json_deviation_l1_align_plus_x_lerp(
    env: ManagerBasedRLEnv,
    pose_path_before: str = "assets/crawl-pose.json",
    pose_path_after: str = "assets/stand-pose.json",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    invert: bool = False,
    blend_start: float = 0.0,
    blend_end: float = 1.0,
) -> torch.Tensor:
    """L1 penalty that blends between two target poses based on orientation alignment.

    Uses the projected gravity in base frame ``g_b = asset.data.projected_gravity_b`` and
    computes alignment with +X axis. The alignment score is mapped to a blend weight in [0, 1]:

        blend = clamp( (align + 1) / 2, 0, 1 ) = clamp( 1 - 0.25 * ||g_b - [1,0,0]||^2, 0, 1 )

    By default (invert=False):
      - blend -> 0 when misaligned (use ``pose_path_before``)
      - blend -> 1 when aligned with +X (use ``pose_path_after``)

    If ``invert=True``, the mapping is flipped (i.e., aligned -> before, misaligned -> after).

    Interpolation is controlled by a misalignment "percent" window [blend_start, blend_end] in [0, 1],
    where percent p is based on ``misalign_projected_gravity_plus_x_l2`` normalized and clamped to [0, 1]:

        p = clamp( 0.5 * ||g_b - [1,0,0]||^2, 0, 1 ) = clamp(1 - cos(theta), 0, 1)

    p=0 when aligned with +X (crawl), p≈1 when orthogonal (stand), and p=1 when opposite (clamped).
    The blend saturates to 1 below blend_start and to 0 above blend_end, with linear interpolation in between.

    Args:
        env: RL environment.
        pose_path_before: JSON pose for low-alignment regime.
        pose_path_after: JSON pose for high-alignment regime.
        asset_cfg: Scene entity for the robot asset.
        invert: If True, swap which pose is selected by high alignment.

    Returns:
        Tensor of shape (num_envs,) with L1 deviation penalty from orientation-dependent target pose.
    """
    # Joint-space targets
    asset: Articulation = env.scene[asset_cfg.name]
    target_before_full = _get_full_target_vector(pose_path_before, asset)
    target_after_full = _get_full_target_vector(pose_path_after, asset)

    joint_pos_sel = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target_before_sel = target_before_full[asset_cfg.joint_ids].to(
        device=joint_pos_sel.device, dtype=joint_pos_sel.dtype
    )
    target_after_sel = target_after_full[asset_cfg.joint_ids].to(
        device=joint_pos_sel.device, dtype=joint_pos_sel.dtype
    )

    if not torch.isfinite(joint_pos_sel).all():
        raise RuntimeError("Non-finite joint positions in pose_json_deviation_l1_align_plus_x_lerp")

    # Per-branch L1 penalties
    penalty_before = torch.sum(torch.abs(joint_pos_sel - target_before_sel), dim=1)
    penalty_after = torch.sum(torch.abs(joint_pos_sel - target_after_sel), dim=1)

    # Orientation-based blend using projected gravity in base frame
    rob: RigidObject = env.scene[asset_cfg.name]
    g_b = rob.data.projected_gravity_b  # (num_envs, 3)
    target = torch.tensor([1.0, 0.0, 0.0], dtype=g_b.dtype, device=g_b.device)
    dist_sq = torch.sum(torch.square(g_b - target), dim=1)  # in [0, 4]

    # Misalignment percent p in [0, 1]: 0 at perfect alignment, ~1 at orthogonal, 1 at opposite (clamped)
    misalign = 0.5 * dist_sq  # = 1 - cos(theta)
    p = torch.clamp(misalign, min=0.0, max=1.0)

    start = float(blend_start)
    end = float(blend_end)
    if not (0.0 <= start < end <= 1.0):
        raise RuntimeError("pose_json_deviation_l1_align_plus_x_lerp requires 0 <= blend_start < blend_end <= 1")
    denom = max(1e-6, (end - start))

    # Linear map p from [start..end] -> [0..1], then convert to blend (1 - t)
    t = (p - start) / denom
    t = torch.clamp(t, min=0.0, max=1.0)
    blend = 1.0 - t
    if invert:
        blend = 1.0 - blend

    return (1.0 - blend) * penalty_before + blend * penalty_after