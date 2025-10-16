from __future__ import annotations

import math
import torch
import time
import numpy as np
from typing import TYPE_CHECKING, Literal
import os
import json

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt


# Try to import debug drawing - gracefully handle headless mode
try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    DEBUG_DRAW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # In headless mode or when debug drawing is not available
    omni_debug_draw = None
    DEBUG_DRAW_AVAILABLE = False

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from .observations import compute_animation_phase_and_frame
from ..g1 import get_animation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Global variable to track when push lines were last drawn
_push_lines_timestamp = 0.0
_push_lines_drawn = False
# Separate tracker for site point visualization
_site_points_timestamp = 0.0
_site_points_drawn = False
# Separate tracker for base position points visualization
_base_points_timestamp = 0.0
_base_points_drawn = False
# Separate tracker for per-step animation site visualization
_anim_sites_timestamp = 0.0
_anim_sites_drawn = False

# Separate tracker for forward-velocity world-frame arrows
_vel_arrows_timestamp = 0.0
_vel_arrows_drawn = False
# Separate tracker for heading world-XY arrows
_heading_arrows_timestamp = 0.0
_heading_arrows_drawn = False


def is_visualization_available() -> bool:
    """
    Check if visualization is available (not in headless mode).
    
    Returns:
        bool: True if debug drawing is available, False if in headless mode or not available
    """
    return DEBUG_DRAW_AVAILABLE


def push_by_setting_velocity_with_viz(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # Print the z (height) component of the root position for the selected envs
    # print("Root pos_w z (height) for envs:", asset.data.root_pos_w[env_ids, 2])

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    sampled_velocities = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    vel_w += sampled_velocities
    
    # Add debug visualization for the applied velocities
    visualize_applied_velocities(
        asset_positions=asset.data.root_pos_w[env_ids],
        applied_velocities=sampled_velocities,
        env_ids=env_ids,
        duration=1.0,
        arrow_scale=2.0
    )
    
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def visualize_applied_velocities(
    asset_positions: torch.Tensor,
    applied_velocities: torch.Tensor, 
    env_ids: torch.Tensor,
    duration: float = 1.0,
    arrow_scale: float = 2.0,
    height_offset: float = 0.5,
    min_magnitude_threshold: float = 0.01,
    arrowhead_length: float = 0.3,
    arrowhead_angle: float = 25.0,
    arrow_color_override: tuple[float, float, float, float] | None = None,
    print_debug_info: bool = False
):
    """
    Visualize applied velocities as colored arrows using Isaac Sim's debug drawing interface.
    
    This function creates arrow visualizations showing the direction and magnitude of applied velocities.
    Arrows are colored based on magnitude (red = strong, yellow = weak) and include arrowheads for clarity.
    
    Note: Visualization is automatically disabled in headless mode (during training) to prevent errors.
    
    Args:
        asset_positions: Positions of assets in world frame. Shape: [num_envs, 3]
        applied_velocities: Applied velocity vectors. Shape: [num_envs, 6] (linear + angular)
        env_ids: Environment IDs being affected
        duration: How long to show the arrows (seconds)
        arrow_scale: Scaling factor for arrow length visualization
        height_offset: Height offset above asset position to draw arrows
        min_magnitude_threshold: Minimum velocity magnitude to visualize
        arrowhead_length: Length of arrowhead lines
        arrowhead_angle: Angle of arrowhead lines (degrees)
        arrow_color_override: Fixed color for all arrows (R,G,B,A). If None, uses magnitude-based coloring
        print_debug_info: Whether to print debug information
    
    Usage example in other event functions:
        ```python
        # After applying some effect to assets
        visualize_applied_velocities(
            asset_positions=asset.data.root_pos_w[env_ids],
            applied_velocities=sampled_forces,  # or velocities, or any vector
            env_ids=env_ids,
            duration=2.0,  # Show for 2 seconds
            arrow_scale=1.5
        )
        ```
    """
    # Early return if visualization is not available (headless mode)
    if not is_visualization_available():
        if print_debug_info and len(env_ids) <= 4:
            print(f"[DEBUG VIZ] Headless mode - skipping visualization for {len(env_ids)} environments")
        return
    
    global _push_lines_timestamp, _push_lines_drawn
    
    # Get debug drawing interface
    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    
    # Clear old visualization if duration has passed
    current_time = time.time()
    if _push_lines_drawn and (current_time - _push_lines_timestamp) > duration:
        draw_interface.clear_lines()
        _push_lines_drawn = False

    # Prepare lists for line drawing
    line_start_points = []
    line_end_points = []
    line_colors = []
    line_sizes = []
    
    # Convert angle to radians for calculations
    angle_rad = np.radians(arrowhead_angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    for i, env_id in enumerate(env_ids):
        # Get the applied velocity for this environment (only linear components)
        applied_vel = applied_velocities[i, :3].cpu().numpy()  # x, y, z components
        robot_pos = asset_positions[i].cpu().numpy()
        
        # Skip if velocity magnitude is too small to visualize
        vel_magnitude = float(torch.norm(applied_velocities[i, :3]))
        if vel_magnitude < min_magnitude_threshold:
            continue
            
        # Calculate arrow start and end points
        arrow_start = [
            float(robot_pos[0]), 
            float(robot_pos[1]), 
            float(robot_pos[2] + height_offset)
        ]
        arrow_end = [
            float(robot_pos[0] + applied_vel[0] * arrow_scale),
            float(robot_pos[1] + applied_vel[1] * arrow_scale), 
            float(robot_pos[2] + height_offset + applied_vel[2] * arrow_scale)
        ]
        
        # Color based on velocity magnitude (red = strong, yellow = weak) or use override
        if arrow_color_override is not None:
            arrow_color = arrow_color_override
        else:
            color_intensity = min(vel_magnitude / 2.0, 1.0)  # Normalize to 0-1
            arrow_color = (1.0, 1.0 - color_intensity, 0.0, 1.0)  # Red to yellow gradient
        
        # Add main arrow line
        line_start_points.append(arrow_start)
        line_end_points.append(arrow_end)
        line_colors.append(arrow_color)
        line_sizes.append(4.0)
        
        # Create arrowhead in horizontal plane for better visibility
        direction_2d = np.array([arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1]])
        if np.linalg.norm(direction_2d) > 0:
            direction_2d = direction_2d / np.linalg.norm(direction_2d)
            
            # Left arrowhead line
            arrowhead_dir1 = np.array([
                -direction_2d[0] * cos_angle + direction_2d[1] * sin_angle,
                -direction_2d[1] * cos_angle - direction_2d[0] * sin_angle
            ])
            arrowhead_end1 = [
                arrow_end[0] + arrowhead_dir1[0] * arrowhead_length,
                arrow_end[1] + arrowhead_dir1[1] * arrowhead_length,
                arrow_end[2]
            ]
            line_start_points.append(arrow_end)
            line_end_points.append(arrowhead_end1)
            line_colors.append(arrow_color)
            line_sizes.append(3.0)
            
            # Right arrowhead line
            arrowhead_dir2 = np.array([
                -direction_2d[0] * cos_angle - direction_2d[1] * sin_angle,
                -direction_2d[1] * cos_angle + direction_2d[0] * sin_angle
            ])
            arrowhead_end2 = [
                arrow_end[0] + arrowhead_dir2[0] * arrowhead_length,
                arrow_end[1] + arrowhead_dir2[1] * arrowhead_length,
                arrow_end[2]
            ]
            line_start_points.append(arrow_end)
            line_end_points.append(arrowhead_end2)
            line_colors.append(arrow_color)
            line_sizes.append(3.0)
    
    # Draw all the arrows if any were created
    if line_start_points:
        # Clear previous lines before drawing new ones
        if _push_lines_drawn:
            draw_interface.clear_lines()
        
        draw_interface.draw_lines(line_start_points, line_end_points, line_colors, line_sizes)
        _push_lines_timestamp = current_time
        _push_lines_drawn = True
        
        # Optional debug info (limited to avoid spam)
        if print_debug_info and len(env_ids) <= 4:
            for i, env_id in enumerate(env_ids):
                vel_mag = float(torch.norm(applied_velocities[i, :3]))
                if vel_mag > min_magnitude_threshold:
                    print(f"[DEBUG VIZ] Applied velocity to env {env_id.item()}: magnitude = {vel_mag:.3f}")


# ===== HOW TO ADD VISUALIZATION TO OTHER EVENT FUNCTIONS =====
"""
To add visualization to any event function, follow this simple pattern:

1. Do your event logic normally
2. Call visualize_applied_velocities() with your data

Example:
```python
def my_event_with_viz(env, env_ids, params, asset_cfg):
    # Your normal event logic
    asset = env.scene[asset_cfg.name]
    sampled_values = sample_something(...)
    apply_changes(asset, sampled_values, env_ids)
    
    # Add visualization (just 4 lines!)
    viz_vectors = torch.zeros(len(env_ids), 6, device=asset.device)
    viz_vectors[:, :3] = sampled_values * scale_factor
    visualize_applied_velocities(asset.data.root_pos_w[env_ids], viz_vectors, env_ids)
```

Note: Visualization automatically detects headless mode and gracefully skips drawing
to prevent import errors during training.
"""

def reset_root_state_to_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# ===== Pose JSON-based joint reset =====
def _load_pose_from_json_events(json_path: str) -> dict:
    """Load pose entry from JSON file (same format used in rewards).

    Expected structure:
    {"poses": [ {"joints": {...}, "base_pos": [x,y,z], "base_rpy": [r,p,y]}, ... ]}
    Returns the first pose entry dict. Fails loudly if required fields are missing.
    """
    # Resolve project root relative to this file (mirrors rewards.py logic)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
    full_path = os.path.join(project_root, json_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Pose JSON not found at: {full_path}")
    with open(full_path, "r") as f:
        data = json.load(f)
    if "poses" not in data or len(data["poses"]) == 0:
        raise ValueError(f"No poses found in {json_path}")
    entry = data["poses"][0]
    if not isinstance(entry.get("joints"), dict) or not entry.get("joints"):
        raise ValueError(f"Pose JSON missing 'joints' map in first pose: {json_path}")
    # base_pos/base_rpy are required for this reset method
    if "base_pos" not in entry or "base_rpy" not in entry:
        raise ValueError(f"Pose JSON must include 'base_pos' and 'base_rpy': {json_path}")
    if len(entry["base_pos"]) != 3 or len(entry["base_rpy"]) != 3:
        raise ValueError(f"'base_pos' and 'base_rpy' must be length-3 arrays: {json_path}")
    return entry


def _build_target_vector_from_pose_dict_events(asset: Articulation, pose: dict) -> torch.Tensor:
    """Build target joint vector in `asset.joint_names` order. Fail if any joint missing."""
    names = asset.joint_names
    missing = [n for n in names if n not in pose]
    if len(missing) > 0:
        # Fail loudly; caller should ensure pose JSON covers all joints
        sample = ", ".join(missing[:8])
        more = "..." if len(missing) > 8 else ""
        raise KeyError(f"Pose dict missing joint keys: {sample}{more}")
    values = [pose[n] for n in names]
    return torch.tensor(values, dtype=asset.data.joint_pos.dtype, device=asset.data.joint_pos.device)


def reset_to_pose_json(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    json_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    position_range: tuple[float, float] | None = None,
    joint_velocity_range: tuple[float, float] | None = None,
):
    """Reset robot to pose from JSON with optional noise/scaling for root and joints.

    Root state (base):
    - Root pose/velocity are set from the JSON file (base_pos + base_rpy)
    - Optional uniform noise via `pose_range` and `velocity_range` (same as reset_root_state_uniform)
    
    Joint state:
    - Joint positions are taken from `json_path` 
    - Optional scaling via `position_range` (same as reset_joints_by_scale)
    - Joint velocities default to zero unless `joint_velocity_range` is provided
    - Joint positions/velocities are clamped to limits
    - Fails loudly if any joint is missing in the pose JSON
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    # Load full pose entry (joints + base_pos + base_rpy)
    entry = _load_pose_from_json_events(json_path)
    joints_map = entry["joints"]
    base_pos = entry["base_pos"]
    base_rpy = entry["base_rpy"]


    # ===== ROOT STATE RESET (following reset_root_state_uniform pattern) =====
    num_envs = len(env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    
    # Set base pose from JSON
    pos_t = torch.tensor(base_pos, device=device, dtype=root_states.dtype).view(1, 3).expand(num_envs, -1)
    roll = torch.full((num_envs,), float(base_rpy[0]), device=device, dtype=root_states.dtype)
    pitch = torch.full((num_envs,), float(base_rpy[1]), device=device, dtype=root_states.dtype)
    yaw = torch.full((num_envs,), float(base_rpy[2]), device=device, dtype=root_states.dtype)
    quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw).to(dtype=root_states.dtype)
    
    positions = env.scene.env_origins[env_ids] + pos_t
    orientations = quat
    
    # Add pose noise if specified
    if pose_range is not None:
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=device)
        
        positions = positions + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(orientations, orientations_delta)
    
    # Set velocities (zero or noise)
    velocities = root_states[:, 7:13].clone()
    velocities[:] = 0.0
    
    if velocity_range is not None:
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=device)
        velocities = velocities + rand_samples
    
    # Write root state to sim
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    # ===== JOINT STATE RESET (following reset_joints_by_scale pattern) =====
    # cast env_ids to allow broadcasting when joint_ids are specified
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids
    
    # Get joint positions from JSON
    target_vec = _build_target_vector_from_pose_dict_events(asset, joints_map)
    
    # # Handle joint_ids: if specific joints are requested, select only those
    if asset_cfg.joint_ids != slice(None):
        # Select only the specified joints from the full target vector
        joint_pos = target_vec[asset_cfg.joint_ids].view(1, -1).expand(num_envs, -1).clone()
    else:
        # Use all joints
        joint_pos = target_vec.view(1, -1).expand(num_envs, -1).clone()
    
    joint_vel = torch.zeros_like(joint_pos, device=device)

    # joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    # joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()
    
    # Apply position scaling if specified
    if position_range is not None:
        joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, device)
    
    # Apply velocity scaling if specified
    if joint_velocity_range is not None:
        joint_vel *= math_utils.sample_uniform(*joint_velocity_range, joint_vel.shape, device)
    
    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    
    # Clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    
    # Write joint state to sim
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_from_pose_array_with_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    json_path: str,
    frame_range: tuple[int, int] = (0, -1),
    home_frame: int = 0,
    home_frame_prob: float = 0.3,
    end_home_frame: int | None = None,
    end_home_frame_prob: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    position_range: tuple[float, float] | None = None,
    joint_velocity_range: tuple[float, float] | None = None,
):
    """Reset robot from a pose array JSON with curriculum-based pose sampling.
    
    This function works with pose_viewer.py style JSON files that contain a list of poses
    with base_pos, base_rpy, and joint positions. It samples poses within a specified range,
    with special handling for "anchor" poses (start home and/or end home) that are always available.
    
    Pose array format:
    {
      "poses": [
        {
          "base_pos": [x, y, z],
          "base_rpy": [roll, pitch, yaw],
          "joints": {"joint_name": value, ...}
        },
        ...
      ]
    }
    
    Sampling strategy:
    - With probability `home_frame_prob`, sample the start home pose
    - With probability `end_home_frame_prob`, sample the end home pose
    - Otherwise, sample uniformly from [min_pose, max_pose] range
    - The frame_range can be dynamically expanded by a curriculum term
    
    Args:
        env: The environment
        env_ids: Environment IDs to reset
        json_path: Path to pose array JSON (e.g., "assets/animation_mocap_rc0_poses_sorted.json")
        frame_range: (min_pose, max_pose) to sample from. Use -1 for max to mean "all poses"
        home_frame: The start "anchor" pose (typically 0 for stand2crawl, 5795 for crawl2stand)
        home_frame_prob: Probability of sampling home_frame (0.0 = never, 1.0 = always)
        end_home_frame: Optional end "anchor" pose (e.g., 5795 for stand2crawl). If None, not used.
        end_home_frame_prob: Probability of sampling end_home_frame (only if end_home_frame is set)
        asset_cfg: Asset configuration
        pose_range: Optional uniform noise for base pose
        velocity_range: Optional uniform noise for base velocity
        position_range: Optional scaling range for joint positions
        joint_velocity_range: Optional scaling range for joint velocities
    
    Example curriculum usage (stand2crawl - forward):
        Start: frame_range=(0, 0), home_frame=0, end_home_frame=5795
        Mid: frame_range=(0, 2897), home_frame=0, end_home_frame=5795
        End: frame_range=(0, 5795), home_frame=0, end_home_frame=5795
        Samples: 30% pose 0, 10% pose 5795, 60% from curriculum range
    
    Example curriculum usage (crawl2stand - reverse):
        Start: frame_range=(5795, 5795), home_frame=5795, end_home_frame=0
        Mid: frame_range=(2897, 5795), home_frame=5795, end_home_frame=0
        End: frame_range=(0, 5795), home_frame=5795, end_home_frame=0
        Samples: 30% pose 5795, 10% pose 0, 60% from curriculum range
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = len(env_ids)
    
    # Load pose array with caching
    poses = _load_pose_array_with_cache(json_path)
    num_poses = len(poses)
    
    # Determine actual pose range
    min_pose = max(0, int(frame_range[0]))
    max_pose = int(frame_range[1]) if frame_range[1] >= 0 else num_poses - 1
    max_pose = min(max_pose, num_poses - 1)
    
    # Ensure home_frame is valid
    home_frame = max(0, min(home_frame, num_poses - 1))
    
    # Ensure end_home_frame is valid if provided
    if end_home_frame is not None:
        end_home_frame = max(0, min(end_home_frame, num_poses - 1))
    
    # Normalize probabilities if both anchors are used
    total_anchor_prob = home_frame_prob + (end_home_frame_prob if end_home_frame is not None else 0.0)
    if total_anchor_prob > 1.0:
        raise ValueError(f"home_frame_prob ({home_frame_prob}) + end_home_frame_prob ({end_home_frame_prob}) must be <= 1.0")
    
    # Sample poses with anchor bias
    sampled_poses = []
    for _ in range(num_envs):
        rand_val = torch.rand(1).item()
        
        # Check start home frame first
        if rand_val < home_frame_prob:
            pose_idx = home_frame
        # Check end home frame second
        elif end_home_frame is not None and rand_val < (home_frame_prob + end_home_frame_prob):
            pose_idx = end_home_frame
        # Otherwise sample from curriculum range
        else:
            # Sample uniformly from [min_pose, max_pose]
            if min_pose == max_pose:
                pose_idx = min_pose
            else:
                pose_idx = torch.randint(min_pose, max_pose + 1, (1,)).item()
        
        sampled_poses.append(poses[pose_idx])
    
    # ===== ROOT STATE RESET =====
    root_states = asset.data.default_root_state[env_ids].clone()
    
    # Extract base positions and orientations from sampled poses
    base_pos_list = []
    base_quat_list = []
    for pose in sampled_poses:
        base_pos = pose["base_pos"]
        base_rpy = pose["base_rpy"]
        
        base_pos_list.append([float(base_pos[0]), float(base_pos[1]), float(base_pos[2])])
        
        # Convert RPY to quaternion
        roll = torch.tensor(float(base_rpy[0]), device=device, dtype=root_states.dtype)
        pitch = torch.tensor(float(base_rpy[1]), device=device, dtype=root_states.dtype)
        yaw = torch.tensor(float(base_rpy[2]), device=device, dtype=root_states.dtype)
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        base_quat_list.append(quat.cpu().tolist())
    
    base_pos_t = torch.tensor(base_pos_list, device=device, dtype=root_states.dtype)
    base_quat_t = torch.tensor(base_quat_list, device=device, dtype=root_states.dtype)
    
    positions = env.scene.env_origins[env_ids] + base_pos_t
    orientations = base_quat_t
    
    # Add pose noise if specified
    if pose_range is not None:
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=device)
        
        positions = positions + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(orientations, orientations_delta)
    
    # Set velocities (zero or noise)
    velocities = root_states[:, 7:13].clone()
    velocities[:] = 0.0
    
    if velocity_range is not None:
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=device)
        velocities = velocities + rand_samples
    
    # Write root state to sim
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    
    # ===== JOINT STATE RESET =====
    # cast env_ids for joint access
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids
    
    # Build joint position tensor from sampled poses
    joint_names = asset.data.joint_names
    joint_pos_list = []
    for pose in sampled_poses:
        joints_dict = pose["joints"]
        joint_pos_row = []
        for jname in joint_names:
            if jname not in joints_dict:
                raise KeyError(f"Joint '{jname}' not found in pose. Available: {list(joints_dict.keys())}")
            joint_pos_row.append(float(joints_dict[jname]))
        joint_pos_list.append(joint_pos_row)
    
    joint_pos = torch.tensor(joint_pos_list, device=device, dtype=asset.data.joint_pos.dtype)
    
    # Handle joint_ids: if specific joints are requested, select only those
    if asset_cfg.joint_ids != slice(None):
        joint_pos = joint_pos[:, asset_cfg.joint_ids]
    
    joint_vel = torch.zeros_like(joint_pos, device=device)
    
    # Apply position scaling if specified
    if position_range is not None:
        joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, device)
    
    # Apply velocity scaling if specified
    if joint_velocity_range is not None:
        joint_vel *= math_utils.sample_uniform(*joint_velocity_range, joint_vel.shape, device)
    
    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    
    # Clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    
    # Write joint state to sim
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


# ===== Pose array loader (for pose_viewer.py style files) =====

# Cache for loaded pose arrays (keyed by json_path)
_POSE_ARRAY_CACHE: dict[str, list] = {}


def _load_pose_array_with_cache(json_path: str) -> list[dict]:
    """Load pose array JSON with caching (pose_viewer.py format).
    
    Expected structure:
    {
      "poses": [
        {
          "base_pos": [x, y, z],
          "base_rpy": [roll, pitch, yaw],
          "joints": {"joint_name": value, ...}
        },
        ...
      ]
    }
    
    Returns list of pose dicts.
    """
    if json_path in _POSE_ARRAY_CACHE:
        return _POSE_ARRAY_CACHE[json_path]
    
    # Resolve project root relative to this file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
    full_path = os.path.join(project_root, json_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Pose array JSON not found at: {full_path}")
    
    with open(full_path, "r") as f:
        data = json.load(f)
    
    if "poses" not in data:
        raise ValueError(f"Pose array JSON missing 'poses' key: {json_path}")
    
    poses = data["poses"]
    if not isinstance(poses, list) or len(poses) == 0:
        raise ValueError(f"'poses' must be a non-empty list: {json_path}")
    
    # Validate pose structure
    for i, pose in enumerate(poses):
        if not isinstance(pose, dict):
            raise ValueError(f"Pose {i} must be a dict: {json_path}")
        if "base_pos" not in pose or "base_rpy" not in pose or "joints" not in pose:
            raise ValueError(f"Pose {i} missing required keys (base_pos, base_rpy, joints): {json_path}")
        if not isinstance(pose["joints"], dict) or len(pose["joints"]) == 0:
            raise ValueError(f"Pose {i} 'joints' must be a non-empty dict: {json_path}")
    
    # Cache and return
    _POSE_ARRAY_CACHE[json_path] = poses
    return poses


# ===== Animation-based reset helpers and event =====
from ..g1 import get_animation

# Cache for loaded animation data (keyed by json_path)
_ANIMATION_CACHE: dict[str, dict] = {}


def _load_animation_with_cache(json_path: str) -> dict:
    """Load animation JSON with caching to avoid repeated file I/O.
    
    Animation JSON structure (dict with keys):
    - "frames": qpos frames [[qpos0...], [qpos1...], ...]
    - "metadata": dict with "joints", "qpos_labels", "base", etc.
    - "site_positions": optional per-frame site positions
    - "contact_flags": optional per-frame contact flags
    
    Returns cached dict with keys: "qpos_frames", "metadata", "num_frames"
    """
    if json_path in _ANIMATION_CACHE:
        return _ANIMATION_CACHE[json_path]
    
    # Resolve project root relative to this file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
    full_path = os.path.join(project_root, json_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Animation JSON not found at: {full_path}")
    
    with open(full_path, "r") as f:
        raw_data = json.load(f)
    
    # Parse structure: dict with "frames" and "metadata" keys
    if not isinstance(raw_data, dict):
        raise ValueError(f"Animation JSON must be a dict: {json_path}")
    
    if "frames" not in raw_data:
        raise ValueError(f"Animation JSON missing 'frames' key: {json_path}")
    
    if "metadata" not in raw_data:
        raise ValueError(f"Animation JSON missing 'metadata' key: {json_path}")
    
    qpos_frames = raw_data["frames"]
    if not isinstance(qpos_frames, list) or len(qpos_frames) == 0:
        raise ValueError(f"Animation 'frames' must be a non-empty list: {json_path}")
    
    metadata = raw_data["metadata"]
    if "joints" not in metadata:
        raise ValueError(f"Animation metadata missing 'joints' info: {json_path}")
    
    # Cache parsed data
    cached = {
        "qpos_frames": qpos_frames,
        "metadata": metadata,
        "num_frames": len(qpos_frames),
    }
    _ANIMATION_CACHE[json_path] = cached
    return cached


def _build_joint_index_map(asset: Articulation, joints_meta, qpos_labels):
    robot_joint_names = asset.data.joint_names
    index_map: list[int] = []
    missing: list[str] = []

    name_to_qposadr: dict[str, int] = {}
    if isinstance(joints_meta, dict):
        for name, qposadr in joints_meta.items():
            try:
                name_to_qposadr[str(name)] = int(qposadr)
            except Exception:
                continue
    elif isinstance(joints_meta, list):
        for item in joints_meta:
            if not isinstance(item, dict):
                continue
            jname = item.get("name")
            jtype = item.get("type")
            adr = item.get("qposadr")
            dim = item.get("qposdim", 1)
            if jname is not None and jtype in ("hinge", "slide") and isinstance(adr, int) and int(dim) == 1:
                name_to_qposadr[str(jname)] = int(adr)

    label_lookup: dict[str, int] = {}
    if qpos_labels is not None:
        for i, lbl in enumerate(qpos_labels):
            s = str(lbl)
            label_lookup[s] = i
            if s.startswith("joint:"):
                label_lookup[s[len("joint:"):]] = i

    for jn in robot_joint_names:
        qidx = -1
        if jn in name_to_qposadr:
            qidx = name_to_qposadr[jn]
        else:
            if jn in label_lookup:
                qidx = label_lookup[jn]
            elif ("joint:" + jn) in label_lookup:
                qidx = label_lookup["joint:" + jn]
        index_map.append(qidx)
        if qidx == -1:
            missing.append(jn)

    if missing:
        print(f"[WARN] Missing qpos indices for {len(missing)} joints (will keep defaults)")
    return index_map


def _build_joint_velocity_index_map(asset: Articulation, joints_meta, qvel_labels, qpos_labels=None):
    """Build per-joint mapping into the animation's qvel vector.

    Preference order per joint:
    1) joints_meta.qveladr if provided
    2) joints_meta.qposadr (hinge/slide 1-DoF often match)
    3) label lookup in qvel_labels
    4) fallback to qpos label lookup
    """
    robot_joint_names = asset.data.joint_names
    index_map: list[int] = []
    missing: list[str] = []

    name_to_qveladr: dict[str, int] = {}
    if isinstance(joints_meta, dict):
        for name, adr in joints_meta.items():
            try:
                # Allow either qveladr directly or legacy qposadr
                if isinstance(adr, dict):
                    if "qveladr" in adr and isinstance(adr["qveladr"], int):
                        name_to_qveladr[str(name)] = int(adr["qveladr"])  # type: ignore[index]
                    elif "qposadr" in adr and isinstance(adr["qposadr"], int):
                        name_to_qveladr[str(name)] = int(adr["qposadr"])  # type: ignore[index]
                else:
                    name_to_qveladr[str(name)] = int(adr)
            except Exception:
                continue
    elif isinstance(joints_meta, list):
        for item in joints_meta:
            if not isinstance(item, dict):
                continue
            jname = item.get("name")
            jtype = item.get("type")
            vadr = item.get("qveladr")
            padr = item.get("qposadr")
            vdim = item.get("qveldim", 1)
            pdim = item.get("qposdim", 1)
            # Only 1-DoF joints are considered here
            if (
                jname is not None
                and jtype in ("hinge", "slide")
                and ((isinstance(vadr, int) and int(vdim) == 1) or (isinstance(padr, int) and int(pdim) == 1))
            ):
                if isinstance(vadr, int) and int(vdim) == 1:
                    name_to_qveladr[str(jname)] = int(vadr)
                elif isinstance(padr, int) and int(pdim) == 1:
                    name_to_qveladr[str(jname)] = int(padr)

    # Label lookup from qvel labels, with fallback to qpos labels
    label_lookup: dict[str, int] = {}
    if qvel_labels is not None:
        for i, lbl in enumerate(qvel_labels):
            s = str(lbl)
            label_lookup[s] = i
            if s.startswith("joint:"):
                label_lookup[s[len("joint:"):]] = i
    if not label_lookup and qpos_labels is not None:
        for i, lbl in enumerate(qpos_labels):
            s = str(lbl)
            label_lookup[s] = i
            if s.startswith("joint:"):
                label_lookup[s[len("joint:"):]] = i

    for jn in robot_joint_names:
        qidx = -1
        if jn in name_to_qveladr:
            qidx = name_to_qveladr[jn]
        else:
            if jn in label_lookup:
                qidx = label_lookup[jn]
            elif ("joint:" + jn) in label_lookup:
                qidx = label_lookup["joint:" + jn]
        index_map.append(qidx)
        if qidx == -1:
            missing.append(jn)

    if missing:
        print(f"[WARN] Missing qvel indices for {len(missing)} joints (will keep default zeros)")
    return index_map


def reset_from_animation_frame(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    json_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    position_range: tuple[float, float] | None = None,
    joint_velocity_range: tuple[float, float] | None = None,
):
    """Reset robot to a random frame from an animation JSON with optional noise/scaling.
    
    This function loads animation data (with caching for performance), samples a random
    frame for each environment, and resets the robot to that frame's state.
    
    Animation structure:
    - qpos frames contain: [base_x, base_y, base_z, quat_w, quat_x, quat_y, quat_z, joint0, joint1, ...]
    - metadata provides joint mapping information
    
    Args:
        env: The environment
        env_ids: Environment IDs to reset
        json_path: Path to animation JSON (e.g., "assets/animation_rc4.json")
        asset_cfg: Asset configuration
        pose_range: Optional uniform noise for base pose (same as reset_root_state_uniform)
        velocity_range: Optional uniform noise for base velocity
        position_range: Optional scaling range for joint positions (e.g., (0.9, 1.1))
        joint_velocity_range: Optional scaling range for joint velocities
    
    Root state (base):
    - Base pose/velocity are taken from the sampled animation frame
    - Optional uniform noise via `pose_range` and `velocity_range`
    
    Joint state:
    - Joint positions are taken from the sampled animation frame
    - Optional scaling via `position_range`
    - Joint velocities default to zero unless `joint_velocity_range` is provided
    - Joint positions/velocities are clamped to limits
    - Fails loudly if any joint is missing in the animation
    
    Note: Animation data is cached after first load to avoid repeated file I/O during training.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = len(env_ids)
    
    # Load animation with caching
    anim_data = _load_animation_with_cache(json_path)
    qpos_frames = anim_data["qpos_frames"]
    metadata = anim_data["metadata"]
    num_frames = anim_data["num_frames"]
    
    # Build joint index map (also cached via function-level logic)
    joints_meta = metadata["joints"]
    qpos_labels = metadata.get("qpos_labels", None)
    joint_index_map = _build_joint_index_map(asset, joints_meta, qpos_labels)
    
    # Sample random frames for each env
    frame_indices = torch.randint(0, num_frames, (num_envs,), device="cpu")


def reset_from_animation_frame_with_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    json_path: str,
    frame_range: tuple[int, int] = (0, -1),
    home_frame: int = 0,
    home_frame_prob: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    position_range: tuple[float, float] | None = None,
    joint_velocity_range: tuple[float, float] | None = None,
):
    """Reset robot to a frame from an animation JSON with curriculum-based frame sampling.
    
    This function samples frames within a specified range, with special handling for a "home base"
    frame that's always available and preferentially sampled. This enables curriculum learning
    where easy poses are introduced first, then progressively harder ones are mixed in.
    
    Sampling strategy:
    - With probability `home_frame_prob`, sample the home frame (typically frame 0)
    - Otherwise, sample uniformly from [min_frame, max_frame] range
    - The frame_range can be dynamically expanded by a curriculum term
    
    Args:
        env: The environment
        env_ids: Environment IDs to reset
        json_path: Path to animation JSON (e.g., "assets/animation_mocap_rc0_poses_sorted.json")
        frame_range: (min_frame, max_frame) to sample from. Use -1 for max to mean "all frames"
        home_frame: The "safe" baseline frame to always include (typically 0 for default pose)
        home_frame_prob: Probability of sampling home_frame instead of range (0.0 = uniform, 1.0 = always home)
        asset_cfg: Asset configuration
        pose_range: Optional uniform noise for base pose
        velocity_range: Optional uniform noise for base velocity
        position_range: Optional scaling range for joint positions
        joint_velocity_range: Optional scaling range for joint velocities
    
    Example curriculum usage:
        Start with frame_range=(0, 0) - only samples home frame
        Expand to frame_range=(0, 50) - samples from first 50 frames, with 30% bias to home
        Eventually frame_range=(0, -1) - samples from all frames, still 30% home frame
    
    Note: Animation data is cached after first load to avoid repeated file I/O during training.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = len(env_ids)
    
    # Load animation with caching
    anim_data = _load_animation_with_cache(json_path)
    qpos_frames = anim_data["qpos_frames"]
    metadata = anim_data["metadata"]
    num_frames = anim_data["num_frames"]
    
    # Build joint index map
    joints_meta = metadata["joints"]
    qpos_labels = metadata.get("qpos_labels", None)
    joint_index_map = _build_joint_index_map(asset, joints_meta, qpos_labels)
    
    # Determine actual frame range
    min_frame = max(0, int(frame_range[0]))
    max_frame = int(frame_range[1]) if frame_range[1] >= 0 else num_frames - 1
    max_frame = min(max_frame, num_frames - 1)
    
    # Ensure home_frame is valid
    home_frame = max(0, min(home_frame, num_frames - 1))
    
    # Sample frames with home frame bias
    frame_indices = []
    for _ in range(num_envs):
        # Decide whether to sample home frame or from range
        if torch.rand(1).item() < home_frame_prob:
            frame_idx = home_frame
        else:
            # Sample uniformly from [min_frame, max_frame]
            if min_frame == max_frame:
                frame_idx = min_frame
            else:
                frame_idx = torch.randint(min_frame, max_frame + 1, (1,)).item()
        frame_indices.append(frame_idx)
    
    frame_indices = torch.tensor(frame_indices, dtype=torch.long, device="cpu")
    
    # Extract base poses and joint positions for sampled frames
    # qpos structure: [base_x, base_y, base_z, quat_w, quat_x, quat_y, quat_z, joint0, joint1, ...]
    base_pos_list = []
    base_quat_list = []
    joint_pos_list = []
    
    for idx in frame_indices:
        qpos = qpos_frames[int(idx.item())]
        
        # Extract base pose (first 7 elements: xyz + wxyz quaternion)
        base_pos_list.append([float(qpos[0]), float(qpos[1]), float(qpos[2])])
        base_quat_list.append([float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])])
        
        # Extract joint positions using index map
        joint_positions = []
        for joint_qpos_idx in joint_index_map:
            # qpos[0:7] is base, so joints start at index 7
            val = float(qpos[joint_qpos_idx])
            joint_positions.append(val)
        joint_pos_list.append(joint_positions)
    
    # Convert to tensors
    base_pos_t = torch.tensor(base_pos_list, device=device, dtype=asset.data.root_pos_w.dtype)
    base_quat_t = torch.tensor(base_quat_list, device=device, dtype=asset.data.root_quat_w.dtype)
    
    # ===== ROOT STATE RESET =====
    positions = env.scene.env_origins[env_ids] + base_pos_t
    orientations = base_quat_t
    
    # Add pose noise if specified
    if pose_range is not None:
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=device)
        
        positions = positions + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(orientations, orientations_delta)
    
    # Set velocities (zero or noise)
    root_states = asset.data.default_root_state[env_ids].clone()
    velocities = root_states[:, 7:13].clone()
    velocities[:] = 0.0
    
    if velocity_range is not None:
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=device)
        velocities = velocities + rand_samples
    
    # Write root state to sim
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    
    # ===== JOINT STATE RESET =====
    # cast env_ids for joint access
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids
    
    # Convert joint positions list to tensor
    joint_pos = torch.tensor(joint_pos_list, device=device, dtype=asset.data.joint_pos.dtype)
    
    # Handle joint_ids: if specific joints are requested, select only those
    if asset_cfg.joint_ids != slice(None):
        joint_pos = joint_pos[:, asset_cfg.joint_ids]
    
    joint_vel = torch.zeros_like(joint_pos, device=device)
    
    # Apply position scaling if specified
    if position_range is not None:
        joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, device)
    
    # Apply velocity scaling if specified
    if joint_velocity_range is not None:
        joint_vel *= math_utils.sample_uniform(*joint_velocity_range, joint_vel.shape, device)
    
    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    
    # Clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    
    # Write joint state to sim
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


# ===== Animation playback speed helpers =====
def set_animation_playback_speed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    speed: float | torch.Tensor,
):
    """Set per-env animation playback speed multiplier.

    speed: float or tensor broadcastable to [len(env_ids)] with values > 0.
    """
    if not hasattr(env, "_anim_playback_speed"):
        raise RuntimeError("_anim_playback_speed not initialized. Ensure init_animation_phase_offsets ran at startup.")
    if isinstance(speed, (float, int)):
        val = float(speed)
        if not math.isfinite(val) or val <= 0.0:
            raise RuntimeError(f"Invalid playback speed: {speed}")
        env._anim_playback_speed[env_ids] = float(val)  # type: ignore[attr-defined]
    elif isinstance(speed, torch.Tensor):
        if speed.numel() == 1:
            val = float(speed.item())
            if not math.isfinite(val) or val <= 0.0:
                raise RuntimeError(f"Invalid playback speed tensor value: {val}")
            env._anim_playback_speed[env_ids] = float(val)  # type: ignore[attr-defined]
        else:
            if int(speed.shape[0]) != int(len(env_ids)):
                raise RuntimeError("Speed tensor length must match env_ids length")
            if (speed <= 0).any() or not torch.isfinite(speed).all():
                raise RuntimeError("Speed tensor must be positive finite values")
            env._anim_playback_speed[env_ids] = speed.to(device=env._anim_playback_speed.device, dtype=env._anim_playback_speed.dtype)  # type: ignore[attr-defined]
    else:
        raise TypeError(f"Unsupported speed type: {type(speed).__name__}")


def randomize_animation_playback_speed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    min_speed: float = 0.5,
    max_speed: float = 1.5,
):
    """Randomize per-env playback speed uniformly in [min_speed, max_speed]."""
    if min_speed <= 0.0 or max_speed <= 0.0 or max_speed < min_speed:
        raise RuntimeError("Invalid speed range for randomization")
    if not hasattr(env, "_anim_playback_speed"):
        raise RuntimeError("_anim_playback_speed not initialized. Ensure init_animation_phase_offsets ran at startup.")
    rng = torch.rand(len(env_ids), device=env._anim_playback_speed.device)  # type: ignore[attr-defined]
    speeds = min_speed + (max_speed - min_speed) * rng
    env._anim_playback_speed[env_ids] = speeds  # type: ignore[attr-defined]


def update_animation_playback_speed_from_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "base_velocity",
    component: str = "vz",
    base_speed: float = 1.0,
    scale: float | None = None,
    min_speed: float = 0.25,
    max_speed: float = 3.0,
    abs_value: bool = True,
    use_animation_nominal: bool = True,
):
    """Update per-env animation playback speed from a command term.

    Mapping strategies:
    - If use_animation_nominal and animation metadata provides 'base_forward_velocity_mps' (v_nominal > 0):
        speed = clamp(base_speed * (|cmd_vz| / v_nominal), min_speed, max_speed)
      where 'component' selects which command element to use. Defaults to 'vz'.
    - Else, fallback to affine mapping:
        speed = clamp(base_speed + (scale or 0.0) * (|cmd_component|), min_speed, max_speed)
    """
    if not hasattr(env, "command_manager"):
        raise RuntimeError("command_manager not available on env. Ensure commands are configured in EnvCfg.")
    if not hasattr(env, "_anim_playback_speed"):
        raise RuntimeError("_anim_playback_speed not initialized. Ensure init_animation_phase_offsets ran at startup.")

    cmd = env.command_manager.get_command(command_name)
    # command shape from CrawlVelocityCommand: [vz, vy, roll]
    comp_idx = 0 if component.lower() in ("vz", "z", "lin_z") else 1 if component.lower() in ("vy", "y", "lin_y") else 2
    comp = cmd[:, comp_idx]
    if abs_value:
        comp = torch.abs(comp)

    speeds: torch.Tensor
    used_nominal = False
    if use_animation_nominal:
        try:
            anim = get_animation()
            meta = anim.get("metadata", {}) or {}
            v_nominal = float(meta.get("base_forward_velocity_mps", 0.0) or 0.0)
        except Exception:
            v_nominal = 0.0
        if v_nominal > 0.0 and math.isfinite(v_nominal):
            factor = (comp / float(v_nominal)).to(device=env._anim_playback_speed.device)  # type: ignore[attr-defined]
            speeds = float(base_speed) * factor
            used_nominal = True
        else:
            speeds = None  # type: ignore[assignment]
    else:
        speeds = None  # type: ignore[assignment]

    if speeds is None:
        # Fallback affine mapping
        s = float(scale) if (scale is not None) else 0.0
        speeds = float(base_speed) + s * comp.to(device=env._anim_playback_speed.device)  # type: ignore[attr-defined]

    # Clamp and write
    speeds = torch.clamp(speeds, min=float(min_speed), max=float(max_speed))
    env._anim_playback_speed[env_ids] = speeds[env_ids].to(
        device=env._anim_playback_speed.device, dtype=env._anim_playback_speed.dtype  # type: ignore[attr-defined]
    )


def init_animation_phase_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None = None,
):
    """Initialize per-env animation phase offsets to zeros on device.

    This ensures observations that use phase can run before any reset-based randomization.
    """
    print("init_animation_phase_offsets")
    asset: Articulation = env.scene["robot"]
    device = asset.device
    num_envs = env.num_envs
    # Per-env static phase offset in [0,1)
    setattr(env, "_anim_phase_offset", torch.zeros(num_envs, device=device, dtype=torch.float32))
    # Per-env playback speed multiplier (1.0 = realtime)
    setattr(env, "_anim_playback_speed", torch.ones(num_envs, device=device, dtype=torch.float32))
    # Per-env accumulated phase increment (wraps in [0,1))
    setattr(env, "_anim_phase_accum", torch.zeros(num_envs, device=device, dtype=torch.float32))
    # Step tracking to advance accumulator once per step
    setattr(env, "_anim_last_step", int(-1))
 # if not hasattr(env, "_anim_phase_offset"):
    #     setattr(env, "_anim_phase_offset", torch.zeros(env.num_envs, device=asset.device, dtype=torch.float32))
    


def viz_animation_sites_step(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_envs: int = 4,
    throttle_steps: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Visualize animation site points each step for a few envs, centered at current base pose.

    - Uses compute_animation_phase_and_frame(env) for consistent phase.
    - Offsets animation sites by env origin and current base world position.
    - Throttled by throttle_steps and limited to max_envs for performance.
    """
    if not is_visualization_available():
        return
    # Throttle by common step counter
    if throttle_steps > 1 and (getattr(env, "common_step_counter", 0) % throttle_steps != 0):
        return

    anim = get_animation()
    sites = anim.get("site_positions", None)
    nsite = int(anim.get("nsite", 0) or 0)
    if sites is None or nsite <= 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    # Determine which envs to draw
    draw_env_ids = env_ids[: max_envs]
    if len(draw_env_ids) == 0:
        return

    # Compute frame indices
    _, frame_idx = compute_animation_phase_and_frame(env)
    # Current base positions
    base_pos_w = asset.data.root_pos_w[draw_env_ids]
    env_origins = env.scene.env_origins[draw_env_ids]

    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    # Overwrite any previous point drawings for a clean per-frame update
    draw_interface.clear_points()
    current_time = time.time()
    global _anim_sites_timestamp, _anim_sites_drawn

    point_list = []
    colors = []
    sizes = []
    color = (0.1, 0.9, 0.2, 1.0)
    size = 6
    # Build points per env
    fi_sel = frame_idx[draw_env_ids].detach().cpu()
    base_pos_cpu = base_pos_w.detach().cpu()
    origins_cpu = env_origins.detach().cpu()
    for i in range(len(draw_env_ids)):
        fi = int(fi_sel[i].item())
        pts = sites[fi].detach().cpu()  # [nsite, 3]
        origin = origins_cpu[i]
        base = base_pos_cpu[i]
        # Center around current base: shift animation sites by (origin + base)
        for j in range(nsite):
            x = float(pts[j, 0].item() + origin[0].item() + base[0].item())
            y = float(pts[j, 1].item() + origin[1].item() + base[1].item())
            z = float(pts[j, 2].item() + origin[2].item() + base[2].item())
            point_list.append((x, y, z))
            colors.append(color)
            sizes.append(size)

    if point_list:
        try:
            draw_interface.draw_points(point_list, colors, sizes)
            _anim_sites_timestamp = current_time
            _anim_sites_drawn = True
        except Exception as e:
            print(f"[DEBUG VIZ] draw_points (anim sites) unavailable: {e}")


# ===== World forward-velocity visualization (desired vs. actual) =====
def viz_forward_velocity_world_step(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_envs: int = 4,
    throttle_steps: int = 1,
    duration: float = 1.0,
    arrow_scale: float = 1.5,
    height_offset: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Visualize world-frame forward velocity: desired (from animation) and actual (measured).

    - Desired vector: +X world with magnitude from animation metadata key 'base_forward_velocity_mps'.
    - Actual vector: robot base linear velocity in world frame (vx, vy, vz) projected in XY.

    Draws two arrows per env from a point above the base: desired (green) and actual (orange).
    """
    if not is_visualization_available():
        return
    # Throttle by common step counter
    if throttle_steps > 1 and (getattr(env, "common_step_counter", 0) % throttle_steps != 0):
        return

    anim = get_animation()
    meta = anim.get("metadata", {}) or {}
    if "base_forward_velocity_mps" not in meta or meta["base_forward_velocity_mps"] is None:
        raise RuntimeError("Animation metadata missing 'base_forward_velocity_mps' for forward-velocity viz")
    base_target = float(meta["base_forward_velocity_mps"])  # must be > 0
    if not np.isfinite(base_target) or base_target <= 0.0:
        raise RuntimeError("'base_forward_velocity_mps' must be a finite positive float")

    asset: Articulation = env.scene[asset_cfg.name]
    draw_env_ids = env_ids[: max_envs]
    if len(draw_env_ids) == 0:
        return

    # Positions and velocities
    base_pos_w = asset.data.root_pos_w[draw_env_ids]
    vel_w = asset.data.root_lin_vel_w[draw_env_ids]

    # Prepare draw interface and optional timed clear
    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    current_time = time.time()
    global _vel_arrows_timestamp, _vel_arrows_drawn
    if _vel_arrows_drawn and (current_time - _vel_arrows_timestamp) > duration:
        draw_interface.clear_lines()
        _vel_arrows_drawn = False

    # Build line lists for both desired and actual in one shot to avoid internal clears
    line_start_points: list[list[float]] = []
    line_end_points: list[list[float]] = []
    line_colors: list[tuple[float, float, float, float]] = []
    line_sizes: list[float] = []

    # Arrowhead parameters
    arrowhead_length = 0.3
    arrowhead_angle_deg = 25.0
    angle_rad = np.radians(arrowhead_angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Colors: desired (green-ish), actual (orange)
    desired_color = (0.1, 0.9, 0.2, 1.0)
    actual_color = (1.0, 0.6, 0.1, 1.0)

    def _append_arrow(start_xyz: np.ndarray, vec_xyz: np.ndarray, color: tuple[float, float, float, float], size_main: float = 4.0, size_head: float = 3.0):
        # main arrow
        end_xyz = start_xyz + vec_xyz * arrow_scale
        line_start_points.append([float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])])
        line_end_points.append([float(end_xyz[0]), float(end_xyz[1]), float(end_xyz[2])])
        line_colors.append(color)
        line_sizes.append(size_main)
        # arrowhead projected in XY plane for clarity
        direction_2d = np.array([end_xyz[0] - start_xyz[0], end_xyz[1] - start_xyz[1]], dtype=float)
        norm_2d = np.linalg.norm(direction_2d)
        if norm_2d > 1e-9:
            direction_2d = direction_2d / norm_2d
            # Left head
            arrowhead_dir1 = np.array([
                -direction_2d[0] * cos_angle + direction_2d[1] * sin_angle,
                -direction_2d[1] * cos_angle - direction_2d[0] * sin_angle,
            ])
            head1 = end_xyz.copy()
            head1[0] += arrowhead_dir1[0] * arrowhead_length
            head1[1] += arrowhead_dir1[1] * arrowhead_length
            line_start_points.append([float(end_xyz[0]), float(end_xyz[1]), float(end_xyz[2])])
            line_end_points.append([float(head1[0]), float(head1[1]), float(head1[2])])
            line_colors.append(color)
            line_sizes.append(size_head)
            # Right head
            arrowhead_dir2 = np.array([
                -direction_2d[0] * cos_angle - direction_2d[1] * sin_angle,
                -direction_2d[1] * cos_angle + direction_2d[0] * sin_angle,
            ])
            head2 = end_xyz.copy()
            head2[0] += arrowhead_dir2[0] * arrowhead_length
            head2[1] += arrowhead_dir2[1] * arrowhead_length
            line_start_points.append([float(end_xyz[0]), float(end_xyz[1]), float(end_xyz[2])])
            line_end_points.append([float(head2[0]), float(head2[1]), float(head2[2])])
            line_colors.append(color)
            line_sizes.append(size_head)

    # Per-env desired targets scaled by playback speed (if available)
    if not hasattr(env, "_anim_playback_speed"):
        desired_targets = None
    else:
        speed = env._anim_playback_speed  # type: ignore[attr-defined]
        if speed.dim() == 0:
            speed = speed.view(1).expand(asset.data.root_lin_vel_w.shape[0])
        desired_targets = (base_target * speed.to(device=asset.device, dtype=asset.data.root_lin_vel_w.dtype)).detach().cpu().numpy()

    # Build arrows
    base_pos_cpu = base_pos_w.detach().cpu().numpy()
    vel_w_cpu = vel_w.detach().cpu().numpy()
    for i in range(len(draw_env_ids)):
        start = base_pos_cpu[i].copy()
        start[2] += float(height_offset)
        # Desired: along +X world
        v_des = float(desired_targets[i]) if desired_targets is not None else float(base_target)
        desired_vec = np.array([v_des, 0.0, 0.0], dtype=float)
        _append_arrow(start, desired_vec, desired_color)
        # Actual: measured world velocity (use XY components; keep Z for 3D continuity)
        actual_vec = np.array([vel_w_cpu[i, 0], vel_w_cpu[i, 1], 0.0], dtype=float)
        _append_arrow(start, actual_vec, actual_color)

    # Draw
    if line_start_points:
        # Clear previous lines for this viz before drawing new ones
        if _vel_arrows_drawn:
            draw_interface.clear_lines()
        draw_interface.draw_lines(line_start_points, line_end_points, line_colors, line_sizes)
        _vel_arrows_timestamp = current_time
        _vel_arrows_drawn = True


def viz_heading_world_xy_step(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_envs: int = 4,
    throttle_steps: int = 1,
    duration: float = 1.0,
    arrow_scale: float = 1.5,
    height_offset: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Visualize robot heading: base +Z axis projected to world XY vs. world +X.

    - Actual heading: quat_apply(base_quat_w, [0,0,1]) -> project to XY plane -> normalize.
    - Reference heading: world +X = [1,0,0].

    Draws two arrows per env from above the base: actual (blue) and reference (white).
    """
    if not is_visualization_available():
        return
    if throttle_steps > 1 and (getattr(env, "common_step_counter", 0) % throttle_steps != 0):
        return

    asset: Articulation = env.scene[asset_cfg.name]
    draw_env_ids = env_ids[: max_envs]
    if len(draw_env_ids) == 0:
        return

    base_pos_w = asset.data.root_pos_w[draw_env_ids]
    base_quat_w = asset.data.root_quat_w[draw_env_ids]

    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    current_time = time.time()
    global _heading_arrows_timestamp, _heading_arrows_drawn
    if _heading_arrows_drawn and (current_time - _heading_arrows_timestamp) > duration:
        draw_interface.clear_lines()
        _heading_arrows_drawn = False

    line_start_points: list[list[float]] = []
    line_end_points: list[list[float]] = []
    line_colors: list[tuple[float, float, float, float]] = []
    line_sizes: list[float] = []

    # Arrowhead params
    arrowhead_length = 0.3
    arrowhead_angle_deg = 25.0
    angle_rad = np.radians(arrowhead_angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Colors
    actual_color = (0.1, 0.4, 1.0, 1.0)   # blue
    ref_color = (1.0, 1.0, 1.0, 1.0)      # white

    def _append_arrow(start_xyz: np.ndarray, vec_xyz: np.ndarray, color: tuple[float, float, float, float], size_main: float = 4.0, size_head: float = 3.0):
        end_xyz = start_xyz + vec_xyz * arrow_scale
        line_start_points.append([float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])])
        line_end_points.append([float(end_xyz[0]), float(end_xyz[1]), float(end_xyz[2])])
        line_colors.append(color)
        line_sizes.append(size_main)
        # arrowhead in XY plane
        direction_2d = np.array([end_xyz[0] - start_xyz[0], end_xyz[1] - start_xyz[1]], dtype=float)
        norm_2d = np.linalg.norm(direction_2d)
        if norm_2d > 1e-9:
            direction_2d = direction_2d / norm_2d
            arrowhead_dir1 = np.array([
                -direction_2d[0] * cos_angle + direction_2d[1] * sin_angle,
                -direction_2d[1] * cos_angle - direction_2d[0] * sin_angle,
            ])
            head1 = end_xyz.copy()
            head1[0] += arrowhead_dir1[0] * arrowhead_length
            head1[1] += arrowhead_dir1[1] * arrowhead_length
            line_start_points.append([float(end_xyz[0]), float(end_xyz[1]), float(end_xyz[2])])
            line_end_points.append([float(head1[0]), float(head1[1]), float(head1[2])])
            line_colors.append(color)
            line_sizes.append(size_head)
            arrowhead_dir2 = np.array([
                -direction_2d[0] * cos_angle - direction_2d[1] * sin_angle,
                -direction_2d[1] * cos_angle + direction_2d[0] * sin_angle,
            ])
            head2 = end_xyz.copy()
            head2[0] += arrowhead_dir2[0] * arrowhead_length
            head2[1] += arrowhead_dir2[1] * arrowhead_length
            line_start_points.append([float(end_xyz[0]), float(end_xyz[1]), float(end_xyz[2])])
            line_end_points.append([float(head2[0]), float(head2[1]), float(head2[2])])
            line_colors.append(color)
            line_sizes.append(size_head)

    # Compute and draw per env
    start_cpu = base_pos_w.detach().cpu().numpy()
    quat_cpu = base_quat_w.detach().cpu()
    z_axis_b = torch.tensor([0.0, 0.0, 1.0], device=base_quat_w.device, dtype=base_quat_w.dtype).unsqueeze(0).repeat(len(draw_env_ids), 1)
    z_axis_w = math_utils.quat_apply(base_quat_w, z_axis_b).detach().cpu().numpy()  # [N,3]
    for i in range(len(draw_env_ids)):
        start = start_cpu[i].copy()
        start[2] += float(height_offset)
        # Actual heading: project +Z_w to XY and normalize
        hx, hy = float(z_axis_w[i, 0]), float(z_axis_w[i, 1])
        norm_xy = (hx * hx + hy * hy) ** 0.5
        if norm_xy < 1e-9:
            continue
        actual_vec = np.array([hx / norm_xy, hy / norm_xy, 0.0], dtype=float)
        _append_arrow(start, actual_vec, actual_color)
        # Reference: +X world
        ref_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        _append_arrow(start, ref_vec, ref_color)

    if line_start_points:
        if _heading_arrows_drawn:
            draw_interface.clear_lines()
        draw_interface.draw_lines(line_start_points, line_end_points, line_colors, line_sizes)
        _heading_arrows_timestamp = current_time
        _heading_arrows_drawn = True

# def viz_base_positions_step(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     max_envs: int = 4,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """Visualize base world positions each step for a few envs to confirm interval events run.

#     Draws points at current base positions; no clearing to avoid wiping other debug draws.
#     """
#     if not is_visualization_available():
#         return
#     # if throttle_steps > 1 and (getattr(env, "common_step_counter", 0) % throttle_steps != 0):
#     #     return
    
#     asset: Articulation = env.scene[asset_cfg.name]
#     draw_env_ids = env_ids[: max_envs]
#     if len(draw_env_ids) == 0:
#         return
#     draw_interface = omni_debug_draw.acquire_debug_draw_interface()
#     base_pos_w = asset.data.root_pos_w[draw_env_ids].detach().cpu()
#     point_list = []
#     colors = []
#     sizes = []
#     color = (1.0, 0.1, 0.1, 1.0)
#     size = 12
#     for i in range(len(draw_env_ids)):
#         p = base_pos_w[i]
#         point_list.append((float(p[0].item()), float(p[1].item()), float(p[2].item())))
#         colors.append(color)
#         sizes.append(size)
#     draw_interface.draw_points(point_list, colors, sizes)
