# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs.mdp.curriculums import modify_term_cfg  

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel_crawl(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())



#     terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_crawl)
def override_value(env, env_ids, data, value, num_steps):
    # if env.common_step_counter % 1000 == 0:  # print every 1000 steps
    # print(f"[curriculum probe] common_step_counter={env.common_step_counter}, "
    #     f"num_steps={num_steps}")
    if env.common_step_counter > num_steps:
        # print(f"[curriculum trigger] triggered at step {env.common_step_counter}")
        return value
    return modify_term_cfg.NO_CHANGE


def expand_frame_range_linear(env, env_ids, data, total_frames, start_frames=1, warmup_steps=50000):
    """Expand animation frame range linearly from start_frames to total_frames over warmup_steps.
    
    This curriculum function gradually expands the range of animation frames available for reset sampling.
    It starts with only the first few frames (easier poses) and progressively includes more challenging
    poses as training progresses.
    
    Args:
        env: The learning environment
        env_ids: Environment IDs (not used, but required by curriculum API)
        data: Current value (not used for this curriculum)
        total_frames: Maximum number of frames to reach at the end of warmup
        start_frames: Number of frames to start with (default 1 = only home frame)
        warmup_steps: Number of training steps to reach full frame range
    
    Returns:
        Tuple (min_frame, max_frame) representing the current frame range.
        min_frame is always 0 (home frame always available).
        max_frame increases from start_frames to total_frames over warmup_steps.
    
    Example:
        At step 0: returns (0, 1) - only frames 0-1 available
        At step 25000: returns (0, 150) - frames 0-150 available (halfway through 300)
        At step 50000+: returns (0, 300) - all frames available
    """
    # Calculate progress (0.0 at start, 1.0 at warmup_steps)
    progress = min(1.0, float(env.common_step_counter) / float(warmup_steps))
    
    # Linearly interpolate from start_frames to total_frames
    max_frame = int(start_frames + (total_frames - start_frames) * progress)
    
    # Always start from frame 0 (home frame)
    min_frame = 0
    
    # Only update if we've progressed beyond the initial state
    if env.common_step_counter > 0:
        return (min_frame, max_frame)
    else:
        return modify_term_cfg.NO_CHANGE


def expand_frame_range_exponential(env, env_ids, data, total_frames, start_frames=1, warmup_steps=50000, exponent=2.0):
    """Expand animation frame range exponentially for slower early progress, faster later.
    
    Similar to expand_frame_range_linear, but uses an exponential curve. This keeps the robot
    practicing easier poses for longer before introducing harder ones more rapidly later in training.
    
    Args:
        env: The learning environment
        env_ids: Environment IDs (not used, but required by curriculum API)
        data: Current value (not used for this curriculum)
        total_frames: Maximum number of frames to reach at the end of warmup
        start_frames: Number of frames to start with (default 1 = only home frame)
        warmup_steps: Number of training steps to reach full frame range
        exponent: Exponential power (2.0 = quadratic, higher = slower start)
    
    Returns:
        Tuple (min_frame, max_frame) representing the current frame range.
    
    Example with exponent=2.0:
        At step 0: returns (0, 1)
        At step 25000: returns (0, 75) - only 25% of frames (slower than linear)
        At step 43000: returns (0, 225) - 75% of frames (catching up)
        At step 50000+: returns (0, 300) - all frames
    """
    # Calculate progress (0.0 at start, 1.0 at warmup_steps)
    progress = min(1.0, float(env.common_step_counter) / float(warmup_steps))
    
    # Apply exponential curve
    progress_exp = progress ** exponent
    
    # Interpolate from start_frames to total_frames
    max_frame = int(start_frames + (total_frames - start_frames) * progress_exp)
    
    min_frame = 0
    
    if env.common_step_counter > 0:
        return (min_frame, max_frame)
    else:
        return modify_term_cfg.NO_CHANGE


def expand_frame_range_linear_reverse(env, env_ids, data, total_frames, start_frames=1, warmup_steps=50000):
    """Expand animation frame range linearly in REVERSE (for crawl2stand).
    
    This curriculum function gradually expands the range BACKWARD from the end of the sorted poses.
    It starts with only the last pose and progressively includes earlier poses as training progresses.
    
    Perfect for crawl2stand training where:
    - Pose 5794 is crawling (easiest end state)
    - Pose 0 is standing (hardest start state)
    
    Args:
        env: The learning environment
        env_ids: Environment IDs (not used, but required by curriculum API)
        data: Current value (not used for this curriculum)
        total_frames: Maximum number of frames (should match total poses, e.g., 5795)
        start_frames: Number of frames to start with (default 1 = only last pose)
        warmup_steps: Number of training steps to reach full frame range
    
    Returns:
        Tuple (min_frame, max_frame) representing the current frame range.
        max_frame is always total_frames-1 (last pose always available).
        min_frame decreases from total_frames-1 to 0 over warmup_steps.
    
    Example with total_frames=5795:
        At step 0: returns (5794, 5794) - only last pose (crawling)
        At step 25000: returns (2897, 5794) - second half of poses
        At step 50000+: returns (0, 5794) - all poses (includes standing)
    
    Usage for crawl2stand:
        Set home_frame=5794 (crawling anchor) and this curriculum expands backward to include standing.
    """
    # Calculate progress (0.0 at start, 1.0 at warmup_steps)
    progress = min(1.0, float(env.common_step_counter) / float(warmup_steps))
    
    # Max frame is always the last pose
    max_frame = total_frames - 1
    
    # Min frame decreases from (total_frames - start_frames) to 0
    initial_min = total_frames - start_frames
    min_frame = int(initial_min - (initial_min - 0) * progress)
    
    if env.common_step_counter > 0:
        return (min_frame, max_frame)
    else:
        return modify_term_cfg.NO_CHANGE


def expand_frame_range_exponential_reverse(env, env_ids, data, total_frames, start_frames=1, warmup_steps=50000, exponent=2.0):
    """Expand animation frame range exponentially in REVERSE (for crawl2stand).
    
    Similar to expand_frame_range_linear_reverse, but uses an exponential curve for slower
    early progress. This keeps the robot practicing the crawling end state longer before
    introducing more challenging intermediate poses.
    
    Args:
        env: The learning environment
        env_ids: Environment IDs (not used, but required by curriculum API)
        data: Current value (not used for this curriculum)
        total_frames: Maximum number of frames (should match total poses)
        start_frames: Number of frames to start with (default 1 = only last pose)
        warmup_steps: Number of training steps to reach full frame range
        exponent: Exponential power (2.0 = quadratic, higher = slower start)
    
    Returns:
        Tuple (min_frame, max_frame) representing the current frame range.
    
    Example with total_frames=5795, exponent=2.0:
        At step 0: returns (5794, 5794)
        At step 25000: returns (4346, 5794) - only last 25% expanded (slower than linear)
        At step 50000+: returns (0, 5794) - all poses
    """
    # Calculate progress (0.0 at start, 1.0 at warmup_steps)
    progress = min(1.0, float(env.common_step_counter) / float(warmup_steps))
    
    # Apply exponential curve
    progress_exp = progress ** exponent
    
    # Max frame is always the last pose
    max_frame = total_frames - 1
    
    # Min frame decreases exponentially from (total_frames - start_frames) to 0
    initial_min = total_frames - start_frames
    min_frame = int(initial_min - (initial_min - 0) * progress_exp)
    
    if env.common_step_counter > 0:
        return (min_frame, max_frame)
    else:
        return modify_term_cfg.NO_CHANGE


def ramp_end_anchor_probability(env, env_ids, data, target_prob=0.1, start_step=40000, end_step=50000):
    """Gradually increase end anchor probability in later training stages.
    
    This curriculum function keeps the end anchor at 0% for most of training, then
    ramps it up linearly in the final stages. This allows the policy to focus on
    forward progression first, then add the goal state anchor for final refinement.
    
    Args:
        env: The learning environment
        env_ids: Environment IDs (not used, but required by curriculum API)
        data: Current value (not used for this curriculum)
        target_prob: Final probability to reach (e.g., 0.1 = 10%)
        start_step: Step to start ramping up (default 40000 = 80% of 50k warmup)
        end_step: Step to reach target_prob (default 50000)
    
    Returns:
        Float probability value for end_home_frame_prob.
    
    Example with target_prob=0.1, start_step=40000, end_step=50000:
        At step 0-39999: returns 0.0 (no end anchor)
        At step 40000: returns 0.0 (just starting ramp)
        At step 45000: returns 0.05 (halfway through ramp)
        At step 50000+: returns 0.1 (full end anchor)
    
    Usage for stand2crawl:
        Focuses on learning standing→intermediate poses first (80% of training),
        then adds crawling goal anchor in final 20% for refinement.
    """
    step = env.common_step_counter
    
    # Before start_step: no end anchor
    if step < start_step:
        prob = 0.0
    # After end_step: full target probability
    elif step >= end_step:
        prob = target_prob
    # During ramp: linear interpolation
    else:
        progress = (step - start_step) / (end_step - start_step)
        prob = target_prob * progress
    
    if env.common_step_counter > 0:
        return prob
    else:
        return modify_term_cfg.NO_CHANGE


def ramp_weight_linear(env, env_ids, data, initial_weight=0.0, target_weight=-10.0, start_step=0, end_step=50000):
    """Linearly ramp a reward weight from initial to target value over training.
    
    This curriculum function gradually changes a reward term weight, useful for
    introducing penalties progressively as the policy learns the basic task first.
    
    Args:
        env: The learning environment
        env_ids: Environment IDs (not used, but required by curriculum API)
        data: Current value (not used for this curriculum)
        initial_weight: Starting weight value (e.g., 0.0 for no penalty)
        target_weight: Final weight value (e.g., -10.0 for full penalty)
        start_step: Step to start ramping (default 0 = ramp from beginning)
        end_step: Step to reach target_weight
    
    Returns:
        Float weight value for the reward term.
    
    Example with initial_weight=0.0, target_weight=-10.0, start_step=0, end_step=50000:
        At step 0: returns 0.0 (no penalty)
        At step 25000: returns -5.0 (half penalty)
        At step 50000+: returns -10.0 (full penalty)
    
    Usage:
        Allows the robot to learn basic locomotion first without harsh penalties,
        then progressively adds penalties like undesired contact as it improves.
    """
    step = env.common_step_counter
    
    # Before start_step: use initial weight
    if step < start_step:
        weight = initial_weight
    # After end_step: use full target weight
    elif step >= end_step:
        weight = target_weight
    # During ramp: linear interpolation
    else:
        progress = (step - start_step) / (end_step - start_step)
        weight = initial_weight + (target_weight - initial_weight) * progress
    
    if env.common_step_counter > 0:
        return weight
    else:
        return modify_term_cfg.NO_CHANGE

# from __future__ import annotations

# from collections.abc import Sequence
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedRLEnv


# def modify_event_parameter(env: ManagerBasedRLEnv, env_ids: Sequence[int], num_steps: int):
#     """Curriculum that modifies an event parameter after a given number of steps.

#     Args:
#         env: The learning environment.
#         env_ids: Not used since all environments are affected.
#         term_name: The name of the event term.
#         weight: The new value to set for the event parameter.
#         num_steps: The number of steps after which the change should be applied.
#     """
    
#     if env.common_step_counter > num_steps:
#         # print("Changing event parameter!!" )
#         # obtain term settings
#         term_cfg = env.event_manager.get_term_cfg("push_robot")
#         # update term settings
#         term_cfg.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
#         env.event_manager.set_term_cfg("push_robot", term_cfg)
#     else:
#         term_cfg = env.event_manager.get_term_cfg("push_robot")
#         # update term settings
#         term_cfg.params["velocity_range"] = {"x": (0.,0.), "y": (0.,0.)}
#         env.event_manager.set_term_cfg("push_robot", term_cfg)