# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from .g1 import G1_CFG

##
# Scene definition
##


@configclass
class G1Stand2CrawlSceneCfg(InteractiveSceneCfg):
  # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAAC_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # imu_pelvis = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base",offset)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.CrawlVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0,10.0),
        rel_standing_envs=0.02,
        debug_vis=True,
        ranges=mdp.CrawlVelocityCommandCfg.Ranges(
            heading=(0.0,0.0),
            # Crawling fields used by the command implementation
            lin_vel_z=(0.0, 1.5),
            lin_vel_y=(0.,0.),
            ang_vel_x=(-1.0, 1.0)
        )
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action_with_log)
        # phase = ObsTerm(func=mdp.animation_phase)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action_with_log)
        # phase = ObsTerm(func=mdp.animation_phase)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: ObsGroup = PolicyCfg()
    critic: ObsGroup = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # Floor / foot friction: =U(0.4, 1.0) - matching MuJoCo rand_dynamics
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Ensure animation phase offsets are initialized for observations before any reset
    init_anim_phase = EventTerm(
        func=mdp.init_animation_phase_offsets,
        mode="startup",
        params={},
    )

    # Scale all link masses: *U(0.9, 1.1) - matching MuJoCo rand_dynamics
    randomize_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # Add mass to torso: +U(-1.0, 1.0) - matching MuJoCo rand_dynamics
    add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    # Scale static friction: *U(0.5, 2.0) - matching MuJoCo rand_dynamics joint friction
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )

    # Scale armature: *U(1.0, 1.05) - matching MuJoCo rand_dynamics
    randomize_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (1.0, 1.05),
            "operation": "scale",
        },
    )


    # reset
    # Curriculum-aware pose array reset with progressive difficulty
    # Uses pose_viewer.py format (pose array with base_pos/base_rpy/joints)
    reset_base = EventTerm(
        func=mdp.reset_from_pose_array_with_curriculum,
        mode="reset",
        params={
            "json_path": "assets/animation_mocap_rc0_poses_sorted.json",
            
            # Curriculum parameters: start with only home frame, expand over time
            "frame_range": (0, 0),  # Start with frame 0 only - curriculum will expand this
            "home_frame": 0,  # Frame 0 is the "home base" default pose
            "home_frame_prob": 0.3,  # Always maintain 30% probability of sampling home frame

            # Small random offsets on root pose at reset (position in meters, angles in radians)
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.10, 0.10),
            },
            # Small random root velocity at reset (linear m/s, angular rad/s)
            "velocity_range": {
                "x": (-0.20, 0.20),
                "y": (-0.20, 0.20),
                "z": (-0.2, 0.2),
                "roll": (-0.30, 0.30),
                "pitch": (-0.30, 0.30),
                "yaw": (-0.30, 0.30),
            },
            "position_range": (0.9, 1.1),
            "joint_velocity_range": (0.0, 0.0),

        },
    )
    # reset_base = EventTerm(
    #     func=mdp.reset_from_animation,
    #     mode="reset",
    #     params={},
    # )


    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_with_viz,
        mode="interval",
        interval_range_s=(1000,1000),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_crawl)
def override_value(env, env_ids, data, value, num_steps):
    # if env.common_step_counter % 1000 == 0:  # print every 1000 steps
    # print(f"[curriculum probe] common_step_counter={env.common_step_counter}, "
    #     f"num_steps={num_steps}")
    if env.common_step_counter > num_steps:
        # print(f"[curriculum trigger] triggered at step {env.common_step_counter}")
        return value
    return mdp.modify_term_cfg.NO_CHANGE


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
        return mdp.modify_term_cfg.NO_CHANGE


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
        return mdp.modify_term_cfg.NO_CHANGE



@configclass
class CurriculumCfg:
    """Curriculum terms for progressive difficulty."""
    
    # Expand pose array range from easy to hard poses over training
    animation_difficulty = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.reset_base.params.frame_range",
            "modify_fn": expand_frame_range_linear,
            "modify_params": {
                "total_frames": 5795,  # Total poses in animation_mocap_rc0_poses_sorted.json
                "start_frames": 1,     # Start with just pose 0 (home pose)
                "warmup_steps": 50000, # Reach all poses by 50k steps (~3.5M env steps with 4096 envs)
            }
        }
    )
    
    # Push event curriculum (existing)
    push_event_freq = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.push_robot.interval_range_s",
            "modify_fn": override_value,
            "modify_params": {"value": (3, 10), "num_steps": 36000}
        }
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    #follow commands (base YZ plane and roll about X)
    track_lin_vel_yz_exp = RewTerm(
        func=mdp.track_lin_vel_yz_base_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_x_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.25}
    )
    flat_orientation_l2 = RewTerm(func=mdp.align_projected_gravity_plus_x_l2, weight=.2)
    
    
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-.1,
        params={
            "target_height": 0.22,
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            # "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    joint_deviation_all = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    #limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    #regulatorization
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    bellyhead_drag_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names= "torso_link",
            ),
            "threshold": 1.0,  # in Newtons (normal force magnitude)
        },
    )

    undesired_body_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="^(?!.*ankle_roll_link|.*wrist_link|torso_link).*",
            ),
            "threshold": 1.0,  # in Newtons (normal force magnitude)
        },
    )

    slippage = RewTerm(
        func=mdp.feet_slide, 
        weight=-.2,
                params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link", ".*_wrist_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link", ".*_wrist_link"]),
        },
    )

    both_feet_air = RewTerm(
        func=mdp.both_feet_air,
        weight=-.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    

    both_hand_air = RewTerm(
        func=mdp.both_feet_air,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wrist_link"),
        },
    )

    both_left_air = RewTerm(
        func=mdp.both_feet_air,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wrist_link", "left_ankle_roll_link"]),
        },
    )

    both_right_air = RewTerm(
        func=mdp.both_feet_air,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_link", "right_ankle_roll_link"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class G1Stand2CrawlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: G1Stand2CrawlSceneCfg = G1Stand2CrawlSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self) -> None:
        """Post initialization."""

        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        # self.sim.dt = 0.002
        self.sim.dt = 0.005
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

# 
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Set terrain to plane and disable height scanning
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.scene.height_scanner = None

        # if getattr(self.curriculum, "terrain_levels", None) is not None:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = True
       
