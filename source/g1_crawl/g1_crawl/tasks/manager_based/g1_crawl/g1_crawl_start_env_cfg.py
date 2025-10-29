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
from .g1 import G1_STAND_CFG

##
# Constants
##

CRAWL_POSE_PATH = "assets/crawl-pose.json"


##
# Scene definition
##


@configclass
class G1CrawlStartSceneCfg(InteractiveSceneCfg):
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
    robot: ArticulationCfg = G1_STAND_CFG.copy().replace(prim_path="{ENV_REGEX_NS}/Robot")
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

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)
        ),
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
        # boolean_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "boolean_command"})
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
        # boolean_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "boolean_command"})
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

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "y": (-0.1, 0.1),
                "x": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )


    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )


    # reset_robot = EventTerm(
    #     func=mdp.reset_from_pose_array_with_curriculum,
    #     mode="reset",
    #     params={
    #         "json_path": "assets/sorted-poses-rc3.json",
            
    #         # Curriculum parameters: start with crawling, expand BACKWARD to include standing
    #         "frame_range": (4000, 5796),  # Start with last pose only - curriculum will expand backward
    #         "home_frame": 5796,           # Pose 5796 (crawling) is the start anchor
    #         "home_frame_prob": 0.1,       # 30% always sample crawling pose
            
    #         # Optional: Add standing as end anchor (starts at 0, ramped up by curriculum)
    #         "end_home_frame": 0,          # Pose 0 (standing) is the end anchor
    #         "end_home_frame_prob": 0.0,   # Start at 0% - curriculum will ramp this up near the end
    #         # Curriculum will increase this to ~0.1 in final stages

    #         # Small random offsets on root pose at reset (position in meters, angles in radians)
    #         "pose_range": {
    #             "x": (-0.05, 0.05),
    #             "y": (-0.05, 0.05),
    #             "z": (-0.05, 0.05),
    #             "roll": (-0.10, 0.10),
    #             "pitch": (-0.10, 0.10),
    #             "yaw": (-3.14, 3.14),
    #         },
    #         # Small random root velocity at reset (linear m/s, angular rad/s)
    #         "velocity_range": {
    #             "x": (-0.05, 0.05),
    #             "y": (-0.05, 0.05),
    #             "z":(-0.05, 0.05),
    #             "roll": (-0.05, 0.05),
    #             "pitch":(-0.05, 0.05),
    #             "yaw":(-0.05, 0.05)
    #         },
    #         "position_range": (0.95, 1.05),
    #         "joint_velocity_range": (0.0, 0.0),
    #     },
    # )


    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity_with_viz,
    #     mode="interval",
    #     interval_range_s=(3.0, 5.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)

    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7, params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #         )
    #     }
    # )
    # Safety-aligned penalties (match deploy/runtime warnings)
    # Penalize absolute position jump rate violations (|Δq|/dt > 15 rad/s)
    # pos_rate_violation = RewTerm(
    #     func=mdp.joint_position_rate_violation_penalty,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "rate_limit": 15.0,
    #     },
    # )
    # # Penalize velocity spikes (|qdot| > 25 rad/s)
    # vel_violation = RewTerm(
    #     func=mdp.joint_velocity_violation_penalty,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "vel_limit": 25.0,
    #     },
    # )
    # Penalize acceleration spikes (|qddot| > 1500 rad/s^2)
    # acc_violation = RewTerm(
    #     func=mdp.joint_acceleration_violation_penalty,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "acc_limit": 1500.0,
    #     },
    # )
    # body_velocity_regularizer = RewTerm(
    #     func=mdp.body_lin_acc_l2, 
    #     weight=-0.1,  # Negative weight for penalty
    #     params={"asset_cfg": SceneEntityCfg("robot")}  # All body parts
    # )
    

    # both_feet_air = RewTerm(
    #     func=mdp.both_feet_air,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #     },
    # )

    # desired_body_contact_penalty = RewTerm(
    #     func=mdp.desired_contacts,
    #     weight=3.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=".*(ankle_roll_link|wrist_link)",
    #         ),
    #     }
    # )

    undesired_body_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="^(?!.*ankle_roll_link|.*wrist_link).*",
            ),
            "threshold": 1.0,  # in Newtons (normal force magnitude)
        },
    )

    # flat_orientation_l2 = RewTerm(func=mdp.align_projected_gravity_plus_x_l2, weight=3.)
    misalign_projected_gravity_plus_x_l2 = RewTerm(
        func=mdp.misalign_projected_gravity_plus_x_l2, 
        weight=-3.0
    )

    # base_height_l2 = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-3.,
    #     params={
    #         "target_height": 0.22,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
    #         # "sensor_cfg": SceneEntityCfg("height_scanner"),
    #     },
    # )

    # joint_deviation_all = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )


    # delayed_pose_deviation = RewTerm(
    #     func=mdp.pose_json_deviation_l1_after_delay,
    #     weight=-0.1,
    #     params={
    #         "pose_path": "assets/crawl-pose.json",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "delay_s": 1.,
    #         "ramp_s": 0.1,
    #     }
    # )

    pose_deviation = RewTerm(
        func=mdp.pose_json_deviation_l1_two_stage,
        weight=-0.1,
        params={
            "pose_path_before": "assets/default-pose.json",
            "pose_path_after": "assets/crawl-pose.json", 
            "delay_s": 2.,
            "ramp_s": 0.25,
        }
    )
        
    
    # either_foot_off_ground = RewTerm(
    #     func=mdp.either_foot_off_ground,
    #     weight=-0.3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=".*_ankle_roll_link",
    #         ),
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class G1CrawlStartEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: G1CrawlStartSceneCfg = G1CrawlStartSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self) -> None:
        """Post initialization."""

        # general settings
        self.decimation = 4
        self.episode_length_s = 8.0 #7.0
        # simulation settings
        self.sim.dt = 0.005
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

# 
        # self.scene.robot = G1_STAND_CFG.copy().replace(prim_path="{ENV_REGEX_NS}/Robot")

        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Set terrain to plane and disable height scanning
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
       
