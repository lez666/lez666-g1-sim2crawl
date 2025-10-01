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
class G1CrawlTransitionSceneCfg(InteractiveSceneCfg):
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

    boolean_command = mdp.BooleanCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),
        true_probability=0.5,
        debug_vis=False
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
        boolean_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "boolean_command"})
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
        boolean_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "boolean_command"})
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

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "z": (-0.5, 0.5),
    #             "roll": (-0.5, 0.5),
    #             "pitch": (-0.5, 0.5),
    #             "yaw": (-0.5, 0.5),
    #         },
    #     },
    # )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    # reset
    # Reset robot to pose from JSON with optional noise/scaling for both root and joints
    reset_robot = EventTerm(
        func=mdp.reset_to_pose_json,
        mode="reset",
        params={
            "json_path": "assets/default-pose.json",
            # Root pose noise (position in meters, angles in radians)
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.1),
                # "roll": (-0.5, 0.5),
                # "pitch": (-0.5, 0.5),
                # "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },

            # "pose_range": {
            #     "x": (-0.2, 0.2),
            #     "y": (-0.2, 0.2),
            #     "z": (-0.2, 0.2),
            #     "roll": (0.0, 0.0),
            #     "pitch": (0.0, 0.0),
            #     "yaw": (-3.14, 3.14),
            # },
            # # Root velocity noise (linear m/s, angular rad/s)
            # "velocity_range": {
            #     "x": (-0.5, 0.5),
            #     "y": (-0.5, 0.5),
            #     "z": (-0.5, 0.5),
            #     "roll": (-0.5, 0.5),
            #     "pitch": (-0.5, 0.5),
            #     "yaw": (-0.5, 0.5),
            # },
            # Joint position scaling (multiplies JSON pose values)
            "position_range": (0.5, 1.5),
            # Joint velocity scaling (multiplies joint velocities, 0 means no velocity)
            "joint_velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_with_viz,
        mode="interval",
        interval_range_s=(2.0, 2.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity_with_viz,
    #     mode="interval",
    #     interval_range_s=(5,10),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )
# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_crawl)
# def override_value(env, env_ids, data, value, num_steps):
#         if env.common_step_counter > num_steps:
#             return value
#         return mdp.modify_term_cfg.NO_CHANGE 

# @configclass
# class CurriculumCfg:


#     command_object_pose_xrange_adr = CurrTerm(
#                 func=mdp.modify_term_cfg,
#                 params={
#                     "address": "commands.base_velocity.ranges.lin_vel_z",   # note: `_manager.cfg` is omitted
#                     "modify_fn": override_value,
#                     "modify_params": {"value": (0.0,2.0), "num_steps": 25000}
#                 }
#             )

#     push_event_freq = CurrTerm(
#                 func=mdp.modify_term_cfg,
#                 params={
#                     "address": "events.push_robot.interval_range_s",   # note: `_manager.cfg` is omitted
#                     "modify_fn": override_value,
#                     "modify_params": {"value": (2,5), "num_steps": 25000}
#                 }
#             )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    feet_on_ground = RewTerm(
        func=mdp.both_feet_on_ground, 
        weight=10.0,       
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    lin_vel_l2 = RewTerm(func=mdp.lin_vel_l2, weight=-5.0)
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-5.0)

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-.1,
        params={
            "target_height": 0.74,
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        },
    )

    # joint_deviation_hip_waist = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", "waist_yaw_joint"])},
    # )

    # joint_deviation_upper_body = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"])},
    # )

    #limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    #regulatorization
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    slippage = RewTerm(
        func=mdp.feet_slide, 
        weight=-1.0,
                params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )


    # Command-based joint deviation penalty - UPPER BODY (shoulders/arms) - lower weight
    command_joint_deviation_upper = RewTerm(
        func=mdp.command_based_joint_deviation_l1,
        weight=-.2,
        params={
            "command_name": "boolean_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]),
            "command_0_pose_path": "assets/default-pose.json",
            "command_1_pose_path": "assets/default-pose.json",
        },
    )
    
    # Command-based joint deviation penalty - MID/LOWER BODY (waist/hips/legs) - higher weight
    command_joint_deviation_hip_waist = RewTerm(
        func=mdp.command_based_joint_deviation_l1,
        weight=-.1,
        params={
            "command_name": "boolean_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", "waist_yaw_joint"]),
            "command_0_pose_path": "assets/default-pose.json",
            "command_1_pose_path": "assets/default-pose.json",
        },
    )
    


    # Orientation tied to the boolean command: crawl(0) vs default(1  # - Negative L2 penalty (switches target based on command)
    # commanded_orientation_l2_penalty = RewTerm(
    #     func=mdp.command_based_orientation_l2_penalty,
    #     weight=-0.25,
    #     params={
    #         "command_name": "boolean_command",
    #     },
    # )

    # # - Exponential proximity bonus with per-branch stds
    # commanded_orientation_bonus = RewTerm(
    #     func=mdp.command_based_orientation_bonus_exp,
    #     weight=1.25,
    #     params={
    #         "command_name": "boolean_command",
    #         "std_crawl": 0.1,
    #         "std_stand": 0.1,
    #     },
    # )

    # base_height_l2 = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-.2,
    #     params={
    #         "target_height": 0.78,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
    #     },
    # )

    # command_joint_deviation_all = RewTerm(
    #     func=mdp.command_based_joint_deviation_l1,
    #     weight=-0.2,
    #     params={
    #         "command_name": "boolean_command",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "command_0_pose_path": "assets/stand-pose.json",
    #         "command_1_pose_path": "assets/stand-pose.json",
    #     },
    # )
    
    # # Bonus for being close to command-based goal joint positions
    # command_pose_proximity_bonus = RewTerm(
    #     func=mdp.command_based_pose_proximity_bonus_exp,
    #     weight=1.0,
    #     params={
    #         "command_name": "boolean_command",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "std": 0.1,  # Adjust this value: smaller = sharper falloff, larger = more forgiving
    #         "command_0_pose_path": "assets/stand-pose.json",
    #         "command_1_pose_path": "assets/stand-pose.json",
    #     },
    # )
    


    # # Proximity bonus - UPPER BODY (shoulders/arms) - lower weight
    # command_pose_proximity_bonus_upper = RewTerm(
    #     func=mdp.command_based_pose_proximity_bonus_exp,
    #     weight=0.5,
    #     params={
    #         "command_name": "boolean_command",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]),
    #         "std": 0.1,
    #         "command_0_pose_path": "assets/stand-pose-rc2.json",
    #         "command_1_pose_path": "assets/stand-pose-rc2.json",
    #     },
    # )
    
    # # Proximity bonus - MID/LOWER BODY (waist/hips/legs) - higher weight
    # command_pose_proximity_bonus_lower = RewTerm(
    #     func=mdp.command_based_pose_proximity_bonus_exp,
    #     weight=1.5,
    #     params={
    #         "command_name": "boolean_command",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*", ".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"]),
    #         "std": 0.1,
    #         "command_0_pose_path": "assets/stand-pose-rc2.json",
    #         "command_1_pose_path": "assets/stand-pose-rc2.json",
    #     },
    # )
    
    # #limits
    # dof_pos_limits = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    # torque_limits = RewTerm(
    #     func=mdp.applied_torque_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # #regulatorization
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    # bellyhead_drag_penalty = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-5.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names= "torso_link",
    #         ),
    #         "threshold": 1.0,  # in Newtons (normal force magnitude)
    #     },
    # )

    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


    # undesired_body_contact_penalty = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-2.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names="^(?!.*ankle_roll_link|.*wrist_link|torso_link).*",
    #         ),
    #         "threshold": 1.0,  # in Newtons (normal force magnitude)
    #     },
    # )

    # slippage = RewTerm(
    #     func=mdp.feet_slide, 
    #     weight=-.2,
    #             params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link", ".*_wrist_link"]),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link", ".*_wrist_link"]),
    #     },
    # )

    # both_feet_air = RewTerm(
    #     func=mdp.both_feet_air,
    #     weight=-.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #     },
    # )
    

    # both_hand_air = RewTerm(
    #     func=mdp.both_feet_air,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wrist_link"),
    #     },
    # )

    # both_left_air = RewTerm(
    #     func=mdp.both_feet_air,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wrist_link", "left_ankle_roll_link"]),
    #     },
    # )

    # both_right_air = RewTerm(
    #     func=mdp.both_feet_air,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_link", "right_ankle_roll_link"]),
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

@configclass
class G1CrawlTransitionEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: G1CrawlTransitionSceneCfg = G1CrawlTransitionSceneCfg(num_envs=4096, env_spacing=4.0)
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
        self.episode_length_s = 20.0 #7.0
        # simulation settings
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
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"

        # self.scene.height_scanner = None

        # if getattr(self.curriculum, "terrain_levels", None) is not None:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = True
       
