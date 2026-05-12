import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR

from legged_lab.assets.a1 import A1_LEGS_V1_CFG
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

KEY_BODY_NAMES = [
    "Link_L6",
    "Link_R6",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 8


@configclass
class A1AmpRewards:
    """Reward terms for A1 AMP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_R[5-6]", ".*_L[5-6]"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        # params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R1", "joint_R3", "joint_L1", "joint_L3"])},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R1", "joint_R3", "joint_L1", "joint_L3"])},
    )

    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.5,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"]),
    #         "threshold": 0.4,
    #     },
    # )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"]),
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.1,
        params={"command_name": "base_velocity"},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class A1AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the A1 AMP environment."""

    rewards: A1AmpRewards = A1AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 256
        self.scene.robot = A1_LEGS_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "a1_12dof", "amp", "walk1_subject5"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "walk1_subject5_turn1": 1.0,
            "walk1_subject5_walk1": 1.0,
            "walk1_subject5_walk2": 1.0,
            # "walk2_subject1_lowspeed": 1.0,
            # "walk3_subject2_back": 1.0,
        }
        # self.motion_data.motion_dataset.motion_data_dir = os.path.join(
        #     LEGGED_LAB_ROOT_DIR, "data", "MotionData", "a1_12dof", "amp", "ACCAD_walk"
        # )
        # self.motion_data.motion_dataset.motion_data_weights = {
        #     "Walk_B10_-_Walk_turn_left_45": 1.0,
        #     "Walk_B13_-_Walk_turn_right_45": 1.0,
        #     "Walk_B15_-_Walk_turn_around": 1.0,
        #     "Walk_B16_-_Walk_turn_change": 1.0,
        #     "Walk_B22_-_Side_step_left": 2.0,
        #     "Walk_B23_-_Side_step_right": 2.0,
        #     "Walk_B4_-_Stand_to_Walk_Back": 2.0,
        # }


        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        self.terminal_obs_groups = ("disc",)

        # policy observations
        # self.observations.policy.key_body_pos_b.params = {
        #     "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        # }

        # critic observations
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }

        # discriminator observations
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.history_length = AMP_NUM_STEPS

        # discriminator demonstration observations
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base"]
        self.events.reset_from_ref.params = {"animation": ANIMATION_TERM_NAME, "height_offset": 0.1}

        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None

        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        self.terminations.base_contact = None


@configclass
class A1AmpEnvCfg_PLAY(A1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
