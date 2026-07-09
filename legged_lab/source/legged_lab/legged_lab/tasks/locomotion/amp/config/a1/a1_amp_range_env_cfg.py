import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
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
AMP_NUM_STEPS = 4


@configclass
class A1AmpRangeRewards:
    """Reward terms for the standalone A1 walk/turn range task.

    Self-contained snapshot of what the (previously inherited) range task actually
    used — no inheritance from the fixed-speed config. Final weights are baked in
    here; __post_init__ only sets motion/commands/terminations/events.
    """

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.36)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # NOTE: this weight is POSITIVE (2e-6) — carried over verbatim from the range
    # config. A positive torque weight *rewards* torque; if that was a typo, flip to
    # -2.0e-6 (the original A1AmpRewards value).
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    # only penalize ankle-pitch (joint 5) hitting its limit, not joint 6
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_R5", ".*_L5"])},
    )

    # hip-yaw (joint 3) deviation
    joint_deviation_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R3", "joint_L3"])},
    )
    # ankle-pitch (joint 5) deviation
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R5", "joint_L5"])},
    )
    # hip-abduction (joint 2) deviation — keep legs from splaying
    joint_deviation_hip2 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R2", "joint_L2"])},
    )

    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-2.0,
        params={"command_name": "base_velocity"},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)

    # full-sole flatness (pitch+roll); off by default here (weight 0)
    feet_flat = RewTerm(
        func=mdp.feet_orientation_l2,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"]),
        },
    )

    feet_flat_orientation = RewTerm(
        func=mdp.feet_flat_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"])},
    )

    # NOTE: joint_deviation_hip (joint 1) and feet_roll are intentionally NOT included
    # (the range task had set them to None).


@configclass
class A1AmpRangeEnvCfg(LocomotionAmpEnvCfg):
    """Standalone range-velocity walk/turn AMP for A1 (no inheritance from fixed)."""

    rewards: A1AmpRangeRewards = A1AmpRangeRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 256
        self.scene.robot = A1_LEGS_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------
        # motion data — continuous walk + turns (a1_cal), all weights = 1
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "a1_cal"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            # --- straight walk (~0.5) ---
            "B3_-_walk1": 1.0,
            # --- continuous walk turns (L/R, various angles) ---
            "B10_-_walk_turn_left_(45)": 1.0,
            "B9_-__Walk_turn_left_90": 1.0,
            "B11_-__Walk_turn_left_135": 1.0,
            "B13_-__Walk_turn_right_90": 1.0,
            "B14_-_walk_turn_right_(135)": 1.0,
            "B15_-_walk_turn_around_(same_direction)": 1.0,
        }

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
        self.terminal_obs_groups = ("disc",)
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events (incl. cold-start: 20% reset to default stand, 80% mid-motion ref)
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base"]
        self.events.reset_from_ref.func = mdp.reset_from_ref_or_default
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1,
            "ref_ratio": 0.8,
        }

        # ------------------------------------------------------
        # Commands — direct (vx, vy, wz); forward+turn only, no lateral/backward
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.05

        # ------------------------------------------------------
        # Curriculum — none (full range from step 0)
        # ------------------------------------------------------
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None

        # ------------------------------------------------------
        # Terminations — "fell down" resets (base_contact enabled, tight gates)
        # ------------------------------------------------------
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0},
        )
        self.terminations.bad_orientation.params["limit_angle"] = math.radians(45.0)
        self.terminations.base_height.params["minimum_height"] = 0.35


@configclass
class A1AmpRangeEnvCfg_PLAY(A1AmpRangeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
