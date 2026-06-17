import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.velocity.mdp import curriculums as vel_curr
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
class A1AmpRewards:
    """Reward terms for A1 AMP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.36)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
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
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R1", "joint_L1"])},
    )

    joint_deviation_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R3", "joint_L3"])},
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
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.125,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"]),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"]),
    #     },
    # )

    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)


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
            LEGGED_LAB_ROOT_DIR, "data", "a1_4"
        )
        # Gait distribution for command range vx in [-0.8, 1.5].
        # Policy: ALL weights = 1.0 (uniform sampling). Distribution is shaped by the NUMBER
        # of clips per category, NOT by weights — to emphasize a regime, add more distinct
        # clips of it. Disc obs are body-local (imitate GAIT not speed), so each gait family's
        # clips cover its whole speed range. Counts: forward 7 | backward 5 | turn 5 | run-turn 4 | side 2.
        # (B3 is the lone weight=2.0 exception — see below.) Run-turns added to flatten the vx
        # histogram: the 1.0-1.5 jog band was starved (~4%); they lift it to ~10%.
        self.motion_data.motion_dataset.motion_data_weights = {
            # --- forward: walk -> run, dense coverage up to ~1.5 ---
            # "B1_-_stand_to_walk": 1.0,       # accel 0 -> 0.5
            # "B3_-_walk1": 2.0,               # steady walk 0.55 (backbone)
            # "B2_-_walk_to_stand": 1.0,       # decel -> 0
            # "C1_-_stand_to_run": 1.5,        # ramps 0 -> 2.1, sweeps the WHOLE forward range in one clip
            # "C5_-_walk_to_run": 1.5,         # walk->run, mid-speed 0.7-1.2
            # "C4_-_Run_to_walk1": 1.5,        # run->walk decel
            # "C3_-_run": 1.5,                 # steady run ~1.7 (high-speed gait template for ~1.5)
            # # --- backward walk (rebalanced UP: was under-represented vs forward (4.0 vs 10.0),
            # # and SN's gentler disc under-enforced the sparse backward direction -> backward
            # # drifted off-style. These walk-back clips' PEAKS reach -0.8~-0.9 (disc imitates the
            # # GAIT not the speed, so the walk-back gait covers the full backward range). Dropped
            # # C8_run_backwards to keep backward style pure walk. New backward total ~8.0.) ---
            # "B5_-__Walk_backwards": 3.0,          # steady backwalk, mean -0.59 / peak -0.95 (main)
            # "B5_-_walk_backwards": 1.5,           # steady backwalk, mean -0.36 (long, 691f)
            # "B4_-_Stand_to_Walk_backwards": 1.5,  # brisk backwalk, peak -0.90 (covers the -0.8 band)
            # "B4_-_stand_to_walk_back": 1.0,       # stand->backwalk accel
            # "B6_-_walk_backwards_to_stand": 1.0,  # backwalk->stand decel
            # # --- turns (~0.5) ---
            # "B9_-__Walk_turn_left_90": 1.0,
            # "B13_-__Walk_turn_right_90": 1.0,
            # "B11_-__Walk_turn_left_135": 1.0,
            # "B15_-__Walk_turn_around": 1.0,
            # "B16_-_walk_turn_change_direction": 1.0,
            # # --- side step (~0.4 vy) ---
            # "B22_-_side_step_left": 1.5,
            # "B23_-_side_step_right": 1.5,
            # forward: walk -> run (7).  B3 kept at 2.0 as a SPECIAL CASE: it is the only
            # clean steady-walk clip in a1_4, so its gait can't be reinforced by adding more
            # clips (count method fails for an under-supplied category) — weight compensates.
            "B1_-_stand_to_walk": 1.0,
            "B3_-_walk1": 2.0,
            "B2_-_walk_to_stand": 1.0,
            "C1_-_stand_to_run": 1.0,
            "C5_-_walk_to_run": 1.0,
            "C4_-_Run_to_walk1": 1.0,
            "C3_-_run": 1.0,
            # backward walk (5) — peaks reach -0.8~-0.9; C8 run-back dropped to keep style pure walk
            "B5_-__Walk_backwards": 1.0,
            "B5_-_walk_backwards": 1.0,
            "B4_-_Stand_to_Walk_backwards": 1.0,
            "B4_-_stand_to_walk_back": 1.0,
            "B6_-_walk_backwards_to_stand": 1.0,
            # turns ~0.5 (5)
            "B9_-__Walk_turn_left_90": 1.0,
            "B13_-__Walk_turn_right_90": 1.0,
            "B11_-__Walk_turn_left_135": 1.0,
            "B15_-__Walk_turn_around": 1.0,
            "B16_-_walk_turn_change_direction": 1.0,
            # high-speed run-turns (4): forward vx ~1.1-1.6, best available fill for the
            # starved 1.0-1.5 band (dataset is walk-heavy, the jog band is otherwise empty);
            # they also add high-speed turning gait (walk-turns above are only ~0.5).
            "C11_-__run_turn_left_(90)": 1.0,
            "C12_-_run_turn_left_45": 1.0,
            "C14_-_run_turn_right_90": 1.0,
            "C16_-_run_turn_right_135": 1.0,
            # side step ~0.4 vy (2)
            "B22_-_side_step_left": 1.0,
            "B23_-_side_step_right": 1.0,
        }
        # self.motion_data.motion_dataset.motion_data_dir = os.path.join(
        #     LEGGED_LAB_ROOT_DIR, "data", "MotionData", "a1_12dof", "amp", "walk1_subject5"
        # )
        # self.motion_data.motion_dataset.motion_data_weights = {
        #     # "Walk_B10_-_Walk_turn_left_45": 1.0,
        #     # "Walk_B13_-_Walk_turn_right_45": 1.0,
        #     # "Walk_B15_-_Walk_turn_around": 1.0,
        #     # "Walk_B16_-_Walk_turn_change": 1.0,
        #     # "Walk_B22_-_Side_step_left": 2.0,
        #     # "Walk_B23_-_Side_step_right": 2.0,
        #     # "Walk_B4_-_Stand_to_Walk_Back": 2.0,
        #     "walk1_subject5_turn":1.0,
        #     "walk1_subject5_walk":1.0,
        #     "walk3_subject2_back":1.0,
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
        # [DIAG TEST] Narrowed cmd range to match demo coverage — checking whether
        # style-reward collapse is caused by commanding off-manifold motions.
        # Original (full range) kept below for easy revert.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        # self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.8)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        # cmd-range curriculum — starts from the narrow (demo-aligned) ranges set above
        # and expands by +-0.1 each episode whenever the tracking reward exceeds
        # reward_threshold_ratio * weight, clamped to the full target range. Threshold is
        # lowered to 0.6 (default 0.8) because AMP's task reward ceiling is capped by the
        # style split, so 0.8 is rarely reached and expansion would stall.
        #
        # Expansion is gated on TASK reward only. Watch AMP/mean_style_reward + disc_loss
        # as the range grows: if style genuinely collapses at some range (disc_demo_score
        # unchanged, task can't keep up), that range is a DATA GAP — add demo motions via
        # GMR rather than tuning. Optionally add a style-reward floor gate inside
        # velocity/mdp/curriculums.py to stop expanding into gaps automatically.
        self.curriculum.lin_vel_cmd_levels = CurrTerm(
            func=vel_curr.lin_vel_cmd_levels,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "lin_vel_x_limit": [-0.8, 1.5],
                "lin_vel_y_limit": [-0.5, 0.5],
                "reward_threshold_ratio": 0.7,
            },
        )
        self.curriculum.ang_vel_cmd_levels = CurrTerm(
            func=vel_curr.ang_vel_cmd_levels,
            params={
                "reward_term_name": "track_ang_vel_z_exp",
                "ang_vel_z_limit": [-1.0, 1.0],
                "reward_threshold_ratio": 0.5,
            },
        )

        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        self.terminations.base_contact = None
        # self.terminations.bad_orientation = None


@configclass
class A1AmpEnvCfg_PLAY(A1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
