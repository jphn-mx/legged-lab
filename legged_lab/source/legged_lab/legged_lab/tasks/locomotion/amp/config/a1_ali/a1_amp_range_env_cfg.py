import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.locomotion.amp.config.a1.a1_amp_fixed_env_cfg import A1AmpFixedEnvCfg


@configclass
class A1AmpRangeEnvCfg(A1AmpFixedEnvCfg):
    """Range-velocity, multi-clip AMP for A1.

    Reuses the tuned rewards / terminations / actuator setup from
    :class:`A1AmpFixedEnvCfg`, but replaces the single fixed forward speed with a
    full velocity command range (forward/backward, lateral, turning) and a curated
    multi-clip demonstration set from the ``a1_cal`` dataset. No curriculum — the
    full command range is commanded from step 0.
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------
        # motion data — curated multi-clip set from a1_cal
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "a1_cal"
        )
        # Disc obs are body-local (imitate GAIT, not world speed). All weights = 1.
        # CONTINUOUSLY-WALKING clips only. No lateral/side-step for now (dropped —
        # robot wasn't learning it), so no diagonal clips either.
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
        # Commands — walk only, forward 0.5-0.7 (no run, no backward)
        # ------------------------------------------------------
        # Walk data tops out ~0.5 straight / ~0.66 in turns, so forward is capped at
        # 0.7. Standing envs still give the 0 command. Lateral +-0.3 (B22/B23),
        # turn +-0.8 (walk-turn ~0.76).
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        # Direct (vx, vy, wz) command instead of heading-tracking: "give vx, wz=0"
        # walks straight. ang_vel_z range is now actually sampled (matches the
        # steady-yaw walk-turn clips). heading range is unused when this is False.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        # Cold-start via standing envs: 15% of envs get a 0-command; when it resamples
        # to a forward speed mid-episode the robot must start from ~rest, teaching it to
        # walk from a standstill (the play reset is a static stand). reset_from_ref stays
        # as-is (all envs start mid-motion) — trying the rel_standing_envs route first.
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.events.reset_from_ref.func = mdp.reset_from_ref_or_default
        self.events.reset_from_ref.params = {"animation": "animation", "height_offset": 0.1, "ref_ratio": 0.8}

        # self.rewards.track_lin_vel_xy_exp["std"] = math.sqrt(0.49)
        self.rewards.stand_still.weight = -2.0
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.feet_roll = None
        self.rewards.dof_torques_l2.weight = 2.0e-6
        self.rewards.joint_deviation_ankle.weight = -0.2
        self.rewards.joint_deviation_hip = None
        self.rewards.feet_flat_orientation.weight = -0.5

        # extra hip-abduction (joint 2, R2/L2) deviation penalty — keep legs from splaying
        self.rewards.joint_deviation_hip2 = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R2", "joint_L2"])},
        )
        self.rewards.feet_slide =  RewTerm(
            func=mdp.feet_slide,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["Link_R6", "Link_L6"]),
            },
        )
        # Curriculum stays disabled (inherited None from A1AmpFixedEnvCfg):
        # full range is commanded from the start.

        # ------------------------------------------------------
        # Terminations — re-enable "fell down" resets
        # ------------------------------------------------------
        # A1AmpEnvCfg disables base_contact (=None) and the inherited base_height
        # threshold (0.2) sits far below the ~0.45 walking height, so a collapsed
        # robot never terminated. Rebuild base_contact and tighten the gates:
        # walking base height is ~0.45, so 0.35 catches a collapse with margin.
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

        # keep the same command range at play time (walk only, forward 0.5-0.7)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
