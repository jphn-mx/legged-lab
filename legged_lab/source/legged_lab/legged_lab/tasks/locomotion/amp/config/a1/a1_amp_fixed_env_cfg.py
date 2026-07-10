import math
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.locomotion.amp.config.a1.a1_amp_env_cfg import A1AmpEnvCfg

# Command locked to a single steady forward speed. 0.5 m/s matches the native
# steady-walk speed of B3_-_walk1 (~0.47-0.49 m/s), so task and style rewards
# pull in the same direction instead of fighting each other.
FIXED_LIN_VEL_X = 0.5


@configclass
class A1AmpFixedEnvCfg(A1AmpEnvCfg):
    """A1 AMP for a single fixed forward speed, imitating one steady-walk clip.

    Instead of learning a whole velocity band, the policy learns to walk forward
    at a single commanded speed (FIXED_LIN_VEL_X). The demonstration dataset is
    reduced to the single steady-walk clip closest to that speed, and the
    command-range curriculum is disabled.
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------
        # motion data — one steady-walk clip only
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "b3_walk1"
        )
        # Single clip: the only clean steady straight-line walk (~0.47-0.49 m/s).
        self.motion_data.motion_dataset.motion_data_weights = {
            "B3_-_walk1": 1.0,
        }

        # ------------------------------------------------------
        # Commands — either stand still, or walk straight forward at FIXED_LIN_VEL_X
        # ------------------------------------------------------
        # Two commanded modes only: a fraction of envs get a zero command (stand
        # still), the rest get the single fixed forward speed. No turning, no
        # lateral, no intermediate speeds.
        self.rewards.stand_still.weight = -3.0
        self.rewards.feet_flat_orientation.weight = -2.0
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.feet_flat.weight = 0.0
        # switch feet_roll from L2 to L1 (constant gradient toward zero lateral tilt)
        # self.rewards.feet_roll.func = mdp.feet_roll_l1
        self.rewards.joint_deviation_ankle.weight = -0.4
        self.rewards.joint_deviation_yaw.weigt = -0.4
        self.rewards.joint_deviation_hip.weigt = -0.4
        self.rewards.feet_roll.weigt  = 0.0


        # only penalize the ankle-pitch (joint 5) hitting its limit, not joint 6
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_R5", ".*_L5"]
        )
        self.commands.base_velocity.ranges.lin_vel_x = (FIXED_LIN_VEL_X, FIXED_LIN_VEL_X)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        # ~15% of envs are commanded to stand still. Standing gets no style reward
        # (the demo set is walk-only), but the zero-velocity tracking reward is
        # near-max and stand_still_joint_deviation_l1 (weight -1.0, inherited)
        # drives the default standing pose — with task_style_lerp=0.6 the task
        # reward dominates for these envs, so standing is learned without a stand clip.
        self.commands.base_velocity.rel_standing_envs = 0.15

        # ------------------------------------------------------
        # Curriculum — none (fixed target, nothing to expand)
        # ------------------------------------------------------
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None


@configclass
class A1AmpFixedEnvCfg_PLAY(A1AmpFixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # keep the same fixed command at play time
        self.commands.base_velocity.ranges.lin_vel_x = (FIXED_LIN_VEL_X, FIXED_LIN_VEL_X)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
