from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.a1 import A1_ACTION_SCALE, A1_LEGS_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class A1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = A1_LEGS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = A1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base"
        self.commands.motion.body_names = [
            "base",
            "Link_R2",
            "Link_R4",
            "Link_R6",
            "Link_L2",
            "Link_L4",
            "Link_L6",
        ]

        # Override undesired_contacts: penalize all contacts except feet
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[r"^(?!Link_R6$)(?!Link_L6$).+$"],
        )

        # Override ee_body_pos termination: A1 end effectors are Link_R6 and Link_L6
        self.terminations.ee_body_pos.params["body_names"] = ["Link_R6", "Link_L6"]

        # Override base_com event: use "base" instead of "torso_link"
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base")

        # Foot synchronization rewards for jump
        # Only for double jump
        self.rewards.feet_height_sync = RewTerm(
            func=mdp.feet_height_diff_l2,
            weight=-20.0,
            params={"command_name": "motion", "body_names": ["Link_R6", "Link_L6"]},
        )
        self.rewards.feet_contact_sync = RewTerm(
            func=mdp.feet_contact_sync,
            weight=-5.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_R6", "Link_L6"]),
                "threshold": 10.0,
            },
        )
        self.rewards.feet_vz_sync = RewTerm(
            func=mdp.feet_vertical_velocity_diff,
            weight=-2.0,
            params={"command_name": "motion", "body_names": ["Link_R6", "Link_L6"]},
        )

        # Self-collision penalty
        self.rewards.self_collision = RewTerm(
            func=mdp.self_collision_penalty,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("self_contact_forces"),
                "threshold": 1.0,
            },
        )

@configclass
class A1FlatWoStateEstimationEnvCfg(A1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class A1FlatDanceEnvCfg(A1FlatEnvCfg):
    """A1 config for dance motions from video estimation (noisier, asymmetric)."""

    def __post_init__(self):
        super().__post_init__()

        # Remove feet sync rewards (dance has asymmetric foot movements)
        self.rewards.feet_height_sync = None
        self.rewards.feet_contact_sync = None
        self.rewards.feet_vz_sync = None

        # Relax tracking std for noisy video-estimated data
        self.rewards.motion_global_anchor_pos.params["std"] = 0.5
        self.rewards.motion_body_pos.params["std"] = 0.6
        self.rewards.motion_body_ori.params["std"] = 0.7
        self.rewards.motion_body_lin_vel.params["std"] = 2.5
        self.rewards.motion_body_ang_vel.params["std"] = 5.0

        # Softer action penalty to allow more expressive motion
        self.rewards.action_rate_l2.weight = -5e-3


@configclass
class A1FlatDanceWoStateEstimationEnvCfg(A1FlatDanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class A1FlatGetUpEnvCfg(A1FlatEnvCfg):
    """A1 config for getting up from ground (fall recovery)."""

    def __post_init__(self):
        super().__post_init__()

        # Remove feet sync rewards (irrelevant for getup)
        self.rewards.feet_height_sync = None
        self.rewards.feet_contact_sync = None
        self.rewards.feet_vz_sync = None

        # Remove undesired_contacts penalty (body touches ground during getup)
        self.rewards.undesired_contacts = None

        # Relax tracking std (getup is a large-range motion)
        self.rewards.motion_global_anchor_pos.params["std"] = 0.4
        self.rewards.motion_global_anchor_ori.params["std"] = 0.6
        self.rewards.motion_body_pos.params["std"] = 0.5
        self.rewards.motion_body_ori.params["std"] = 0.6
        self.rewards.motion_body_lin_vel.params["std"] = 2.0
        self.rewards.motion_body_ang_vel.params["std"] = 4.0

        # Increase position tracking weight (critical for getup progress)
        self.rewards.motion_global_anchor_pos.weight = 3.0
        self.rewards.motion_body_pos.weight = 2.0

        # Relax termination thresholds (large pose changes during getup)
        self.terminations.anchor_pos.params["threshold"] = 0.6
        self.terminations.anchor_ori.params["threshold"] = 1.2
        self.terminations.ee_body_pos.params["threshold"] = 0.6

        # Longer episode for slow getup motion
        self.episode_length_s = 15.0

        # Disable push perturbation (don't push robot while getting up)
        self.events.push_robot = None

        # Penalize self-collision (legs hitting each other)
        self.rewards.self_collision = RewTerm(
            func=mdp.self_collision_penalty,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("self_contact_forces"),
                "threshold": 1.0,
            },
        )


@configclass
class A1FlatGetUpWoStateEstimationEnvCfg(A1FlatGetUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
