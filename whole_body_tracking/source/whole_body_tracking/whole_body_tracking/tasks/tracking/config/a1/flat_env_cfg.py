from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

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

        # self.commands.motion.adaptive_uniform_ratio = 0.3
        # self.terminations.anchor_pos.params["threshold"] = 0.8
@configclass
class A1FlatWoStateEstimationEnvCfg(A1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
