"""Configuration for A1-legs_V1 bipedal robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab import LEGGED_LAB_ROOT_DIR


A1_LEGS_V1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/A1/A1-legs_V1.urdf",
        fix_base=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=2.0,
            ),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            "joint_R1": -0.22,
            "joint_R2": 0.0,
            "joint_R3": 0.0,
            "joint_R4": 0.70,
            "joint_R5": -0.48,
            "joint_R6": 0.0,
            "joint_L1": -0.22,
            "joint_L2": 0.0,
            "joint_L3": 0.0,
            "joint_L4": 0.70,
            "joint_L5": -0.48,
            "joint_L6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hip_knee": ImplicitActuatorCfg(
            joint_names_expr=["joint_R[1-4]", "joint_L[1-4]"],
            effort_limit_sim=27.0,
            velocity_limit_sim=36.0,
            stiffness=30.0,
            damping=0.3,
            armature=0.01,
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=["joint_R[5-6]", "joint_L[5-6]"],
            effort_limit_sim=7.0,
            velocity_limit_sim=120.0,
            stiffness=30.0,
            damping=0.3,
            armature=0.01,
        ),
    },
)
