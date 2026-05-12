import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

A1_LEGS_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/A1-legs_V1/A1-legs_V1.urdf",
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
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
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
    soft_joint_pos_limit_factor=0.9,
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

A1_ACTION_SCALE = {}
for a in A1_LEGS_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            A1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
