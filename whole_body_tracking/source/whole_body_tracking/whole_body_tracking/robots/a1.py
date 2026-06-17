import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# Heuristic PD gains following the WBT paper (2508.08241v4, Supp. S1):
#   k_p = I * w^2,  k_d = 2 * I * zeta * w,  with I = g^2 * J_rotor (reflected armature).
# Motors (Damiao DM-J series, rotor inertia from datasheet):
#   DM-J4340  hip/knee (joint 1-4, peak torque 27 N.m): g=40, J_rotor=2.0e-5
#   DM-J4310  ankle    (joint 5-6, peak torque  7 N.m): g=10, J_rotor=1.8e-5
ROTOR_INERTIA_4340 = 2.00e-5
ROTOR_INERTIA_4310 = 1.80e-5
GEAR_4340 = 40.0
GEAR_4310 = 10.0

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz -> w ~= 62.83 rad/s (low value promotes compliance)
DAMPING_RATIO = 2.0  # zeta = 2 (overdamped, compensates for inertia underestimation)

ARMATURE_4340 = ROTOR_INERTIA_4340 * GEAR_4340**2  # ~= 0.0320
ARMATURE_4310 = ROTOR_INERTIA_4310 * GEAR_4310**2  # ~= 0.0018

STIFFNESS_4340 = ARMATURE_4340 * NATURAL_FREQ**2  # ~= 126
STIFFNESS_4310 = ARMATURE_4310 * NATURAL_FREQ**2  # ~= 7.1
DAMPING_4340 = 2.0 * DAMPING_RATIO * ARMATURE_4340 * NATURAL_FREQ  # ~= 8.0
DAMPING_4310 = 2.0 * DAMPING_RATIO * ARMATURE_4310 * NATURAL_FREQ  # ~= 0.45

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
        pos=(0.0, 0.0, 0.5),
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
            stiffness=STIFFNESS_4340,
            damping=DAMPING_4340,
            armature=ARMATURE_4340,
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=["joint_R[5-6]", "joint_L[5-6]"],
            effort_limit_sim=7.0,
            velocity_limit_sim=120.0,
            stiffness=STIFFNESS_4310,
            damping=DAMPING_4310,
            armature=ARMATURE_4310,
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
