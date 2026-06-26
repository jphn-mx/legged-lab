"""Configuration for A1-legs_V1 bipedal robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg,DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets import unitree_actuators


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
DAMPING_RATIO = 1.2  # zeta = 2 (overdamped, compensates for inertia underestimation)

ARMATURE_4340 = ROTOR_INERTIA_4340 * GEAR_4340**2  # ~= 0.0320
ARMATURE_4310 = ROTOR_INERTIA_4310 * GEAR_4310**2  # ~= 0.0018

STIFFNESS_4340 = ARMATURE_4340 * NATURAL_FREQ**2  # ~= 126
STIFFNESS_4310 = ARMATURE_4310 * NATURAL_FREQ**2  # ~= 7.1
DAMPING_4340 = 2.0 * DAMPING_RATIO * ARMATURE_4340 * NATURAL_FREQ  # ~= 8.0
DAMPING_4310 = 2.0 * DAMPING_RATIO * ARMATURE_4310 * NATURAL_FREQ  # ~= 0.45


A1_LEGS_V1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # asset_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/A1/A1-legs_V1.urdf",
        asset_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/A1_V2/A1_legs_V2.urdf",
        # asset_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/A1_V2/A1_legs_V2_ball.urdf",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        # joint_pos={
        #     "joint_R1": -0.22,
        #     "joint_R2": 0.0,
        #     "joint_R3": 0.0,
        #     "joint_R4": 0.70,
        #     "joint_R5": -0.48,
        #     "joint_R6": 0.0,
        #     "joint_L1": -0.22,
        #     "joint_L2": 0.0,
        #     "joint_L3": 0.0,
        #     "joint_L4": 0.70,
        #     "joint_L5": -0.48,
        #     "joint_L6": 0.0,
        # },
        joint_pos={
            "joint_R1": -0.2,
            "joint_R2": 0.0,
            "joint_R3": 0.0,
            "joint_R4": 0.40,
            "joint_R5": -0.2,
            "joint_R6": 0.0,
            "joint_L1": -0.2,
            "joint_L2": 0.0,
            "joint_L3": 0.0,
            "joint_L4": 0.4,
            "joint_L5": -0.2,
            "joint_L6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # actuators={
    #     "hip_knee": ImplicitActuatorCfg(
    #         joint_names_expr=["joint_R[1-4]", "joint_L[1-4]"],
    #         effort_limit_sim=27.0,
    #         velocity_limit_sim=5.5,  # DM-J4340 no-load speed: 52.5 rpm = 5.50 rad/s (was 36, unrealistic)
    #         stiffness=STIFFNESS_4340,
    #         damping=DAMPING_4340,
    #         armature=ARMATURE_4340,
    #     ),
    #     "ankle": ImplicitActuatorCfg(
    #         joint_names_expr=["joint_R[5-6]", "joint_L[5-6]"],
    #         effort_limit_sim=7.0,
    #         velocity_limit_sim=20.9,  # DM-J4310 no-load speed: 200 rpm = 20.94 rad/s (was 120, unrealistic)
    #         stiffness=STIFFNESS_4310,
    #         damping=DAMPING_4310,
    #         armature=ARMATURE_4310,
    #     ),
    # },
    actuators={
        "hip_knee": DelayedPDActuatorCfg(
            joint_names_expr=["joint_R[1-4]", "joint_L[1-4]"],
            effort_limit_sim=27.0,
            velocity_limit_sim=5.5,  # DM-J4340 no-load speed: 52.5 rpm = 5.50 rad/s (was 36, unrealistic)
            stiffness=200.0,#STIFFNESS_4340,
            damping=5.0,#DAMPING_4340,
            armature=ARMATURE_4340,
            min_delay=1,
            max_delay=4,
        ),
        "ankle": DelayedPDActuatorCfg(
            joint_names_expr=["joint_R[5-6]", "joint_L[5-6]"],
            effort_limit_sim=7.0,
            velocity_limit_sim=20.9,  # DM-J4310 no-load speed: 200 rpm = 20.94 rad/s (was 120, unrealistic)
            stiffness=40.0,#STIFFNESS_4310,
            damping=0.5,#DAMPING_4310,
            armature=ARMATURE_4310,
            min_delay=1,
            max_delay=4,
        ),
    },
)


# Variant of A1_LEGS_V1_CFG that drives the joints with the Damiao torque-speed
# (T-N curve) + friction actuator model instead of the implicit fixed-limit PD.
# Same robot/spawn/init_state; only the actuators dict is swapped. The original
# A1_LEGS_V1_CFG above is left untouched.
A1_LEGS_V1_CFG_DAMIAO = A1_LEGS_V1_CFG.replace(
    actuators={
        "hip_knee": unitree_actuators.DamiaoActuatorCfg_DM4340(
            joint_names_expr=["joint_R[1-4]", "joint_L[1-4]"],
            stiffness=STIFFNESS_4340,
            damping=DAMPING_4340,
            # armature, Y1/X1/X2 come from DamiaoActuatorCfg_DM4340
        ),
        "ankle": unitree_actuators.DamiaoActuatorCfg_DM4310(
            joint_names_expr=["joint_R[5-6]", "joint_L[5-6]"],
            stiffness=STIFFNESS_4310,
            damping=DAMPING_4310,
            # armature, Y1/X1/X2 come from DamiaoActuatorCfg_DM4310
        ),
    },
)
