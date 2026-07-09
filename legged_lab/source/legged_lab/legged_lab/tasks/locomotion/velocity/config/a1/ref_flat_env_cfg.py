# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""A1 flat velocity env with an additional sinusoidal reference joint-trajectory reward.

Registered as:
  LeggedLab-Isaac-RefVel-A1-v0       (training)
  LeggedLab-Isaac-RefVel-A1-Play-v0  (evaluation)
"""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.velocity.mdp as mdp
from legged_lab.tasks.locomotion.velocity.config.a1.flat_env_cfg import (
    A1FlatEnvCfg,
    A1RewardsCfg,
    FEET_BODY_NAMES,
    GAIT_PERIOD,
)
GAIT_PERIOD = 0.7
# ---------------------------------------------------------------------------
# Sinusoidal reference trajectory parameters (A1 bipedal, 12 DOF)
#
# Joint naming:  joint_{L,R}1 hip-pitch  | joint_{L,R}2 hip-roll  | joint_{L,R}3 hip-yaw
#                joint_{L,R}4 knee       | joint_{L,R}5 ankle-pitch | joint_{L,R}6 ankle-roll
#
# The gait is anti-phase: left leg phase offset = 0.0, right leg = 0.5.
# Amplitudes and phase offsets are intentionally conservative so this term
# acts as a soft motion prior rather than a hard constraint.
# ---------------------------------------------------------------------------
_REF = [
    # hip pitch (*1, default=-0.2, limits=±1.05)
    # phase_offset=pi: global_phase<0.5=stance(hip extended), global_phase>=0.5=swing(hip flexed)
    # amplitude=0.35 -> range [-0.55, +0.15] rad  (~20° forward, ~20° behind default)
    {"joint_names": ["joint_L1"], "leg_phase_offset": 0.0, "amplitude": 0.35, "phase_offset": math.pi},
    {"joint_names": ["joint_R1"], "leg_phase_offset": 0.5, "amplitude": 0.35, "phase_offset": math.pi},
    # knee (*4, default=0.4, limits=[0, 1.92])
    # amplitude must be < 0.40 (trough = default - amp > 0)
    # amplitude=0.35 -> range [0.05, 0.75] rad  (swing peak ~43°, stance near straight)
    {"joint_names": ["joint_L4"], "leg_phase_offset": 0.0, "amplitude": 0.35, "phase_offset": math.pi + math.pi / 4},
    {"joint_names": ["joint_R4"], "leg_phase_offset": 0.5, "amplitude": 0.35, "phase_offset": math.pi + math.pi / 4},
    # ankle pitch (*5, default=-0.2, limits=±0.52)
    # amplitude=0.20 -> range [-0.40, 0.0] rad  (push-off plantarflexion ~23°)
    # {"joint_names": ["joint_L5"], "leg_phase_offset": 0.0, "amplitude": 0.20, "phase_offset": math.pi - math.pi / 3},
    # {"joint_names": ["joint_R5"], "leg_phase_offset": 0.5, "amplitude": 0.20, "phase_offset": math.pi - math.pi / 3},
]

@configclass
class A1RefFlatRewardsCfg(A1RewardsCfg):
    """Extends A1RewardsCfg with a sinusoidal reference joint-trajectory term.

    Disables joint_deviation terms for joints covered by the reference (hip pitch *1, ankle pitch *5)
    since they directly oppose the reference and suppress tracking reward below 0.5.
    """

    joint_pos_reference = RewTerm(
        func=mdp.joint_pos_reference_tracking,
        weight=0.5,
        params={
            "period": GAIT_PERIOD,
            "reference": _REF,
            "std": 0.5,  # wider kernel: reference is a soft prior, not a hard constraint
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )

    # Disable deviation penalties for joints that the reference already guides.
    # joint_deviation_hip_pitch pulls joint_*1 back to default, fighting ±0.35 rad reference swing.
    # joint_deviation_ankle    pulls joint_*5 back to default, fighting ±0.20 rad reference swing.
    # Removing them lets the reference reward become the sole "desired pose" signal for these joints.
    # joint_deviation_hip_pitch: RewTerm | None = None
    # joint_deviation_ankle: RewTerm | None = None

    # Higher foot clearance target: push the swing foot to reach 0.20 m (vs 0.15 m in flat task).
    # Knee amplitude is already at its safe upper limit (0.35 rad, trough barely above joint limit 0),
    # so the only remaining lever for more lift is the clearance reward target height.
    # feet_clearance = RewTerm(
    #     func=mdp.feet_clearance_swing,
    #     weight=2.5,
    #     params={
    #         "std": 0.25,
    #         "target_height": 0.20,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
    #     },
    # )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-2.0,
        params={"command_name": "base_velocity"},
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R[1-6]", "joint_L[1-6]"])},
    )

    


@configclass
class A1RefFlatEnvCfg(A1FlatEnvCfg):
    """A1 velocity env augmented with a sinusoidal reference joint-trajectory reward.

    All scene, observation, action, termination, event and curriculum settings
    are inherited unchanged from A1FlatEnvCfg.  Only the rewards field is
    replaced with A1RefFlatRewardsCfg (which is a strict superset of A1RewardsCfg).
    """

    rewards: A1RefFlatRewardsCfg = A1RefFlatRewardsCfg()


    def __post_init__(self):
        super().__post_init__()
        # rewards field is already an A1RefFlatRewardsCfg instance declared above;
        # super().__post_init__() does not reassign rewards, so no further action needed.
        # self.rewards.joint_deviation_ankle.weight = -0.4
        # self.rewards.joint_deviation_hip.weight = -0.4

        # Override gait_phase obs to use this task's GAIT_PERIOD (0.7s) and idle-freeze.
        # The inherited obs uses flat_env_cfg's GAIT_PERIOD (0.5s) without command gating.
        _gait_phase_term = ObsTerm(
            func=mdp.gait_phase,
            params={"period": GAIT_PERIOD, "command_name": "base_velocity", "command_threshold": 0.1},
        )
        self.rewards.feet_slide.weight = -1.0
        self.rewards.joint_deviation_hip.weight = -0.4
        self.rewards.joint_deviation_ankle.weight = -0.4
        self.rewards.joint_deviation_yaw.weight = -0.8
        self.rewards.feet_flat_orientation.weight = -2.0
        self.rewards.feet_air_time.weight = 4.0
        self.rewards.feet_clearance.weight = 2.5
        self.observations.policy.gait_phase = _gait_phase_term
        self.observations.critic.gait_phase = _gait_phase_term

        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)


class A1RefFlatEnvCfg_PLAY(A1RefFlatEnvCfg):
    """Play / evaluation version: smaller scene, no domain randomisation."""

    def __post_init__(self):
        super().__post_init__()

        # smaller scene for visualisation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable sensor noise
        self.observations.policy.enable_corruption = False

        # disable all domain randomisation
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_rotor_inertia = None
        self.events.joint_zero_offset = None

        # disable curriculum growth at play time
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None

        # zero command range for static evaluation (override as needed)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
