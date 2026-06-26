import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.velocity.mdp as mdp
from legged_lab.assets.a1 import A1_LEGS_V1_CFG
from legged_lab.tasks.locomotion.velocity.velocity_env_cfg import EventCfg, LocomotionVelocityEnvCfg

# A1-legs_V1 bipedal naming (12 DOF, 2 legs x 6 joints):
#   joint_*1 hip pitch | joint_*2 hip roll | joint_*3 hip yaw
#   joint_*4 knee | joint_*5 ankle pitch | joint_*6 ankle roll
#   feet bodies: Link_R6 / Link_L6 ; trunk body: base
FEET_BODY_NAMES = ["Link_R6", "Link_L6"]

# Observation latency range, in control steps (1 step = decimation * sim.dt = 0.02 s here).
OBS_DELAY_STEPS = {"min_delay_steps": 0, "max_delay_steps": 0}

# Gait clock: the robot is rewarded for stepping at this frequency at ALL times, even when the
# velocity command is zero (marching in place). GAIT_PERIOD is the full stride period in seconds;
# GAIT_OFFSETS places the two feet in anti-phase. Shared by the phase observation and feet_gait reward.
GAIT_PERIOD = 0.5
GAIT_OFFSETS = [0.0, 0.5]


def _delayed(func, **term_kwargs):
    """ObsTerm wrapper that returns ``func`` delayed by a random per-env lag (sim-to-real latency)."""
    return ObsTerm(func=mdp.delayed_obs, params={"func": func, **OBS_DELAY_STEPS}, **term_kwargs)


@configclass
class A1ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (no base_lin_vel — not measurable on hardware).

        Sensor-derived terms are wrapped with a random per-env latency (``delayed_obs``) to
        emulate IMU/encoder/communication delay. Commands and last action are not delayed.
        Note: each wrapped term draws its lag independently.
        """

        # observation terms (order preserved)
        base_ang_vel = _delayed(mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = _delayed(mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = _delayed(mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = _delayed(mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # gait clock (sin/cos): an exact internal signal, not delayed/corrupted
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class A1RewardsCfg:
    """Pure-RL reward terms for A1 bipedal velocity tracking (no imitation/style term)."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # -- base
    # base orientation is already nearly level in practice (projected_gravity x ~0.045 => ~2.6deg
    # pitch), so this is NOT the lever for the visible "lean" -- that lives in the hip-pitch posture
    # (see joint_deviation_hip_pitch below). Kept at a moderate value to hold the torso level.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # -- joints
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R[1-4]", "joint_L[1-4]"])},
    )
    # Soft thermal limit: penalize ONLY the torque ABOVE each motor's rated (continuous) value, so
    # brief peaks (impacts/push-off, up to the hard effort_limit) stay free but sustained high torque
    # is punished. Rated: DM-J4340 hip/knee = 9 N*m, DM-J4310 ankle = 3 N*m. excess^2 can be large
    # (hip can reach (27-9)^2=324), so the weights are small -- tune by watching the torque histogram:
    # raise if the policy parks above rated, lower if the gait gets too weak/slow.
    joint_torque_over_rated_hip_knee = RewTerm(
        func=mdp.joint_torque_over_limit_l2,
        weight=-2.0e-4,
        params={"limit": 9.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R[1-4]", "joint_L[1-4]"])},
    )
    joint_torque_over_rated_ankle = RewTerm(
        func=mdp.joint_torque_over_limit_l2,
        weight=-2.0e-4,
        params={"limit": 3.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R[5-6]", "joint_L[5-6]"])},
    )
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R[5-6]", "joint_L[5-6]"])},
    )

    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=3.0,
        params={
            # None => reward stepping even at zero command, consistent with the always-on gait
            "command_name": None,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_swing,
        weight=1.5,
        params={
            "std": 0.25,
            "target_height": 0.15,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
        },
    )
    # keep the soles parallel to the ground at all times (flat feet). Penalizes the horizontal
    # components of gravity projected into each foot frame -> 0 when the sole is level.
    feet_flat_orientation = RewTerm(
        func=mdp.feet_flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES)},
    )
    # penalize uneven swing height between the two legs (one foot lifts high, the other drags).
    # CAUTION: this penalty is minimized by NOT lifting either foot (peak_L = peak_R = 0). Keep the
    # weight small enough that even clumsy/asymmetric early lifting stays net-positive against the
    # feet_clearance (+1.5) / feet_air_time (+1.0) rewards, otherwise the policy collapses to "don't
    # lift" to avoid the penalty. Only raise it once a stepping gait has emerged.
    feet_swing_height_symmetry = RewTerm(
        func=mdp.feet_swing_height_symmetry,
        weight=0.0,  # DISABLED until a real stepping gait emerges; then bump to ~-2 to refine symmetry
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )

    # Left/right step-timing symmetry: penalize the variance (i.e. the difference, for 2 feet) of the
    # two feet's last air-time AND last contact-time, so both legs swing and stand for equal durations.
    # This is a temporal symmetry (correct for an anti-phase gait) -- it does NOT fight the alternation
    # the way an instantaneous |q_L - q_R| posture penalty would. Same caveat as the height symmetry:
    # its minimum is "both feet motionless", so keep the weight gentle. Raise once the gait is stable.
    feet_air_time_symmetry = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=0.0,  # DISABLED until a real stepping gait emerges; then bump to ~-0.5 to refine symmetry
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES)},
    )

    # -- posture
    # joint_*1 is the hip PITCH (Y-axis), the main leg-swing joint. With the torso staying level, the
    # visible forward "lean" is actually this joint raking the whole leg forward from its default
    # (it was freed earlier to allow striding). A SMALL deviation penalty pulls the mean hip pitch
    # back toward the upright default without killing the walking swing. Raise toward -0.5 if the
    # leg still rakes forward; lower if stride/forward speed suffers.
    joint_deviation_hip_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R1", "joint_L1"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R2", "joint_L2"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R5", "joint_L5"])},
    )
    # joint_*3 is the hip YAW (Z-axis) joint -- the primary DOF for redirecting/turning the leg.
    # Command-gated: a LARGE weight keeps the yaw joints centered while going straight (prevents the
    # duck-footed / splayed "外八" stance), but the penalty switches OFF when a yaw rate is commanded
    # so it never fights turning. After the turn the yaw command returns to ~0 and the feet re-center.
    joint_deviation_yaw = RewTerm(
        func=mdp.joint_deviation_l1_straight,
        weight=-0.8,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R3", "joint_L3"]),
            "ang_vel_threshold": 1.0,
        },
    )
    # joint_*6 is the ankle ROLL (X-axis) joint -- keep it near default so the foot does not invert/
    # evert (roll side-to-side). Command-gated like the yaw term: penalty off while turning so it
    # leaves room to adjust the foot during a turn, on (and re-centering) while going straight.
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1_straight,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R6", "joint_L6"]),
            "ang_vel_threshold": 1.0,
        },
    )

    # NOTE: stand_still is intentionally disabled. It penalizes joint deviation when the command is
    # ~0, which directly conflicts with the requirement to keep stepping (marching in place) at cmd 0.
    # Re-enable only if you switch to "stand still when idle" behavior and drop the always-on feet_gait.
    # stand_still = RewTerm(
    #     func=mdp.stand_still_joint_deviation_l1,
    #     weight=-0.5,
    #     params={"command_name": "base_velocity"},
    # )

    # -- gait: track the phase clock at ALL times (command_name=None => active even when cmd == 0,
    # so the robot keeps marching in place). Rewards each foot for being in stance during the first
    # half of its phase and in swing during the second half. Keep `period` == GAIT_PERIOD used by
    # the gait_phase observation so the policy's clock matches the rewarded contact pattern.
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=1.2,
        params={
            "period": GAIT_PERIOD,
            "offset": GAIT_OFFSETS,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True),
            "threshold": 0.5,
            "command_name": None,
        },
    )


@configclass
class A1EventCfg(EventCfg):
    """Domain-randomization events for A1: inherits base (mass/COM/friction/push) and adds
    PD-gain randomization and a joint zero-point (encoder calibration) offset."""

    # randomize PD gains (kp/kd) by +-20% once at startup (implicit actuators -> startup only)
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # randomize motor rotor inertia (reflected armature) by +-20% once at startup.
    # CPU-side write -> startup only (same restriction as the gain randomization above).
    randomize_rotor_inertia = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Bridge the UNMEASURED dry (Coulomb) friction with domain randomization instead of a bench test.
    # The implicit actuators have no Fs/Fd; PhysX joint `friction` acts as a static friction torque
    # (N*m), so we sample an absolute value per joint at startup. operation="abs" (NOT scale -- the
    # base friction is 0, so scaling would stay 0). Ranges are conservative guesses bounded by motor
    # size; tighten the center/width later once a bench sweep gives the real Fs.
    #   hip/knee = DM-J4340 (40:1, peak 27 N*m) -> larger gearbox friction
    #   ankle    = DM-J4310 (10:1, peak  7 N*m) -> smaller
    # randomize_joint_friction_hip_knee = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R[1-4]", "joint_L[1-4]"]),
    #         "friction_distribution_params": (0.2, 1.5),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )
    # randomize_joint_friction_ankle = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_R[5-6]", "joint_L[5-6]"]),
    #         "friction_distribution_params": (0.1, 0.6),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # Domain-randomize the custom Damiao/Unitree actuator's INTERNAL friction torque (Fs/Fd),
    # applied inside UnitreeActuator.compute() as `effort -= Fs*tanh(v/Va) + Fd*v`.
    # NOTE: this is the motor-shaft friction of the DamiaoActuator (A1_LEGS_V1_CFG hip_knee/ankle),
    # which is a DIFFERENT channel from the PhysX joint `friction` randomized just above
    # (randomize_joint_friction_*) and from the foot/ground contact friction (physics_material).
    # The DamiaoActuator does NOT read the PhysX joint friction, so its Fs/Fd were 0 (modeling gap);
    # this term fills that gap. One value per env is drawn at reset and held for the whole episode.
    # Ranges from PDF analysis (Fs ~= 2-3% of peak torque, Fd ~= Fs/10):
    #   hip_knee (DM4340, peak 27 N.m): Fs in [0.3, 0.9],  Fd in [0.03, 0.09]
    #   ankle    (DM4310, peak  7 N.m): Fs in [0.1, 0.25], Fd in [0.01, 0.025]
    # randomize_actuator_friction_hip_knee = EventTerm(
    #     func=mdp.randomize_actuator_friction,
    #     mode="reset",
    #     params={
    #         "actuator_names": ["hip_knee"],
    #         "static_friction_range": (0.3, 0.9),
    #         "dynamic_friction_range": (0.03, 0.09),
    #     },
    # )
    # randomize_actuator_friction_ankle = EventTerm(
    #     func=mdp.randomize_actuator_friction,
    #     mode="reset",
    #     params={
    #         "actuator_names": ["ankle"],
    #         "static_friction_range": (0.1, 0.25),
    #         "dynamic_friction_range": (0.01, 0.025),
    #     },
    # )

    # slight per-robot joint zero-point offset (encoder/motor miscalibration)
    joint_zero_offset = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "offset_range": (-0.05, 0.05),
        },
    )


@configclass
class A1FlatEnvCfg(LocomotionVelocityEnvCfg):
    observations: A1ObservationsCfg = A1ObservationsCfg()
    rewards: A1RewardsCfg = A1RewardsCfg()
    events: A1EventCfg = A1EventCfg()

    def __post_init__(self):
        super().__post_init__()

        # -----------------------------------------------------------------------------
        # Scene
        # -----------------------------------------------------------------------------
        self.scene.robot = A1_LEGS_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # -----------------------------------------------------------------------------
        # Action
        # -----------------------------------------------------------------------------
        self.actions.joint_pos.scale = 0.25

        # -----------------------------------------------------------------------------
        # Commands
        # -----------------------------------------------------------------------------
        # Start NARROW and forward-biased; the velocity curriculum below expands the range
        # symmetrically (+-0.1/episode) toward the full targets once tracking is good. This biases
        # early discovery toward walking forward and avoids collapsing into the backward local optimum.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        # self.commands.base_velocity.ranges.lin_vel_x = (-0., 0.)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0., 0.)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0., 0.)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # -----------------------------------------------------------------------------
        # Events
        # -----------------------------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        # Base mass DR as a multiplicative factor (WBT-style minimal mass DR): base mass *= U[0.8,1.2].
        # base nominal ~0.62kg -> sampled ~0.50..0.74kg (+-0.12kg). NOTE: this scales ONLY the light
        # base link, so the absolute change is tiny vs the 8.45kg whole-robot mass and does NOT emulate
        # whole-body payload variation (for that you'd add ballast or scale a heavier body). Inertia is
        # recomputed by the same ratio (recompute_inertia default True).
        self.events.add_base_mass.params["operation"] = "scale"
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.base_com.params["asset_cfg"].body_names = "base"
        # widen fore-aft (x) CoM randomization so the policy can't rely on a fixed forward lean to
        # balance -> forces an upright posture that is robust to the real robot's true CoM (sim2real).
        self.events.base_com.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        # Ground/contact friction + restitution DR aligned to WBT (BeyondMimic Table S2). The base
        # EventCfg default was static/dynamic U[0.1,0.5], restitution 0 (fixed) -- too low to cover
        # the sim2sim/sim2real deployment friction (~0.8) and gave no contact-restitution robustness.
        # Overridden here (A1 flat only; base velocity_env_cfg untouched so rough/other tasks unchanged).
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.6)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.5)

        # -----------------------------------------------------------------------------
        # Terminations
        # -----------------------------------------------------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        self.terminations.root_height_below_minimum.params["minimum_height"] = 0.3

        # -----------------------------------------------------------------------------
        # Curriculums
        # -----------------------------------------------------------------------------
        self.curriculum.terrain_levels = None
        # Velocity command curriculum: grow the command ranges from the narrow forward-biased start
        # above toward the full targets, one +-0.1 step per episode, gated on the tracking reward
        # exceeding reward_threshold_ratio * weight. lin expands x AND y together; ang expands yaw.
        # self.curriculum.lin_vel_cmd_levels = CurrTerm(
        #     func=mdp.lin_vel_cmd_levels,
        #     params={
        #         "reward_term_name": "track_lin_vel_xy_exp",
        #         "lin_vel_x_limit": [-0.7, 0.7],
        #         "lin_vel_y_limit": [-0.5, 0.5],
        #         "reward_threshold_ratio": 0.7,
        #     },
        # )
        # self.curriculum.ang_vel_cmd_levels = CurrTerm(
        #     func=mdp.ang_vel_cmd_levels,
        #     params={
        #         "reward_term_name": "track_ang_vel_z_exp",
        #         "ang_vel_z_limit": [-0.6, 0.6],
        #         "reward_threshold_ratio": 0.6,
        #     },
        # )


class A1FlatEnvCfg_PLAY(A1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # disable ALL domain randomization for play (keep only reset_base / reset_robot_joints,
        # which are needed to spawn and reset the robot each episode).
        self.events.physics_material = None          # friction / restitution randomization
        self.events.add_base_mass = None             # random payload mass
        self.events.base_com = None                  # random COM offset
        self.events.base_external_force_torque = None
        self.events.push_robot = None                # random pushes
        self.events.randomize_actuator_gains = None  # PD kp/kd randomization
        self.events.randomize_rotor_inertia = None   # armature randomization
        self.events.joint_zero_offset = None         # encoder zero-point offset

        # fixed command range for evaluation (no curriculum growth at play time)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0., 0.)
        self.commands.base_velocity.ranges.lin_vel_y = (-0., 0.)
        self.commands.base_velocity.ranges.ang_vel_z = (-0., 0.)
