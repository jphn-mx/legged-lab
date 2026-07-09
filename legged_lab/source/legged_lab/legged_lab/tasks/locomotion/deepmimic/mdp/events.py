from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def reset_from_ref(
    env: ManagerBasedAnimationEnv,
    env_ids: torch.Tensor,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.1,
):
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    offset = torch.tensor([0.0, 0.0, height_offset], device=env.device, dtype=torch.float32).unsqueeze(0)  # (1, 3)
    position = animation_term.get_root_pos_w(env_ids)[:, 0, :] + env.scene.env_origins[env_ids, :] + offset
    orientation = animation_term.get_root_quat(env_ids)[:, 0, :]
    lin_vel = animation_term.get_root_vel_w(env_ids)[:, 0, :]
    ang_vel = animation_term.get_root_ang_vel_w(env_ids)[:, 0, :]

    pos = torch.cat([position, orientation], dim=-1)
    vel = torch.cat([lin_vel, ang_vel], dim=-1)

    robot.write_root_pose_to_sim(pos, env_ids=env_ids)
    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)

    dof_pos = animation_term.get_dof_pos(env_ids)[:, 0, :]
    dof_vel = animation_term.get_dof_vel(env_ids)[:, 0, :]
    robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

def reset_from_ref_or_default(
    env: ManagerBasedAnimationEnv,
    env_ids: torch.Tensor,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.1,
    ref_ratio: float = 0.8,
):
    """Reset a ``ref_ratio`` fraction of envs from a reference motion frame (mid-motion,
    already moving) and the rest to the robot's default standing pose (zero velocity).

    The default-pose resets force the policy to learn cold-start (walk from a dead stop),
    which pure ``reset_from_ref`` never exercises because it always drops the robot into a
    moving reference frame. This matches the play reset (static stand) so the trained policy
    can actually start walking from standstill.
    """
    if len(env_ids) == 0:
        return

    robot: Articulation = env.scene[asset_cfg.name]

    rand = torch.rand(len(env_ids), device=env.device)
    ref_ids = env_ids[rand < ref_ratio]
    default_ids = env_ids[rand >= ref_ratio]

    # moving reference-frame resets
    if len(ref_ids) > 0:
        reset_from_ref(env, ref_ids, animation, asset_cfg, height_offset)

    # default standing-pose resets (cold-start practice)
    if len(default_ids) > 0:
        root_state = robot.data.default_root_state[default_ids].clone()
        root_state[:, :3] += env.scene.env_origins[default_ids]
        robot.write_root_pose_to_sim(root_state[:, :7], env_ids=default_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=default_ids)
        joint_pos = robot.data.default_joint_pos[default_ids].clone()
        joint_vel = robot.data.default_joint_vel[default_ids].clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=default_ids)

def randomize_actuator_friction(
    env: ManagerBasedAnimationEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    actuator_names: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Domain-randomize the *internal* friction torque (Fs/Fd) of the custom UnitreeActuator.

    Unlike :func:`isaaclab.envs.mdp.events.randomize_joint_parameters`, which writes the PhysX
    joint friction coefficient, the Damiao/Unitree actuator applies its own Coulomb + viscous
    friction inside ``UnitreeActuator.compute()``::

        applied_effort -= Fs * tanh(joint_vel / Va) + Fd * joint_vel

    via the per-env tensors ``_friction_static`` (Fs) and ``_friction_dynamic`` (Fd), each of
    shape (num_envs, num_joints). PhysX-level randomization never touches these, so we sample
    Fs/Fd here and overwrite those tensors directly. A single value per env is drawn (uniform),
    so friction stays fixed for the whole episode when this term runs on ``startup``/``reset``.

    Args:
        static_friction_range: (low, high) for Fs [N*m].
        dynamic_friction_range: (low, high) for Fd [N*m*s/rad].
        actuator_names: which actuator groups to randomize. ``None`` -> every actuator group
            that exposes ``_friction_static`` (i.e. all UnitreeActuator-derived groups).
    """
    robot: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=robot.device)

    if actuator_names is None:
        actuator_names = list(robot.actuators.keys())

    n = env_ids.shape[0]
    for name in actuator_names:
        actuator = robot.actuators.get(name, None)
        # only UnitreeActuator-derived groups carry the internal friction tensors
        if actuator is None or not hasattr(actuator, "_friction_static"):
            continue

        fs = actuator._friction_static
        fd = actuator._friction_dynamic
        num_joints = fs.shape[1]

        fs_new = torch.empty((n, num_joints), device=fs.device).uniform_(*static_friction_range)
        fd_new = torch.empty((n, num_joints), device=fd.device).uniform_(*dynamic_friction_range)

        fs[env_ids] = fs_new.to(fs.dtype)
        fd[env_ids] = fd_new.to(fd.dtype)
