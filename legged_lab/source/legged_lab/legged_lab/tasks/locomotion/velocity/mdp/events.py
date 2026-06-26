"""Custom events (domain randomization) for the velocity locomotion environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset_range: tuple[float, float] = (-0.03, 0.03),
) -> None:
    """Add a small per-environment offset to the default joint positions ("zero-point" offset).

    This emulates the joint encoder / motor zero-calibration error of a real robot. The position
    action term caches its offset (``default_joint_pos``) once at construction, *before* startup
    events run, so perturbing ``default_joint_pos`` here does **not** move the commanded neutral
    pose. What changes is the reference used by ``joint_pos_rel`` observations: the policy therefore
    observes a persistent, per-robot bias between its action zero and the reported joint zero --
    exactly the miscalibration seen on hardware.

    Intended to be used as a ``startup`` event so each robot keeps a fixed offset for its lifetime.

    Args:
        offset_range: Uniform sampling range (radians) for the additive offset per joint.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    default_joint_pos = asset.data.default_joint_pos
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        num_joints = default_joint_pos.shape[1]
        offsets = torch.empty((len(env_ids), num_joints), device=asset.device).uniform_(*offset_range)
        default_joint_pos[env_ids] = default_joint_pos[env_ids] + offsets
    else:
        joint_ids_t = torch.as_tensor(joint_ids, dtype=torch.long, device=asset.device)
        offsets = torch.empty((len(env_ids), len(joint_ids_t)), device=asset.device).uniform_(*offset_range)
        default_joint_pos[env_ids[:, None], joint_ids_t] = (
            default_joint_pos[env_ids[:, None], joint_ids_t] + offsets
        )


def randomize_actuator_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    actuator_names: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Domain-randomize the *internal* friction torque (Fs/Fd) of the custom UnitreeActuator.

    Unlike :func:`isaaclab.envs.mdp.events.randomize_joint_parameters`, which writes the PhysX
    joint friction coefficient, the Damiao/Unitree actuator applies its own Coulomb + viscous
    friction inside ``UnitreeActuator.compute()``::

        applied_effort -= Fs * tanh(joint_vel / Va) + Fd * joint_vel

    via the per-env tensors ``_friction_static`` (Fs) and ``_friction_dynamic`` (Fd), each of
    shape (num_envs, num_joints). PhysX-level randomization never touches these, so we sample
    Fs/Fd here and overwrite those tensors directly. A single value per env/joint is drawn
    (uniform), so friction stays fixed for the whole episode when this term runs on
    ``startup``/``reset``.

    Args:
        static_friction_range: (low, high) for Fs [N*m].
        dynamic_friction_range: (low, high) for Fd [N*m*s/rad].
        actuator_names: which actuator groups to randomize. ``None`` -> every actuator group
            that exposes ``_friction_static`` (i.e. all UnitreeActuator-derived groups).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if actuator_names is None:
        actuator_names = list(asset.actuators.keys())

    n = env_ids.shape[0]
    for name in actuator_names:
        actuator = asset.actuators.get(name, None)
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
