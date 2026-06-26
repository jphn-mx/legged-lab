# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class feet_swing_height_symmetry(ManagerTermBase):
    """Penalize uneven swing-lift between the two feet (one leg lifting high, the other dragging).

    For each foot it tracks the peak clearance reached during a swing -- measured relative to the
    height at that foot's most recent ground contact, so the metric is terrain-agnostic and a foot
    that never leaves the ground reads ~0. The peak is *latched* at touchdown (swing -> contact), so
    the term compares the two feet's most recently completed strides. It returns the absolute
    difference of those peaks (metres), so a **negative** weight drives both legs to lift to the same
    height. Being a penalty (not a positive reward), it never rewards "not lifting at all" -- the
    feet_gait / feet_clearance / feet_air_time terms remain responsible for producing the lift.

    Expects exactly two foot bodies; ``sensor_cfg`` (contact sensor) and ``asset_cfg`` (robot bodies)
    must list them in the same order.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._sensor_ids = sensor_cfg.body_ids
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._asset_ids = asset_cfg.body_ids
        shape = (self.num_envs, 2)
        self._contact_z = torch.zeros(shape, device=self.device)
        self._running_peak = torch.zeros(shape, device=self.device)
        self._latched_peak = torch.zeros(shape, device=self.device)
        self._prev_contact = torch.ones(shape, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        foot_z = self._asset.data.body_pos_w[:, self._asset_ids, 2]
        self._contact_z[env_ids] = foot_z[env_ids]
        self._running_peak[env_ids] = 0.0
        self._latched_peak[env_ids] = 0.0
        self._prev_contact[env_ids] = True

    def __call__(self, env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        foot_z = self._asset.data.body_pos_w[:, self._asset_ids, 2]
        in_contact = self._contact_sensor.data.current_contact_time[:, self._sensor_ids] > 0.0
        # remember the ground height at each foot while it is in contact
        self._contact_z = torch.where(in_contact, foot_z, self._contact_z)
        clearance = (foot_z - self._contact_z).clamp(min=0.0)
        # accumulate the peak clearance over the current swing
        self._running_peak = torch.where(in_contact, self._running_peak, torch.maximum(self._running_peak, clearance))
        # latch the completed-swing peak at touchdown, then clear the running accumulator
        touchdown = in_contact & (~self._prev_contact)
        self._latched_peak = torch.where(touchdown, self._running_peak, self._latched_peak)
        self._running_peak = torch.where(in_contact, torch.zeros_like(self._running_peak), self._running_peak)
        self._prev_contact = in_contact
        return torch.abs(self._latched_peak[:, 0] - self._latched_peak[:, 1])


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def joint_torque_over_limit_l2(
    env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize (squared) ONLY the part of the applied joint torque that exceeds ``limit`` (N*m).

    Soft thermal limit: the hard ``effort_limit`` still allows brief peak torque (impacts, push-off),
    but this term pushes the *continuous* torque back toward the motor's rated value. Below ``limit``
    the penalty is exactly zero, so transient peaks are free while sustained high torque is punished.

    ``limit`` is the per-group rated (continuous) torque -- DM-J4340 hip/knee = 9 N*m,
    DM-J4310 ankle = 3 N*m.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    excess = (torque.abs() - limit).clamp(min=0.0)
    return torch.sum(torch.square(excess), dim=-1)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env, command_name: str | None, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If ``command_name`` is given, the reward is zeroed when that command is small (the agent is not
    supposed to step). Pass ``command_name=None`` to keep rewarding stepping even at zero command
    (e.g. when an always-on gait should march in place).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command (skipped when command_name is None -> always rewarded)
    if command_name is not None:
        reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def joint_energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


# def feet_clearance_reward(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, base_height:float, target_feet_height: float, std: float, tanh_mult: float
# ) -> torch.Tensor:
#     """Reward the swinging feet for clearing a specified height off the ground"""
#     asset: Articulation = env.scene[asset_cfg.name]
#     foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - (asset.data.root_pos_w[:, 2:3] - base_height + target_feet_height))
#     foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
#     reward = foot_z_target_error * foot_velocity_tanh
#     return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_clearance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_clearance_swing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
) -> torch.Tensor:
    """Reward swing feet (off the ground) for reaching a target clearance height.

    Unlike :func:`feet_clearance`, which weights the height error by the foot's *horizontal* speed,
    this gates by contact state. A foot in the air is required to reach ``target_height`` regardless
    of the velocity command, so the robot still picks up its feet when marching in place at zero
    command (where the horizontal-speed weighting gives no signal). Feet in contact are not penalized.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0  # [N, F]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [N, F]
    error = torch.square(foot_z - target_height) * (~in_contact)
    return torch.exp(-torch.sum(error, dim=1) / std)


def feet_flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize foot tilt so the soles stay parallel to the ground (flat feet).

    The foot link's local z-axis is the sole normal (the collision box is thin in z). The world
    gravity direction is projected into each foot frame; when the sole is horizontal the gravity
    vector is purely vertical in that frame, so its xy components are zero. Returns the summed
    squared horizontal components over all listed feet -- use a negative weight. Analogous to
    :func:`flat_orientation_l2` but per-foot instead of for the base.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # [N, F, 4] (wxyz, world)
    num_envs, num_feet = foot_quat.shape[0], foot_quat.shape[1]
    grav_w = torch.zeros(num_envs, num_feet, 3, device=foot_quat.device)
    grav_w[..., 2] = -1.0
    foot_grav = quat_apply_inverse(foot_quat.reshape(-1, 4), grav_w.reshape(-1, 3)).reshape(num_envs, num_feet, 3)
    return torch.sum(torch.sum(torch.square(foot_grav[..., :2]), dim=-1), dim=1)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def joint_deviation_l1_straight(
    env, command_name: str, asset_cfg: SceneEntityCfg, ang_vel_threshold: float = 0.3
) -> torch.Tensor:
    """L1 joint deviation from default, applied ONLY when the yaw command is small (going straight).

    Lets a large weight keep the hip-yaw joints centered for straight walking (prevents a duck-footed /
    splayed stance) without fighting turning: when a yaw rate above ``ang_vel_threshold`` is commanded
    the penalty switches off, so the policy is free to use the hip-yaw joints to steer. Once the turn
    finishes (yaw command returns to ~0) the penalty re-engages and pulls the feet back to forward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(angle), dim=1)
    yaw_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return deviation * (yaw_cmd < ang_vel_threshold)
