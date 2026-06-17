"""Left-right symmetry for the A1-legs_V1 velocity task (12 DOF).

Mirror function for RSL-RL PPO symmetry augmentation (data augmentation / mirror loss). It MUST
match the exact policy observation layout defined in ``A1ObservationsCfg.PolicyCfg``. Each term is
repeated over ``history_length`` and the terms are concatenated in definition order:

    base_ang_vel(3), projected_gravity(3), velocity_commands(3),
    joint_pos(12), joint_vel(12), actions(12), gait_phase(2)

Per-term mirror under a left-right (sagittal, y -> -y) reflection:
    base_ang_vel       pseudovector -> [-1,  1, -1]
    projected_gravity  polar vector -> [ 1, -1,  1]
    velocity_commands  [vx, vy, wz] -> [ 1, -1, -1]
    joint_pos/vel/act  swap L<->R legs, negate roll & yaw joints
    gait_phase         [sin, cos]   -> [-1, -1]   (mirroring legs == half-period phase shift,
                                                   sin/cos(2*pi*(phi+0.5)) = -sin/-cos)

Only the ``policy`` group is mirrored (the critic half stays the duplicated original, which is
consistent: the value targets are repeated from the originals). Keep this file in sync with the
PolicyCfg if its terms / order / history_length ever change.

A1 Lab joint order (12 DOF), used by the swap below:
 0 joint_L1 hip pitch | 1 joint_R1 hip pitch | 2 joint_L2 hip roll | 3 joint_R2 hip roll
 4 joint_L3 hip yaw   | 5 joint_R3 hip yaw   | 6 joint_L4 knee      | 7 joint_R4 knee
 8 joint_L5 ankle pitch | 9 joint_R5 ankle pitch | 10 joint_L6 ankle roll | 11 joint_R6 ankle roll
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# Must match A1ObservationsCfg.PolicyCfg.__post_init__.history_length and term dims/order.
HISTORY_LEN = 5
ANG_VEL_DIM = 3
PROJ_GRAV_DIM = 3
VEL_CMD_DIM = 3
JOINT_POS_DIM = 12
JOINT_VEL_DIM = 12
LAST_ACTIONS_DIM = 12
GAIT_PHASE_DIM = 2


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(obs["policy"][:])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _switch_a1_12dof_joints_left_right(actions.clone())
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device
    ang_vel_sign = torch.tensor([-1.0, 1.0, -1.0], device=device)
    proj_grav_sign = torch.tensor([1.0, -1.0, 1.0], device=device)
    vel_cmd_sign = torch.tensor([1.0, -1.0, -1.0], device=device)
    gait_phase_sign = torch.tensor([-1.0, -1.0], device=device)

    end_idx = 0
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * ang_vel_sign
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + PROJ_GRAV_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * proj_grav_sign
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * vel_cmd_sign
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_a1_12dof_joints_left_right(obs[:, start_idx:end_idx])
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_a1_12dof_joints_left_right(obs[:, start_idx:end_idx])
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_a1_12dof_joints_left_right(obs[:, start_idx:end_idx])
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + GAIT_PHASE_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * gait_phase_sign

    return obs


def _switch_a1_12dof_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data_switched = torch.zeros_like(joint_data)

    left_indices = [0, 2, 4, 6, 8, 10]
    right_indices = [1, 3, 5, 7, 9, 11]
    roll_indices = [2, 3, 10, 11]
    yaw_indices = [4, 5]

    joint_data_switched[..., left_indices] = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]

    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0

    return joint_data_switched
