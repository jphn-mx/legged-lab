"""Functions to specify the symmetry in the observation and action space for A1-legs_V1 12dof."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


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
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"][:]
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device
    joint_num = 12

    HISTORY_LEN = 5
    ANG_VEL_DIM = 3
    ROT_TAN_NORM = 6
    VEL_CMD_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num

    end_idx = 0
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)

    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ROT_TAN_NORM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1, 1, -1, 1], device=device)

    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)

    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_a1_12dof_joints_left_right(obs[:, start_idx:end_idx])

    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_a1_12dof_joints_left_right(obs[:, start_idx:end_idx])

    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_a1_12dof_joints_left_right(obs[:, start_idx:end_idx])

    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_a1_12dof_joints_left_right(actions[:])
    return actions


"""
A1 Lab joint order (12 DOF):
 0 - joint_L1 (hip pitch,   axis 0 1 0)
 1 - joint_R1 (hip pitch,   axis 0 1 0)
 2 - joint_L2 (hip roll,    axis 1 0 0)
 3 - joint_R2 (hip roll,    axis 1 0 0)
 4 - joint_L3 (hip yaw,     axis 0 0 1)
 5 - joint_R3 (hip yaw,     axis 0 0 1)
 6 - joint_L4 (knee pitch,  axis 0 1 0)
 7 - joint_R4 (knee pitch,  axis 0 1 0)
 8 - joint_L5 (ankle pitch, axis 0 1 0)
 9 - joint_R5 (ankle pitch, axis 0 1 0)
10 - joint_L6 (ankle roll,  axis 1 0 0)
11 - joint_R6 (ankle roll,  axis 1 0 0)
"""


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
