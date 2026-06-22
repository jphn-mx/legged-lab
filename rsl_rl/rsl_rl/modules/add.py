from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd
from tensordict import TensorDict

from rsl_rl.modules.amp import AMPDiscriminator, LossType
from rsl_rl.networks.diff_normalization import DiffNormalizer


class ADDDiscriminator(AMPDiscriminator):
    """Adversarial Differential Discriminator.

    Instead of classifying absolute observations, this discriminator operates on
    the difference between demonstration and agent observations. Zero difference
    (perfect imitation) is the positive class.
    """

    def __init__(
        self,
        disc_obs_dim: int,
        disc_obs_steps: int,
        obs_groups: dict,
        loss_type: LossType = LossType.GAN,
        hidden_dims=[256, 256, 256],
        activation="relu",
        style_reward_scale=1.0,
        task_style_lerp=0.0,
        disc_logit_reg: float = 0.01,
        diff_normalizer_min_diff: float = 1e-4,
        diff_normalizer_clip: float = float("inf"),
        device="cpu",
    ):
        super().__init__(
            disc_obs_dim=disc_obs_dim,
            disc_obs_steps=disc_obs_steps,
            obs_groups=obs_groups,
            loss_type=loss_type,
            hidden_dims=hidden_dims,
            activation=activation,
            style_reward_scale=style_reward_scale,
            task_style_lerp=task_style_lerp,
            device=device,
        )

        self.disc_logit_reg = disc_logit_reg

        self.diff_normalizer = DiffNormalizer(
            shape=self.disc_obs_dim,
            min_diff=diff_normalizer_min_diff,
            clip=diff_normalizer_clip,
        ).to(device)

        self.register_buffer(
            "_pos_diff", torch.zeros(1, self.input_dim, device=device)
        )

    def predict_style_reward_add(
        self, disc_obs: torch.Tensor, disc_demo_obs: torch.Tensor, dt: float
    ):
        """Compute style reward from differential observations.

        Args:
            disc_obs: Agent observations [num_envs, disc_obs_steps, disc_obs_dim]
            disc_demo_obs: Demo observations [num_envs, disc_obs_steps, disc_obs_dim]
            dt: Environment step time

        Returns:
            Tuple of (style_reward, disc_score)
        """
        was_training = self.training
        with torch.no_grad():
            self.eval()

            obs_diff = disc_demo_obs - disc_obs  # [num_envs, steps, dim]
            obs_diff_reshaped = obs_diff.reshape(-1, self.disc_obs_dim)
            norm_diff = self.diff_normalizer(obs_diff_reshaped)
            norm_diff = norm_diff.reshape(-1, self.input_dim)

            disc_score = self.forward(norm_diff)  # [num_envs, 1]

            # Use disc score directly as reward (clamped to non-negative).
            # LSGAN trains disc to output ~1 for zero-diff, ~0 for actual-diff,
            # so score directly indicates imitation quality.
            rew = torch.clamp(disc_score, min=0)

            style_reward = dt * self.style_reward_scale * rew

            if was_training:
                self.train()

        return style_reward.squeeze(-1), disc_score.squeeze(-1)

    def get_logit_weights(self) -> torch.Tensor:
        return torch.flatten(self.disc_linear.weight)
