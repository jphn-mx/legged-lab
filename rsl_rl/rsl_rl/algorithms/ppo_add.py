from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent
from rsl_rl.modules.add import ADDDiscriminator
from rsl_rl.modules.amp import LossType
from rsl_rl.storage import RolloutStorage, CircularBuffer
from rsl_rl.algorithms.ppo_amp import PPOAMP


class PPOADD(PPOAMP):
    """PPO with Adversarial Differential Discriminator (ADD).

    Extends PPOAMP by computing a differential observation (demo - agent) and
    classifying the difference instead of absolute observations. Zero difference
    is the positive class (perfect imitation).
    """

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN,
        storage: RolloutStorage,
        disc_obs_buffer: CircularBuffer,
        disc_demo_obs_buffer: CircularBuffer,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        amp_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        # Call PPOAMP.__init__ which sets up everything including AMPDiscriminator.
        # We will replace the discriminator afterwards.
        super().__init__(
            policy=policy,
            storage=storage,
            disc_obs_buffer=disc_obs_buffer,
            disc_demo_obs_buffer=disc_demo_obs_buffer,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            amp_cfg=amp_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        # Replace AMPDiscriminator with ADDDiscriminator
        add_disc_kwargs = self.amp_cfg.get("amp_discriminator", {})
        self.disc_logit_reg = self.amp_cfg.get("disc_logit_reg", 0.01)

        self.amp_discriminator = ADDDiscriminator(
            disc_obs_dim=self.amp_cfg["disc_obs_dim"],
            disc_obs_steps=self.amp_cfg["disc_obs_steps"],
            obs_groups=self.policy.obs_groups,
            loss_type=self.loss_type,
            reward_type=self.reward_type,
            device=device,
            disc_logit_reg=self.disc_logit_reg,
            diff_normalizer_min_diff=self.amp_cfg.get("diff_normalizer_min_diff", 1e-4),
            diff_normalizer_clip=self.amp_cfg.get("diff_normalizer_clip", float("inf")),
            **add_disc_kwargs,
        ).to(self.device)

        # Rebuild discriminator optimizer for the new discriminator
        params = [
            {
                "name": "disc_trunk",
                "params": self.amp_discriminator.disc_trunk.parameters(),
                "weight_decay": self.amp_cfg["disc_trunk_weight_decay"],
            },
            {
                "name": "disc_linear",
                "params": self.amp_discriminator.disc_linear.parameters(),
                "weight_decay": self.amp_cfg["disc_linear_weight_decay"],
            },
        ]
        self.disc_optimizer = optim.Adam(
            params,
            lr=self.amp_cfg["disc_learning_rate"],
        )

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)

        if "terminal_obs" in extras:
            terminal_disc_obs = self.amp_discriminator.get_disc_obs(extras["terminal_obs"], flatten_history_dim=False)
            done_mask = dones.to(dtype=torch.bool)
            if torch.any(done_mask):
                disc_obs = disc_obs.clone()
                disc_obs[done_mask] = terminal_disc_obs[done_mask]

        # Compute style reward using differential approach
        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward_add(
            disc_obs, disc_demo_obs, dt=self.amp_cfg["step_dt"]
        )
        # Lerp task + style
        self.rewards_lerp = self.amp_discriminator.lerp_reward(task_reward=rewards, style_reward=self.style_rewards)

        # Store paired observations: concatenate along last dim for paired mini-batch retrieval
        # disc_obs_buffer stores agent obs, disc_demo_obs_buffer stores demo obs (same append order = paired)
        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)

        # Call grandparent (PPO) process_env_step with lerped rewards
        from rsl_rl.algorithms.ppo import PPO
        PPO.process_env_step(self, obs, self.rewards_lerp, dones, extras)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_disc_loss = 0
        mean_disc_grad_penalty = 0
        mean_disc_logit_loss = 0
        mean_disc_pos_logit = 0
        mean_disc_neg_logit = 0

        # Get mini batch generators
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # For ADD, we need paired (agent, demo) samples. Since both buffers are appended
        # simultaneously in process_env_step, they have the same ordering.
        # We use the same generator parameters to get the same indices.
        disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_demo_obs_generator = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        for samples, disc_obs_batch, disc_demo_obs_batch in zip(generator, disc_obs_generator, disc_demo_obs_generator):
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
            ) = samples

            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Policy forward pass
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # PPO losses
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # --- ADD discriminator loss ---
            mini_batch_size = disc_obs_batch.shape[0]

            # Compute differential observations
            obs_diff = disc_demo_obs_batch - disc_obs_batch  # [batch, steps, dim]
            obs_diff_flat = obs_diff.reshape(mini_batch_size, -1)

            # Normalize difference
            diff_normalizer = self.amp_discriminator.diff_normalizer
            norm_diff = diff_normalizer.normalize(
                obs_diff.reshape(-1, self.amp_discriminator.disc_obs_dim)
            ).reshape(mini_batch_size, -1)
            norm_diff.requires_grad_(True)

            # Negative class: actual difference
            disc_neg_logit = self.amp_discriminator(norm_diff)  # [batch, 1]

            # Positive class: zero difference
            pos_diff = self.amp_discriminator._pos_diff.expand(mini_batch_size, -1).clone()
            pos_diff.requires_grad_(True)
            disc_pos_logit = self.amp_discriminator(pos_diff)  # [batch, 1]

            # Discriminator loss with label smoothing (prevents perfect convergence)
            if self.loss_type == LossType.LSGAN:
                disc_loss_pos = 0.5 * torch.mean(torch.square(disc_pos_logit - 1.0))
                disc_loss_neg = 0.5 * torch.mean(torch.square(disc_neg_logit - 0.))
            else:
                bce = torch.nn.BCEWithLogitsLoss()
                disc_loss_pos = bce(disc_pos_logit, torch.ones_like(disc_pos_logit))
                disc_loss_neg = bce(disc_neg_logit, torch.zeros_like(disc_neg_logit))

            disc_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

            # Logit regularization
            logit_weights = self.amp_discriminator.get_logit_weights()
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss = disc_loss + self.disc_logit_reg * disc_logit_loss

            # Gradient penalty on both positive and negative
            disc_neg_grad = torch.autograd.grad(
                disc_neg_logit.sum(), norm_diff,
                create_graph=True, retain_graph=True,
            )[0]
            disc_neg_grad_penalty = torch.mean(torch.sum(torch.square(disc_neg_grad), dim=-1))

            disc_pos_grad = torch.autograd.grad(
                disc_pos_logit.sum(), pos_diff,
                create_graph=True, retain_graph=True,
            )[0]
            disc_pos_grad_penalty = torch.mean(torch.sum(torch.square(disc_pos_grad), dim=-1))

            disc_grad_penalty = 0.5 * (disc_neg_grad_penalty + disc_pos_grad_penalty)
            disc_total_loss = disc_loss + self.amp_cfg["grad_penalty_scale"] * disc_grad_penalty

            # Backward passes
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            self.disc_optimizer.zero_grad()
            disc_total_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            nn.utils.clip_grad_norm_(self.amp_discriminator.parameters(), self.disc_max_grad_norm)
            self.disc_optimizer.step()

            # Update diff normalizer
            diff_normalizer.record(obs_diff.reshape(-1, self.amp_discriminator.disc_obs_dim))
            diff_normalizer.update()

            # Accumulate metrics
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            mean_disc_loss += disc_loss.item()
            mean_disc_grad_penalty += disc_grad_penalty.item()
            mean_disc_logit_loss += disc_logit_loss.item()
            mean_disc_pos_logit += disc_pos_logit.mean().item()
            mean_disc_neg_logit += disc_neg_logit.mean().item()

        # Average metrics
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_disc_loss /= num_updates
        mean_disc_grad_penalty /= num_updates
        mean_disc_logit_loss /= num_updates
        mean_disc_pos_logit /= num_updates
        mean_disc_neg_logit /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        loss_dict["add/disc_loss"] = mean_disc_loss
        loss_dict["add/disc_grad_penalty"] = mean_disc_grad_penalty
        loss_dict["add/disc_logit_loss"] = mean_disc_logit_loss
        loss_dict["add/disc_pos_logit"] = mean_disc_pos_logit
        loss_dict["add/disc_neg_logit"] = mean_disc_neg_logit

        return loss_dict
