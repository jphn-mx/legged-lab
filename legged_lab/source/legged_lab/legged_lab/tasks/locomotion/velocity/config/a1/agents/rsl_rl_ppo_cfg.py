# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class A1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    empirical_normalization = False
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "a1_flat"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Left/right (sagittal-plane) symmetry. We enable ONLY the mirror loss (a soft regularizer
        # that penalizes the actor for not producing the mirrored action on the mirrored obs); data
        # augmentation is left off so the on-policy statistics (advantages/returns) are not altered.
        # mirror_loss_coeff=0.5 is a moderate start (range ~0.5-1.0); raise toward 1.0 if the gait
        # stays asymmetric, lower if it suppresses the learned gait. The augmentation function is
        # referenced by dotted path and resolved by rsl_rl via string_to_callable.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=0.5,
            data_augmentation_func=(
                "legged_lab.tasks.locomotion.velocity.config.a1.agents.symmetry:a1_symmetry_augmentation"
            ),
        ),
    )


