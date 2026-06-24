from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlSymmetryCfg

from legged_lab.rsl_rl import RslRlAddCfg, RslRlAmpCfg, RslRlPpoAddAlgorithmCfg, RslRlPpoAmpAlgorithmCfg
from legged_lab.tasks.locomotion.amp.mdp.symmetry import a1


@configclass
class A1RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    empirical_normalization = True
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "a1_v2_amp"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        # class_name="PPOAMP",
        # value_loss_coef=1.0,
        # use_clipped_value_loss=True,
        # clip_param=0.2,
        # entropy_coef=0.01,
        # num_learning_epochs=5,
        # num_mini_batches=4,
        # learning_rate=1.0e-4,
        # schedule="adaptive",
        # gamma=0.99,
        # lam=0.95,
        # desired_kl=0.01,
        # max_grad_norm=1.0,
        # amp_cfg=RslRlAmpCfg(
        #     disc_obs_buffer_size=2000,
        #     grad_penalty_scale=15.0,
        #     disc_trunk_weight_decay=1e-4,
        #     disc_linear_weight_decay=1.0e-2,
        #     disc_learning_rate=1.0e-5,
        #     disc_max_grad_norm=1.0,
        #     amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
        #         hidden_dims=[256, 128], activation="elu", style_reward_scale=3.0, task_style_lerp=0.4
        #     ),
        #     loss_type="LSGAN",
        # ),
        class_name="PPOAMP",
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
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100, # 1000
            grad_penalty_scale=10.0, # 20.0
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1e-3,  # was 5e-6; safe to raise now that spectral_norm caps disc saturation
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[512, 256], activation="elu", style_reward_scale=2.0, task_style_lerp=0.4,
                use_spectral_norm=False,  # cap disc Lipschitz -> stop it saturating to +-0.98 -> keep style alive
            ),
            loss_type="LSGAN",
            # loss_type="WGAN",
            reward_type="LSGAN",  # "LSGAN" (saturating), "GAIL" (non-saturating), "AIRL" (unbounded)
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=a1.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )


@configclass
class A1RslRlOnPolicyRunnerAddCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "a1_add"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAddAlgorithmCfg(
        class_name="PPOADD",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAddCfg(
            disc_obs_buffer_size=500,
            grad_penalty_scale=10.0,
            disc_logit_reg=0.01,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAddCfg.AMPDiscriminatorCfg(
                hidden_dims=[256, 128], activation="elu", style_reward_scale=5.0, task_style_lerp=0.3
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            data_augmentation_func=a1.compute_symmetric_states,
            use_mirror_loss=False,
            mirror_loss_coeff=0.1,
        ),
    )
