from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlAmpCfg:
    """Configuration class for the AMP (Adversarial Motion Priors) in the training"""

    disc_obs_buffer_size: int = 1000
    """Size of the replay buffer for storing discriminator observations"""

    grad_penalty_scale: float = 10.0
    """Scale for the gradient penalty in AMP training"""

    disc_trunk_weight_decay: float = 1.0e-4
    """Weight decay for the discriminator trunk network"""

    disc_linear_weight_decay: float = 1.0e-2
    """Weight decay for the discriminator linear network"""

    disc_learning_rate: float = 1.0e-5
    """Learning rate for the discriminator networks"""

    disc_max_grad_norm: float = 1.0
    """Maximum gradient norm for the discriminator networks"""

    @configclass
    class AMPDiscriminatorCfg:
        """Configuration for the AMP discriminator network."""

        hidden_dims: list[int] = MISSING
        """The hidden dimensions of the AMP discriminator network."""

        activation: str = "elu"
        """The activation function for the AMP discriminator network."""

        style_reward_scale: float = 1.0
        """Scale for the style reward in the training"""

        task_style_lerp: float = 0.0
        """Linear interpolation factor for the task style reward in the AMP training."""

        use_spectral_norm: bool = False
        """If True, wrap every discriminator Linear layer in spectral normalization to
        bound its Lipschitz constant. This stops the discriminator from saturating into
        over-confident +-1 scores (which flattens the style reward to ~0 and kills the
        style gradient mid-run). Does not change the reward formula. Default False."""

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    """Configuration for the AMP discriminator network."""

    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "LSGAN"
    """Type of loss function used for the AMP discriminator training (e.g., 'GAN', 'LSGAN', 'WGAN')"""

    reward_type: Literal["LSGAN", "GAIL", "AIRL"] = "LSGAN"
    """Type of reward formula for the policy. 'LSGAN' saturates when disc is strong;
    'GAIL' (-log(1-sigmoid(d))) is non-saturating; 'AIRL' (logit) is unbounded."""
