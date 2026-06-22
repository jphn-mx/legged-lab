from isaaclab.utils import configclass

from .amp_cfg import RslRlAmpCfg


@configclass
class RslRlAddCfg(RslRlAmpCfg):
    """Configuration class for ADD (Adversarial Differential Discriminator) training.

    Inherits all AMP discriminator hyperparameters and adds ADD-specific ones.
    """

    disc_logit_reg: float = 0.01
    """L2 regularization coefficient for discriminator output layer weights."""

    diff_normalizer_clip: float = float("inf")
    """Clipping value for the differential normalizer output."""

    diff_normalizer_min_diff: float = 1e-4
    """Minimum denominator for the differential normalizer (prevents division by zero)."""
