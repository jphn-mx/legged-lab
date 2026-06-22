from __future__ import annotations

import torch
from torch import nn


class DiffNormalizer(nn.Module):
    """Normalize values by running mean absolute value.

    Designed for differential observations where the expected mean is zero.
    Instead of subtracting mean and dividing by std, this simply divides by
    the running mean of absolute values.
    """

    def __init__(self, shape: int | tuple[int], min_diff: float = 1e-4, clip: float = float("inf")) -> None:
        super().__init__()
        self.min_diff = min_diff
        self.clip = clip

        if isinstance(shape, int):
            shape = (shape,)

        self.register_buffer("_mean_abs", torch.ones(shape))
        self.register_buffer("_count", torch.zeros(1, dtype=torch.long))
        self._new_count: int = 0
        self._new_sum_abs: torch.Tensor | None = None

    @property
    def mean_abs(self) -> torch.Tensor:
        return self._mean_abs.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        diff = torch.clamp_min(self._mean_abs, self.min_diff)
        norm_x = x / diff
        norm_x = torch.clamp(norm_x, -self.clip, self.clip)
        return norm_x

    def record(self, x: torch.Tensor) -> None:
        """Accumulate statistics from a batch of observations."""
        if not self.training:
            return
        flat = x.reshape(-1, *self._mean_abs.shape)
        self._new_count += flat.shape[0]
        batch_sum_abs = torch.sum(torch.abs(flat), dim=0)
        if self._new_sum_abs is None:
            self._new_sum_abs = torch.zeros_like(self._mean_abs)
        self._new_sum_abs += batch_sum_abs

    def update(self) -> None:
        """Update running mean absolute value from recorded data."""
        if self._new_count == 0 or self._new_sum_abs is None:
            return

        new_mean_abs = self._new_sum_abs / self._new_count
        new_total = self._count + self._new_count
        w_old = self._count.float() / new_total.float()
        w_new = float(self._new_count) / new_total.float()

        self._mean_abs.copy_(w_old * self._mean_abs + w_new * new_mean_abs)
        self._count.copy_(new_total)

        self._new_count = 0
        self._new_sum_abs.zero_()
