from __future__ import annotations

import math
import torch
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def gait_phase(env: ManagerBasedEnv, period: float) -> torch.Tensor:
    """Periodic gait-clock observation: ``[sin(2*pi*phi), cos(2*pi*phi)]``.

    The phase ``phi = (t % period) / period`` advances continuously with episode time and is
    independent of the velocity command, so the policy always has a clock to step to (even when the
    command is zero). Use the **same** ``period`` here and in :func:`mdp.feet_gait` so the commanded
    stance/swing pattern matches what the policy observes.
    """
    phi = (env.episode_length_buf * env.step_dt) % period / period  # [N] in [0, 1)
    angle = 2.0 * math.pi * phi
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # [N, 2]


class delayed_obs(ManagerTermBase):
    """Return a stale (delayed) version of an inner observation to emulate sensor/comms latency.

    On every reset, a new per-environment integer lag in ``[min_delay_steps, max_delay_steps]``
    control steps is sampled and held constant for the episode. One control step equals
    ``decimation * sim.dt`` seconds (e.g. 0.02 s with decimation 4 and dt 0.005).

    Configure as a class-based observation term. The wrapped observation is passed via ``func``
    (and optional ``func_params``); any noise/clip/scale on the term is applied by the manager
    *after* this term returns, so keep the noise spec on the delayed term as usual::

        joint_pos = ObsTerm(
            func=mdp.delayed_obs,
            params={"func": mdp.joint_pos_rel, "min_delay_steps": 0, "max_delay_steps": 2},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._min_delay = int(cfg.params.get("min_delay_steps", 0))
        self._max_delay = int(cfg.params.get("max_delay_steps", 0))
        self._buffer = DelayBuffer(self._max_delay, self.num_envs, self.device)
        self._resample(torch.arange(self.num_envs, device=self.device))

    def _resample(self, env_ids: torch.Tensor) -> None:
        if self._max_delay <= 0:
            return
        lags = torch.randint(
            self._min_delay, self._max_delay + 1, (len(env_ids),), device=self.device, dtype=torch.int
        )
        self._buffer.set_time_lag(lags, env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._buffer.reset(env_ids)
        self._resample(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        func: Callable,
        func_params: dict | None = None,
        min_delay_steps: int = 0,
        max_delay_steps: int = 0,
    ) -> torch.Tensor:
        data = func(env, **(func_params or {}))
        return self._buffer.compute(data)


def height_scan_ch(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.

    add a channel dimension to the output tensor, so that it can be used as a 2D image

    ref: isaaclab.envs.mdp.observations.height_scan
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    ordering = sensor.cfg.pattern_cfg.ordering
    """Specifies the ordering of points in the generated grid. Defaults to ``"xy"``.

    Consider a grid pattern with points at :math:`(x, y)` where :math:`x` and :math:`y` are the grid indices.
    The ordering of the points can be specified as "xy" or "yx". This determines the inner and outer loop order
    when iterating over the grid points.

    * If "xy" is selected, the points are ordered with inner loop over "x" and outer loop over "y".
    * If "yx" is selected, the points are ordered with inner loop over "y" and outer loop over "x".

    For example, the grid pattern points with :math:`X = (0, 1, 2)` and :math:`Y = (3, 4)`:

    * "xy" ordering: :math:`[(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (2, 4)]`
    * "yx" ordering: :math:`[(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]`
    """

    shape = sensor.cfg.shape  # define in RayCasterArrayCfg

    # height scan: height = sensor_height - hit_point_z - offset
    scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    # TODO: check
    if ordering == "yx":
        scan = scan.reshape(-1, shape[0], shape[1])
    elif ordering == "xy":
        scan = scan.reshape(-1, shape[1], shape[0]).transpose(1, 2)
    else:
        raise ValueError(f"Invalid ordering: {ordering}. Expected 'xy' or 'yx'.")

    return scan.unsqueeze(-1)  # add a channel dimension to the output tensor
