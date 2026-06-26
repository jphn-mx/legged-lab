# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sagittal-plane (left/right) symmetry augmentation for the A1 bipedal velocity task.

The robot faces +x; the mirror is the reflection about the x-z plane (flip y). This module
provides ``a1_symmetry_augmentation`` which is plugged into rsl_rl's PPO via
``RslRlSymmetryCfg.data_augmentation_func``. It builds, per observation group and for the
policy action, an index-permutation tensor and a sign tensor that together map a sample to
its mirror image, then concatenates the mirrored samples after the originals along the batch
dimension (the convention rsl_rl's PPO expects: first ``B`` rows = original, next ``B`` = mirror).

# ---------------------------------------------------------------------------------------------
# Joint mirror table (Lab joint order: L1,R1,L2,R2,L3,R3,L4,R4,L5,R5,L6,R6)
# ---------------------------------------------------------------------------------------------
# A1 leg joints (per leg, joint_*N):
#   1 hip pitch  (URDF axis 0 1 0, y)  -> sagittal swing   -> sign +1
#   2 hip roll   (URDF axis 1 0 0, x)  -> lateral          -> sign -1
#   3 hip yaw    (URDF axis 0 0 1, z)  -> yaw              -> sign -1
#   4 knee       (URDF axis 0 1 0, y)  -> sagittal         -> sign +1
#   5 ankle pitch(URDF axis 0 1 0, y)  -> sagittal         -> sign +1
#   6 ankle roll (URDF axis 1 0 0, x)  -> lateral          -> sign -1
# Both legs use IDENTICAL local axis directions in the URDF (verified A1_legs_V2.urdf), and the
# joint_*2 limits are mirror-negated (R2 [-1.05,0.26] vs L2 [-0.26,1.05]), confirming that the
# roll/yaw joints must flip sign under the sagittal mirror while the pitch joints keep their sign.
#
# Mirror of a 12-d joint vector q (Lab order) =
#     for each joint type N: swap L_N <-> R_N, then multiply by JOINT_SIGN[N].
# In Lab order [L1,R1,L2,R2,...] the swap is just adjacent-pair swap (index 0<->1, 2<->3, ...).
# ---------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import torch

# Per-joint sign under the sagittal mirror, indexed by joint number 1..6 (axis from URDF).
#   pitch (1,4,5): +1   |   roll/yaw (2,3,6): -1
_JOINT_TYPE_SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: 1.0, 5: 1.0, 6: -1.0}

# Lab joint order: [L1, R1, L2, R2, L3, R3, L4, R4, L5, R5, L6, R6]
# (term index, joint side, joint number) for documentation; built into the buffers below.
_LAB_JOINT_ORDER = [
    ("L", 1), ("R", 1),
    ("L", 2), ("R", 2),
    ("L", 3), ("R", 3),
    ("L", 4), ("R", 4),
    ("L", 5), ("R", 5),
    ("L", 6), ("R", 6),
]


def _build_joint_perm_sign() -> tuple[list[int], list[float]]:
    """Permutation + sign that mirror a 12-d joint vector in Lab order.

    Returns ``(perm, sign)`` such that ``mirrored[i] = sign[i] * x[perm[i]]``.
    Swapping L_N<->R_N in Lab order is an adjacent-pair swap; the sign is the joint-type sign.
    """
    perm = [0] * 12
    sign = [1.0] * 12
    # map (side, num) -> index in Lab order
    index_of = {sn: i for i, sn in enumerate(_LAB_JOINT_ORDER)}
    for i, (side, num) in enumerate(_LAB_JOINT_ORDER):
        other_side = "R" if side == "L" else "L"
        perm[i] = index_of[(other_side, num)]
        sign[i] = _JOINT_TYPE_SIGN[num]
    return perm, sign


_JOINT_PERM, _JOINT_SIGN = _build_joint_perm_sign()


def _build_frame_perm_sign(terms: list[str]) -> tuple[list[int], list[float]]:
    """Build the per-frame permutation/sign for a single observation frame.

    ``terms`` is the ordered list of observation-term names that make up ONE frame. The layout
    of each single-frame observation (47-d for policy, 50-d for critic) is the concatenation of
    these terms in order. Returns ``(perm, sign)`` over the full frame width such that
    ``mirrored_frame[i] = sign[i] * frame[perm[i]]``.

    Per-term mirror rules (robot faces +x, mirror flips y):
      base_lin_vel       [vx,vy,vz] (true vector)      -> [ vx, -vy,  vz]
      base_ang_vel       [wx,wy,wz] (pseudovector)     -> [-wx,  wy, -wz]
      projected_gravity  [gx,gy,gz] (true vector)      -> [ gx, -gy,  gz]
      velocity_commands  [vx,vy,yaw]                   -> [ vx, -vy, -yaw]
      joint_pos/joint_vel/actions (12, Lab order)      -> L<->R swap + joint-type sign
      gait_phase         [sin,cos] (two legs anti-phase, phi -> phi+0.5)
                                                       -> [-sin, -cos]
    """
    # local per-term (perm, sign), perm relative to the term's own start
    term_specs = {
        "base_lin_vel": ([0, 1, 2], [1.0, -1.0, 1.0]),
        "base_ang_vel": ([0, 1, 2], [-1.0, 1.0, -1.0]),
        "projected_gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),
        "velocity_commands": ([0, 1, 2], [1.0, -1.0, -1.0]),
        "joint_pos": (_JOINT_PERM, _JOINT_SIGN),
        "joint_vel": (_JOINT_PERM, _JOINT_SIGN),
        "actions": (_JOINT_PERM, _JOINT_SIGN),
        # gait_phase = [sin(2*pi*phi), cos(2*pi*phi)]; swapping legs shifts phi by half a period
        # (phi -> phi + 0.5) => sin -> -sin, cos -> -cos. Missing this makes the symmetry loss
        # fight the anti-phase gait clock.
        "gait_phase": ([0, 1], [-1.0, -1.0]),
    }

    perm: list[int] = []
    sign: list[float] = []
    offset = 0
    for term in terms:
        if term not in term_specs:
            raise KeyError(f"a1_symmetry_augmentation: no mirror spec for observation term '{term}'")
        local_perm, local_sign = term_specs[term]
        perm.extend(p + offset for p in local_perm)
        sign.extend(local_sign)
        offset += len(local_perm)
    return perm, sign


# Single-frame term layout. MUST match A1ObservationsCfg (config/a1/flat_env_cfg.py).
#   PolicyCfg  : base_ang_vel | projected_gravity | velocity_commands | joint_pos | joint_vel | actions | gait_phase  (47)
#   CriticCfg  : base_lin_vel | base_ang_vel | projected_gravity | velocity_commands | joint_pos | joint_vel | actions | gait_phase  (50)
_POLICY_FRAME_TERMS = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "gait_phase",
]
_CRITIC_FRAME_TERMS = ["base_lin_vel"] + _POLICY_FRAME_TERMS

_POLICY_FRAME_PERM, _POLICY_FRAME_SIGN = _build_frame_perm_sign(_POLICY_FRAME_TERMS)
_CRITIC_FRAME_PERM, _CRITIC_FRAME_SIGN = _build_frame_perm_sign(_CRITIC_FRAME_TERMS)

_FRAME_WIDTH = {"policy": len(_POLICY_FRAME_PERM), "critic": len(_CRITIC_FRAME_PERM)}
_FRAME_SPEC = {
    "policy": (_POLICY_FRAME_PERM, _POLICY_FRAME_SIGN),
    "critic": (_CRITIC_FRAME_PERM, _CRITIC_FRAME_SIGN),
}

# cache of (perm, sign) tensors per (group, device, dtype, history_length)
_CACHE: dict = {}


def _get_obs_tensors(group: str, history_length: int, device, dtype):
    """Tile the single-frame (perm, sign) across the flattened history.

    The observation manager uses term-major (default) concatenation when ``concatenate_terms`` is
    True and ``interleave_by_time`` is False (the A1 config): each term's full history is flattened
    [frame0..frame_{H-1}] and then terms are concatenated. Because EVERY frame of a term shares the
    same per-frame layout, the correct full-vector mirror is the single-frame (perm, sign) repeated
    once per history frame, with the perm shifted by the frame offset. This produces an identical
    result for the alternative interleaved layout too, since both are just H copies of the same
    per-frame block; only the block ordering differs and the permutation is built to match the
    term-major order used here.
    """
    key = (group, device, dtype, history_length)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    frame_perm, frame_sign = _FRAME_SPEC[group]
    width = _FRAME_WIDTH[group]
    full_perm: list[int] = []
    full_sign: list[float] = []
    for f in range(history_length):
        base = f * width
        full_perm.extend(p + base for p in frame_perm)
        full_sign.extend(frame_sign)

    perm_t = torch.tensor(full_perm, dtype=torch.long, device=device)
    sign_t = torch.tensor(full_sign, dtype=dtype, device=device)
    _CACHE[key] = (perm_t, sign_t)
    return perm_t, sign_t


def _get_action_tensors(device, dtype):
    key = ("action", device, dtype)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    perm_t = torch.tensor(_JOINT_PERM, dtype=torch.long, device=device)
    sign_t = torch.tensor(_JOINT_SIGN, dtype=dtype, device=device)
    _CACHE[key] = (perm_t, sign_t)
    return perm_t, sign_t


def _mirror_flat(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """mirrored[..., i] = sign[i] * x[..., perm[i]] for a flat last dim."""
    return x[..., perm] * sign


def a1_symmetry_augmentation(env=None, obs=None, actions=None, obs_type: str = "policy"):
    """Symmetry data-augmentation function for the A1 bipedal velocity task.

    Signature matches :class:`isaaclab_rl.rsl_rl.RslRlSymmetryCfg.data_augmentation_func` and the
    way rsl_rl's PPO calls it (keyword args ``env=``, ``obs=``, ``actions=``).

    Args:
        env: the (unused) vec-env handle injected by rsl_rl.
        obs: a TensorDict keyed by observation group ("policy", "critic"), each value a flat
            ``[B, group_dim]`` tensor (history already flattened), OR ``None``.
        actions: a ``[B, 12]`` action tensor (Lab joint order), OR ``None``.
        obs_type: name of the obs group when a single tensor is passed (default "policy").

    Returns:
        ``(obs_out, actions_out)``: the original samples with their mirrored copies concatenated
        along the batch dimension (``[2B, ...]``). Either may be ``None`` if the input was ``None``.
    """
    obs_out = None
    if obs is not None:
        # obs is a TensorDict of {group_name: [B, dim]}. Mirror every group and stack [orig; mirror].
        if hasattr(obs, "keys") and not torch.is_tensor(obs):
            mirrored = obs.clone()
            for group in obs.keys():
                tensor = obs[group]
                width = _FRAME_WIDTH.get(group)
                if width is None:
                    # unknown group: pass through unchanged (cannot mirror safely)
                    mirrored[group] = tensor.clone()
                    continue
                history_length = tensor.shape[-1] // width
                if history_length * width != tensor.shape[-1]:
                    raise ValueError(
                        f"a1_symmetry_augmentation: group '{group}' width {tensor.shape[-1]} "
                        f"is not a multiple of frame width {width}."
                    )
                perm, sign = _get_obs_tensors(group, history_length, tensor.device, tensor.dtype)
                mirrored[group] = _mirror_flat(tensor, perm, sign)
            obs_out = torch.cat([obs, mirrored], dim=0)
        else:
            # plain tensor for a single obs group
            tensor = obs
            width = _FRAME_WIDTH[obs_type]
            history_length = tensor.shape[-1] // width
            perm, sign = _get_obs_tensors(obs_type, history_length, tensor.device, tensor.dtype)
            obs_out = torch.cat([tensor, _mirror_flat(tensor, perm, sign)], dim=0)

    actions_out = None
    if actions is not None:
        perm, sign = _get_action_tensors(actions.device, actions.dtype)
        actions_out = torch.cat([actions, _mirror_flat(actions, perm, sign)], dim=0)

    return obs_out, actions_out
