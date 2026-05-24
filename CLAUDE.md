# CLAUDE.md

## Project Overview

Legged Lab — a research platform for training and deploying locomotion policies on legged robots using reinforcement learning. Built on NVIDIA Isaac Lab + RSL-RL (PPO).

## Architecture

```
legged_lab/          Isaac Lab extension — RL environments (AMP, DeepMimic, Velocity, Animation)
whole_body_tracking/ Isaac Lab extension — whole-body motion tracking
GMR/                 General Motion Retargeting — IK-based human→robot motion retargeting
rsl_rl/              RSL-RL framework (PPO, AMP discriminator, runners)
human/               MuJoCo sim deployment (ONNX inference + PD control)
scripts/             Top-level utilities (auto_tune agent)
A1-legs_V1/          URDF model
A1-legs_V1_mjcf/     MuJoCo XML model
```

## Key Conventions

### Environment Registration

Isaac Lab environments are registered via `gymnasium.register()` in `__init__.py` files under each robot's config directory:
- `legged_lab/source/legged_lab/legged_lab/tasks/locomotion/{method}/config/{robot}/__init__.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/{robot}/__init__.py`

Task IDs follow the pattern: `LeggedLab-Isaac-{Method}-{Robot}-v0` or `Tracking-Flat-{Robot}-v0`.

### Config Pattern (Isaac Lab)

All environment and agent configs use `@configclass` decorator from `isaaclab.utils`:
- `*_env_cfg.py` — environment config (scene, observations, actions, rewards, terminations)
- `rsl_rl_ppo_cfg.py` — training algorithm config (PPO, AMP discriminator, network architecture)

Base environment configs are in `{method}_env_cfg.py`; robot-specific ones inherit and override in `config/{robot}/`.

### Joint Ordering

**Critical**: Isaac Lab and MuJoCo use different joint orders for A1:
- Isaac Lab (ONNX): `L1, R1, L2, R2, L3, R3, L4, R4, L5, R5, L6, R6`
- MuJoCo: `R1, R2, R3, R4, R5, R6, L1, L2, L3, L4, L5, L6`

Deployment configs (`human/configs/*.yaml`) define the mapping via `obs_index_in_lab` / `obs_index_in_mj`.

### Motion Data Formats

| Format | Used By | Description |
|--------|---------|-------------|
| GMR pkl | GMR output | `{root_pos, root_rot(xyzw), dof_pos, fps}` |
| Lab pkl | legged_lab AMP/DeepMimic | `{root_pos, root_rot(wxyz), dof_pos, key_body_pos, fps, loop_mode}` |
| CSV | intermediate | `[x, y, z, qx, qy, qz, qw, dof_1..12]` per row |
| npz | whole_body_tracking | `{fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w}` |

Note: GMR uses xyzw quaternion convention; Lab uses wxyz. The `gmr_to_lab.py` script handles this conversion.

### Data Conversion Pipeline

```
GMR pkl ──► gmr_to_lab.py / dataset_retarget.py ──► Lab pkl (for AMP/DeepMimic)
GMR pkl ──► pkl_to_csv.py ──► csv_to_npz.py ──► npz (for whole-body tracking)
```

Conversion scripts:
- `legged_lab/scripts/tools/retarget/gmr_to_lab.py` — single file GMR→Lab
- `legged_lab/scripts/tools/retarget/dataset_retarget.py` — batch GMR→Lab
- `whole_body_tracking/scripts/pkl_to_csv.py` — GMR pkl→CSV
- `whole_body_tracking/scripts/csv_to_npz.py` — CSV→npz with FK

Config files for retargeting: `legged_lab/scripts/tools/retarget/config/{a1_12dof,g1_29dof}.yaml`

## Common Commands

```bash
# Training (requires Isaac Sim + Isaac Lab)
cd legged_lab
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-A1-v0 --num_envs 4096
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-A1-Play-v0

# Whole-body tracking training
cd whole_body_tracking
python scripts/rsl_rl/train.py --task Tracking-Flat-A1-v0 --num_envs 4096

# Motion retargeting
cd GMR
python scripts/smplx_to_robot.py --smplx_file <path> --robot a1_legs_v1 --save_path output.pkl

# Data conversion (GMR → legged_lab)
cd legged_lab
python scripts/tools/retarget/single_retarget.py --robot a1 \
    --input_file <gmr.pkl> --output_file <lab.pkl> \
    --config_file scripts/tools/retarget/config/a1_12dof.yaml

# MuJoCo deployment
cd human
python deploy_a1_amp.py          # AMP locomotion
python deploy_a1_wbt.py          # WBT motion playback
python deploy_a1_combined.py     # Combined AMP + WBT skills
```

## Installation

```bash
# Isaac Lab extensions (editable install)
pip install -e legged_lab/source/legged_lab
pip install -e whole_body_tracking/source/whole_body_tracking
pip install -e rsl_rl
pip install -e GMR

# MuJoCo deployment
pip install mujoco onnxruntime scipy pyyaml
```

## Code Style

- **legged_lab / whole_body_tracking**: Black (line-length 120), isort (profile=black), flake8
- **rsl_rl**: Ruff (line-length 120, target py39), extensive lint rules (see `rsl_rl/ruff.toml`)
- **GMR / human**: No enforced formatter

## Robots

| Robot | DOF | Config Location |
|-------|-----|-----------------|
| A1-legs-V1 (bipedal) | 12 | `legged_lab/.../config/a1/`, `whole_body_tracking/.../config/a1/` |
| G1 (humanoid) | 29 | `legged_lab/.../config/g1/`, `whole_body_tracking/.../config/g1/` |
| Go2 (quadruped) | 12 | `legged_lab/.../velocity/config/go2/` |

## AMP Training Notes

- Discriminator learning rate sweet spot: 3e-5 ~ 5e-5 for A1. Too high → pure task behavior; too low → pure style imitation.
- `task_style_lerp` controls task vs style reward balance (0 = all task, 1 = all style).
- Symmetry augmentation available via `RslRlSymmetryCfg` (data augmentation or mirror loss).
- AMP observation groups: `policy`, `critic`, `discriminator`, `discriminator_demonstration`.

## Directories to Ignore

- `TienKung-Lab/` — separate project, not part of this repo
- `legged_lab/logs/`, `legged_lab/outputs/` — training outputs (gitignored)
- `*/motion_data/` (most subdirs) — large datasets (gitignored, keep only examples)
