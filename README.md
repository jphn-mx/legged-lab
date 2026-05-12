# Legged Lab

A research platform for legged robot locomotion using reinforcement learning, built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl).

## Overview

This repository integrates multiple modules for training, retargeting, and deploying locomotion policies on legged robots:

| Module | Description |
|--------|-------------|
| `legged_lab/` | Core RL training environments (AMP, DeepMimic, Velocity Tracking, Animation) built on Isaac Lab |
| `whole_body_tracking/` | Whole-body motion tracking for humanoid and quadruped robots |
| `GMR/` | General Motion Retargeting — real-time human-to-robot motion retargeting |
| `human/` | MuJoCo-based sim deployment for trained policies (A1 bipedal robot) |

## Supported Robots

- **Unitree A1-legs-V1** — custom 12-DOF bipedal configuration
- **Unitree G1** — 29-DOF humanoid
- **Unitree Go2** — quadruped
- Multiple humanoids via GMR (Booster T1/K1, Fourier N1, ENGINEAI PM01, etc.)

## Project Structure

```
legged_lab/
├── legged_lab/           # Isaac Lab extension for RL training
│   ├── source/           # Python package (envs, tasks, managers, sensors)
│   ├── scripts/          # Training & evaluation scripts
│   └── data/motions/     # Motion reference data
├── whole_body_tracking/  # Whole-body tracking extension
│   ├── source/           # Tracking environments & MDP
│   └── scripts/          # Train/play/data conversion scripts
├── GMR/                  # General Motion Retargeting
│   ├── general_motion_retargeting/  # Core retargeting library
│   ├── scripts/          # Retargeting entry points
│   └── assets/           # Robot models (a1_legs_v1 example included)
├── human/                # MuJoCo deployment
│   ├── deploy_a1_amp.py  # Real-time policy deployment with keyboard control
│   ├── configs/          # Deployment YAML configs
│   ├── model/            # MuJoCo XML models
│   └── policy/           # Trained ONNX/PT policy weights
├── A1-legs_V1/           # URDF model
├── A1-legs_V1_mjcf/      # MuJoCo XML model
└── A1_legs_V1/           # Configuration files
```

## Key Features

### AMP (Adversarial Motion Priors)
- Style-guided locomotion learning from motion capture data
- Discriminator-based reward for natural motion generation
- Symmetry-augmented training for A1 and G1 robots

### DeepMimic
- Reference motion imitation with phase-based observation
- Joint position, velocity, and key-body tracking rewards

### Velocity Tracking
- Command-conditioned locomotion (linear velocity + yaw rate)
- Curriculum-based terrain difficulty scaling

### Whole-Body Tracking
- Full-body motion tracking for humanoid robots
- Support for SMPL-based human motion references

### Motion Retargeting (GMR)
- Real-time SMPL-X / BVH to robot retargeting at 60-70 FPS
- IK-based solver supporting 9+ robot platforms
- Dataset-level batch processing for RL training data

### MuJoCo Deployment
- ONNX/TorchScript policy inference
- Keyboard teleoperation (WASD + velocity arrows visualization)
- PD control with configurable gains and torque limits

## Installation

### Prerequisites
- NVIDIA Isaac Sim 4.x
- Isaac Lab (installed as extension)
- Python 3.10+
- PyTorch, CUDA

### Setup

```bash
# 1. Install Isaac Lab following official docs
# https://isaac-sim.github.io/IsaacLab/

# 2. Install legged_lab extension
cd legged_lab
python -m pip install -e source/legged_lab

# 3. Install whole_body_tracking extension
cd ../whole_body_tracking
python -m pip install -e source/whole_body_tracking

# 4. Install GMR (optional, for motion retargeting)
cd ../GMR
pip install -e .

# 5. Install MuJoCo deployment dependencies (optional)
pip install mujoco onnxruntime scipy
```

## Usage

### Training (Isaac Lab)

```bash
cd legged_lab

# Train AMP locomotion policy (A1 bipedal)
python scripts/rsl_rl/train.py --task Lab-Locomotion-Amp-A1-v0 --num_envs 4096

# Train velocity tracking (G1 humanoid)
python scripts/rsl_rl/train.py --task Lab-Locomotion-Velocity-Flat-G1-v0 --num_envs 4096

# Play/evaluate trained policy
python scripts/rsl_rl/play.py --task Lab-Locomotion-Amp-A1-v0
```

### Whole-Body Tracking

```bash
cd whole_body_tracking

# Convert motion CSV to npz format (for training)
python scripts/csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 \
    --frame_range 122 722 --output_file ./scripts/motions/dance1.npz --output_fps 50

# Train whole-body tracking policy (A1)
python scripts/rsl_rl/train.py --task Tracking-Flat-A1-v0 --num_envs 4096

# Train without state estimation
python scripts/rsl_rl/train.py --task Tracking-Flat-A1-Wo-State-Estimation-v0 --num_envs 4096

# Play/evaluate trained tracking policy
python scripts/rsl_rl/play.py --task Tracking-Flat-A1-v0

# Replay motion npz file for visualization
python scripts/replay_npz.py
```

### Motion Retargeting (GMR)

```bash
cd GMR

# Retarget SMPL-X motion to robot
python scripts/smplx_to_robot.py --smplx_file <path> --robot a1_legs_v1 --save_path output.pkl

# Retarget BVH motion
python scripts/bvh_to_robot.py --bvh_file <path> --robot unitree_g1 --save_path output.pkl
```

### MuJoCo Deployment

```bash
cd human

# Deploy AMP policy with keyboard control
python deploy_a1_amp.py

# Controls: W/S = forward/backward, A/D = yaw, J/L = lateral, Space = stop, R = reset
```

## Training Pipeline

```
Human Motion Data (SMPL-X / BVH)
        │
        ▼
   GMR Retargeting ──────► Robot Motion References (.pkl/.npz)
                                    │
                                    ▼
                    Isaac Lab Training (AMP / DeepMimic)
                                    │
                                    ▼
                         Trained Policy (.pt / .onnx)
                                    │
                                    ▼
                    MuJoCo Deployment (human/deploy_a1_amp.py)
```

## License

MIT License

## Acknowledgements

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — Simulation framework
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) — RL training framework
- [GMR](https://github.com/YanjieZe/GMR) — General Motion Retargeting
- [MuJoCo](https://mujoco.org/) — Physics simulation for deployment
