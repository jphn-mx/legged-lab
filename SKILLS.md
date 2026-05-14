# Robot Skills

This document describes the locomotion skills that can be trained and deployed on legged robots in this project.

## Supported Robots

| Robot | Type | DOF | Skills Available |
|-------|------|-----|-----------------|
| Unitree A1-legs-V1 | Bipedal | 12 | AMP Walking, Velocity Tracking, Whole-Body Tracking |
| Unitree G1 | Humanoid | 29 | AMP Walking, DeepMimic, Velocity Tracking |
| Unitree Go2 | Quadruped | 12 | Velocity Tracking |

---

## Skill 1: AMP Locomotion (Adversarial Motion Priors)

**Method:** Style-transfer via adversarial discriminator trained on motion capture demonstrations.

**Description:**  
The robot learns natural walking gaits by imitating human/animal motion data. A discriminator network distinguishes between the robot's behavior and reference motion clips, providing a style reward signal in addition to task rewards (velocity tracking).

**Available Motions (A1 Bipedal):**

| Motion | Source | Description |
|--------|--------|-------------|
| walk1_subject5_walk1 | OptiTrack | Forward walking |
| walk1_subject5_walk2 | OptiTrack | Forward walking variant |
| walk1_subject5_turn1 | OptiTrack | Walking with turns |
| Walk_turn_left_45 | ACCAD (CMU) | 45° left turn while walking |
| Walk_turn_right_45 | ACCAD (CMU) | 45° right turn while walking |
| Walk_turn_around | ACCAD (CMU) | 180° turn while walking |
| Side_step_left | ACCAD (CMU) | Lateral stepping left |
| Side_step_right | ACCAD (CMU) | Lateral stepping right |
| Stand_to_Walk_Back | ACCAD (CMU) | Backward walking |
| Walk_to_hop_to_walk | ACCAD (CMU) | Hopping transition |
| Walk_to_leap_to_walk | ACCAD (CMU) | Leaping transition |
| Walk_to_crouch | ACCAD (CMU) | Crouching transition |
| Walk_to_skip | ACCAD (CMU) | Skipping gait |

**Observation Space:**
- Base angular velocity (3D)
- Root local rotation (tangent + normal vectors, 6D)
- Velocity commands (vx, vy, yaw_rate)
- Joint positions (12D)
- Joint velocities (12D)
- Last action (12D)
- Key body positions in base frame (optional, 6D for feet)

**Command Space:**
- Linear velocity X: [-0.5, 1.2] m/s
- Linear velocity Y: [-0.6, 0.6] m/s
- Angular velocity Z: [-1.2, 1.2] rad/s

**Training:**
```bash
cd legged_lab
python scripts/rsl_rl/train.py --task Lab-Locomotion-Amp-A1-v0 --num_envs 4096
```

---

## Skill 2: Velocity Tracking

**Method:** Reward shaping for command-conditioned locomotion with curriculum learning.

**Description:**  
The robot learns to follow velocity commands (forward, lateral, yaw) using exponential tracking rewards, regularization penalties, and optional terrain curriculum.

**Reward Components:**
- `track_lin_vel_xy_exp` — Exponential tracking of XY velocity commands
- `track_ang_vel_z_exp` — Exponential tracking of yaw rate command
- `flat_orientation_l2` — Penalize non-flat body orientation
- `lin_vel_z_l2` — Penalize vertical bouncing
- `action_rate_l2` — Smooth actions
- `feet_slide` — Penalize foot sliding during contact
- `stand_still` — Minimize joint deviation when zero command

**Training:**
```bash
cd legged_lab
# G1 Humanoid
python scripts/rsl_rl/train.py --task Lab-Locomotion-Velocity-Flat-G1-v0 --num_envs 4096
# Go2 Quadruped
python scripts/rsl_rl/train.py --task Lab-Locomotion-Velocity-Flat-Go2-v0 --num_envs 4096
```

---

## Skill 3: DeepMimic (Reference Motion Imitation)

**Method:** Phase-based motion imitation with dense tracking rewards.

**Description:**  
The robot precisely tracks reference motion trajectories frame-by-frame. A phase variable synchronizes the policy with the motion clip, enabling highly accurate reproduction of specific motions.

**Tracking Rewards:**
- Joint position tracking
- Joint velocity tracking
- Key body position tracking
- Root orientation tracking

**Training:**
```bash
cd legged_lab
python scripts/rsl_rl/train.py --task Lab-Locomotion-Deepmimic-G1-v0 --num_envs 4096
```

---

## Skill 4: Whole-Body Motion Tracking

**Method:** Full-body kinematic tracking with anchor-relative body positioning.

**Description:**  
The robot tracks pre-recorded whole-body motion sequences specified as body positions, orientations, and velocities. An anchor body (pelvis/base) defines the global reference, and all other bodies are tracked relative to it.

**Tracking Objectives:**
- `motion_global_anchor_position_error_exp` — Track anchor (root) position in world
- `motion_global_anchor_orientation_error_exp` — Track anchor orientation
- `motion_relative_body_position_error_exp` — Track limb positions relative to anchor
- `motion_relative_body_orientation_error_exp` — Track limb orientations relative to anchor
- `motion_global_body_linear_velocity_error_exp` — Track body linear velocities

**Motion Data Format (npz):**
```python
{
    "fps": 50,                    # Playback framerate
    "joint_pos": (T, N_joints),   # Joint angles per frame
    "joint_vel": (T, N_joints),   # Joint velocities per frame
    "body_pos_w": (T, N_bodies, 3),   # Body positions in world
    "body_quat_w": (T, N_bodies, 4),  # Body orientations (w,x,y,z)
    "body_lin_vel_w": (T, N_bodies, 3),  # Body linear velocities
    "body_ang_vel_w": (T, N_bodies, 3),  # Body angular velocities
}
```

**Training:**
```bash
cd whole_body_tracking
# A1 bipedal
python scripts/rsl_rl/train.py --task Tracking-Flat-A1-v0 --num_envs 4096 \
    --motion_name a1-walk-turn-left-45
# G1 humanoid
python scripts/rsl_rl/train.py --task Tracking-Flat-G1-v0 --num_envs 4096
```

---

## Skill Pipeline: From Human Motion to Robot Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Motion Capture                                              │
│     - OptiTrack / Xsens / AMASS dataset                         │
│     - Formats: SMPL-X, BVH, C3D/CSV                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Motion Retargeting (GMR)                                    │
│     - IK-based human → robot mapping                            │
│     - Automatic height scaling                                  │
│     - Output: robot joint trajectories (.pkl)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Data Conversion                                             │
│     - pkl_to_csv.py (GMR pkl → CSV intermediate)                │
│     - csv_to_npz.py (CSV → npz for whole-body tracking)        │
│     - gmr_to_lab.py (pkl → npz for AMP/DeepMimic)              │
│     - Frame rate resampling & range selection                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. RL Training (Isaac Lab + RSL-RL)                            │
│     - AMP: style reward from discriminator                      │
│     - DeepMimic: dense tracking reward                          │
│     - Velocity: command-conditioned with curriculum             │
│     - Whole-Body: anchor-relative multi-body tracking           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Policy Export & Deployment (MuJoCo)                         │
│     - Export: .pt (TorchScript) / .onnx                         │
│     - deploy_a1_amp.py: velocity locomotion (keyboard teleop)   │
│     - deploy_a1_wbt.py: motion playback (trajectory tracking)   │
│     - deploy_a1_combined.py: multi-skill (AMP + WBT switching)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deployment Modes (MuJoCo)

### Mode 1: AMP Locomotion (`deploy_a1_amp.py`)

Velocity-commanded walking with keyboard teleoperation.

| Key | Action |
|-----|--------|
| W / S | Increase / decrease forward velocity |
| A / D | Increase / decrease yaw rate |
| J / L | Increase / decrease lateral velocity |
| Space | Stop (zero all commands) |
| R | Reset robot to default pose |

Velocity arrows are rendered above the robot:
- **Green arrow** — commanded velocity direction
- **Blue arrow** — actual velocity direction

### Mode 2: Whole-Body Tracking (`deploy_a1_wbt.py`)

Autonomous motion playback — the robot executes a pre-recorded trajectory (e.g., jump) frame by frame using the WBT policy.

| Key | Action |
|-----|--------|
| R | Reset robot and replay from beginning |
| Q | Quit |

The policy takes `(obs, time_step)` as input and outputs actions along with reference trajectory data (joint positions, body positions/orientations).

### Mode 3: Combined AMP + WBT (`deploy_a1_combined.py`)

Multi-skill deployment — AMP locomotion as the base skill, with WBT motions triggered on demand.

| Key | Action |
|-----|--------|
| W / S / A / D / J / L / Space | AMP velocity control (same as Mode 1) |
| 1 | Execute jump motion (WBT policy), auto-returns to AMP |
| R | Reset robot to default pose |

This demonstrates skill composition: the robot walks using AMP, then seamlessly transitions to a WBT motion (jump) and back.
