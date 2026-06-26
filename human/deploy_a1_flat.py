"""Sim2sim deployment of the pure-RL velocity policy (A1FlatEnvCfg) in MuJoCo.

PolicyCfg observation order (history_length=1, total 47):
    base_ang_vel(3) | projected_gravity(3) | velocity_commands(3) |
    joint_pos_rel(12) | joint_vel(12) | last_action(12) | gait_phase(2)
"""
import time
import sys
import os
import termios
import tty
import fcntl

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
import onnxruntime as ort

from math_utils import pd_control, get_gravity_orientation

# Sensor terms wrapped with delayed_obs in flat_env_cfg.py (commands/action/gait not delayed)
DELAYED_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")


def damiao_clip_effort(effort, joint_vel, y1, y2, x1, x2):
    """Replicate UnitreeActuator._clip_effort (torque-speed / T-N curve limiting).

    Strict port of unitree_actuators.py::UnitreeActuator._clip_effort / _compute_effort_limit:
      - same_direction (vel*effort > 0) -> max_effort = Y1 ; else Y2
      - |vel| < X1 -> keep max_effort (full peak torque)
      - |vel| >= X1 -> linear derate: k = -max_effort/(X2 - X1);
                       limit = k*(|vel| - X1) + max_effort, clipped to >= 0
      - return clip(effort, -max_effort, +max_effort)
    All args are per-joint numpy arrays in the SAME (MuJoCo) order as `effort`.
    """
    same_direction = (joint_vel * effort) > 0
    max_effort = np.where(same_direction, y1, y2)
    abs_vel = np.abs(joint_vel)
    # derated limit beyond the knee speed X1 (matches _compute_effort_limit)
    k = -max_effort / (x2 - x1)
    derated = np.clip(k * (abs_vel - x1) + max_effort, 0.0, None)
    max_effort = np.where(abs_vel < x1, max_effort, derated)
    return np.clip(effort, -max_effort, max_effort)


def damiao_apply_actuator(effort, joint_vel, tn, fric):
    """Replicate the full UnitreeActuator.compute() torque pipeline (in MuJoCo joint order):
      1. T-N curve clip of the PD effort   (UnitreeActuator._clip_effort)
      2. subtract friction AFTER the clip  (UnitreeActuator.compute):
            effort -= Fs*tanh(vel/Va) + Fd*vel
    `tn` and `fric` are dicts of per-joint numpy arrays (Y1,Y2,X1,X2 / Fs,Fd,Va), MuJoCo order.
    NOTE: friction is intentionally NOT re-clipped, matching the training-side ordering.
    """
    clipped = damiao_clip_effort(effort, joint_vel, tn["Y1"], tn["Y2"], tn["X1"], tn["X2"])
    clipped = clipped - (fric["Fs"] * np.tanh(joint_vel / fric["Va"]) + fric["Fd"] * joint_vel)
    return clipped


class ObsDelay:
    """Per-term observation latency, matching IsaacLab's delayed_obs: an independent integer lag
    (in control steps) is sampled in [min_delay, max_delay] once per episode and held constant."""

    def __init__(self, term_names, min_delay, max_delay):
        self.term_names = set(term_names)
        self.min_delay = int(min_delay)
        self.max_delay = int(max_delay)
        self.reset()

    def reset(self):
        self.buffers = {n: [] for n in self.term_names}
        self.lags = {n: np.random.randint(self.min_delay, self.max_delay + 1) for n in self.term_names}

    def apply(self, name, value):
        if name not in self.term_names or self.max_delay <= 0:
            return value
        buf = self.buffers[name]
        buf.append(value.copy())
        if len(buf) > self.max_delay + 1:
            buf.pop(0)
        return buf[max(0, len(buf) - 1 - self.lags[name])]


class TermGroupedHistory:
    """Per-term history matching IsaacLab (interleave_by_time=False): concatenate each term's full
    history first, then concatenate all terms -> [t1_h0..t1_hN | t2_h0..t2_hN | ...] (h0 oldest)."""

    def __init__(self, term_dims, hist_len):
        self.hist_len = hist_len
        self.buffers = [np.zeros((hist_len, d), dtype=np.float32) for d in term_dims]
        self.initialized = False

    def update(self, term_obs_list):
        for i, obs in enumerate(term_obs_list):
            if not self.initialized:
                self.buffers[i][:] = obs
            else:
                self.buffers[i][:-1] = self.buffers[i][1:]
                self.buffers[i][-1] = obs
        self.initialized = True
        return np.concatenate([b.flatten() for b in self.buffers])


# ---------------------------------------------------------------- keyboard ctrl
fd = sys.stdin.fileno()
old_term = termios.tcgetattr(fd)
tty.setcbreak(fd)
old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

x_vel = y_vel = yaw = 0.0
reset_requested = False


def get_key():
    global x_vel, y_vel, yaw, reset_requested
    try:
        ch = sys.stdin.read(1)
    except IOError:
        return
    if not ch:
        return
    if ch == "w":
        x_vel += 0.1
    elif ch == "s":
        x_vel -= 0.1
    elif ch == "d":
        yaw -= 0.1
    elif ch == "a":
        yaw += 0.1
    elif ch == "l":
        y_vel -= 0.1
    elif ch == "j":
        y_vel += 0.1
    elif ch == " ":
        x_vel = y_vel = yaw = 0.0
    elif ch == "r":
        reset_requested = True


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/a1_flat.yaml"
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"]
    xml_path = config["xml_path"]
    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)  # Lab order
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions = config["num_actions"]
    num_history = config["num_history"]
    gait_period = config["gait_period"]
    step_dt = config["step_dt"]
    model_type = config["model_type"]
    obs_index = config["obs_index"]

    # Lab <-> MuJoCo joint-order maps
    mj2lab = [config["obs_index_in_mj"].index(n) for n in config["obs_index_in_lab"]]
    lab2mj = [config["action_index_in_lab"].index(n) for n in config["action_index_in_mj"]]

    # Effort limits (IsaacLab order) -> MuJoCo order
    effort_limits = np.array([27, 27, 27, 27, 27, 27, 27, 27, 7, 7, 7, 7], dtype=np.float32)
    torque_limits = effort_limits[lab2mj]

    # --- Damiao actuator model (T-N derate + friction), see use_damiao_actuator_model switch ---
    use_damiao = bool(config.get("use_damiao_actuator_model", False))
    damiao_tn = damiao_fric = None
    if use_damiao:
        # Build per-joint params in IsaacLab order, then map to MuJoCo order (like torque_limits).
        # Lab order = [L1,R1,L2,R2,L3,R3,L4,R4,L5,R5,L6,R6]:
        #   indices 0-7  -> hip/knee (joints *1-*4) = DM4340
        #   indices 8-11 -> ankle    (joints *5-*6) = DM4310
        hk, ak = config["damiao_tn"]["hip_knee"], config["damiao_tn"]["ankle"]
        hkf, akf, va = (config["damiao_friction"]["hip_knee"], config["damiao_friction"]["ankle"],
                        config["damiao_friction"]["Va"])

        def _lab_group_array(hip_knee_val, ankle_val):
            arr = np.empty(12, dtype=np.float32)
            arr[0:8] = hip_knee_val   # hip/knee joints in Lab order
            arr[8:12] = ankle_val     # ankle joints in Lab order
            return arr[lab2mj]        # -> MuJoCo order

        damiao_tn = {
            "Y1": _lab_group_array(hk["Y1"], ak["Y1"]),
            "Y2": _lab_group_array(hk["Y2"], ak["Y2"]),
            "X1": _lab_group_array(hk["X1"], ak["X1"]),
            "X2": _lab_group_array(hk["X2"], ak["X2"]),
        }
        damiao_fric = {
            "Fs": _lab_group_array(hkf["Fs"], akf["Fs"]),
            "Fd": _lab_group_array(hkf["Fd"], akf["Fd"]),
            "Va": np.float32(va),
        }

    cmd = np.array(config["cmd_init"], dtype=np.float32)
    actions = np.zeros(num_actions, dtype=np.float32)  # Lab order
    target_dof_pos = default_angles[lab2mj].copy()
    obs_delay = ObsDelay(DELAYED_TERMS, config["min_delay_steps"], config["max_delay_steps"])
    # action/setpoint execution delay (single signal, in CONTROL steps) -- reuse ObsDelay with one "term"
    action_delay = ObsDelay(("action",), config["action_min_delay_steps"], config["action_max_delay_steps"])

    # per-term dims (must match obs_index), for history stacking
    term_dims_map = {
        "base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3,
        "joint_pos": num_actions, "joint_vel": num_actions, "last_action": num_actions,
        "gait_phase": 2,
    }
    term_dims = [term_dims_map[n] for n in obs_index]
    obs_hist = TermGroupedHistory(term_dims, num_history)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    initial_qpos = d.qpos.copy()
    initial_qpos[7:] = default_angles[lab2mj]
    d.qpos[:] = initial_qpos
    mujoco.mj_forward(m, d)

    if model_type == "jit":
        policy = torch.jit.load(policy_path)
    else:
        if not policy_path.endswith(".onnx"):
            policy_path += ".onnx"
        policy = ort.InferenceSession(policy_path)
    print("policy loaded",policy_path)
    counter = 0          # sim steps
    ctrl_steps = 0       # policy steps (advances gait clock)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            if reset_requested:
                d.qpos[:] = initial_qpos
                d.qvel[:] = 0.0
                d.ctrl[:] = 0.0
                cmd[:] = 0.0
                x_vel = y_vel = yaw = 0.0
                actions[:] = 0.0
                target_dof_pos = default_angles[lab2mj].copy()
                counter = ctrl_steps = 0
                reset_requested = False
                obs_delay.reset()
                action_delay.reset()
                obs_hist = TermGroupedHistory(term_dims, num_history)
                mujoco.mj_forward(m, d)
                viewer.sync()
                continue

            get_key()
            cmd[0], cmd[1], cmd[2] = x_vel, y_vel, yaw

            # PD control (MuJoCo order). NOTE: yaml kps/kds are ALREADY in MuJoCo joint order
            # (R1..R6,L1..L6 -> ankle gains at idx 4,5,10,11), same order as d.qpos[7:]/d.qvel[6:],
            # so they are used directly WITHOUT [lab2mj] (a second permutation here was a bug that
            # mis-mapped gains: hip-yaw R3/L3 got the ankle kp=7 and ankle-pitch R5/L5 got 126).
            tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                             np.zeros_like(kds), d.qvel[6:], kds)
            if use_damiao:
                # Replicate training-side UnitreeActuator: T-N curve derate + friction (MuJoCo order)
                tau = damiao_apply_actuator(tau, d.qvel[6:], damiao_tn, damiao_fric)
            else:
                # Legacy behavior: fixed torque clip (+-27 / +-7), no T-N derate, no friction
                tau = np.clip(tau, -torque_limits, torque_limits)
            
            d.ctrl = tau

            print(f"x_vel:{cmd[0]:.2f}  y_vel:{cmd[1]:.2f}  yaw:{cmd[2]:.2f}\r", end="")
            mujoco.mj_step(m, d)
            counter += 1

            if counter % control_decimation == 0:
                quat_wxyz = d.qpos[3:7]
                qj = d.qpos[7:][mj2lab] - default_angles  # joint_pos_rel (Lab order)
                dqj = d.qvel[6:][mj2lab]                   # joint_vel (Lab order)
                omega_b = d.qvel[3:6].copy()
                phi = (ctrl_steps * step_dt) % gait_period / gait_period
                angle = 2.0 * np.pi * phi

                terms = {
                    "base_ang_vel": omega_b,
                    "projected_gravity": get_gravity_orientation(quat_wxyz),
                    "velocity_commands": cmd * cmd_scale,
                    "joint_pos": qj,
                    "joint_vel": dqj,
                    "last_action": actions.copy(),
                    "gait_phase": np.array([np.sin(angle), np.cos(angle)], dtype=np.float32),
                }                
                term_obs_list = [obs_delay.apply(name, terms[name]) for name in obs_index]
                total_obs = obs_hist.update(term_obs_list).astype(np.float32)
                obs_tensor = torch.from_numpy(total_obs).unsqueeze(0)
                if model_type == "jit":
                    actions = policy(obs_tensor).detach().cpu().numpy().squeeze()
                else:
                    input_name = policy.get_inputs()[0].name
                    actions = policy.run(None, {input_name: obs_tensor.numpy()})[0].squeeze()

                # actions (Lab order) -> PD target (MuJoCo order)
                target_dof_pos = (actions * action_scale + default_angles)[lab2mj]
                # delay the joint-position setpoint by a per-episode random lag IN CONTROL STEPS,
                # matching training's DelayedPDActuatorCfg (delays the setpoint, then PD runs each sim step)
                joint_lower_limits = np.array([-1.05, -0.26, -1.0, 0.0, -0.52, -0.35,
                     -1.05, -1.05, -1.0, 0.0, -0.52, -0.35])
                joint_upper_limits= np.array([ 1.05,  1.05,  1.0, 1.92, 0.52,  0.35,
                      1.05,  0.26,  1.0, 1.92, 0.52,  0.35])
                target_dof_pos = np.clip(target_dof_pos, joint_lower_limits, joint_upper_limits)
                print(target_dof_pos)
                target_dof_pos = action_delay.apply("action", target_dof_pos)
                # target_dof_pos=default_angles[lab2mj].copy()
                ctrl_steps += 1

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
