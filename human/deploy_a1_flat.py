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

    cmd = np.array(config["cmd_init"], dtype=np.float32)
    actions = np.zeros(num_actions, dtype=np.float32)  # Lab order
    target_dof_pos = default_angles[lab2mj].copy()
    obs_delay = ObsDelay(DELAYED_TERMS, config["min_delay_steps"], config["max_delay_steps"])

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
                obs_hist = TermGroupedHistory(term_dims, num_history)
                mujoco.mj_forward(m, d)
                viewer.sync()
                continue

            get_key()
            cmd[0], cmd[1], cmd[2] = x_vel, y_vel, yaw

            # PD control (MuJoCo order)
            tau = pd_control(target_dof_pos, d.qpos[7:], kps[lab2mj],
                             np.zeros_like(kds), d.qvel[6:], kds[lab2mj])
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
                ctrl_steps += 1

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
