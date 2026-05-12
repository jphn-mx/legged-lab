import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

from math_utils import pd_control

import sys
import termios
import tty
import fcntl
import os


def normalize_policy_output(policy_output):
    if isinstance(policy_output, torch.Tensor):
        return policy_output
    if isinstance(policy_output, np.ndarray):
        return torch.from_numpy(policy_output)
    if isinstance(policy_output, (list, tuple)):
        for item in policy_output:
            if isinstance(item, torch.Tensor):
                return item
            if isinstance(item, np.ndarray):
                return torch.from_numpy(item)
        raise TypeError(f"Policy output sequence does not contain a tensor or ndarray: {type(policy_output)}")
    raise TypeError(f"Unsupported policy output type: {type(policy_output)}")


class TermGroupedHistory:
    """Per-term history buffers matching IsaacLab's default (interleave_by_time=False).

    IsaacLab concatenates each term's full history first, then concatenates all terms:
      [term1_t0|term1_t1|...|term1_tN | term2_t0|...|term2_tN | ...]
    """

    def __init__(self, term_dims: list[int], hist_len: int):
        self.term_dims = term_dims
        self.hist_len = hist_len
        self.buffers = [np.zeros((hist_len, dim), dtype=np.float32) for dim in term_dims]
        self.initialized = False

    def update(self, term_obs_list: list[np.ndarray]):
        if not self.initialized:
            for i, obs in enumerate(term_obs_list):
                self.buffers[i][:] = obs
            self.initialized = True
        else:
            for i, obs in enumerate(term_obs_list):
                self.buffers[i][:-1] = self.buffers[i][1:]
                self.buffers[i][-1] = obs
        return np.concatenate([buf.flatten() for buf in self.buffers])

    def reset(self):
        for buf in self.buffers:
            buf[:] = 0.0
        self.initialized = False


def yaw_quat(quat_wxyz):
    """Extract yaw-only quaternion from full quaternion (w,x,y,z)."""
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def quat_conjugate(q):
    """Conjugate of quaternion (w,x,y,z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    """Multiply two quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rotmat(q):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_apply_inverse(q, v):
    """Apply inverse quaternion rotation to vector: q^{-1} * v."""
    q_conj = quat_conjugate(q)
    return R.from_quat([q_conj[1], q_conj[2], q_conj[3], q_conj[0]]).apply(v)


def compute_root_local_rot_tan_norm(quat_wxyz):
    """Compute root_local_rot_tan_norm: remove yaw, extract tangent and normal vectors."""
    yaw_q = yaw_quat(quat_wxyz)
    local_q = quat_mul(quat_conjugate(yaw_q), quat_wxyz)
    rotm = quat_to_rotmat(local_q)
    tan_vec = rotm[:, 0]
    norm_vec = rotm[:, 2]
    return np.concatenate([tan_vec, norm_vec])


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Kinematic chains from MuJoCo XML: (offset_from_parent, rotation_func or None for fixed)
_RIGHT_LEG_CHAIN = [
    (np.array([0.0, -0.021, -0.055]),   None),    # R0 fixed
    (np.array([0.0, -0.04125, 0.0]),     _rot_y),  # R1 axis=0 1 0
    (np.array([0.0, -0.0987, 0.0]),      _rot_x),  # R2 axis=1 0 0
    (np.array([0.0,  0.0, -0.1215]),     _rot_z),  # R3 axis=0 0 1
    (np.array([0.0,  0.0, -0.112]),      _rot_y),  # R4 axis=0 1 0
    (np.array([0.0,  0.0, -0.221]),      _rot_y),  # R5 axis=0 1 0
    (np.array([0.0,  0.0, -0.05]),       _rot_x),  # R6 axis=1 0 0
]

_LEFT_LEG_CHAIN = [
    (np.array([0.0, 0.021, -0.055]),     None),    # L0 fixed
    (np.array([0.0, 0.04125, 0.0]),      _rot_y),  # L1 axis=0 1 0
    (np.array([0.0, 0.0987, 0.0]),       _rot_x),  # L2 axis=1 0 0
    (np.array([-0.0001, 0.0, -0.122]),   _rot_z),  # L3 axis=0 0 1
    (np.array([0.0,  0.0, -0.1115]),     _rot_y),  # L4 axis=0 1 0
    (np.array([0.0,  0.0, -0.221]),      _rot_y),  # L5 axis=0 1 0
    (np.array([0.0,  0.0, -0.05]),       _rot_x),  # L6 axis=1 0 0
]


def _fk_foot_pos(joint_angles_6, chain):
    """Forward kinematics: compute last link position in base frame."""
    rot = np.eye(3)
    pos = np.zeros(3)
    j = 0
    for offset, rot_func in chain:
        pos = pos + rot @ offset
        if rot_func is not None:
            rot = rot @ rot_func(joint_angles_6[j])
            j += 1
    return pos


def compute_key_body_pos_b_fk(joint_angles_mj):
    """Compute key body positions in base frame using FK (no simulator needed).

    Args:
        joint_angles_mj: 12 joint angles in MuJoCo order [R1..R6, L1..L6]
    Returns:
        Concatenated [L6_pos, R6_pos] (6D), matching key_body_names=["Link_L6","Link_R6"]
    """
    r6_pos = _fk_foot_pos(joint_angles_mj[:6], _RIGHT_LEG_CHAIN)
    l6_pos = _fk_foot_pos(joint_angles_mj[6:], _LEFT_LEG_CHAIN)
    return np.concatenate([l6_pos, r6_pos])


def _arrow_rot_mat(direction):
    """Build a 3x3 rotation matrix whose Z-axis aligns with *direction*."""
    d = np.asarray(direction, dtype=np.float64)
    length = np.linalg.norm(d)
    if length < 1e-8:
        return np.eye(3).flatten()
    z = d / length
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]).flatten()


def draw_velocity_arrows(viewer, robot_pos, quat_wxyz, cmd_body, actual_vel_w,
                         arrow_scale=0.5, z_offset=0.5):
    """Draw command (green) and actual velocity (blue) arrows above the robot.

    Arrows are centered directly above the robot and elongate symmetrically.
    """
    viewer.user_scn.ngeom = 0
    center = robot_pos.copy()
    center[2] += z_offset

    yaw = np.arctan2(2.0 * (quat_wxyz[0] * quat_wxyz[3] + quat_wxyz[1] * quat_wxyz[2]),
                     1.0 - 2.0 * (quat_wxyz[2] ** 2 + quat_wxyz[3] ** 2))
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cmd_w = np.array([cos_y * cmd_body[0] - sin_y * cmd_body[1],
                      sin_y * cmd_body[0] + cos_y * cmd_body[1],
                      0.0])

    for vec, rgba in [
        (cmd_w * arrow_scale, np.array([0.0, 0.8, 0.0, 1.0])),
        (np.array([actual_vel_w[0], actual_vel_w[1], 0.0]) * arrow_scale,
         np.array([0.0, 0.3, 1.0, 1.0])),
    ]:
        length = np.linalg.norm(vec)
        if length < 0.01:
            continue
        half = length / 2.0
        size = np.array([0.015, 0.03, half])
        idx = viewer.user_scn.ngeom
        mujoco.mjv_initGeom(viewer.user_scn.geoms[idx],
                            mujoco.mjtGeom.mjGEOM_ARROW,
                            size, center, _arrow_rot_mat(vec), rgba)
        viewer.user_scn.ngeom += 1


# Terminal setup
fd = sys.stdin.fileno()
old_term = termios.tcgetattr(fd)
tty.setcbreak(fd)
old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

x_vel = 0.0
y_vel = 0.0
yaw = 0.0
reset_requested = False


def restore_terminal():
    termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)


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
    config_file = "configs/a1_amp.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"]
    xml_path = config["xml_path"]
    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    num_history = config["num_history"]
    model_type = config["model_type"]
    obs_index = config["obs_index"]
    key_body_names = config["key_body_names"]

    obs_index_in_lab = config["obs_index_in_lab"]
    obs_index_in_mj = config["obs_index_in_mj"]
    action_index_in_lab = config["action_index_in_lab"]
    action_index_in_mj = config["action_index_in_mj"]

    mj2lab = [obs_index_in_mj.index(name) for name in obs_index_in_lab]
    lab2mj = [action_index_in_lab.index(name) for name in action_index_in_mj]
    print("mj2lab =", mj2lab)
    print("lab2mj =", lab2mj)

    # Effort limits per joint in IsaacLab order, converted to MuJoCo order
    effort_limits = np.array([27, 27, 27, 27, 27, 27, 27, 27, 7, 7, 7, 7], dtype=np.float32)
    torque_limits = effort_limits[lab2mj]

    cmd = np.array(config["cmd_init"], dtype=np.float32)

    # Dimension of each observation term (must match obs_index order)
    obs_term_dims = {
        "base_ang_vel": 3,
        "root_local_rot_tan_norm": 6,
        "velocity_commands": 3,
        "joint_pos": num_actions,
        "joint_vel": num_actions,
        "last_action": num_actions,
        "key_body_pos_b": len(key_body_names) * 3,
    }
    term_dims = [obs_term_dims[name] for name in obs_index]

    obs_hist = TermGroupedHistory(term_dims, num_history)
    actions = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles[lab2mj].copy()

    print(f"xml_path = {xml_path}")
    print(f"num_actions = {num_actions}, num_obs = {num_obs}, history = {num_history}")
    print(f"total policy input dim = {num_obs * num_history}")

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    initial_qpos = d.qpos.copy()
    initial_qpos[7:] = default_angles[lab2mj]
    d.qpos[:] = initial_qpos
    mujoco.mj_forward(m, d)
    initial_qvel = d.qvel.copy()

    if model_type == "jit":
        policy = torch.jit.load(policy_path)
        print(f"Loaded JIT model from {policy_path}")
    elif model_type == "onnx":
        if not policy_path.endswith(".onnx"):
            policy_path += ".onnx"
        policy = ort.InferenceSession(policy_path)
        print(f"Loaded ONNX model from {policy_path}")

    counter = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            if reset_requested:
                d.qpos[:] = initial_qpos[:]
                d.qvel[:] = 0.0
                d.ctrl[:] = 0.0
                d.qacc[:] = 0.0
                d.qacc_warmstart[:] = 0.0
                d.xfrc_applied[:] = 0.0
                cmd[:] = 0.0
                x_vel = y_vel = yaw = 0.0
                actions[:] = 0.0
                target_dof_pos = default_angles[lab2mj].copy()
                obs_hist = TermGroupedHistory(term_dims, num_history)
                counter = 0
                reset_requested = False
                print("Robot state reset")
                mujoco.mj_forward(m, d)
                viewer.sync()
                continue

            get_key()
            cmd[0] = x_vel
            cmd[1] = y_vel
            cmd[2] = yaw

            # PD control (in MuJoCo order)
            tau = pd_control(
                target_dof_pos,
                d.qpos[7:],
                kps[lab2mj],
                np.zeros_like(kds),
                d.qvel[6:],
                kds[lab2mj],
            )
            tau = np.clip(tau, -torque_limits, torque_limits)
            d.ctrl = tau

            print(f"x_vel:{cmd[0]:.2f}  y_vel:{cmd[1]:.2f}  yaw:{cmd[2]:.2f}\r", end="")
            mujoco.mj_step(m, d)
            counter += 1

            if counter % control_decimation == 0:
                # Read MuJoCo state, reorder to Lab order
                qj = d.qpos[7:][mj2lab]  # absolute joint positions in Lab order
                dqj = d.qvel[6:][mj2lab]  # joint velocities in Lab order

                # Root state (MuJoCo quat is w,x,y,z)
                quat_wxyz = d.qpos[3:7]
                root_pos = d.qpos[:3]

                # MuJoCo free joint: qvel[3:6] is angular velocity in body frame
                omega_b = d.qvel[3:6].copy()

                # Build per-term observations following AMP order
                term_obs_list = []
                for idx in obs_index:
                    if idx == "base_ang_vel":
                        term_obs_list.append(omega_b)
                    elif idx == "root_local_rot_tan_norm":
                        term_obs_list.append(compute_root_local_rot_tan_norm(quat_wxyz))
                    elif idx == "velocity_commands":
                        term_obs_list.append(cmd * cmd_scale)
                    elif idx == "joint_pos":
                        term_obs_list.append(qj)
                    elif idx == "joint_vel":
                        term_obs_list.append(dqj)
                    elif idx == "last_action":
                        term_obs_list.append(actions.copy())
                    elif idx == "key_body_pos_b":
                        term_obs_list.append(compute_key_body_pos_b_fk(d.qpos[7:]))

                total_obs = obs_hist.update(term_obs_list)

                obs_tensor = torch.clip(torch.from_numpy(total_obs).unsqueeze(0).float(), -18.0, 18.0)

                if model_type == "jit":
                    policy_output = normalize_policy_output(policy(obs_tensor))
                    actions = torch.clip(policy_output, -18.0, 18.0).detach().cpu().numpy().squeeze()
                elif model_type == "onnx":
                    input_name = policy.get_inputs()[0].name
                    obs_np = obs_tensor.numpy()
                    outputs = policy.run(None, {input_name: obs_np})
                    actions = np.clip(outputs[0], -18.0, 18.0).squeeze()

                # Actions are in Lab order, convert to MuJoCo order for PD target
                target_dof = actions * action_scale + default_angles
                target_dof_pos = target_dof[lab2mj]

            draw_velocity_arrows(viewer, d.qpos[:3], d.qpos[3:7], cmd, d.qvel[:3])
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    restore_terminal()
