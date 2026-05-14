"""Combined AMP + WBT deployment for A1 in MuJoCo.

Default: AMP locomotion with velocity commands (WASD).
Press '1': execute jump motion (WBT policy), auto-return to AMP when done.

Usage:
    python deploy_a1_combined.py [configs/a1_combined.yaml]
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
import yaml
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

from math_utils import pd_control


# ==================== Quaternion utilities ====================

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_apply(q, v):
    q_vec = q[1:]
    t = 2.0 * np.cross(q_vec, v)
    return v + q[0] * t + np.cross(q_vec, t)


def quat_apply_inverse(q, v):
    return quat_apply(quat_conjugate(q), v)


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def subtract_frame_transforms(t01, q01, t02, q02):
    q01_inv = quat_conjugate(q01)
    q_rel = quat_mul(q01_inv, q02)
    pos_rel = quat_apply(q01_inv, t02 - t01)
    return pos_rel, q_rel


def ori_6d_from_quat(q):
    mat = quat_to_rotmat(q)
    return mat[:, :2].flatten()


def yaw_quat(quat_wxyz):
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def compute_root_local_rot_tan_norm(quat_wxyz):
    yaw_q = yaw_quat(quat_wxyz)
    local_q = quat_mul(quat_conjugate(yaw_q), quat_wxyz)
    rotm = quat_to_rotmat(local_q)
    return np.concatenate([rotm[:, 0], rotm[:, 2]])


# ==================== AMP helpers ====================

class TermGroupedHistory:
    def __init__(self, term_dims: list, hist_len: int):
        self.term_dims = term_dims
        self.hist_len = hist_len
        self.buffers = [np.zeros((hist_len, dim), dtype=np.float32) for dim in term_dims]
        self.initialized = False

    def update(self, term_obs_list: list):
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


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_RIGHT_LEG_CHAIN = [
    (np.array([0.0, -0.021, -0.055]), None),
    (np.array([0.0, -0.04125, 0.0]), _rot_y),
    (np.array([0.0, -0.0987, 0.0]), _rot_x),
    (np.array([0.0, 0.0, -0.1215]), _rot_z),
    (np.array([0.0, 0.0, -0.112]), _rot_y),
    (np.array([0.0, 0.0, -0.221]), _rot_y),
    (np.array([0.0, 0.0, -0.05]), _rot_x),
]

_LEFT_LEG_CHAIN = [
    (np.array([0.0, 0.021, -0.055]), None),
    (np.array([0.0, 0.04125, 0.0]), _rot_y),
    (np.array([0.0, 0.0987, 0.0]), _rot_x),
    (np.array([-0.0001, 0.0, -0.122]), _rot_z),
    (np.array([0.0, 0.0, -0.1115]), _rot_y),
    (np.array([0.0, 0.0, -0.221]), _rot_y),
    (np.array([0.0, 0.0, -0.05]), _rot_x),
]


def _fk_foot_pos(joint_angles_6, chain):
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
    r6_pos = _fk_foot_pos(joint_angles_mj[:6], _RIGHT_LEG_CHAIN)
    l6_pos = _fk_foot_pos(joint_angles_mj[6:], _LEFT_LEG_CHAIN)
    return np.concatenate([l6_pos, r6_pos])


# ==================== Visualization ====================

def _arrow_rot_mat(direction):
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
    viewer.user_scn.ngeom = 0
    center = robot_pos.copy()
    center[2] += z_offset

    yaw_angle = np.arctan2(2.0 * (quat_wxyz[0] * quat_wxyz[3] + quat_wxyz[1] * quat_wxyz[2]),
                           1.0 - 2.0 * (quat_wxyz[2] ** 2 + quat_wxyz[3] ** 2))
    cos_y, sin_y = np.cos(yaw_angle), np.sin(yaw_angle)
    cmd_w = np.array([cos_y * cmd_body[0] - sin_y * cmd_body[1],
                      sin_y * cmd_body[0] + cos_y * cmd_body[1], 0.0])

    for vec, rgba in [
        (cmd_w * arrow_scale, np.array([0.0, 0.8, 0.0, 1.0])),
        (np.array([actual_vel_w[0], actual_vel_w[1], 0.0]) * arrow_scale,
         np.array([0.0, 0.3, 1.0, 1.0])),
    ]:
        length = np.linalg.norm(vec)
        if length < 0.01:
            continue
        size = np.array([0.015, 0.03, length / 2.0])
        idx = viewer.user_scn.ngeom
        mujoco.mjv_initGeom(viewer.user_scn.geoms[idx],
                            mujoco.mjtGeom.mjGEOM_ARROW,
                            size, center, _arrow_rot_mat(vec), rgba)
        viewer.user_scn.ngeom += 1


# ==================== WBT motion data loading ====================

def load_wbt_motion(policy_path):
    """Extract motion trajectory data from WBT ONNX model."""
    import onnx
    from onnx import numpy_helper

    onnx_model = onnx.load(policy_path)

    const_outputs = {}
    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.type == 4:
                    const_outputs[node.output[0]] = numpy_helper.to_array(attr.t)

    motion = {}
    for node in onnx_model.graph.node:
        if node.op_type == "Gather" and len(node.output) > 0:
            out_name = node.output[0]
            data_input = node.input[0]
            if out_name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                            "body_lin_vel_w", "body_ang_vel_w") \
               and data_input in const_outputs:
                motion[out_name] = const_outputs[data_input]

    assert "joint_pos" in motion, "Could not find joint_pos in ONNX graph"
    return motion


# ==================== Main ====================

if __name__ == "__main__":
    # Terminal input (non-blocking)
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

    MODE_AMP = 0
    MODE_WBT = 1
    current_mode = MODE_AMP

    x_vel = 0.0
    y_vel = 0.0
    yaw_cmd = 0.0
    reset_requested = False
    jump_requested = False
    quit_requested = False

    def restore_terminal():
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)

    def get_key():
        global x_vel, y_vel, yaw_cmd, reset_requested, jump_requested, quit_requested
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
            yaw_cmd -= 0.1
        elif ch == "a":
            yaw_cmd += 0.1
        elif ch == "l":
            y_vel -= 0.1
        elif ch == "j":
            y_vel += 0.1
        elif ch == " ":
            x_vel = y_vel = yaw_cmd = 0.0
        elif ch == "r":
            reset_requested = True
        elif ch == "1":
            jump_requested = True
        elif ch == "q":
            quit_requested = True

    # Load config
    config_file = "configs/a1_combined.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Shared params
    xml_path = config["xml_path"]
    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    num_actions = config["num_actions"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    effort_limits = np.array(config["effort_limits"], dtype=np.float32)

    obs_index_in_lab = config["obs_index_in_lab"]
    obs_index_in_mj = config["obs_index_in_mj"]
    action_index_in_lab = config["action_index_in_lab"]
    action_index_in_mj = config["action_index_in_mj"]

    mj2lab = [obs_index_in_mj.index(name) for name in obs_index_in_lab]
    lab2mj = [action_index_in_lab.index(name) for name in action_index_in_mj]
    torque_limits = effort_limits[lab2mj]

    # AMP params
    amp_policy_path = config["amp_policy_path"]
    amp_action_scale = config["amp_action_scale"]
    amp_num_obs = config["amp_num_obs"]
    amp_num_history = config["amp_num_history"]
    amp_obs_index = config["amp_obs_index"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    key_body_names = config["key_body_names"]

    obs_term_dims = {
        "base_ang_vel": 3,
        "root_local_rot_tan_norm": 6,
        "velocity_commands": 3,
        "joint_pos": num_actions,
        "joint_vel": num_actions,
        "last_action": num_actions,
        "key_body_pos_b": len(key_body_names) * 3,
    }
    amp_term_dims = [obs_term_dims[name] for name in amp_obs_index]

    # WBT params
    wbt_policy_path = config["wbt_policy_path"]
    wbt_action_scale = np.array(config["wbt_action_scale"], dtype=np.float32)
    wbt_num_obs = config["wbt_num_obs"]

    # Load AMP policy
    amp_policy = ort.InferenceSession(amp_policy_path)
    print(f"[AMP] Loaded: {amp_policy_path}")

    # Load WBT policy + motion data
    wbt_policy = ort.InferenceSession(wbt_policy_path)
    wbt_input_names = [inp.name for inp in wbt_policy.get_inputs()]
    print(f"[WBT] Loaded: {wbt_policy_path}")

    motion_data = load_wbt_motion(wbt_policy_path)
    motion_joint_pos = motion_data["joint_pos"]
    motion_joint_vel = motion_data["joint_vel"]
    motion_body_pos_w = motion_data["body_pos_w"]
    motion_body_quat_w = motion_data["body_quat_w"]
    time_step_total = motion_joint_pos.shape[0]
    print(f"[WBT] Motion length: {time_step_total} steps")

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    initial_qpos = d.qpos.copy()
    initial_qpos[7:] = default_angles[lab2mj]
    d.qpos[:] = initial_qpos
    mujoco.mj_forward(m, d)

    # State variables
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    amp_obs_hist = TermGroupedHistory(amp_term_dims, amp_num_history)
    amp_actions = np.zeros(num_actions, dtype=np.float32)
    wbt_actions = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles[lab2mj].copy()

    # WBT state
    wbt_time_step = 0
    wbt_quat_offset = np.array([1.0, 0.0, 0.0, 0.0])

    counter = 0

    print(f"\nControls: WASD=move, JL=strafe, Space=stop, 1=jump, R=reset, Q=quit")
    print(f"Mode: AMP (locomotion)\n")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            get_key()

            if quit_requested:
                break

            if reset_requested:
                d.qpos[:] = initial_qpos
                d.qvel[:] = 0.0
                d.ctrl[:] = 0.0
                cmd[:] = 0.0
                x_vel = y_vel = yaw_cmd = 0.0
                amp_actions[:] = 0.0
                wbt_actions[:] = 0.0
                target_dof_pos = default_angles[lab2mj].copy()
                amp_obs_hist = TermGroupedHistory(amp_term_dims, amp_num_history)
                current_mode = MODE_AMP
                counter = 0
                reset_requested = False
                mujoco.mj_forward(m, d)
                print("\n[Reset] Back to AMP mode")
                viewer.sync()
                continue

            # Handle jump request (only from AMP mode)
            if jump_requested and current_mode == MODE_AMP:
                jump_requested = False
                current_mode = MODE_WBT
                wbt_time_step = 0
                wbt_actions[:] = 0.0

                # Compute orientation offset so motion aligns with robot's current heading
                robot_quat = d.qpos[3:7].copy()
                motion_quat_0 = motion_body_quat_w[0, 0].copy()
                wbt_quat_offset = quat_mul(robot_quat, quat_conjugate(motion_quat_0))

                print("\n[Jump] WBT mode activated!")
            else:
                jump_requested = False

            cmd[0] = x_vel
            cmd[1] = y_vel
            cmd[2] = yaw_cmd

            # PD control + physics step
            tau = pd_control(
                target_dof_pos,
                d.qpos[7:],
                kps[lab2mj],
                np.zeros_like(kds),
                d.qvel[6:],
                kds[lab2mj],
            )
            tau = np.clip(tau, -torque_limits, torque_limits)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            counter += 1

            # Policy query at control frequency
            if counter % control_decimation == 0:
                quat_wxyz = d.qpos[3:7]
                root_pos = d.qpos[:3]
                omega_b = d.qvel[3:6].copy()
                qj = d.qpos[7:][mj2lab]
                dqj = d.qvel[6:][mj2lab]

                if current_mode == MODE_AMP:
                    # ---------- AMP policy ----------
                    term_obs_list = []
                    for idx in amp_obs_index:
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
                            term_obs_list.append(amp_actions.copy())
                        elif idx == "key_body_pos_b":
                            term_obs_list.append(compute_key_body_pos_b_fk(d.qpos[7:]))

                    total_obs = amp_obs_hist.update(term_obs_list)
                    obs_np = np.clip(total_obs, -18.0, 18.0).reshape(1, -1).astype(np.float32)

                    input_name = amp_policy.get_inputs()[0].name
                    outputs = amp_policy.run(None, {input_name: obs_np})
                    amp_actions = np.clip(outputs[0].squeeze(), -18.0, 18.0)

                    target_dof = amp_actions * amp_action_scale + default_angles
                    target_dof_pos = target_dof[lab2mj]

                    print(f"[AMP] vx:{cmd[0]:.2f} vy:{cmd[1]:.2f} yaw:{cmd[2]:.2f}\r", end="")

                else:
                    # ---------- WBT policy ----------
                    ref_joint_pos = motion_joint_pos[wbt_time_step]
                    ref_joint_vel = motion_joint_vel[wbt_time_step]
                    ref_body_quat_w = motion_body_quat_w[wbt_time_step]

                    command_obs = np.concatenate([ref_joint_pos, ref_joint_vel])

                    # Apply orientation offset to align motion with robot's heading at trigger
                    adjusted_anchor_quat = quat_mul(wbt_quat_offset, ref_body_quat_w[0])
                    _, anchor_quat_b = subtract_frame_transforms(
                        root_pos, quat_wxyz, root_pos, adjusted_anchor_quat
                    )
                    anchor_ori_b = ori_6d_from_quat(anchor_quat_b)

                    joint_pos_rel = qj - default_angles

                    obs = np.concatenate([
                        command_obs,          # 24
                        anchor_ori_b,         # 6
                        omega_b,              # 3
                        joint_pos_rel,        # 12
                        dqj,                  # 12
                        wbt_actions,          # 12
                    ]).astype(np.float32)
                    obs = np.clip(obs, -18.0, 18.0)

                    obs_input = obs.reshape(1, -1)
                    time_step_input = np.array([[wbt_time_step]], dtype=np.float32)

                    onnx_outputs = wbt_policy.run(None, {
                        wbt_input_names[0]: obs_input,
                        wbt_input_names[1]: time_step_input,
                    })
                    wbt_actions = np.clip(onnx_outputs[0].squeeze(), -18.0, 18.0)

                    target_dof = wbt_actions * wbt_action_scale + default_angles
                    target_dof_pos = target_dof[lab2mj]

                    wbt_time_step += 1
                    print(f"[WBT] t={wbt_time_step:3d}/{time_step_total}\r", end="")

                    # Motion complete -> switch back to AMP
                    if wbt_time_step >= time_step_total:
                        current_mode = MODE_AMP
                        amp_actions[:] = 0.0
                        amp_obs_hist = TermGroupedHistory(amp_term_dims, amp_num_history)
                        print("\n[Jump done] Back to AMP mode")

            draw_velocity_arrows(viewer, d.qpos[:3], d.qpos[3:7], cmd, d.qvel[:3])
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    restore_terminal()
    print("\nDone.")
