"""Deploy whole_body_tracking (DeepMimic) policy for A1 in MuJoCo.

The ONNX model takes (obs, time_step) and returns (actions, ref_joint_pos, ref_joint_vel,
ref_body_pos_w, ref_body_quat_w, ref_body_lin_vel_w, ref_body_ang_vel_w).

Usage:
    python deploy_a1_wbt.py [configs/a1_wbt.yaml]
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

from math_utils import pd_control


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


def quat_apply(q, v):
    """Rotate vector v by quaternion q (w,x,y,z)."""
    q_vec = q[1:]
    t = 2.0 * np.cross(q_vec, v)
    return v + q[0] * t + np.cross(q_vec, t)


def quat_apply_inverse(q, v):
    """Rotate vector v by inverse of quaternion q (w,x,y,z)."""
    return quat_apply(quat_conjugate(q), v)


def matrix_from_quat(q):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def subtract_frame_transforms(t01, q01, t02, q02):
    """Compute transform of frame 2 relative to frame 1.

    Given world-frame transforms (t01, q01) for frame1 and (t02, q02) for frame2,
    returns (pos_1to2_in_frame1, quat_1to2_in_frame1).

    This matches isaaclab.utils.math.subtract_frame_transforms.
    """
    q01_inv = quat_conjugate(q01)
    q_rel = quat_mul(q01_inv, q02)
    pos_rel = quat_apply(q01_inv, t02 - t01)
    return pos_rel, q_rel


def ori_6d_from_quat(q):
    """Convert quaternion to 6D orientation (first 2 columns of rotation matrix).

    Matches Isaac Lab: mat[..., :2].reshape(-1) which is row-major flattening.
    """
    mat = matrix_from_quat(q)
    return mat[:, :2].flatten()


# ---------- Main ----------
if __name__ == "__main__":
    # ---------- Terminal input (non-blocking) ----------
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

    reset_requested = False
    quit_requested = False

    def restore_terminal():
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)

    def get_key():
        global reset_requested, quit_requested
        try:
            ch = sys.stdin.read(1)
        except IOError:
            return
        if not ch:
            return
        if ch == "r":
            reset_requested = True
        elif ch == "q":
            quit_requested = True

    config_file = "configs/a1_wbt.yaml"
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
    action_scale = np.array(config["action_scale"], dtype=np.float32)
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    effort_limits = np.array(config["effort_limits"], dtype=np.float32)

    obs_index_in_lab = config["obs_index_in_lab"]
    obs_index_in_mj = config["obs_index_in_mj"]
    action_index_in_lab = config["action_index_in_lab"]
    action_index_in_mj = config["action_index_in_mj"]

    mj2lab = [obs_index_in_mj.index(name) for name in obs_index_in_lab]
    lab2mj = [action_index_in_lab.index(name) for name in action_index_in_mj]

    print(f"Config: {config_file}")
    print(f"Policy: {policy_path}")
    print(f"mj2lab = {mj2lab}")
    print(f"lab2mj = {lab2mj}")

    # Load ONNX model
    policy = ort.InferenceSession(policy_path)
    input_names = [inp.name for inp in policy.get_inputs()]
    output_names = [out.name for out in policy.get_outputs()]
    print(f"ONNX inputs: {input_names}")
    print(f"ONNX outputs: {output_names}")

    # Extract motion trajectory data from ONNX graph Constant nodes.
    # The model stores motion arrays as Constant ops that feed into Gather nodes
    # whose outputs are named "joint_pos", "joint_vel", "body_pos_w", "body_quat_w".
    import onnx
    from onnx import numpy_helper

    onnx_model = onnx.load(policy_path)

    # Build map: node output name -> Constant tensor
    _const_outputs = {}
    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.type == 4:  # TENSOR
                    _const_outputs[node.output[0]] = numpy_helper.to_array(attr.t)

    # Find Gather nodes and trace their data input back to a Constant
    _motion = {}
    for node in onnx_model.graph.node:
        if node.op_type == "Gather" and len(node.output) > 0:
            out_name = node.output[0]
            data_input = node.input[0]
            if out_name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                            "body_lin_vel_w", "body_ang_vel_w") \
               and data_input in _const_outputs:
                _motion[out_name] = _const_outputs[data_input]

    assert "joint_pos" in _motion, "Could not find joint_pos in ONNX graph constants"
    motion_joint_pos = _motion["joint_pos"]        # (T, 12)
    motion_joint_vel = _motion["joint_vel"]        # (T, 12)
    motion_body_pos_w = _motion["body_pos_w"]      # (T, num_bodies, 3)
    motion_body_quat_w = _motion["body_quat_w"]    # (T, num_bodies, 4)
    motion_body_lin_vel_w = _motion["body_lin_vel_w"]  # (T, num_bodies, 3)
    motion_body_ang_vel_w = _motion["body_ang_vel_w"]  # (T, num_bodies, 3)
    time_step_total = motion_joint_pos.shape[0]
    print(f"Motion length: {time_step_total} steps")
    print(f"Tracked bodies: {motion_body_pos_w.shape[1]}")
    del onnx_model, _motion

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    def reset_to_motion(t):
        """Reset robot state to match motion reference at time step t."""
        d.qpos[:3] = motion_body_pos_w[t, 0]
        d.qpos[3:7] = motion_body_quat_w[t, 0]
        d.qpos[7:] = motion_joint_pos[t][lab2mj]
        d.qvel[:3] = motion_body_lin_vel_w[t, 0]
        # MuJoCo free joint: qvel[3:6] is angular velocity in body frame
        d.qvel[3:6] = quat_apply_inverse(motion_body_quat_w[t, 0], motion_body_ang_vel_w[t, 0])
        d.qvel[6:] = motion_joint_vel[t][lab2mj]
        mujoco.mj_forward(m, d)

    # Initial state: match motion at t=0
    time_step = 0
    reset_to_motion(time_step)
    actions = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = motion_joint_pos[0][lab2mj].copy()

    counter = 0

    print(f"\nControls: r=reset, q=quit")
    print(f"Motion will loop automatically.\n")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            get_key()

            if quit_requested:
                break

            if reset_requested:
                time_step = 0
                reset_to_motion(time_step)
                d.ctrl[:] = 0.0
                actions[:] = 0.0
                target_dof_pos = motion_joint_pos[0][lab2mj].copy()
                counter = 0
                reset_requested = False
                print("Reset!")
                viewer.sync()
                continue

            # --- Policy query (matches training: observe THEN act) ---
            if counter % control_decimation == 0:
                quat_wxyz = d.qpos[3:7]
                root_pos = d.qpos[:3]
                omega_b = d.qvel[3:6].copy()

                qj = d.qpos[7:][mj2lab]
                dqj = d.qvel[6:][mj2lab]

                ref_joint_pos = motion_joint_pos[time_step]
                ref_joint_vel = motion_joint_vel[time_step]
                ref_body_pos_w = motion_body_pos_w[time_step]
                ref_body_quat_w = motion_body_quat_w[time_step]

                command_obs = np.concatenate([ref_joint_pos, ref_joint_vel])

                motion_anchor_pos_w = ref_body_pos_w[0]
                motion_anchor_quat_w = ref_body_quat_w[0]
                _, anchor_quat_b = subtract_frame_transforms(
                    root_pos, quat_wxyz, motion_anchor_pos_w, motion_anchor_quat_w
                )
                anchor_ori_b = ori_6d_from_quat(anchor_quat_b)

                joint_pos_rel = qj - default_angles

                obs = np.concatenate([
                    command_obs,          # 24
                    anchor_ori_b,         # 6
                    omega_b,              # 3
                    joint_pos_rel,        # 12
                    dqj,                  # 12
                    actions,              # 12
                ]).astype(np.float32)

                obs = np.clip(obs, -18.0, 18.0)

                obs_input = obs.reshape(1, -1)
                time_step_input = np.array([[time_step]], dtype=np.float32)

                onnx_outputs = policy.run(None, {
                    input_names[0]: obs_input,
                    input_names[1]: time_step_input,
                })

                actions = np.clip(onnx_outputs[0].squeeze(), -18.0, 18.0)

                target_dof = actions * action_scale + default_angles
                target_dof_pos = target_dof[lab2mj]

                time_step += 1
                if time_step >= time_step_total:
                    time_step = 0
                    reset_to_motion(time_step)
                    actions[:] = 0.0
                    target_dof_pos = motion_joint_pos[0][lab2mj].copy()
                    print("\n[Loop] Motion restarted (robot reset to t=0)")

                print(f"t={time_step:4d}/{time_step_total}  actions_norm={np.linalg.norm(actions):.3f}\r", end="")

            # --- PD control + physics step ---
            step_start = time.time()
            tau = pd_control(
                target_dof_pos,
                d.qpos[7:],
                kps[lab2mj],
                np.zeros_like(kds),
                d.qvel[6:],
                kds[lab2mj],
            )
            tau = np.clip(tau, -effort_limits[lab2mj], effort_limits[lab2mj])
            d.ctrl[:] = tau

            mujoco.mj_step(m, d)
            counter += 1

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    restore_terminal()
    print("\nDone.")
