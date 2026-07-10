import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Joy
import yaml
import numpy as np
from collections import deque

import onnxruntime as ort

from rl_real_py.utils.math import get_gravity_orientation
# keyboard
import sys
import termios
import tty
import fcntl
from ament_index_python.packages import get_package_share_directory
import os
from rclpy.qos import QoSProfile


# ======================================================================
# 观测工具 (四元数 / 历史缓冲 / 可选 FK)
# ======================================================================
class TermGroupedHistory:
    """按 term 分组的历史 (IsaacLab 默认): 每 term 先拼 N 帧, 再拼所有 term。"""

    def __init__(self, term_dims, hist_len):
        self.hist_len = hist_len
        self.buffers = [np.zeros((hist_len, d), dtype=np.float32) for d in term_dims]
        self.initialized = False

    def update(self, term_obs_list):
        if not self.initialized:
            for i, obs in enumerate(term_obs_list):
                self.buffers[i][:] = obs
            self.initialized = True
        else:
            for i, obs in enumerate(term_obs_list):
                self.buffers[i][:-1] = self.buffers[i][1:]
                self.buffers[i][-1] = obs
        return np.concatenate([b.flatten() for b in self.buffers])


class FlatHistory:
    """整帧拼接历史 (gym/tron 风格)。order: oldest_first | newest_first。"""

    def __init__(self, num_obs, hist_len, order="oldest_first"):
        self.hist_len = hist_len
        self.order = order
        self.frames = None

    def update(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if self.frames is None:
            self.frames = deque([frame.copy() for _ in range(self.hist_len)], maxlen=self.hist_len)
        else:
            self.frames.append(frame)
        seq = list(self.frames)
        if self.order == "newest_first":
            seq = seq[::-1]
        return np.concatenate(seq).astype(np.float32)


def _yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def compute_root_local_rot_tan_norm(quat_wxyz):
    """去除 yaw 后取 tangent + normal 向量 (6D)。"""
    local_q = _quat_mul(_quat_conjugate(_yaw_quat(quat_wxyz)), quat_wxyz)
    rotm = _quat_to_rotmat(local_q)
    return np.concatenate([rotm[:, 0], rotm[:, 2]])


# ---- 可选 key_body_pos_b 的 A1-legs FK (仅 obs_index 用到时才会调用) ----
def _rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_RIGHT_LEG_CHAIN = [
    (np.array([0.0, -0.021, -0.055]), None), (np.array([0.0, -0.04125, 0.0]), _rot_y),
    (np.array([0.0, -0.0987, 0.0]), _rot_x), (np.array([0.0, 0.0, -0.1215]), _rot_z),
    (np.array([0.0, 0.0, -0.112]), _rot_y), (np.array([0.0, 0.0, -0.221]), _rot_y),
    (np.array([0.0, 0.0, -0.05]), _rot_x),
]
_LEFT_LEG_CHAIN = [
    (np.array([0.0, 0.021, -0.055]), None), (np.array([0.0, 0.04125, 0.0]), _rot_y),
    (np.array([0.0, 0.0987, 0.0]), _rot_x), (np.array([-0.0001, 0.0, -0.122]), _rot_z),
    (np.array([0.0, 0.0, -0.1115]), _rot_y), (np.array([0.0, 0.0, -0.221]), _rot_y),
    (np.array([0.0, 0.0, -0.05]), _rot_x),
]


def _fk_foot_pos(angles6, chain):
    rot, pos, j = np.eye(3), np.zeros(3), 0
    for offset, rf in chain:
        pos = pos + rot @ offset
        if rf is not None:
            rot = rot @ rf(angles6[j])
            j += 1
    return pos


def compute_key_body_pos_b_fk(joint_angles_mj):
    """[L6_pos, R6_pos] (6D); 输入为 MuJoCo 序关节角 [R1..R6, L1..L6]。"""
    r6 = _fk_foot_pos(joint_angles_mj[:6], _RIGHT_LEG_CHAIN)
    l6 = _fk_foot_pos(joint_angles_mj[6:], _LEFT_LEG_CHAIN)
    return np.concatenate([l6, r6])


qos = QoSProfile(depth=1)
fd = sys.stdin.fileno()
old_term = termios.tcgetattr(fd)
tty.setcbreak(fd)
old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)


class RL_real(Node):
    """通用 config 驱动实机部署节点。

    观测项 (obs_index)、仿真关节顺序 (joint_index_in_sim)、各种缩放/裁剪/历史方式
    全部由 config 决定; 一份节点代码覆盖 gym / amp / tron 风格策略。
      obs(单帧)  = 按 obs_index 顺序拼接各 term
      history    = term_grouped 或 flat
      action     = policy(history[, obs, commands])   (仿真序)
      target     = default + action·action_scale, 重排到实机序, 再做安全限位裁剪
    """

    def __init__(self, name):
        super().__init__(name)
        self.obs_subscriber = self.create_subscription(JointState, "/left_joint_states", self.obs_callback, 5)
        self.commands_publisher = self.create_publisher(Float64MultiArray, "/dog_joint_pos", qos)
        self.imu_subscriber = self.create_subscription(Imu, '/imu', self.imu_callback, 5)
        self.joy_subscriber = self.create_subscription(Joy, "/joy", self.joy_callback, 5)

        config_file = "common.yaml"
        package_path = get_package_share_directory('rl_real_py')
        # 优先指向源码包目录(改 config/policy 免 colcon build 即时生效); 找不到则回退安装目录
        src_pkg = os.path.normpath(os.path.join(package_path, '..', '..', '..', '..', 'src', 'rl_real_py'))
        base_dir = src_pkg if os.path.isdir(src_pkg) else package_path
        config_path = os.path.join(base_dir, 'configs', config_file)
        print(config_path)
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.cfg = cfg

        self.policy_path = os.path.join(base_dir, cfg['policy_path'])
        self.model_type = cfg["model_type"]
        self.simulation_dt = cfg["simulation_dt"]
        self.control_decimation = cfg["control_decimation"]
        self.ctrl_dt = self.simulation_dt * self.control_decimation

        self.num_obs = cfg["num_obs"]
        self.num_actions = cfg["num_actions"]
        self.num_history = cfg["num_history"]
        self.num_commands = cfg.get("num_commands", 3)

        self.action_scale = cfg["action_scale"]
        self.clip_observations = cfg.get("clip_observations", 100.0)
        self.clip_actions = cfg.get("clip_actions", 100.0)
        self.ang_vel_scale = cfg.get("ang_vel_scale", 1.0)
        self.dof_pos_scale = cfg.get("dof_pos_scale", 1.0)
        self.dof_vel_scale = cfg.get("dof_vel_scale", 1.0)
        self.cmd_scale = np.array(cfg.get("cmd_scale", [1.0] * self.num_commands), dtype=np.float32)

        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)  # 仿真序
        self.obs_index = cfg["obs_index"]
        self.history_type = cfg.get("history_type", "term_grouped")
        self.history_order = cfg.get("history_order", "oldest_first")

        self.joint_lower_limits = np.array(cfg["joint_lower_limits"], dtype=np.float32)  # 实机序
        self.joint_upper_limits = np.array(cfg["joint_upper_limits"], dtype=np.float32)  # 实机序

        # 关节顺序映射 (按名字推导)
        real = cfg["joint_index_in_real"]
        sim = cfg["joint_index_in_sim"]
        self.real2sim = [real.index(n) for n in sim]   # x_sim = x_real[real2sim]
        self.sim2real = [sim.index(n) for n in real]   # x_real = x_sim[sim2real]
        self.default_angles_real = self.default_angles[self.sim2real].astype(np.float32)
        print("real2sim =", self.real2sim)
        print("sim2real =", self.sim2real)

        # 可选: key_body_pos_b FK
        self.key_body_names = cfg.get("key_body_names", [])
        mj = cfg.get("obs_index_in_mj")
        self.real2mj = [real.index(n) for n in mj] if mj else None

        # 可选: gait
        self.gait_freq = float(cfg.get("gait_frequency", 0.0))
        self.gait_params = np.array([cfg.get("gait_frequency", 0.0), cfg.get("gait_offset", 0.0),
                                     cfg.get("gait_duration", 0.0), cfg.get("gait_swing_height", 0.0)],
                                    dtype=np.float32)
        self.gait_zero_thresh = float(cfg.get("gait_zero_speed_thresh", 0.05))
        self.gait_phase = 0.0

        # 各 term 维度 -> 校验
        dim_map = {
            "base_ang_vel": 3, "projected_gravity": 3, "root_local_rot_tan_norm": 6,
            "velocity_commands": self.num_commands, "joint_pos": self.num_actions,
            "joint_pos_rel": self.num_actions, "joint_vel": self.num_actions,
            "joint_vel_rel": self.num_actions, "last_action": self.num_actions,
            "gait_clock": 2, "gait_phase": 2, "gait_params": 4, "key_body_pos_b": len(self.key_body_names) * 3,
        }
        self.term_dims = [dim_map[n] for n in self.obs_index]
        assert sum(self.term_dims) == self.num_obs, \
            f"obs term dims {self.term_dims} 之和 {sum(self.term_dims)} != num_obs {self.num_obs}"

        # 历史缓冲
        if self.history_type == "term_grouped":
            self.hist = TermGroupedHistory(self.term_dims, self.num_history)
        else:
            self.hist = FlatHistory(self.num_obs, self.num_history, self.history_order)

        # 加载策略
        if self.model_type == "onnx":
            if not self.policy_path.endswith(".onnx"):
                self.policy_path += ".onnx"
            self.policy = ort.InferenceSession(self.policy_path)
            self.input_names = [i.name for i in self.policy.get_inputs()]
            self.output_name = self.policy.get_outputs()[0].name
            self.onnx_input_map = cfg.get("onnx_input_map", {})
            print(f"Loaded ONNX model from {self.policy_path}  inputs={self.input_names}")
        elif self.model_type == "jit":
            import torch  # 懒加载, onnx 用户无需安装 torch
            self.torch = torch
            if not self.policy_path.endswith(".pt"):
                self.policy_path += ".pt"
            self.policy = torch.jit.load(self.policy_path)
            print(f"Loaded JIT model from {self.policy_path}")
        else:
            raise ValueError(f"未知 model_type: {self.model_type}")

        # 状态
        self.x_vel = self.y_vel = self.yaw = 0.0
        # get_obs: [0:3] ang_vel, [3:7] quat(wxyz), [7:19] q(实机序), [19:31] qd(实机序)
        self.get_obs = [0.] * 31
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)  # 仿真序
        self.commands = self.default_angles_real.copy()
        self.counter = 0
        self.reset_symbol = False
        self.pause_symbol = False

        self.deadzone = 0.1
        self.speed_scale = 1.0

        kb = cfg.get("keyboard", {})
        self.kb_vx_step = float(kb.get("lin_vel_x_step", 0.1))
        self.kb_vy_step = float(kb.get("lin_vel_y_step", 0.1))
        self.kb_wz_step = float(kb.get("ang_vel_yaw_step", 0.1))
        self.kb_vx_max = float(kb.get("lin_vel_x_max", 1.0))
        self.kb_vy_max = float(kb.get("lin_vel_y_max", 1.0))
        self.kb_wz_max = float(kb.get("ang_vel_yaw_max", 1.0))

        print("rl_real (通用) start ...")
        print(f"obs_index={self.obs_index}  history={self.history_type}  "
              f"input_dim={self.num_obs * self.num_history}")
        print("键盘: W/S=前后 A/D=左右 Q/E=转向 空格=清零 R=复位 P=暂停切换")

        self.timer = self.create_timer(self.simulation_dt, self.timer_callback)

    # ------------------------------------------------------------------
    def _build_terms(self):
        """按 obs_index 逐项构造观测 (仿真序), 返回 term list。"""
        q_sim = np.array(self.get_obs[7:19], dtype=np.float32)[self.real2sim]
        qd_sim = np.array(self.get_obs[19:31], dtype=np.float32)[self.real2sim]
        quat = np.array(self.get_obs[3:7], dtype=np.float32)
        cmd = np.array([self.x_vel, self.y_vel, self.yaw], dtype=np.float32)[:self.num_commands]

        out = []
        for idx in self.obs_index:
            if idx == "base_ang_vel":
                out.append(np.array(self.get_obs[0:3], dtype=np.float32) * self.ang_vel_scale)
            elif idx == "projected_gravity":
                out.append(get_gravity_orientation(self.get_obs[3:7]).astype(np.float32))
            elif idx == "root_local_rot_tan_norm":
                out.append(compute_root_local_rot_tan_norm(quat).astype(np.float32))
            elif idx == "velocity_commands":
                out.append(cmd * self.cmd_scale)
            elif idx == "joint_pos":
                out.append(q_sim)
            elif idx == "joint_pos_rel":
                out.append((q_sim - self.default_angles) * self.dof_pos_scale)
            elif idx in ("joint_vel", "joint_vel_rel"):
                out.append(qd_sim * self.dof_vel_scale)
            elif idx == "last_action":
                out.append(self.last_action.copy())
            elif idx in ("gait_clock", "gait_phase"):
                out.append(np.array([np.sin(2 * np.pi * self.gait_phase),
                                     np.cos(2 * np.pi * self.gait_phase)], dtype=np.float32))
            elif idx == "gait_params":
                out.append(self.gait_params.copy())
            elif idx == "key_body_pos_b":
                out.append(compute_key_body_pos_b_fk(
                    np.array(self.get_obs[7:19], dtype=np.float32)[self.real2mj]).astype(np.float32))
            else:
                raise ValueError(f"未知观测项: {idx}")
        return out

    def _reset_policy(self):
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.gait_phase = 0.0
        if self.history_type == "term_grouped":
            self.hist = TermGroupedHistory(self.term_dims, self.num_history)
        else:
            self.hist = FlatHistory(self.num_obs, self.num_history, self.history_order)

    def _run_policy(self, history_obs, single_obs, commands):
        """模型推理, 返回 action (仿真序)。"""
        if self.model_type == "onnx":
            feed = {}
            single_input = len(self.input_names) == 1
            for name in self.input_names:
                src = self.onnx_input_map.get(name)
                if src is None:  # 自动判断
                    if single_input:
                        src = "history"   # 单输入策略恒为完整历史向量 (不看名字)
                    else:
                        src = ("history" if "history" in name
                               else "commands" if name == "commands"
                               else "obs" if name == "obs" else "history")
                vec = {"history": history_obs, "obs": single_obs, "commands": commands}[src]
                feed[name] = vec[np.newaxis, :].astype(np.float32)
            out = self.policy.run([self.output_name], feed)[0]
            return np.array(out).squeeze().astype(np.float32)
        else:  # jit
            t = self.torch.from_numpy(history_obs[np.newaxis, :]).float()
            out = self.policy(t)
            if isinstance(out, (list, tuple)):
                out = out[0]
            return out.detach().cpu().numpy().squeeze().astype(np.float32)

    def timer_callback(self):
        msg = Float64MultiArray()
        self.counter += 1
        self.get_key()
        print(f'x_vel:{round(self.x_vel,2)}   y_vel:{round(self.y_vel,2)}   yaw:{round(self.yaw,2)}   \r', end="")

        if self.reset_symbol:
            self._reset_policy()
            self.x_vel, self.y_vel, self.yaw = 0., 0., 0.
            self.reset_symbol = False
            print("reset")

        if self.pause_symbol:
            self.commands = self.default_angles_real.copy()
            print('pause                                   ', end='\r')
        elif self.counter % self.control_decimation == 0:
            # gait 相位推进 (仅 obs 用到 gait_clock 时有意义)
            self.gait_phase = (self.gait_phase + self.ctrl_dt * self.gait_freq) % 1.0
            if float(np.linalg.norm([self.x_vel, self.y_vel, self.yaw])) < self.gait_zero_thresh:
                self.gait_phase = 0.0

            term_list = self._build_terms()
            single_obs = np.concatenate(term_list).astype(np.float32)
            if self.history_type == "term_grouped":
                history_obs = self.hist.update(term_list)
            else:
                history_obs = self.hist.update(single_obs)
            history_obs = np.clip(history_obs, -self.clip_observations, self.clip_observations)
            single_obs = np.clip(single_obs, -self.clip_observations, self.clip_observations)
            commands = (np.array([self.x_vel, self.y_vel, self.yaw], dtype=np.float32)[:self.num_commands]
                        * self.cmd_scale)

            action = self._run_policy(history_obs, single_obs, commands)
            action = np.clip(action, -self.clip_actions, self.clip_actions)
            self.last_action = action.copy()

            target_sim = self.default_angles + action * self.action_scale
            target_real = target_sim[self.sim2real]
            self.commands = np.clip(target_real, self.joint_lower_limits, self.joint_upper_limits)

        msg.data = self.commands.tolist()
        self.commands_publisher.publish(msg)

    # ------------------------------------------------------------------
    def obs_callback(self, msg):
        # /left_joint_states 实机序, 直接写入; 重排在 _build_terms 内做
        self.get_obs[7:19] = [msg.position[i] for i in range(12)]
        self.get_obs[19:31] = [msg.velocity[i] for i in range(12)]

    def imu_callback(self, msg):
        self.get_obs[0:3] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.get_obs[3:7] = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]

    def joy_callback(self, msg):
        self.x_vel = msg.axes[1] * self.speed_scale if abs(msg.axes[1]) > self.deadzone else 0.0
        self.y_vel = msg.axes[0] * self.speed_scale if abs(msg.axes[0]) > self.deadzone else 0.0
        self.yaw = msg.axes[2] * self.speed_scale if abs(msg.axes[2]) > self.deadzone else 0.0
        if msg.buttons[0] == 1:
            self.reset_symbol = True
        if msg.buttons[1] == 1:
            self.pause_symbol = True
        if msg.buttons[3] == 1:
            self.pause_symbol = False

    def get_key(self):
        while True:
            try:
                ch = sys.stdin.read(1)
            except (IOError, OSError):
                ch = ''
            if not ch:
                break
            if ch in ('w', 'W'):
                self.x_vel = min(self.kb_vx_max, self.x_vel + self.kb_vx_step)
            elif ch in ('s', 'S'):
                self.x_vel = max(-self.kb_vx_max, self.x_vel - self.kb_vx_step)
            elif ch in ('a', 'A'):
                self.y_vel = min(self.kb_vy_max, self.y_vel + self.kb_vy_step)
            elif ch in ('d', 'D'):
                self.y_vel = max(-self.kb_vy_max, self.y_vel - self.kb_vy_step)
            elif ch in ('q', 'Q'):
                self.yaw = min(self.kb_wz_max, self.yaw + self.kb_wz_step)
            elif ch in ('e', 'E'):
                self.yaw = max(-self.kb_wz_max, self.yaw - self.kb_wz_step)
            elif ch == ' ':
                self.x_vel = self.y_vel = self.yaw = 0.0
            elif ch in ('r', 'R'):
                self.reset_symbol = True
            elif ch in ('p', 'P'):
                self.pause_symbol = not self.pause_symbol


def restore_terminal():
    termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RL_real("rl_real")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        restore_terminal()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
