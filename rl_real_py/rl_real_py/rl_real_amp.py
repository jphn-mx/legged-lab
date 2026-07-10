import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
import yaml
import numpy as np
import csv
import time
from sensor_msgs.msg import Joy

# keyboard
import sys
import termios
import tty
import fcntl
from ament_index_python.packages import get_package_share_directory
import os
from rclpy.qos import QoSProfile


# ======================================================================
# 纯函数 / 工具 (移植自 deploy_a1_amp.py, 本地保留以与源对齐)
# ======================================================================
class TermGroupedHistory:
    """按 term 分组的历史缓冲, 匹配 IsaacLab 默认 (interleave_by_time=False)。

    IsaacLab 先把每个 term 的完整历史拼起来, 再把所有 term 拼起来:
      [term1_t0|term1_t1|...|term1_tN | term2_t0|...|term2_tN | ...]
    """

    def __init__(self, term_dims, hist_len):
        self.term_dims = term_dims
        self.hist_len = hist_len
        self.buffers = [np.zeros((hist_len, dim), dtype=np.float32) for dim in term_dims]
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
        return np.concatenate([buf.flatten() for buf in self.buffers])

    def reset(self):
        for buf in self.buffers:
            buf[:] = 0.0
        self.initialized = False


def yaw_quat(quat_wxyz):
    """从完整四元数 (w,x,y,z) 提取只含 yaw 的四元数。"""
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def quat_conjugate(q):
    """四元数 (w,x,y,z) 的共轭。"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    """两个四元数 (w,x,y,z) 相乘。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rotmat(q):
    """四元数 (w,x,y,z) 转 3x3 旋转矩阵。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def compute_root_local_rot_tan_norm(quat_wxyz):
    """计算 root_local_rot_tan_norm: 去除 yaw 后取 tangent 与 normal 向量。"""
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


# MuJoCo XML 里的运动链: (相对父的偏移, 旋转函数或 None 表示固定关节)
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
    """正运动学: 计算末端 link 在 base 系下的位置。"""
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
    """用 FK 计算 key body 在 base 系下的位置 (无需仿真器)。

    Args:
        joint_angles_mj: 12 个关节角, MuJoCo 序 [R1..R6, L1..L6]
    Returns:
        拼接的 [L6_pos, R6_pos] (6D), 对应 key_body_names=["Link_L6","Link_R6"]
    """
    r6_pos = _fk_foot_pos(joint_angles_mj[:6], _RIGHT_LEG_CHAIN)
    l6_pos = _fk_foot_pos(joint_angles_mj[6:], _LEFT_LEG_CHAIN)
    return np.concatenate([l6_pos, r6_pos])


# 各观测 term 的维度 (与 obs_index 配合使用)
def _obs_term_dims(num_actions, n_key_bodies):
    return {
        "base_ang_vel": 3,
        "root_local_rot_tan_norm": 6,
        "velocity_commands": 3,
        "joint_pos": num_actions,
        "joint_vel": num_actions,
        "last_action": num_actions,
        "key_body_pos_b": n_key_bodies * 3,
    }


qos = QoSProfile(depth=1)
fd = sys.stdin.fileno()
# 保存终端状态
old_term = termios.tcgetattr(fd)
tty.setcbreak(fd)
# 设置为非阻塞
old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)


class RL_real(Node):
    """A1-legs 双足机器人 AMP 策略实机部署节点。

    把 deploy_a1_amp.py 的观测/推理管线迁移到 ROS2 (骨架对齐 rl_real_tron.py):
      obs(48) = [base_ang_vel(3), root_local_rot_tan_norm(6), velocity_commands(3),
                 joint_pos(12, 绝对/不缩放), joint_vel(12, 不缩放), last_action(12)]
      obs_history(240) = TermGroupedHistory: 每 term 先拼 5 帧, 再拼所有 term
      action(12, Lab 序) = policy(obs_history)
      target_q = default + action·action_scale, 再做关节安全限位裁剪

    关节顺序: /left_joint_states (入) 与 /dog_joint_pos (出) 均为 L0..L5,R0..R5
    (== joint_L1..L6, joint_R1..R6); AMP 策略用 Isaac Lab 交错序 (L1,R1,L2,R2,...),
    映射 real<->lab<->mj 在 __init__ 内按关节名自动推导。实机电机自带 PD,
    因此仿真里的 PD/力矩部分丢弃, 本节点只发布目标关节角。
    """

    def __init__(self, name):
        super().__init__(name)
        self.obs_subscriber = self.create_subscription(JointState, "/left_joint_states", self.obs_callback, 5)
        self.commands_publisher = self.create_publisher(Float64MultiArray, "/dog_joint_pos", qos)  # L0-L5 R0-R5
        self.imu_subscriber = self.create_subscription(Imu, '/imu', self.imu_callback, 5)
        self.joy_subscriber = self.create_subscription(Joy, "/joy", self.joy_callback, 5)

        config_file = "a1_amp.yaml"
        package_path = get_package_share_directory('rl_real_py')
        config_path = os.path.join(
            package_path,
            '..', '..', '..', '..',
            'src',
            'rl_real_py',
            'configs',
            config_file
        )
        print(config_path)
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.policy_path = os.path.join(package_path, config['policy_path'])
            self.model_type = config["model_type"]

            self.simulation_dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]

            self.num_obs = config["num_obs"]
            self.num_actions = config["num_actions"]
            self.num_history = config["num_history"]

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)  # Lab 序
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.clip_observations = config["clip_observations"]
            self.clip_actions = config["clip_actions"]

            self.obs_index = config["obs_index"]
            self.key_body_names = config["key_body_names"]

            self.joint_index_in_real = config["joint_index_in_real"]
            self.obs_index_in_lab = config["obs_index_in_lab"]
            self.obs_index_in_mj = config["obs_index_in_mj"]

            self.joint_lower_limits = np.array(config["joint_lower_limits"], dtype=np.float32)  # 实机序
            self.joint_upper_limits = np.array(config["joint_upper_limits"], dtype=np.float32)  # 实机序
            # PD 增益 (实机序), 仅用于"理论 PD 力矩"记录
            self.joint_kp = np.array(config["joint_kp"], dtype=np.float32)
            self.joint_kd = np.array(config["joint_kd"], dtype=np.float32)

            # 键盘控制参数 (步长 / 上限)
            kb = config.get("keyboard", {})
            self.kb_vx_step = float(kb.get("lin_vel_x_step", 0.1))
            self.kb_vy_step = float(kb.get("lin_vel_y_step", 0.1))
            self.kb_wz_step = float(kb.get("ang_vel_yaw_step", 0.1))
            self.kb_vx_max = float(kb.get("lin_vel_x_max", 1.0))
            self.kb_vy_max = float(kb.get("lin_vel_y_max", 1.0))
            self.kb_wz_max = float(kb.get("ang_vel_yaw_max", 1.0))

        # 关节重排映射 (按名字推导, 勿手填)
        # real obs -> lab: lab[k] = real[real2lab[k]]
        self.real2lab = [self.joint_index_in_real.index(n) for n in self.obs_index_in_lab]
        # lab action/target -> real (发布): real[k] = lab[lab2real[k]]
        self.lab2real = [self.obs_index_in_lab.index(n) for n in self.joint_index_in_real]
        # real obs -> mj (FK key_body_pos_b 用): mj[k] = real[real2mj[k]]
        self.real2mj = [self.joint_index_in_real.index(n) for n in self.obs_index_in_mj]
        print("real2lab =", self.real2lab)
        print("lab2real =", self.lab2real)
        print("real2mj  =", self.real2mj)

        # default_angles 实机序版本 (默认站姿 / pause 时发布)
        self.default_angles_real = self.default_angles[self.lab2real].astype(np.float32)

        # 观测各 term 维度 -> obs_index 对应的 term_dims
        term_dim_map = _obs_term_dims(self.num_actions, len(self.key_body_names))
        self.term_dims = [term_dim_map[name] for name in self.obs_index]
        assert sum(self.term_dims) == self.num_obs, \
            f"obs term dims {self.term_dims} 之和 {sum(self.term_dims)} != num_obs {self.num_obs}"

        # 加载策略
        print(self.policy_path)
        if self.model_type == "jit":
            import torch
            self.policy = torch.jit.load(self.policy_path + ".pt"
                                         if not self.policy_path.endswith(".pt") else self.policy_path)
            self.policy.eval()
            print(f"Loaded JIT model from {self.policy_path}")
        else:
            import onnxruntime as ort
            path = self.policy_path + ".onnx" if not self.policy_path.endswith(".onnx") else self.policy_path
            self.policy = ort.InferenceSession(path)
            self.input_name = self.policy.get_inputs()[0].name
            self.output_name = self.policy.get_outputs()[0].name
            print(f"Loaded ONNX model from {path}")

        # 状态
        self.x_vel = self.y_vel = self.yaw = 0.0
        # get_obs: [0:3] ang_vel, [3:7] quat(wxyz), [7:19] q (实机序), [19:31] qd (实机序)
        self.get_obs = [0.] * 31
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)  # Lab 序
        self.obs_hist = TermGroupedHistory(self.term_dims, self.num_history)
        self._initialized = False
        self.actions = np.zeros(self.num_actions, dtype=np.float32)

        self.reset_symbol = False
        self.pause_symbol = False
        self.run_model = False   # 模型门控: 默认不跑策略, 按 G 启动
        self.commands = self.default_angles_real.copy()  # 实机序发布
        self.counter = 0

        self.deadzone = 0.1   # 摇杆死区
        self.speed_scale = 1.0  # 速度缩放因子
        self._prev_buttons = []  # 手柄按钮上升沿检测

        print("rl_real (A1-legs AMP) start ...")
        print(f"simulation_dt:{self.simulation_dt}  control_decimation:{self.control_decimation}")
        print(f"obs_index: {self.obs_index}")
        print(f"term_dims: {self.term_dims}  total policy input = {self.num_obs * self.num_history}")
        print("键盘控制 (聚焦终端): W/S=前后  A/D=左右  Q/E=转向  空格=清零  R=复位  P=暂停切换")
        print("模型开关: G=启动/停止模型 (默认不跑, 仅保持默认站姿)")
        print("记录控制: M=开始记录(每次按新建文件)  N=停止记录")

        # ---- effort / 理论 PD 力矩 记录 (M 开始 / N 停止) ----
        # get_effort / _last_q_real / _last_qd_real: 实机序 L1..L6,R1..R6
        self.get_effort = np.zeros(self.num_actions, dtype=np.float32)
        self._eff_has_data = False         # 话题是否带 effort 字段
        self._last_q_real = None           # 最近一帧实机序关节角 (原始)
        self._last_qd_real = np.zeros(self.num_actions, dtype=np.float32)  # 实机序关节速度
        self._logging = False
        self._log_file = None
        self._log_writer = None
        self._log_t0 = 0.0
        self._log_path = None
        self._log_rows = []   # 内存暂存, 停止时出曲线图
        self._log_jn = [f"L{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 7)]  # 实机序

        self.timer = self.create_timer(self.simulation_dt, self.timer_callback)

    # ------------------------------------------------------------------
    # 推理流水线 (移植自 deploy_a1_amp.py)
    # ------------------------------------------------------------------
    def _build_term_obs(self):
        """按 obs_index 逐项构造观测 (返回 term list, 供 TermGroupedHistory)。"""
        q_real = np.array(self.get_obs[7:19], dtype=np.float32)
        qd_real = np.array(self.get_obs[19:31], dtype=np.float32)
        qj = q_real[self.real2lab]    # Lab 序绝对关节角
        dqj = qd_real[self.real2lab]  # Lab 序关节速度
        quat_wxyz = np.array(self.get_obs[3:7], dtype=np.float32)

        term_obs_list = []
        for idx in self.obs_index:
            if idx == "base_ang_vel":
                term_obs_list.append(np.array(self.get_obs[0:3], dtype=np.float32))
            elif idx == "root_local_rot_tan_norm":
                term_obs_list.append(compute_root_local_rot_tan_norm(quat_wxyz).astype(np.float32))
            elif idx == "velocity_commands":
                # term_obs_list.append(
                #     np.array([self.x_vel, self.y_vel, self.yaw], dtype=np.float32) * self.cmd_scale)
                # if self.x_vel < 0.5:
                #     self.x_vel = 0.5
                # else:
                #     self.x_vel = 0.0
                # term_obs_list.append(
                #     np.array([1.0, 0.0, 0.0], dtype=np.float32) * self.cmd_scale)
                term_obs_list.append(
                    np.array([self.x_vel, self.y_vel, self.yaw], dtype=np.float32) * self.cmd_scale)
            elif idx == "joint_pos":
                term_obs_list.append(qj)
            elif idx == "joint_vel":
                term_obs_list.append(dqj)
            elif idx == "last_action":
                term_obs_list.append(self.last_action.copy())
            elif idx == "key_body_pos_b":
                term_obs_list.append(
                    compute_key_body_pos_b_fk(q_real[self.real2mj]).astype(np.float32))
        return term_obs_list

    def _reset_policy(self):
        """复位策略状态: obs_hist 重建, last_action 清零 (AMP 无 gait 相位)。"""
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_hist = TermGroupedHistory(self.term_dims, self.num_history)
        self._initialized = False

    def _theory_pd_torque(self):
        """按发布的目标角 self.commands 与实测 q/dq 算理论 PD 力矩 (实机序):
        tau = (target - q) * kp + (0 - dq) * kd  (target_dq=0, 与仿真 pd_control 同式)。"""
        return (self.commands - self._last_q_real) * self.joint_kp \
            + (0.0 - self._last_qd_real) * self.joint_kd

    def _log_step(self):
        """写一行: t, 各关节 target / q / dq / effort(实测) / tau_pd(理论)。
        行尾附 IMU 角速度 (amp 现成可取, 无 gait/projected_gravity 同名项)。"""
        if self._last_q_real is None:
            return
        tau_pd = self._theory_pd_torque()
        ang_vel = self.get_obs[0:3]
        row = [time.time() - self._log_t0] + list(self.commands) + list(self._last_q_real) \
            + list(self._last_qd_real) + list(self.get_effort) + list(tau_pd) \
            + list(ang_vel)
        self._log_writer.writerow([f"{float(v):.6f}" for v in row])
        self._log_file.flush()
        self._log_rows.append([float(v) for v in row])

    def _start_log(self):
        """开始记录: 每次都新建一个时间戳 CSV (若已有打开的先关)。"""
        self._stop_log()
        path = os.path.join(os.getcwd(), time.strftime("amp_torque_%Y%m%d_%H%M%S.csv"))
        self._log_file = open(path, "w", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(
            ["t"] + [f"target_{n}" for n in self._log_jn] + [f"q_{n}" for n in self._log_jn]
            + [f"dq_{n}" for n in self._log_jn] + [f"effort_{n}" for n in self._log_jn]
            + [f"tau_pd_{n}" for n in self._log_jn]
            + ["wx", "wy", "wz"])
        self._log_path = path
        self._log_rows = []
        self._log_t0 = time.time()
        self._logging = True
        if self._last_q_real is None:
            print("\n[记录] 警告: 还没收到 /left_joint_states, 在关节状态到来前不会写入任何数据行")
        elif not self._eff_has_data:
            print("\n[记录] 警告: /left_joint_states 暂无 effort 字段, 实测力矩将记为 0")
        print(f"\n[记录] 开始 -> {path}")

    def _stop_log(self):
        """停止记录, 关闭文件并出 effort vs 理论 PD 力矩 对比图。"""
        was = self._logging
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
            self._log_writer = None
        self._logging = False
        if was:
            print(f"\n[记录] 停止, 已存 CSV: {self._log_path}")
            print("[记录] 出图请用离线脚本(不阻塞控制): "
                  "/home/woan/miniforge/envs/gvhmr/bin/python /home/woan/rl_real/plot_torque.py <该CSV>")
        self._log_rows = []

    def _mark_log(self, text):
        """在 CSV 里插入 空行 + 注释行 作为事件标记 (如模型启停)。仅在记录中时写入。"""
        if self._logging and self._log_file is not None:
            self._log_file.write(f"\n# ==== {text}  t={time.time() - self._log_t0:.3f}s ====\n")
            self._log_file.flush()

    def _save_plot(self):
        """每关节叠画 实测 effort (灰) vs 理论 PD 力矩 (蓝), 共 6x2。"""
        rows, path = self._log_rows, self._log_path
        if not path or len(rows) < 2:
            print("[记录] 数据太少, 不出图")
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import font_manager as fm
        except ImportError:
            print("[记录] 无 matplotlib, 跳过出图")
            return
        # 中文字体: 从已装的 CJK 字体里挑第一个可用的, 否则维持默认(英文)
        avail = {f.name for f in fm.fontManager.ttflist}
        for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
                     "WenQuanYi Micro Hei", "Droid Sans Fallback", "AR PL UMing CN"):
            if cand in avail:
                plt.rcParams["font.sans-serif"] = [cand]
                break
        plt.rcParams["axes.unicode_minus"] = False

        a = np.array(rows, dtype=np.float64)
        t = a[:, 0]
        nj = len(self._log_jn)
        eff0, tau0 = 1 + 3 * nj, 1 + 4 * nj   # effort / tau_pd 列起始下标
        fig, axes = plt.subplots(6, 2, figsize=(12, 13), sharex=True, squeeze=False)
        for j in range(nj):
            ax = axes[j % 6][j // 6]   # 左列 L1-L6, 右列 R1-R6
            eff, tau = a[:, eff0 + j], a[:, tau0 + j]
            ax.plot(t, tau, "b", lw=1.0, label="理论 PD")
            ax.plot(t, eff, "0.5", lw=0.9, label="实测 effort")
            ax.set_ylabel(self._log_jn[j])
            ax.grid(True, alpha=0.3)
            ax.text(0.99, 0.04, f"RMS差={np.sqrt(np.mean((tau-eff)**2)):.2f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.3")
        axes[0][0].legend(loc="upper right", fontsize=8)
        axes[5][0].set_xlabel("t (s)")
        axes[5][1].set_xlabel("t (s)")
        fig.suptitle("各关节: 实测 effort vs 理论 PD 力矩 (N·m)")
        fig.tight_layout()
        p = path.rsplit(".", 1)[0] + ".png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"[记录] 图 -> {p}")

    def timer_callback(self):
        msg = Float64MultiArray()
        self.counter += 1
        self.get_key()
        print(f'x_vel:{round(self.x_vel,2)}     y_vel:{round(self.y_vel,2)}     yaw:{round(self.yaw,2)}     \r', end="")

        if self.reset_symbol:
            self._reset_policy()
            self.x_vel, self.y_vel, self.yaw = 0., 0., 0.
            self.reset_symbol = False
            print("reset")

        # 模型门控: 仅当已用 G 启动 且 未暂停 时才跑策略; 否则保持默认站姿。
        run_now = self.run_model and not self.pause_symbol
        if not run_now:
            self.commands = self.default_angles_real.copy()
            print('pause                                   ' if self.pause_symbol
                  else 'model OFF (按 G 启动)                    ', end='\r')
        elif self.counter % self.control_decimation == 0:
            if not self._initialized:
                self._reset_policy()
                self._initialized = True

            term_list = self._build_term_obs()
            total_obs = self.obs_hist.update(term_list)
            obs_tensor = np.clip(total_obs, -self.clip_observations, self.clip_observations)
            obs_tensor = obs_tensor.astype(np.float32)[np.newaxis, :]

            if self.model_type == "jit":
                import torch
                with torch.no_grad():
                    action = self.policy(torch.from_numpy(obs_tensor)).numpy().squeeze().astype(np.float32)
            else:
                outputs = self.policy.run([self.output_name], {self.input_name: obs_tensor})
                action = outputs[0].squeeze().astype(np.float32)
            action = np.clip(action, -self.clip_actions, self.clip_actions)
            self.last_action = action.copy()  # Lab 序
            self.actions = action

            # 目标关节角 (Lab 序) = 默认角 + action·scale, 重排到实机序, 再做硬件安全限位裁剪
            target_lab = self.default_angles + action * self.action_scale
            target_real = target_lab[self.lab2real]
            self.commands = np.clip(target_real, self.joint_lower_limits, self.joint_upper_limits)

        # 记录: 与模型是否运行无关 (M 开始 / N 停止), 同样按 control_decimation 节流。
        if self._logging and self._log_writer is not None and self.counter % self.control_decimation == 0:
            self._log_step()

        msg.data = self.commands.tolist()
        self.commands_publisher.publish(msg)

    # ------------------------------------------------------------------
    # ROS 回调
    # ------------------------------------------------------------------
    def obs_callback(self, msg):
        # /left_joint_states 顺序 L0-L5,R0-R5 (实机序), 直接写入; 重排在 _build_term_obs 内做
        self.get_obs[7:19] = [msg.position[i] for i in range(12)]
        self.get_obs[19:31] = [msg.velocity[i] for i in range(12)]
        # 记录用的实机序 q/dq (原始, 不重排)
        self._last_q_real = np.array(self.get_obs[7:19], dtype=np.float32)
        self._last_qd_real = np.array(self.get_obs[19:31], dtype=np.float32)
        # 实测力矩 (effort 字段, 实机序 L1..L6,R1..R6; 部分驱动可能不填)
        if len(msg.effort) >= 12:
            self.get_effort = np.array([msg.effort[i] for i in range(12)], dtype=np.float32)
            self._eff_has_data = True

    def imu_callback(self, msg):
        self.get_obs[0:3] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.get_obs[3:7] = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]

    def joy_callback(self, msg):
        # 左摇杆: 前后(axes[1]) / 左右(axes[0]); 右摇杆水平(axes[2]): 转向
        if abs(msg.axes[1]) > self.deadzone:
            self.x_vel = msg.axes[1] * self.speed_scale
        else:
            self.x_vel = 0.0

        if abs(msg.axes[0]) > self.deadzone:
            self.y_vel = msg.axes[0] * self.speed_scale
        else:
            self.y_vel = 0.0

        if abs(msg.axes[2]) > self.deadzone:
            self.yaw = msg.axes[2] * self.speed_scale
        else:
            self.yaw = 0.0

        # 按钮 (Xbox 序): A=0 运行模型, B=1 固定默认角度, X=2 开始记录, Y=3 停止记录。
        # 上升沿触发 (按住只触发一次), 否则按住 A 会每帧重建历史、按住 X 会反复新建日志。
        def pressed(i):
            now = len(msg.buttons) > i and msg.buttons[i] == 1
            was = len(self._prev_buttons) > i and self._prev_buttons[i] == 1
            return now and not was
        if pressed(0):          # A: 运行模型 (干净启动, 与键盘 G 一致)
            self._start_model()
        if pressed(1):          # B: 停回默认角度
            self._stop_model()
        if pressed(3):          # X: 开始记录
            self._start_log()
        if pressed(4):          # Y: 停止记录
            self._stop_log()
        self._prev_buttons = list(msg.buttons)

    def _start_model(self):
        """启动策略: 与键盘 G 启动分支完全一致 (干净重建 obs_history)。"""
        self.run_model = True
        self._initialized = False   # 每次启动都干净重建 obs_history
        self._mark_log("MODEL START")
        print("\n[模型] 启动")

    def _stop_model(self):
        """停止策略, 保持默认站姿 (与键盘 G 停止分支一致)。"""
        self.run_model = False
        self._mark_log("MODEL STOP")
        print("\n[模型] 停止 (保持默认站姿)")

    def get_key(self):
        """非阻塞读取键盘并更新速度指令 (终端已在模块加载时设为 cbreak + 非阻塞)。

        W/S: 前进/后退(vx)   A/D: 左移/右移(vy)   Q/E: 左转/右转(yaw)
        空格: 速度清零   R: 复位   P: 暂停/恢复切换   G: 模型开关   M/N: 记录
        """
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
            elif ch in ('g', 'G'):
                if self.run_model:
                    self._stop_model()
                else:
                    self._start_model()
            elif ch in ('m', 'M'):
                self._start_log()
            elif ch in ('n', 'N'):
                self._stop_log()


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
        if node is not None:
            node._stop_log()
        restore_terminal()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
