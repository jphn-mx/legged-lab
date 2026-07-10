import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
import yaml
import numpy as np
from collections import deque
import csv
import time
import onnxruntime as ort
from sensor_msgs.msg import Joy

from rl_real_py.utils.math import get_gravity_orientation
# keyboard
import sys
import termios
import tty
import fcntl
from ament_index_python.packages import get_package_share_directory
import os
from rclpy.qos import QoSProfile


qos = QoSProfile(depth=1)
fd = sys.stdin.fileno()
# 保存终端状态
old_term = termios.tcgetattr(fd)
tty.setcbreak(fd)
# 设置为非阻塞
old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)


class RL_real(Node):
    """WOAN 双足机器人实机部署节点。

    把 mujoco_woan/sim2sim.py 里的策略推理 (PolicyRunner) 迁移到 ROS2:
      obs(48)  = [base_ang_vel·scale(3), projected_gravity(3),
                  (q-default)·scale(12), qd·scale(12), last_action(12),
                  clock_sin, clock_cos, gait_params(4)]
      obs_history(480) = 最近 10 帧 obs 拼接 (deque 顺序: 旧->新)
      commands(3)      = [vx, vy, wz] * cmd_scale
      action(12)       = policy(obs_history, obs, commands)
      target_q         = default + action·action_scale  (再做安全限位裁剪)

    关节顺序: /left_joint_states (输入) 与 /dog_joint_pos (输出) 均为
    L0..L5, R0..R5, 与策略 joint_order (joint_L1..L6, joint_R1..R6) 一一对应,
    因此无需重排 (identity)。
    """

    def __init__(self, name):
        super().__init__(name)
        self.obs_subscriber = self.create_subscription(JointState, "/left_joint_states", self.obs_callback, 5)
        self.commands_publisher = self.create_publisher(Float64MultiArray, "/dog_joint_pos", qos)  # L0-L5 R0-R5
        self.imu_subscriber = self.create_subscription(Imu, '/imu', self.imu_callback, 5)
        self.joy_subscriber = self.create_subscription(Joy, "/joy", self.joy_callback, 5)

        config_file = "dual_tron.yaml"
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
            self.ctrl_dt = self.simulation_dt * self.control_decimation  # 策略控制周期 (gait 用)

            self.num_obs = config["num_obs"]
            self.num_actions = config["num_actions"]
            self.num_commands = config["num_commands"]
            self.hist_len = config["obs_history_length"]

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)

            self.action_scale = config["action_scale"]
            self.clip_actions = config["clip_actions"]
            self.clip_observations = config["clip_observations"]
            self.user_torque_limit = config["user_torque_limit"]

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            self.gait_params = np.array([
                config["gait_frequency"],
                config["gait_offset"],
                config["gait_duration"],
                config["gait_swing_height"],
            ], dtype=np.float32)
            self.gait_freq = float(config["gait_frequency"])

            self.joint_lower_limits = np.array(config["joint_lower_limits"], dtype=np.float32)
            self.joint_upper_limits = np.array(config["joint_upper_limits"], dtype=np.float32)

            # 键盘控制参数 (步长 / 上限)
            kb = config.get("keyboard", {})
            self.kb_vx_step = float(kb.get("lin_vel_x_step", 0.1))
            self.kb_vy_step = float(kb.get("lin_vel_y_step", 0.1))
            self.kb_wz_step = float(kb.get("ang_vel_yaw_step", 0.1))
            self.kb_vx_max = float(kb.get("lin_vel_x_max", 1.0))
            self.kb_vy_max = float(kb.get("lin_vel_y_max", 1.0))
            self.kb_wz_max = float(kb.get("ang_vel_yaw_max", 1.0))

        # action_clip 使用全 DOF 平均增益 (与 Isaac Gym _action_clip 一致)
        self.kp_mean = float(np.mean(self.kps))
        self.kd_mean = float(np.mean(self.kds))

        # 加载策略 (ONNX, 三输入: obs_history / obs / commands)
        self.policy_path += '.onnx'
        print(self.policy_path)
        self.policy = ort.InferenceSession(self.policy_path)
        self.input_names = [i.name for i in self.policy.get_inputs()]
        self.output_name = self.policy.get_outputs()[0].name
        expected_inputs = {"obs_history", "obs", "commands"}
        assert expected_inputs.issubset(self.input_names), \
            f"ONNX inputs {self.input_names} missing one of {expected_inputs}"
        print(f"Loaded ONNX model from {self.policy_path}")

        # 状态
        self.x_vel = self.y_vel = self.yaw = 0.0
        # get_obs: [0:3] ang_vel, [3:7] quat(wxyz), [7:19] q, [19:31] qd
        self.get_obs = [0.] * 31
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.gait_phase = 0.0
        self.obs_history = deque(maxlen=self.hist_len)
        self._initialized = False
        self.actions = np.zeros(self.num_actions, dtype=np.float32)

        self.reset_symbol = False
        self.pause_symbol = False
        self.run_model = False   # 模型门控: 默认不跑策略, 按 G 启动; 与记录互不影响
        self.commands = self.default_angles.copy()
        self.counter = 0

        self.deadzone = 0.1   # 摇杆死区
        self.speed_scale = 1.0  # 速度缩放因子

        print("rl_real (WOAN) start ...")
        print(f"simulation_dt:{self.simulation_dt}  control_decimation:{self.control_decimation}  ctrl_dt:{self.ctrl_dt}")
        print("键盘控制 (聚焦终端): W/S=前后  A/D=左右  Q/E=转向  空格=清零  R=复位  P=暂停切换")
        print("模型开关: G=启动/停止模型 (默认不跑, 仅保持默认站姿)")
        print("记录控制: M=开始记录(每次按新建文件)  N=停止记录  (记录与模型互不影响)")

        # obs 记录 (键盘 M 开始 / N 停止); 每次开始都新建一个时间戳文件
        self._logging = False
        self._log_file = None
        self._log_writer = None
        self._log_t0 = 0.0
        self._log_path = None
        self._log_rows = []   # 内存暂存, 停止时出曲线图
        self._log_jn = [f"L{i}" for i in range(6)] + [f"R{i}" for i in range(6)]

        self.timer = self.create_timer(self.simulation_dt, self.timer_callback)

    # ------------------------------------------------------------------
    # 推理流水线 (移植自 sim2sim.py PolicyRunner)
    # ------------------------------------------------------------------
    def _build_obs(self):
        """构造 48 维单帧观测。"""
        ang_vel = np.array(self.get_obs[0:3], dtype=np.float32) * self.ang_vel_scale
        projected_gravity = get_gravity_orientation(self.get_obs[3:7]).astype(np.float32)
        q = np.array(self.get_obs[7:19], dtype=np.float32)
        qd = np.array(self.get_obs[19:31], dtype=np.float32)
        dof_pos_err = (q - self.default_angles) * self.dof_pos_scale
        dof_vel = qd * self.dof_vel_scale

        clock_sin = np.sin(2.0 * np.pi * self.gait_phase)
        clock_cos = np.cos(2.0 * np.pi * self.gait_phase)

        obs = np.concatenate([
            ang_vel,                                       # 3
            projected_gravity,                             # 3
            dof_pos_err,                                   # 12
            dof_vel,                                       # 12
            self.last_action,                              # 12
            np.array([clock_sin, clock_cos], dtype=np.float32),  # 2
            self.gait_params,                              # 4
        ]).astype(np.float32)
        return np.clip(obs, -self.clip_observations, self.clip_observations)

    def _action_clip(self, action):
        """PD 可行性裁剪 (与 base_task._action_clip 一致)。"""
        q = np.array(self.get_obs[7:19], dtype=np.float32)
        qd = np.array(self.get_obs[19:31], dtype=np.float32)
        lo = (q - self.default_angles
              + (self.kd_mean * qd - self.user_torque_limit) / self.kp_mean)
        hi = (q - self.default_angles
              + (self.kd_mean * qd + self.user_torque_limit) / self.kp_mean)
        target = np.clip(action * self.action_scale, lo, hi)
        return target / self.action_scale

    def _reset_policy(self):
        """复位策略状态: obs_history 用当前观测重复 hist_len 帧 (匹配 Isaac Gym reset_idx)。"""
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.gait_phase = 0.0
        first_obs = self._build_obs()
        self.obs_history = deque(
            [first_obs.copy() for _ in range(self.hist_len)],
            maxlen=self.hist_len,
        )

    def _start_log(self):
        """开始记录: 每次都新建一个时间戳 CSV (若已有打开的先关)。"""
        self._stop_log()
        path = os.path.join(os.getcwd(), time.strftime("obs_log_%Y%m%d_%H%M%S.csv"))
        self._log_file = open(path, "w", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(
            ["t", "grav_x", "grav_y", "grav_z", "wx", "wy", "wz"]
            + [f"q_{n}" for n in self._log_jn]
            + [f"dq_{n}" for n in self._log_jn]
            + [f"target_{n}" for n in self._log_jn])
        self._log_path = path
        self._log_rows = []
        self._log_t0 = time.time()
        self._logging = True
        print(f"\n[记录] 开始 -> {path}")

    def _stop_log(self):
        """停止记录, 关闭当前文件并把 target 角度出曲线图。"""
        was = self._logging
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
            self._log_writer = None
        self._logging = False
        if was:
            print("\n[记录] 停止")
            self._save_plot()
        self._log_rows = []

    def _mark_log(self, text):
        """在 CSV 里插入 空行 + 注释行 作为事件标记 (如模型启停)。仅在记录中时写入。"""
        if self._logging and self._log_file is not None:
            self._log_file.write(f"\n# ==== {text}  t={time.time() - self._log_t0:.3f}s ====\n")
            self._log_file.flush()

    def _save_plot(self):
        """把记录的 target 角度随时间画成曲线 (所有腿/全部关节)。

        去均值 + 共享 y 轴 => 各关节抖动幅度可直接横向比较, 不会被各子图自动
        缩放误导。蓝=target, 灰=实际 q; 角标给出该关节抖动幅度与稳态跟踪偏置。
        """
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
        q0, tgt0 = 7, 7 + 2 * nj   # q 列 / target 列 在行内的起始下标
        fig, axes = plt.subplots(6, 2, figsize=(12, 13), sharex=True, sharey=True, squeeze=False)
        for j in range(nj):
            ax = axes[j % 6][j // 6]   # 左列 L0-L5, 右列 R0-R5
            tg, q = a[:, tgt0 + j], a[:, q0 + j]
            ax.plot(t, tg - tg.mean(), "b", lw=1.0, label="target")
            ax.plot(t, q - q.mean(), "0.6", lw=0.8, label="actual q")
            ax.set_ylabel(self._log_jn[j])
            ax.grid(True, alpha=0.3)
            ax.text(0.99, 0.04, f"抖={tg.std():.3f}  偏置={(tg - q).mean():+.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.3")
        axes[0][0].legend(loc="upper right", fontsize=8)
        axes[5][0].set_xlabel("t (s)")
        axes[5][1].set_xlabel("t (s)")
        fig.suptitle("各腿关节 target 角度抖动 (去均值, 同一 y 轴尺度直接比幅度)")
        fig.tight_layout()
        p = path.rsplit(".", 1)[0] + "_target.png"
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
            self.commands = self.default_angles.copy()
            print('pause                                   ' if self.pause_symbol
                  else 'model OFF (按 G 启动)                    ', end='\r')
        elif self.counter % self.control_decimation == 0:
            if not self._initialized:
                self._reset_policy()
                self._initialized = True

            # 推进 gait 相位; 速度指令接近 0 时门控为 0 (静止站立)
            self.gait_phase = (self.gait_phase + self.ctrl_dt * self.gait_freq) % 1.0
            if float(np.linalg.norm([self.x_vel, self.y_vel, self.yaw])) < 0.05:
                self.gait_phase = 0.0

            obs = self._build_obs()
            self.obs_history.append(obs)
            obs_hist = np.concatenate(list(self.obs_history)).astype(np.float32)

            commands_scaled = np.array([
                self.x_vel * self.cmd_scale[0],
                self.y_vel * self.cmd_scale[1],
                self.yaw * self.cmd_scale[2],
            ], dtype=np.float32)

            feed = {
                "obs_history": obs_hist[np.newaxis, :],
                "obs":         obs[np.newaxis, :],
                "commands":    commands_scaled[np.newaxis, :],
            }
            action = self.policy.run([self.output_name], feed)[0][0].astype(np.float32)

            action = np.clip(action, -self.clip_actions, self.clip_actions)
            # action = self._action_clip(action)  # 暂时关闭 PD 力矩可行性裁剪, 只保留下方关节限位 safety
            self.last_action = action.copy()
            self.actions = action

            # 目标关节角 = 默认角 + action·scale, 再做硬件安全限位裁剪
            target = self.default_angles + action * self.action_scale
            self.commands = np.clip(target, self.joint_lower_limits, self.joint_upper_limits)

        # 记录: 与模型是否运行无关 (M 开始 / N 停止), 同样按 control_decimation 节流。
        # target 列 = 当前发布指令 (模型在跑=策略输出; 没跑=默认站姿), 能看出模型何时介入。
        if self._logging and self._log_writer is not None and self.counter % self.control_decimation == 0:
            grav = get_gravity_orientation(self.get_obs[3:7])
            row = [time.time() - self._log_t0, grav[0], grav[1], grav[2]] \
                + list(self.get_obs[0:3]) + list(self.get_obs[7:19]) + list(self.get_obs[19:31]) \
                + list(self.commands)
            self._log_writer.writerow([f"{float(v):.6f}" for v in row])
            self._log_file.flush()
            self._log_rows.append([float(v) for v in row])

        msg.data = self.commands.tolist()
        self.commands_publisher.publish(msg)

    # ------------------------------------------------------------------
    # ROS 回调
    # ------------------------------------------------------------------
    def obs_callback(self, msg):
        # /left_joint_states 顺序 L0-L5-R0-R5 == 策略顺序, 直接写入 (无需重排)
        self.get_obs[7:19] = [msg.position[i] for i in range(12)]
        self.get_obs[19:31] = [msg.velocity[i] for i in range(12)]

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

        # A 按钮(buttons[0]): 复位;  B 按钮(buttons[1]): 暂停;  buttons[3]: 解除暂停
        if msg.buttons[0] == 1:
            self.reset_symbol = True
        if msg.buttons[1] == 1:
            self.pause_symbol = True
        if msg.buttons[3] == 1:
            self.pause_symbol = False

    def get_key(self):
        """非阻塞读取键盘并更新速度指令 (终端已在模块加载时设为 cbreak + 非阻塞)。

        W/S: 前进/后退(vx)   A/D: 左移/右移(vy)   Q/E: 左转/右转(yaw)
        空格: 速度清零   R: 复位   P: 暂停/恢复切换
        手柄(/joy)接口保留: 若有手柄发布消息会与键盘共同写入同一组指令。
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
                self.run_model = not self.run_model
                if self.run_model:
                    self._initialized = False   # 每次启动都干净重建 obs_history / gait 相位
                    self._mark_log("MODEL START")
                    print("\n[模型] 启动")
                else:
                    self._mark_log("MODEL STOP")
                    print("\n[模型] 停止 (保持默认站姿)")
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
