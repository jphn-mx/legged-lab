import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
import yaml
import numpy as np
import time
import csv
import torch
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
    """按 term 分组的历史缓冲, 匹配 IsaacLab 默认 (interleave_by_time=False)。

    先把每个 term 的完整历史拼起来, 再拼所有 term:
      [t1_h0..t1_hN | t2_h0..t2_hN | ...]  (h0 最旧, hN 最新)
    首帧用当前观测填满所有历史 (与仿真 reset 行为一致)。
    移植自 deploy_a1_flat.py; 仿真里的 ObsDelay (人为传感器延迟) 不移植 —— 实机自带真实延迟。
    """

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


# 各观测 term 的维度 (与 obs_index 配合, 用于历史拼接)
def _obs_term_dims(num_actions):
    return {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "velocity_commands": 3,
        "joint_pos": num_actions,
        "joint_vel": num_actions,
        "last_action": num_actions,
        "gait_phase": 2,
    }


class LatencyStat:
    """运行统计 (Welford): 计数 / 均值 / 最大 / 最小 / 标准差, 单位秒。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.max = 0.0
        self.min = float('inf')
        self.last = 0.0
        self.neg = 0   # 负延迟次数 (publisher 时钟与本机不同步的信号)

    def add(self, x):
        self.last = x
        if x < 0:
            self.neg += 1
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)
        if x > self.max:
            self.max = x
        if x < self.min:
            self.min = x

    def std(self):
        return (self._m2 / self.n) ** 0.5 if self.n > 1 else 0.0


qos = QoSProfile(depth=1)
fd = sys.stdin.fileno()
# 保存终端状态
old_term = termios.tcgetattr(fd)
tty.setcbreak(fd)
# 设置为非阻塞
old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)


class RL_real(Node):
    """A1-legs 双足机器人 RSL-RL flat (纯速度) 策略实机部署节点。

    把 deploy_a1_flat.py 的观测/推理管线迁移到 ROS2 (单帧 47 维, 含步态时钟):
      obs(47) = [base_ang_vel(3), projected_gravity(3), velocity_commands(3),
                 joint_pos(12, =q-default), joint_vel(12), last_action(12), gait_phase(2)]  (Lab 序)
      gait_phase = [sin(2*pi*phi), cos(2*pi*phi)], phi = (gait_steps*step_dt) % gait_period / gait_period
      action(12, Lab 序) = policy(obs)
      target_q = default + action·action_scale, 重排到实机序, 再做关节安全限位裁剪

    关节顺序: /left_joint_states (入) 与 /dog_joint_pos (出) 均为 L1..L6,R1..R6 (实机序);
    策略用 Isaac Lab 交错序 (L1,R1,L2,R2,...), 映射 real<->sim 在 __init__ 内按关节名自动推导。
    实机电机自带 PD, 因此仿真里的 PD/力矩部分丢弃, 本节点只发布目标关节角。
    """

    def __init__(self, name):
        super().__init__(name)
        self.obs_subscriber = self.create_subscription(JointState, "/left_joint_states", self.obs_callback, 5)
        self.commands_publisher = self.create_publisher(Float64MultiArray, "/dog_joint_pos", qos)  # L1-L6 R1-R6
        self.imu_subscriber = self.create_subscription(Imu, '/imu', self.imu_callback, 5)
        self.joy_subscriber = self.create_subscription(Joy, "/joy", self.joy_callback, 5)

        config_file = "flat.yaml"
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
            self.num_hist = config["num_hist"]

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)  # Lab 序
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            self.gait_period = config["gait_period"]
            self.step_dt = config["step_dt"]
            self.stand_still = config["stand_still"]

            self.obs_index = config["obs_index"]

            self.joint_index_in_sim = config["joint_index_in_sim"]
            self.joint_index_in_real = config["joint_index_in_real"]
            self.joint_action_index_in_sim = config["joint_action_index_in_sim"]
            self.joint_action_index_in_real = config["joint_action_index_in_real"]

            self.joint_lower_limits = np.array(config["joint_lower_limits"], dtype=np.float32)  # 实机序
            self.joint_upper_limits = np.array(config["joint_upper_limits"], dtype=np.float32)  # 实机序
            # PD 增益 (实机序), 仅用于"理论 PD 力矩"记录
            self.joint_kp = np.array(config["joint_kp"], dtype=np.float32)
            self.joint_kd = np.array(config["joint_kd"], dtype=np.float32)

        # 关节重排映射 (按名字推导, 勿手填)
        # real-order 数组 -> sim(Lab)-order:  sim_arr = real_arr[real2sim]
        self.real2sim = [self.joint_action_index_in_real.index(n) for n in self.joint_action_index_in_sim]
        # sim(Lab)-order 数组 -> real-order:  real_arr = sim_arr[sim2real]
        self.sim2real = [self.joint_index_in_sim.index(n) for n in self.joint_index_in_real]
        print("real2sim =", self.real2sim)
        print("sim2real =", self.sim2real)

        # default_angles 实机序版本 (默认站姿 / pause 时发布)
        self.default_angles_real = self.default_angles[self.sim2real].astype(np.float32)

        # 观测各 term 维度 -> obs_index 对应的 term_dims (历史拼接用)
        term_dim_map = _obs_term_dims(self.num_actions)
        self.term_dims = [term_dim_map[n] for n in self.obs_index]
        assert sum(self.term_dims) == self.num_obs, \
            f"obs term dims {self.term_dims} 之和 {sum(self.term_dims)} != num_obs {self.num_obs}"

        # 加载策略
        self.obs_hist = TermGroupedHistory(self.term_dims, self.num_hist)
        self.actions = np.zeros(self.num_actions, dtype=np.float32)  # Lab 序
        if self.model_type == "jit":
            self.policy_path += '.pt'
            self.policy = torch.jit.load(self.policy_path)
            print(f"Loaded JIT model from {self.policy_path}")
        elif self.model_type == "onnx":
            self.policy_path += '.onnx'
            self.policy = ort.InferenceSession(self.policy_path)
            print(f"Loaded ONNX model from {self.policy_path}")

        # 状态
        self.x_vel = self.y_vel = self.yaw = 0.0
        # get_obs: [0:3] ang_vel, [3:7] quat(wxyz), [7:19] q (Lab 序), [19:31] qd (Lab 序)
        self.get_obs = [0.] * 31
        self.reset_symbol = False
        self.run_model = False   # 模型门控: 上电默认不跑策略, 保持默认角度; 手柄 A 启动 / B 停回默认
        self.commands = self.default_angles_real.copy()  # 实机序发布
        self.counter = 0
        self.gait_steps = 0   # 步态时钟步数 (每个策略步 +1, reset 清零)

        self.deadzone = 0.1   # 摇杆死区
        self.speed_scale = 1.0  # 速度缩放因子
        self.kb_step = 0.1    # 键盘每次按键的速度步长
        self.kb_max = 1.0     # 键盘速度上限 (各轴 ±)
        self._prev_buttons = []  # 手柄按钮上升沿检测

        # ---- 通信延迟测量 ----
        # 1) 传感器消息延迟 now-header.stamp;  2) 到达间隔/抖动 (monotonic)
        self.lat = {'imu': LatencyStat(), 'joint': LatencyStat()}   # now - stamp
        self.iat = {'imu': LatencyStat(), 'joint': LatencyStat()}   # 到达间隔
        self._iat_prev = {'imu': None, 'joint': None}
        self._no_stamp = {'imu': False, 'joint': False}
        self._lat_print_every = max(1, int(round(1.0 / self.simulation_dt)))  # ~1s 打印一次
        # 3) 指令->反馈往返测试 (按 T 触发, 仅暂停态; 对单关节发小阶跃)
        self._last_q_real = None          # 最近一帧实机序关节角 (原始)
        self._rt_active = False
        self._rt_joint = 0                # 测试关节 (实机序索引, L1..L6,R1..R6)
        self._rt_delta = 0.1              # 阶跃幅度 (rad)
        self._rt_thresh = 0.02            # 判定"已响应"的位移阈值 (rad)
        self._rt_timeout = 1.0            # 超时 (s)
        self._rt_t0 = 0.0
        self._rt_baseline = 0.0
        self._rt_target = 0.0
        self.rt_lat = LatencyStat()       # 往返延迟统计

        # ---- effort / 理论 PD 力矩 记录 (按 M 开始 / N 停止) ----
        # get_effort: 实机序 L1..L6,R1..R6 的实测力矩 (/left_joint_states.effort)
        self.get_effort = np.zeros(self.num_actions, dtype=np.float32)
        self._eff_has_data = False         # 话题是否带 effort 字段
        self._last_qd_real = np.zeros(self.num_actions, dtype=np.float32)  # 实机序关节速度
        self._logging = False
        self._log_file = None
        self._log_writer = None
        self._log_path = None
        self._log_t0 = 0.0
        self._log_rows = []
        self._log_jn = [f"L{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 7)]  # 实机序

        print("rl_real (A1-legs flat) start ...")
        print(f"simulation_dt:{self.simulation_dt}  control_decimation:{self.control_decimation}")
        print(f"obs_index: {self.obs_index}")
        print("手柄: 左摇杆=移动(vx/vy)  右摇杆横=转向(wz)  A=运行模型  B=固定默认角度  X=开始记录  Y=停止记录")
        print("键盘(聚焦终端): W/S=前后 A/D=左右 Q/E=转向 空格=清零 R=复位 G=运行/停止切换 M/N=记录开始/停止")
        print("延迟测量: L=打印统计  C=清零统计  T=指令→反馈往返测试(需先停模型, 会动 1 个关节)")

        self.timer = self.create_timer(self.simulation_dt, self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()
        self.counter += 1
        self.get_key()
        print(f'x_vel:{round(self.x_vel,2)}     y_vel:{round(self.y_vel,2)}     yaw:{round(self.yaw,2)}     \r', end="")

        if self.reset_symbol:
            self.obs_hist = TermGroupedHistory(self.term_dims, self.num_hist)
            self.actions = np.zeros(self.num_actions, dtype=np.float32)
            self.gait_steps = 0
            self.x_vel, self.y_vel, self.yaw = 0., 0., 0.
            self.reset_symbol = False
            print("reset")

        if self._rt_active:
            # 往返测试进行中: rt_joint 发阶跃, 其余保持默认; 超时则中止
            self.commands = self.default_angles_real.copy()
            self.commands[self._rt_joint] = self._rt_target
            if time.monotonic() - self._rt_t0 > self._rt_timeout:
                self._rt_active = False
                print(f"\n[往返] 超时 {self._rt_timeout:.2f}s 未检测到关节 {self._rt_joint} 响应 "
                      f"(检查关节是否使能/能否动)")
        elif not self.run_model:
            # 模型未运行: 固定默认角度 (上电默认 / 按 B / 按 G 停)
            self.commands = self.default_angles_real.copy()
            print('model OFF (按 A 运行)                     ', end='\r')
        else:
            if self.counter % self.control_decimation == 0:
                # --- read state (Lab 序) ---
                joint_pos = np.array(self.get_obs[7:19], dtype=np.float32)
                joint_vel = np.array(self.get_obs[19:31], dtype=np.float32)
                # 步态相位 (用当前 gait_steps, 推理后再 +1, 与仿真一致)
                phi = (self.gait_steps * self.step_dt) % self.gait_period / self.gait_period
                angle = 2.0 * np.pi * phi
                if self.stand_still:
                    if np.linalg.norm(np.array([self.x_vel, self.y_vel, self.yaw])) < 0.1:
                        gait_phase = np.array([0.0, 1.0], dtype=np.float32)
                    else:
                        gait_phase = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
                else:
                    gait_phase = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)

                # --- assemble observation per obs_index (单帧 47-d, Lab 序; 历史拼接交给 TermGroupedHistory) ---
                term_obs_list = []
                for idx in self.obs_index:
                    if idx == 'base_ang_vel':
                        term_obs_list.append(np.array(self.get_obs[0:3], dtype=np.float32) * self.ang_vel_scale)
                    elif idx == 'projected_gravity':
                        term_obs_list.append(get_gravity_orientation(self.get_obs[3:7]).astype(np.float32))
                    elif idx == 'velocity_commands':
                        term_obs_list.append(np.array([self.x_vel, self.y_vel, self.yaw], dtype=np.float32) * self.cmd_scale)
                    elif idx == 'joint_pos':
                        term_obs_list.append((joint_pos - self.default_angles) * self.dof_pos_scale)
                    elif idx == 'joint_vel':
                        term_obs_list.append(joint_vel * self.dof_vel_scale)
                    elif idx == 'last_action':
                        term_obs_list.append(self.actions.astype(np.float32))
                    elif idx == 'gait_phase':
                        term_obs_list.append(gait_phase)
                total_obs = self.obs_hist.update(term_obs_list)

                # --- policy inference ---
                obs_tensor = torch.clip(torch.from_numpy(total_obs).unsqueeze(0), -100.0, 100.0)
                if self.model_type == "jit":
                    policy_output = normalize_policy_output(self.policy(obs_tensor.float()))
                    self.actions = torch.clip(policy_output, -100.0, 100.0).detach().cpu().numpy().squeeze()
                elif self.model_type == "onnx":
                    input_name = self.policy.get_inputs()[0].name
                    obs_np = obs_tensor.float().numpy()
                    if obs_np.ndim == 1:
                        obs_np = obs_np[np.newaxis, :]
                    outputs = self.policy.run(None, {input_name: obs_np})
                    self.actions = np.clip(outputs[0], -100.0, 100.0).squeeze()

                self.gait_steps += 1   # 推进步态时钟

                # 目标关节角 (Lab 序) = 默认角 + action·scale, 重排到实机序, 再做硬件安全限位裁剪
                target_sim = self.actions * self.action_scale + self.default_angles
                target_real = target_sim[self.sim2real]
                # self.commands = np.clip(target_real, self.joint_lower_limits, self.joint_upper_limits)
                self.commands = target_real

        msg.data = self.commands.tolist()
        self.commands_publisher.publish(msg)

        if self.counter % self._lat_print_every == 0:
            self._print_latency(periodic=True)

        # 力矩记录 (与模型是否运行无关; 按 control_decimation 节流)
        if self._logging and self.counter % self.control_decimation == 0:
            self._log_step()

    def obs_callback(self, msg):
        # /left_joint_states 顺序为实机序 L1..L6,R1..R6; 重排为 Lab 序后存入 get_obs
        q_real = np.array([msg.position[i] for i in range(12)], dtype=np.float32)
        qd_real = np.array([msg.velocity[i] for i in range(12)], dtype=np.float32)
        self.get_obs[7:19] = q_real[self.real2sim]
        self.get_obs[19:31] = qd_real[self.real2sim]
        self._last_q_real = q_real
        self._last_qd_real = qd_real
        # 实测力矩 (effort 字段; 部分驱动可能不填)
        if len(msg.effort) >= 12:
            self.get_effort = np.array([msg.effort[i] for i in range(12)], dtype=np.float32)
            self._eff_has_data = True
        self._record_comm('joint', msg.header)
        # 往返测试: 检测被测关节是否已越过位移阈值
        if self._rt_active and abs(float(q_real[self._rt_joint]) - self._rt_baseline) > self._rt_thresh:
            dt = time.monotonic() - self._rt_t0
            self.rt_lat.add(dt)
            self._rt_active = False
            print(f"\n[往返] 关节 {self._rt_joint} 响应延迟 = {dt * 1000:.1f} ms "
                  f"(均值 {self.rt_lat.mean * 1000:.1f} / 最大 {self.rt_lat.max * 1000:.1f} ms, n={self.rt_lat.n})")

    def imu_callback(self, msg):
        self.get_obs[0:3] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.get_obs[3:7] = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        self._record_comm('imu', msg.header)

    def _record_comm(self, key, header):
        """记录一条消息的 (now - header.stamp) 延迟 与 到达间隔抖动。"""
        # 到达间隔 (本机单调钟)
        now_m = time.monotonic()
        prev = self._iat_prev[key]
        if prev is not None:
            self.iat[key].add(now_m - prev)
        self._iat_prev[key] = now_m
        # 消息延迟 now - header.stamp (需 publisher 用同一系统时钟打戳)
        stamp_ns = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
        if stamp_ns == 0:
            self._no_stamp[key] = True
            return
        self.lat[key].add((self.get_clock().now().nanoseconds - stamp_ns) * 1e-9)

    def _print_latency(self, periodic=False):
        """打印延迟/抖动统计 (单步 = step_dt, 便于换算成 ObsDelay 的延迟步数)。"""
        def fmt(s, no_stamp=False):
            if s.n == 0:
                return "无 stamp" if no_stamp else "无数据"
            extra = f" 负{s.neg}" if s.neg else ""
            return (f"均{s.mean*1000:.1f} 大{s.max*1000:.1f} 小{s.min*1000:.1f} "
                    f"抖{s.std()*1000:.1f}ms(n{s.n}{extra})")
        steps = lambda s: s.mean / self.step_dt if s.n else 0.0
        print("\n[延迟] now-stamp:  imu " + fmt(self.lat['imu'], self._no_stamp['imu'])
              + "  | joint " + fmt(self.lat['joint'], self._no_stamp['joint']))
        print("[延迟] 到达间隔:  imu " + fmt(self.iat['imu']) + "  | joint " + fmt(self.iat['joint']))
        if self.lat['joint'].n:
            print(f"[延迟] joint 观测延迟 ≈ {steps(self.lat['joint']):.2f} 步 "
                  f"(step_dt={self.step_dt}s) → 可填 ObsDelay min/max_delay_steps")
        if not periodic and self.rt_lat.n:
            print(f"[延迟] 往返(指令→反馈): 均 {self.rt_lat.mean*1000:.1f} / 最大 {self.rt_lat.max*1000:.1f} ms, n={self.rt_lat.n}")

    def _start_roundtrip(self):
        """触发一次指令→反馈往返测试 (仅模型停止时; 对单关节发小阶跃)。"""
        if self.run_model:
            print("\n[往返] 请先停止模型 (按 B / G), 再做往返测试 (避免与策略输出冲突)")
            return
        if self._last_q_real is None:
            print("\n[往返] 还没收到 /left_joint_states, 无法测")
            return
        if self._rt_active:
            return
        self._rt_baseline = float(self._last_q_real[self._rt_joint])
        tgt = self._rt_baseline + self._rt_delta
        lo, hi = float(self.joint_lower_limits[self._rt_joint]), float(self.joint_upper_limits[self._rt_joint])
        self._rt_target = float(np.clip(tgt, lo, hi))
        if abs(self._rt_target - self._rt_baseline) < self._rt_thresh:
            self._rt_target = float(np.clip(self._rt_baseline - self._rt_delta, lo, hi))  # 反向 (撞上限时)
        self._rt_t0 = time.monotonic()
        self._rt_active = True
        print(f"\n[往返] 测试关节 {self._rt_joint}: {self._rt_baseline:.3f} -> {self._rt_target:.3f} rad ...")

    # ------------------------------------------------------------------
    # effort / 理论 PD 力矩 记录
    # ------------------------------------------------------------------
    def _theory_pd_torque(self):
        """按发布的目标角 self.commands 与实测 q/dq 算理论 PD 力矩 (实机序):
        tau = (target - q) * kp + (0 - dq) * kd  (target_dq=0, 与仿真 pd_control 同式)。"""
        return (self.commands - self._last_q_real) * self.joint_kp \
            + (0.0 - self._last_qd_real) * self.joint_kd

    def _aligned_line(self, values, header=False):
        """把一行拼成定宽右对齐、逗号分隔的字符串 (表头与数据列对齐)。
        header=True 时 values 是列名(原样右对齐); 否则按 3 位小数格式化。"""
        cells = []
        for v, w in zip(values, self._log_colw):
            s = v if header else f"{float(v):.3f}"
            cells.append(s.rjust(w))
        return ",".join(cells)

    def _log_step(self):
        """写一行: t, 各关节 target / q / dq / effort(实测) / tau_pd(理论)。

        行尾另追加 MuJoCo 标定回放所需字段 (见 _start_log 表头说明):
          step, action(12, Lab 序原始策略输出), target_sim(12, Lab 序未裁剪目标角),
          quat_w/x/y/z(IMU 原始四元数 wxyz)。这些是开环回放的驱动量与初始姿态。
        """
        if self._last_q_real is None:
            return
        tau_pd = self._theory_pd_torque()
        # 行尾追加标量上下文 (放最后, 不影响出图用的 per-joint 列索引)
        ang_vel = self.get_obs[0:3]                                  # 原始 imu 角速度
        grav = get_gravity_orientation(self.get_obs[3:7])           # projected_gravity
        phase = (self.gait_steps * self.step_dt) % self.gait_period / self.gait_period
        gait = [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)]
        # --- MuJoCo 标定回放字段 (Lab 序, 仅记录, 不参与控制) ---
        action = np.asarray(self.actions, dtype=np.float64).reshape(-1)   # 策略原始输出, 未 scale/未裁剪
        target_sim = action * self.action_scale + self.default_angles     # Lab 序目标角, 裁剪/重排前
        quat = self.get_obs[3:7]                                          # IMU 原始四元数 (w, x, y, z)
        row = [time.time() - self._log_t0] + list(self.commands) + list(self._last_q_real) \
            + list(self._last_qd_real) + list(self.get_effort) + list(tau_pd) \
            + list(ang_vel) + list(grav) + [self.x_vel, self.y_vel, self.yaw] + gait \
            + [self.gait_steps] + list(action) + list(target_sim) + list(quat)
        # 定宽对齐写入 (列与表头对齐); 内存 _log_rows 仍存全精度供出图
        self._log_file.write(self._aligned_line(row) + "\n")
        self._log_file.flush()
        self._log_rows.append([float(v) for v in row])

    def _start_log(self):
        """开始记录: 每次新建一个时间戳 CSV。"""
        self._stop_log()
        path = os.path.join(os.getcwd(), time.strftime("flat_torque_%Y%m%d_%H%M%S.csv"))
        self._log_file = open(path, "w", newline="")
        # Lab(策略)序关节名: L1,R1,L2,R2,...; 用于 action / target_sim 列 (区别于行首实机序的 target_/q_/dq_)
        sim_jn = self.joint_index_in_sim
        self._log_header = (
            ["t"] + [f"target_{n}" for n in self._log_jn] + [f"q_{n}" for n in self._log_jn]
            + [f"dq_{n}" for n in self._log_jn] + [f"effort_{n}" for n in self._log_jn]
            + [f"tau_pd_{n}" for n in self._log_jn]
            + ["wx", "wy", "wz", "grav_x", "grav_y", "grav_z",
               "cmd_vx", "cmd_vy", "cmd_wz", "gait_sin", "gait_cos"]
            # --- MuJoCo 标定回放字段 (行尾追加) ---
            + ["step"]                                   # 控制步序号 (= gait_steps, A/G 启动时清零)
            + [f"action_{n}" for n in sim_jn]            # 策略原始输出, Lab 序, 未 scale/未裁剪 (回放驱动量)
            + [f"target_sim_{n}" for n in sim_jn]        # = action*action_scale+default, Lab 序, 裁剪/重排前
            + ["quat_w", "quat_x", "quat_y", "quat_z"])  # IMU 原始姿态四元数, wxyz 序
        # 每列定宽 (表头与数据同宽, 右对齐); 仍逗号分隔, pandas/Excel 可读 (读时用 skipinitialspace=True)
        self._log_colw = [max(len(h), 9) for h in self._log_header]
        self._log_file.write(self._aligned_line(self._log_header, header=True) + "\n")
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
        except ImportError:
            print("[记录] 无 matplotlib, 跳过出图")
            return
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
        # 上升沿触发 (按住只触发一次), 否则按住 A 会每帧重置历史、按住 X 会反复新建日志。
        def pressed(i):
            now = len(msg.buttons) > i and msg.buttons[i] == 1
            was = len(self._prev_buttons) > i and self._prev_buttons[i] == 1
            return now and not was
        if pressed(0):          # A: 运行模型 (干净启动: 重置历史/相位/动作/指令滤波)
            self._start_model()
        if pressed(1):          # B: 停回默认角度
            self._stop_model()
        if pressed(3):          # X: 开始记录
            self._start_log()
        if pressed(4):          # Y: 停止记录
            self._stop_log()
        self._prev_buttons = list(msg.buttons)

    def _start_model(self):
        """启动策略: 干净重置 obs 历史/相位/动作, 清零速度指令。"""
        self.obs_hist = TermGroupedHistory(self.term_dims, self.num_hist)
        self.actions = np.zeros(self.num_actions, dtype=np.float32)
        self.gait_steps = 0
        self.x_vel = self.y_vel = self.yaw = 0.0
        self.run_model = True
        print("\n[模型] 运行")

    def _stop_model(self):
        """停止策略, 固定默认角度。"""
        self.run_model = False
        self.x_vel = self.y_vel = self.yaw = 0.0
        print("\n[模型] 停止 (固定默认角度)")

    def get_key(self):
        """非阻塞读取键盘并更新速度指令 (终端已在模块加载时设为 cbreak + 非阻塞)。

        W/S: 前进/后退(vx)   A/D: 左移/右移(vy)   Q/E: 左转/右转(yaw)
        空格: 速度清零   R: 复位   G: 运行/停止模型切换
        """
        while True:
            try:
                ch = sys.stdin.read(1)
            except (IOError, OSError):
                ch = ''
            if not ch:
                break
            if ch in ('w', 'W'):
                self.x_vel = min(self.kb_max, self.x_vel + self.kb_step)
            elif ch in ('s', 'S'):
                self.x_vel = max(-self.kb_max, self.x_vel - self.kb_step)
            elif ch in ('a', 'A'):
                self.y_vel = min(self.kb_max, self.y_vel + self.kb_step)
            elif ch in ('d', 'D'):
                self.y_vel = max(-self.kb_max, self.y_vel - self.kb_step)
            elif ch in ('q', 'Q'):
                self.yaw = min(self.kb_max, self.yaw + self.kb_step)
            elif ch in ('e', 'E'):
                self.yaw = max(-self.kb_max, self.yaw - self.kb_step)
            elif ch == ' ':
                self.x_vel = self.y_vel = self.yaw = 0.0
            elif ch in ('r', 'R'):
                self.reset_symbol = True
            elif ch in ('g', 'G'):   # 运行/停止模型切换 (= 手柄 A/B)
                if self.run_model:
                    self._stop_model()
                else:
                    self._start_model()
            elif ch in ('l', 'L'):
                self._print_latency(periodic=False)
            elif ch in ('c', 'C'):
                for s in (*self.lat.values(), *self.iat.values(), self.rt_lat):
                    s.reset()
                self._iat_prev = {'imu': None, 'joint': None}
                print("\n[延迟] 统计已清零")
            elif ch in ('t', 'T'):
                self._start_roundtrip()
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
