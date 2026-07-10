import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
import yaml
import numpy as np
import torch
import onnxruntime as ort
from sensor_msgs.msg import Joy

from rl_real_py.obs_history import obs_history_gym
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


qos = QoSProfile(depth=1)
fd = sys.stdin.fileno()
# 保存终端状态
old_term = termios.tcgetattr(fd)
tty.setcbreak(fd)
# 设置为非阻塞
old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)


class RL_real(Node):
    """A1-legs 双足机器人 RSL-RL flat 策略实机部署节点。

    把 test/deploy_mujoco_a1.py 的观测/推理管线迁移到 ROS2 (单帧 49 维):
      obs(49) = [joint_pos_rel(12), joint_vel(12), last_action(12), base_ang_vel(3),
                 base_quat_w(4), velocity_commands(3), projected_gravity(3)]   (Lab 序)
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

        config_file = "yao.yaml"
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
            self.gait_period = config.get("gait_period", 0.7)
            # 指令低通软启动时间常数 (deploy_mujoco_a1(2)); 0 = 不滤波 (忠实 step 指令)
            self.cmd_lin_tau = config.get("command_lin_time_constant", 0.4)
            # 步态相位开关: False = 最早的无相位版本 (obs 去掉 gait_phase, 需配无相位策略)
            self.use_gait_phase = bool(config.get("use_gait_phase", True))

            self.obs_index = config["obs_index"]

            self.joint_index_in_sim = config["joint_index_in_sim"]
            self.joint_index_in_real = config["joint_index_in_real"]
            self.joint_action_index_in_sim = config["joint_action_index_in_sim"]
            self.joint_action_index_in_real = config["joint_action_index_in_real"]

            self.joint_lower_limits = np.array(config["joint_lower_limits"], dtype=np.float32)  # 实机序
            self.joint_upper_limits = np.array(config["joint_upper_limits"], dtype=np.float32)  # 实机序

        # 关掉相位时, 从 obs_index 里剔除 gait_phase (其余 term 顺序不变)
        if not self.use_gait_phase:
            self.obs_index = [t for t in self.obs_index if t != 'gait_phase']

        # 关节重排映射 (按名字推导, 勿手填)
        # real-order 数组 -> sim(Lab)-order:  sim_arr = real_arr[real2sim]
        self.real2sim = [self.joint_action_index_in_real.index(n) for n in self.joint_action_index_in_sim]
        # sim(Lab)-order 数组 -> real-order:  real_arr = sim_arr[sim2real]
        self.sim2real = [self.joint_index_in_sim.index(n) for n in self.joint_index_in_real]
        print("real2sim =", self.real2sim)
        print("sim2real =", self.sim2real)

        # default_angles 实机序版本 (默认站姿 / pause 时发布)
        self.default_angles_real = self.default_angles[self.sim2real].astype(np.float32)

        # 加载策略
        self.obs_hist = obs_history_gym(self.num_obs, self.num_hist)
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
        # 指令低通软启动: self.cmd 是喂给策略的滤波后指令 (target = x_vel/y_vel/yaw)
        ctrl_dt = self.simulation_dt * self.control_decimation
        self.cmd_alpha = 1.0 if self.cmd_lin_tau <= 0.0 else ctrl_dt / (self.cmd_lin_tau + ctrl_dt)
        self.cmd = np.zeros(3, dtype=np.float32)
        # get_obs: [0:3] ang_vel, [3:7] quat(wxyz), [7:19] q (Lab 序), [19:31] qd (Lab 序)
        self.get_obs = [0.] * 31
        self.reset_symbol = False
        self.pause_symbol = False
        self.commands = self.default_angles_real.copy()  # 实机序发布
        self.counter = 0

        self.deadzone = 0.1   # 摇杆死区
        self.speed_scale = 0.4  # 速度缩放因子 (对齐 rl_real_flat)
        self.kb_step = 0.1    # 键盘每次按键的速度步长
        self.kb_max = 1.0     # 键盘速度上限 (各轴 ±)

        print("rl_real (A1-legs flat) start ...")
        print(f"simulation_dt:{self.simulation_dt}  control_decimation:{self.control_decimation}")
        print(f"obs_index: {self.obs_index}")
        _dims = {'joint_pos_rel': 12, 'joint_vel': 12, 'last_action': 12,
                 'base_ang_vel': 3, 'velocity_commands': 3, 'projected_gravity': 3, 'gait_phase': 2}
        print(f"use_gait_phase: {self.use_gait_phase}  (单帧 obs 维度 = {sum(_dims[t] for t in self.obs_index)})")
        print("键盘控制 (聚焦终端): W/S=前后  A/D=左右  Q/E=转向  空格=清零  R=复位  P=暂停切换")

        self.timer = self.create_timer(self.simulation_dt, self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()
        self.counter += 1
        self.get_key()
        print(f'x_vel:{round(self.x_vel,2)}     y_vel:{round(self.y_vel,2)}     yaw:{round(self.yaw,2)}     \r', end="")

        if self.reset_symbol:
            self.obs_hist = obs_history_gym(self.num_obs, self.num_hist)
            self.actions = np.zeros(self.num_actions, dtype=np.float32)
            self.x_vel, self.y_vel, self.yaw = 0., 0., 0.
            self.cmd[:] = 0.0
            self.reset_symbol = False
            print("reset")

        if self.pause_symbol:
            self.commands = self.default_angles_real.copy()
            print('pause                                   ', end='\r')
        else:
            if self.counter % self.control_decimation == 0:
                # --- read state (Lab 序) ---
                joint_pos = np.array(self.get_obs[7:19], dtype=np.float32)
                joint_vel = np.array(self.get_obs[19:31], dtype=np.float32)
                # 指令低通软启动 (deploy_mujoco_a1(2)): vx/vy 低通, yaw 直通; tau=0 关闭。
                cmd_target = np.array([self.x_vel, self.y_vel, self.yaw], dtype=np.float32)
                self.cmd[0:2] += self.cmd_alpha * (cmd_target[0:2] - self.cmd[0:2])
                self.cmd[2] = cmd_target[2]
                # 步态相位: 仅当(滤波后)指令非零时跑时钟, 站立时冻结为 [0,0] (原地不抽动)
                if self.use_gait_phase:
                    if np.linalg.norm(self.cmd[0:2]) > 0.1 or abs(self.cmd[2]) > 0.1:
                        phase = (self.counter * self.simulation_dt) % self.gait_period / self.gait_period
                        angle = 2.0 * np.pi * phase
                        gait_phase = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
                    else:
                        gait_phase = np.zeros(2, dtype=np.float32)

                # --- assemble observation per obs_index (含/不含 gait_phase, Lab 序) ---
                obs = []
                for idx in self.obs_index:
                    if idx == 'joint_pos_rel':
                        obs.append((joint_pos - self.default_angles) * self.dof_pos_scale)
                    elif idx == 'joint_vel':
                        obs.append(joint_vel * self.dof_vel_scale)
                    elif idx == 'last_action':
                        obs.append(self.actions)
                    elif idx == 'base_ang_vel':
                        obs.append(np.array(self.get_obs[0:3], dtype=np.float32) * self.ang_vel_scale)
                    elif idx == 'velocity_commands':
                        obs.append(self.cmd * self.cmd_scale)
                    elif idx == 'projected_gravity':
                        obs.append(get_gravity_orientation(self.get_obs[3:7]))
                    elif idx == 'gait_phase':
                        obs.append(gait_phase)
                obs = np.concatenate(obs, axis=0).astype(np.float32)
                total_obs = self.obs_hist.update(obs)

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

                # 目标关节角 (Lab 序) = 默认角 + action·scale, 重排到实机序, 再做硬件安全限位裁剪
                target_sim = self.actions * self.action_scale + self.default_angles
                target_real = target_sim[self.sim2real]
                self.commands = np.clip(target_real, self.joint_lower_limits, self.joint_upper_limits)

        msg.data = self.commands.tolist()
        self.commands_publisher.publish(msg)

    def obs_callback(self, msg):
        # /left_joint_states 顺序为实机序 L1..L6,R1..R6; 重排为 Lab 序后存入 get_obs
        q_real = np.array([msg.position[i] for i in range(12)], dtype=np.float32)
        qd_real = np.array([msg.velocity[i] for i in range(12)], dtype=np.float32)
        self.get_obs[7:19] = q_real[self.real2sim]
        self.get_obs[19:31] = qd_real[self.real2sim]

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
            elif ch in ('p', 'P'):
                self.pause_symbol = not self.pause_symbol


def restore_terminal():
    termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)


def main(args=None):
    rclpy.init(args=args)
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
