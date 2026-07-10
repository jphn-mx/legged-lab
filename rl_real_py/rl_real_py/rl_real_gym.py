import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
import argparse
import yaml
import numpy as np
import torch
import time
import threading
import onnxruntime as ort
from sensor_msgs.msg import Joy

from rl_real_py.obs_history import obs_history_gym,obs_history_lab
from rl_real_py.utils.math import get_gravity_orientation
# keyboard
import sys
import termios
import tty
import select
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
    def __init__(self,name):
        super().__init__(name)
        self.obs_subscriber = self.create_subscription(JointState,"/left_joint_states",self.obs_callback,5)
        self.commands_publisher = self.create_publisher(Float64MultiArray,"/dog_joint_pos",qos) # index 左前J1J2J3 右前J1J2J3 左后J1J2J3 右后J1J2J3
        # self.commands_publisher2 = self.create_publisher(Float64MultiArray,"/dog_joint_pos2",qos) # index 左前J1J2J3 右前J1J2J3 左后J1J2J3 右后J1J2J3
        self.imu_subscriber = self.create_subscription(Imu,'/imu',self.imu_callback,5)
        self.joy_subscriber = self.create_subscription(Joy,"/joy",self.joy_callback,5)
        config_file = "dog_gym.yaml"

        package_path = get_package_share_directory('rl_real_py')
        # package_path = os.path.abspath(os.path.join(package_path, '..', '..'))
        config_path = os.path.join(
            package_path,
            '..', '..', '..', '..',
            'src',
            'rl_real_py',
            'configs',
            config_file
        )
        print(config_path)
        with open(config_path,'r') as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
            self.policy_path = os.path.join(package_path,config['policy_path'])
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.num_hist = config["num_hist"]

            self.simulation_dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]
            self.model_type = config["model_type"]

            self.default_angles = np.array(config["default_angles"],dtype=np.float32)
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            # self.cmd = np.array(config["cmd_init"], dtype=np.float32)
            self.simulator_type = config["simulator_type"]
            self.joint_index_in_sim = config["joint_index_in_sim"]
            self.joint_index_in_real = config["joint_index_in_real"]
            self.joint_action_index_in_sim = config["joint_action_index_in_sim"]
            self.joint_action_index_in_real = config["joint_action_index_in_real"]
            self.obs_index = config["obs_index"]
        
        self.sim2real = [self.joint_index_in_sim.index(name) for name in self.joint_index_in_real]
        self.real2sim = [self.joint_action_index_in_real.index(name) for name in self.joint_action_index_in_sim]
        print(self.real2sim)
        print(self.sim2real)

        # self.obs_hist = obs_history_gym(self.num_obs,self.num_hist)
        self.obs_hist = obs_history_gym(self.num_obs,self.num_hist)
        self.actions = np.zeros(self.num_actions, dtype=np.float32) 
        if self.model_type == "jit":
            self.policy_path += '.pt'
            self.policy = torch.jit.load(self.policy_path)
            print(f"Loaded JIT model from {self.policy_path}")
        elif self.model_type == "onnx":
            # 检查文件扩展名是否为.onnx
            self.policy_path += '.onnx'
            print(self.policy_path)
            self.policy = ort.InferenceSession(self.policy_path)
            print(f"Loaded ONNX model from {self.policy_path}")     
        # self.policy = torch.jit.load(self.policy_path)   

        self.x_vel = self.y_vel = self.yaw = 0.0
        self.get_obs = [0.]*31 
        self.reset_symbol = False
        self.pause_symbol = False
        self.commands = np.array(self.default_angles[self.sim2real],dtype=np.float32)
        self.counter = 0

        self.start_time = 0.
        self.end_time = 0.

        self.start_model_time = 0.
        self.end_model_time = 0.

        self.current_pos = [0.]*12
        self.deadzone = 0.1  # 摇杆死区
        self.speed_scale = 1.0  # 速度缩放因子

        # commands_threading = threading.Thread(target=self.commands_thread)
        # model_threading = threading.Thread(target=self.model_thread)

        print("rl_real start ...")
        # commands_threading.start()
        # model_threading.start()

        # self.start_commands_time = 0.
        # self.end_commands_time = 0.
        print(f"simulation_dt:{self.simulation_dt}")
        print(f"control_decimation:{self.control_decimation}")

        self.timer = self.create_timer(self.simulation_dt,self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()
        self.counter += 1
        # self.get_key()
        print(f'x_vel:{round(self.x_vel,2)}     y_vel:{round(self.y_vel,2)}     yaw:{round(self.yaw,2)}     \r',end="")
        if self.reset_symbol:
            self.obs_hist.__init__(self.num_obs,self.num_hist)
            self.x_vel,self.y_vel,self.yaw = 0.,0.,0.
            self.reset_symbol = False
            print("reset")
        if self.pause_symbol:
            self.commands = np.array(self.default_angles[self.sim2real],dtype=np.float32)
            print('pause                                   ',end='\r')
        else:
            if self.counter % self.control_decimation == 0:
            
                obs = []
                for idx in self.obs_index:
                    if idx == 'last_action':
                        obs.append(self.actions)
                    if idx == 'velocity_commands':
                        obs.append(np.array([self.x_vel,self.y_vel,self.yaw],dtype=np.float32) * self.cmd_scale)
                    if idx == 'base_ang_vel':
                        obs.append(self.get_obs[:3])
                    if idx == 'projected_gravity':
                        obs.append(get_gravity_orientation(self.get_obs[3:7]))
                    if idx == 'joint_pos_rel':
                        obs.append((self.get_obs[7:19]- self.default_angles) * self.dof_pos_scale )
                    if idx == 'joint_vel_rel':
                        obs.append(np.array(self.get_obs[19:31]) * self.dof_vel_scale )
                obs = np.concatenate(obs,axis=0)
                total_obs = self.obs_hist.update(obs)
                obs_tensor = torch.clip(torch.from_numpy(total_obs).unsqueeze(0),-100.0,100.0)
                if self.model_type == "jit":
                    policy_output = normalize_policy_output(self.policy(obs_tensor.float()))
                    self.actions = torch.clip(policy_output,-100.0,100.0).detach().cpu().numpy().squeeze()
                    # self.actions = torch.clip(self.policy(obs_tensor.float()),-100.0,100.0).detach().numpy().squeeze()
                elif self.model_type == "onnx":
                    # ONNX 推理
                    input_name = self.policy.get_inputs()[0].name
                    obs_np = obs_tensor.float().numpy()
                    if obs_np.ndim == 1:
                        obs_np = obs_np[np.newaxis, :]  # 添加批次维度
                    outputs = self.policy.run(None, {input_name: obs_np})
                    self.actions = np.clip(outputs[0], -100.0, 100.0).squeeze()
                # self.actions = torch.clip(self.policy(obs_tensor.float()),-100.0,100.0).detach().numpy().squeeze()
                commands = (self.actions * self.action_scale + self.default_angles)[self.sim2real]
                # print(commands)
                # print("commands",commands)
                self.commands[0] = np.clip(commands[0],-0.4,0.4)
                self.commands[3] = np.clip(commands[3],-0.4,0.4)
                self.commands[6] = np.clip(commands[6],-0.4,0.4)
                self.commands[9] = np.clip(commands[9],-0.4,0.4)

                self.commands[1] = np.clip(commands[1],-0.0,2.)
                self.commands[4] = np.clip(commands[4],-0.0,2.)
                self.commands[7] = np.clip(commands[7],-0.0,2.)
                self.commands[10] = np.clip(commands[10],-0.0,2)

                self.commands[2] = np.clip(commands[2],-2.,-0.8)
                self.commands[5] = np.clip(commands[5],-2.,-0.8)
                self.commands[8] = np.clip(commands[8],-2.,-0.8)
                self.commands[11] = np.clip(commands[11],-2.,-0.8) ## clip

                self.start_model_time = time.time()
                dt = self.start_model_time - self.end_model_time
                self.end_model_time = self.start_model_time
                # print(f"run model,duration:{dt}")
                # if dt < 0.02:
                #     time.sleep(0.02 - dt)
                # vel = [round(x,2) for x in self.get_obs[19:31]]
                # print("current vel",vel)
        msg.data = self.commands.tolist()
        
        self.commands_publisher.publish(msg)

    def obs_callback(self,msg):
        self.get_obs[7:19] = [msg.position[0],msg.position[1],msg.position[2],
                                msg.position[3],msg.position[4],msg.position[5],
                                msg.position[6],msg.position[7],msg.position[8],
                                msg.position[9],msg.position[10],msg.position[11]][self.real2sim]
        self.current_pos = msg.position 
        self.get_obs[19:31] = [msg.velocity[0],msg.velocity[1],msg.velocity[2],
                                msg.velocity[3],msg.velocity[4],msg.velocity[5],
                                msg.velocity[6],msg.velocity[7],msg.velocity[8],
                                msg.velocity[9],msg.velocity[10],msg.velocity[11]][self.real2sim]
    
    def imu_callback(self,msg):
        self.get_obs[:3] = [msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z]
        # print(self.get_obs[:3] )
        self.get_obs[3:7] = [msg.orientation.w,msg.orientation.x,msg.orientation.y,msg.orientation.z]

### gym
### obs pos index ['joint_YH2', 'joint_YH3', 'joint_YH4', 'joint_YQ2', 'joint_YQ3', 'joint_YQ4', 'joint_ZH2', 'joint_ZH3', 'joint_ZH4', 'joint_ZQ2', 'joint_ZQ3', 'joint_ZQ4']
### action index ['joint_YH2', 'joint_YH3', 'joint_YH4', 'joint_YQ2', 'joint_YQ3', 'joint_YQ4', 'joint_ZH2', 'joint_ZH3', 'joint_ZH4', 'joint_ZQ2', 'joint_ZQ3', 'joint_ZQ4']

    def real2gym(self,real):
        return [real[9],real[10],real[11],
                real[3],real[4],real[5],
                real[6],real[7],real[8],
                real[0],real[1],real[2],
                ]
    def gym2real(self,gym):
        return [gym[9],gym[10],gym[11],
                gym[3],gym[4],gym[5],
                gym[6],gym[7],gym[8],
                gym[0],gym[1],gym[2],
                ]

    def joy_callback(self, msg):
        # 左摇杆控制前后左右
        # 左摇杆垂直方向（通常是axes[1]）控制前进后退
        # 左摇杆水平方向（通常是axes[0]）控制左右移动
        # 右摇杆水平方向（通常是axes[3]）控制转向
        
        # 应用死区
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
        
        # 按钮控制
        # A按钮（通常是buttons[0]）重置
        if msg.buttons[0] == 1:
            self.reset_symbol = True
        # B按钮（通常是buttons[1]）停止
        if msg.buttons[1] == 1:
            self.pause_symbol = True
        if msg.buttons[3] == 1:
            self.pause_symbol = False
        
        # 可以根据需要添加更多按钮功能
        # 例如：X按钮加速，Y按钮减速等
    """
    def get_key(self):
        try:
            ch = sys.stdin.read(1)
        except IOError:
            return  # 没有输入时直接返回

        if not ch:
            return

        if ch == 'w':
            self.x_vel += 0.1
        elif ch == 's':
            self.x_vel -= 0.1
        elif ch == 'd':
            self.yaw -= 0.1
        elif ch == 'a':
            self.yaw += 0.1
        elif ch == 'l':
            self.y_vel -= 0.1
        elif ch == 'j':
            self.y_vel += 0.1
        elif ch == ' ':
            self.x_vel = self.y_vel = self.yaw = 0.0
        elif ch == 'r':
            self.reset_symbol = True
    """
    

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
        rclpy.shutdown()

if __name__ == '__main__':
    main()