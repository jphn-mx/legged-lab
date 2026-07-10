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
from sensor_msgs.msg import Joy
import onnxruntime as ort

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

        config_file = "dog.yaml"

        package_path = get_package_share_directory('rl_real_py')

        config_path = os.path.join(
            package_path,
            'configs',
            config_file
        )
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

        # self.obs_hist = obs_history_gym(self.num_obs,self.num_hist)
        self.obs_hist = obs_history_lab(self.num_obs,self.num_hist)
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

        self.x_vel = self.y_vel = self.yaw = 0.0
        self.get_obs = [0.]*31 
        self.reset_symbol = False
        self.pause_symbol = True
        self.commands = np.array(self.lab2real(self.default_angles),dtype=np.float32)
        self.counter = 0

        self.start_time = 0.
        self.end_time = 0.

        self.start_model_time = 0.
        self.end_model_time = 0.

        self.current_pos = [0.]*12
        self.deadzone = 0.01  # 摇杆死区
        self.speed_scale = 1.0  # 速度缩放因子

        # commands_threading = threading.Thread(target=self.commands_thread)
        # model_threading = threading.Thread(target=self.model_thread)
        
        self.timer = self.create_timer(0.01,self.timer_callback)

        print("rl_real start ...")
        # commands_threading.start()
        # model_threading.start()

    def stand(self,start_pos):
        error = (-np.array(start_pos) + np.array(self.lab2real(self.default_angles)))/1000.0
        msg = Float64MultiArray()
        for i in range(1000):
            commands = np.array(start_pos) + error * (i+1)
            msg.data = commands.tolist()
            self.commands_publisher.publish(msg)
            print(f'start process:{i/1000.0}   ',end='\r')
            time.sleep(0.005)
        print()
        print("stand done")
    
    def joy_callback(self, msg):
        # 左摇杆控制前后左右
        # 左摇杆垂直方向（通常是axes[1]）控制前进后退
        # 左摇杆水平方向（通常是axes[0]）控制左右移动
        # 右摇杆水平方向（通常是axes[2]）控制转向
        
        # 应用死区
        if abs(msg.axes[1]) > self.deadzone:
            self.x_vel = msg.axes[1] * self.speed_scale
        else:
            self.x_vel = 0.0
        
        if abs(msg.axes[0]) > self.deadzone:
            self.y_vel = msg.axes[0] * self.speed_scale  # 反转左右方向
        else:
            self.y_vel = 0.0
        
        if abs(msg.axes[2]) > self.deadzone:
            self.yaw = msg.axes[2] * self.speed_scale  # 反转转向方向
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

    def timer_callback(self):
        msg = Float64MultiArray()
        self.counter += 5
        # self.get_key()
        if self.reset_symbol:
            self.obs_hist.__init__(self.num_obs,self.num_hist)
            self.x_vel = 0.
            self.y_vel = 0.
            self.yaw = 0.
            self.reset_symbol = False
        if self.pause_symbol:
            self.commands = np.array(self.lab2real(self.default_angles),dtype=np.float32)
            print('pause                          ',end='\r')
        else:
            print(f'x_vel:{round(self.x_vel,2)}     y_vel:{round(self.y_vel,2)}     yaw:{round(self.yaw,2)}     \r',end="")
            if self.counter % 8 == 0:
            
                obs = [0.]*45
                obs[0:3] = self.get_obs[:3] # assume omega
                obs[3:6] = get_gravity_orientation(self.get_obs[3:7]) # gravity
                obs[6:9] = np.array([self.x_vel,self.y_vel,self.yaw],dtype=np.float32) * self.cmd_scale
                obs[9:9+self.num_actions] = (self.get_obs[7:19]- self.default_angles) * self.dof_pos_scale # pos
                obs[9+self.num_actions:9+self.num_actions+self.num_actions] = np.array(self.get_obs[19:31]) * self.dof_vel_scale # vel
                obs[-self.num_actions:] = self.actions # last action
                
                total_obs = self.obs_hist.update(obs)
                obs_tensor = torch.clip(torch.from_numpy(total_obs).unsqueeze(0),-100.0,100.0)
                if self.model_type == "jit":
                    self.actions = torch.clip(self.policy(obs_tensor.float()),-100.0,100.0).detach().numpy().squeeze()
                elif self.model_type == "onnx":
                    # ONNX 推理
                    input_name = self.policy.get_inputs()[0].name
                    obs_np = obs_tensor.float().numpy()
                    if obs_np.ndim == 1:
                        obs_np = obs_np[np.newaxis, :]  # 添加批次维度
                    outputs = self.policy.run(None, {input_name: obs_np})
                    self.actions = np.clip(outputs[0], -100.0, 100.0).squeeze()
                    # print("actions",self.actions)
                # self.actions = torch.clip(self.policy(obs_tensor.float()),-100.0,100.0).detach().numpy().squeeze()
                commands = self.lab2real(self.actions * self.action_scale + self.default_angles)
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

        msg.data = self.commands.tolist()
        self.commands_publisher.publish(msg)

    def commands_thread(self):
        while rclpy.ok():
            start_commands_time = time.time()

            msg = Float64MultiArray()
            if self.reset_symbol:
                self.obs_hist.__init__(self.num_obs,self.num_hist)
                self.x_vel,self.y_vel,self.yaw = 0.,0.,0.
                self.commands = self.lab2real(self.default_angles)
                print('reset')
                self.reset_symbol = False
            commands = self.lab2real(self.actions * self.action_scale + self.default_angles)
            self.commands[0] = np.clip(commands[0],-0.4,0.4)
            self.commands[3] = np.clip(commands[3],-0.4,0.4)
            self.commands[6] = np.clip(commands[6],-0.4,0.4)
            self.commands[9] = np.clip(commands[9],-0.4,0.4)

            self.commands[1] = np.clip(commands[1],-0.0,2.)
            self.commands[4] = np.clip(commands[4],-0.0,2.)
            self.commands[7] = np.clip(commands[7],-0.0,2.)
            self.commands[10] = np.clip(commands[10],-0.0,2.)

            self.commands[2] = np.clip(commands[2],-2.,-0.8)
            self.commands[5] = np.clip(commands[5],-2.,-0.8)
            self.commands[8] = np.clip(commands[8],-2.,-0.8)
            self.commands[11] = np.clip(commands[11],-2.,-0.8) ## clip
            msg.data = self.commands.tolist()

            ### test
            # msg.data = [0.0, 0.8000, -1.5, 0.0, 0.800, -1.5, 0.0, 0.8000000, -1.5, 0.0, 0.800000, -1.5]
            # print(msg)
            self.commands_publisher.publish(msg)
            self.get_key()
            print(f'x_vel:{round(self.x_vel,2)}     y_vel:{round(self.y_vel,2)}     yaw:{round(self.yaw,2)}     ',end="\r")

            end_commands_time = time.time()

            dt = end_commands_time - start_commands_time
            if dt < 0.005:
                time.sleep(0.005-dt)
            

    def model_thread(self):
        while rclpy.ok():
            start_time = time.time()

            obs = [0.]*45
            obs[0:3] = self.get_obs[:3] # assume omega
            obs[3:6] = get_gravity_orientation(self.get_obs[3:7]) # gravity
            obs[6:9] = np.array([self.x_vel,self.y_vel,self.yaw],dtype=np.float32) * self.cmd_scale
            obs[9:9+self.num_actions] = (self.get_obs[7:19]- self.default_angles) * self.dof_pos_scale # pos
            obs[9+self.num_actions:9+self.num_actions+self.num_actions] = np.array(self.get_obs[19:31]) * self.dof_vel_scale # vel
            obs[-self.num_actions:] = self.actions # last action
            
            total_obs = self.obs_hist.update(obs)
            obs_tensor = torch.clip(torch.from_numpy(total_obs).unsqueeze(0),-100.0,100.0)
            self.actions = torch.clip(self.policy(obs_tensor.float()),-100.0,100.0).detach().numpy().squeeze()
            # self.commands = self.lab2real(self.actions * self.action_scale + self.default_angles)
        
            end_time = time.time()

            dt = end_time - start_time
            # print("dt",dt)
            if dt < 0.02:
                time.sleep(0.02-dt)
            # else:
            #     print("dt",dt)

    def obs_callback(self,msg):
        self.get_obs[7:19] = self.real2lab([msg.position[0],msg.position[1],msg.position[2],
                                msg.position[3],msg.position[4],msg.position[5],
                                msg.position[6],msg.position[7],msg.position[8],
                                msg.position[9],msg.position[10],msg.position[11]]  )
        self.current_pos = msg.position 

        self.get_obs[19:31] = self.real2lab([msg.velocity[0],msg.velocity[1],msg.velocity[2],
                                msg.velocity[3],msg.velocity[4],msg.velocity[5],
                                msg.velocity[6],msg.velocity[7],msg.velocity[8],
                                msg.velocity[9],msg.velocity[10],msg.velocity[11]])
    
    def imu_callback(self,msg):
        self.get_obs[:3] = [msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z]
        # print(self.get_obs[:3] )
        self.get_obs[3:7] = [msg.orientation.w,msg.orientation.x,msg.orientation.y,msg.orientation.z]


    def real2lab(self,real):
        return [real[9],real[3],real[6],real[0],
                real[10],real[4],real[7],real[1],
                real[11],real[5],real[8],real[2],
                ]
    def lab2real(self,lab):
        return [lab[3],lab[7],lab[11],
                lab[1],lab[5],lab[9],
                lab[2],lab[6],lab[10],
                lab[0],lab[4],lab[8],
                ]

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