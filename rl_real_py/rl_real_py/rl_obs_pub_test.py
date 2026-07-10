import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

class ObsPubTest(Node):
    def __init__(self,name):
        super().__init__(name)
        # self.get_logger().info("obs pub topic %s" %name)
        self.command_publisher = self.create_publisher(JointState,"obs",10)
        self.obs = [i*1.0 for i in range(31)]
        self.timer = self.create_timer(0.005,self.timer_callback)

    
    def timer_callback(self):
        msg = JointState()
        msg.position = self.obs[7:19]
        msg.velocity = self.obs[19:31]
        self.command_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObsPubTest("obs_test")
    rclpy.spin(node)
    rclpy.shutdown()

