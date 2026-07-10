"""IMU 记录 + 出图 (独立节点, 单文件)。

订阅 /imu, 逐帧记录 角速度(wx,wy,wz) 与 重力分量(grav_x,y,z, 由 orientation 算)到 CSV;
Ctrl-C 停止时分别出两张图: 角速度图、重力分量图。

用法:
  ros2 run rl_real_py imu_check
  ros2 run rl_real_py imu_check --topic /imu --outfile ~/imu.csv
"""
import argparse
import csv
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def gravity_from_quat(qw, qx, qy, qz):
    """与 utils.math.get_gravity_orientation 一致 (四元数顺序 w,x,y,z)。"""
    return (2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz))


class ImuCheck(Node):
    def __init__(self, args):
        super().__init__("imu_check")
        self.args = args
        self.f = open(args.outfile, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "wx", "wy", "wz", "grav_x", "grav_y", "grav_z"])
        self.rows = []
        self.t0 = None
        self.n = 0
        self.create_subscription(Imu, args.topic, self.cb, qos_profile_sensor_data)
        print(f"订阅 {args.topic}, 记录角速度+重力分量 -> {args.outfile}  (Ctrl-C 停止并出图)")

    def cb(self, m):
        t = time.time()
        if self.t0 is None:
            self.t0 = t
        o = m.orientation
        g = gravity_from_quat(o.w, o.x, o.y, o.z)
        row = [t - self.t0, m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
               g[0], g[1], g[2]]
        self.rows.append(row)
        self.w.writerow([f"{v:.6f}" for v in row])
        self.f.flush()
        self.n += 1
        if self.n % 50 == 0:
            print(f"\r已记录 {self.n} 帧  w=[{row[1]:+.3f} {row[2]:+.3f} {row[3]:+.3f}]  "
                  f"grav=[{row[4]:+.3f} {row[5]:+.3f} {row[6]:+.3f}]   ", end="")


def save_plots(rows, outfile):
    if len(rows) < 2:
        print("数据太少, 不出图")
        return
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("无 matplotlib, 跳过出图")
        return
    a = np.array(rows)
    base = outfile.rsplit(".", 1)[0]

    # 图1: 角速度
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, (c, col) in enumerate((("wx", "r"), ("wy", "g"), ("wz", "b"))):
        ax.plot(a[:, 0], a[:, 1 + i], col, lw=1.0, label=c)
    ax.set_xlabel("t (s)"); ax.set_ylabel("angular velocity (rad/s)")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    p1 = base + "_angvel.png"
    fig.savefig(p1, dpi=120); plt.close(fig)

    # 图2: 重力分量
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, (c, col) in enumerate((("grav_x", "r"), ("grav_y", "g"), ("grav_z", "b"))):
        ax.plot(a[:, 0], a[:, 4 + i], col, lw=1.0, label=c)
    ax.set_xlabel("t (s)"); ax.set_ylabel("projected gravity")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    p2 = base + "_gravity.png"
    fig.savefig(p2, dpi=120); plt.close(fig)

    print(f"图 -> {p1}\n     {p2}")


def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="/imu")
    p.add_argument("--outfile", default=os.path.join(os.getcwd(), time.strftime("imu_%Y%m%d_%H%M%S.csv")))
    parsed, _ = p.parse_known_args()

    rclpy.init(args=args)
    node = None
    try:
        node = ImuCheck(parsed)
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.f.close()
            print(f"\n停止, 共 {node.n} 帧 -> {parsed.outfile}")
            save_plots(node.rows, parsed.outfile)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
