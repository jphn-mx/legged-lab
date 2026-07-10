"""IMU 3D 姿态实时可视化 — 手动转 IMU, 看屏幕上的 3D 板是否同步转动。

订阅 /imu, 用 orientation 四元数旋转一个 3D 板(标了"前/上"方向)+ 机体坐标轴,
matplotlib 实时刷新。手里怎么转, 屏幕就该怎么转; 不动/乱动 => IMU 数据有问题。

需要图形界面(桌面 / ssh -X)。

用法:
  ros2 run rl_real_py imu_viz
  ros2 run rl_real_py imu_viz --topic /imu
"""
import argparse
import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def quat_to_matrix(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def quat_to_rpy_deg(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# 机体坐标系下的板: X=前(长), Y=左(宽), Z=上(厚)
HX, HY, HZ = 0.6, 0.4, 0.1
_CORNERS = np.array([[sx * HX, sy * HY, sz * HZ]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
# 角点索引: bit (sx,sy,sz) -> 0..7
_FACES = {
    "front(+X)": [4, 5, 7, 6],   # 高亮: 朝前的面
    "back":      [0, 1, 3, 2],
    "left(+Y)":  [2, 3, 7, 6],
    "right":     [0, 1, 5, 4],
    "top(+Z)":   [1, 3, 7, 5],
    "bottom":    [0, 2, 6, 4],
}


class AttitudeViz3D(Node):
    def __init__(self, args):
        super().__init__("imu_viz")
        self.lock = threading.Lock()
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.got = False
        self.create_subscription(Imu, args.topic, self.cb, qos_profile_sensor_data)

    def cb(self, m):
        o = m.orientation
        with self.lock:
            self.quat = [o.w, o.x, o.y, o.z]
            self.got = True

    def snapshot(self):
        with self.lock:
            return list(self.quat), self.got


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/imu")
    parsed, _ = ap.parse_known_args()

    rclpy.init(args=args)
    node = AttitudeViz3D(parsed)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if matplotlib.get_backend().lower() == "agg":
        print("无可用图形界面(headless)。请在桌面运行, 或 ssh -X 开 X 转发。")
        rclpy.shutdown()
        return

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")

    def draw_world():
        for vec, c in ((np.array([1, 0, 0]), "r"), (np.array([0, 1, 0]), "g"), (np.array([0, 0, 1]), "b")):
            ax.quiver(0, 0, 0, *vec, color=c, alpha=0.25, lw=1, arrow_length_ratio=0.08)

    def update(_):
        q, got = node.snapshot()
        R = quat_to_matrix(*q)
        ax.cla()
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("world X"); ax.set_ylabel("world Y"); ax.set_zlabel("world Z")
        draw_world()  # 世界系参考(淡)

        # 旋转后的板
        v = (R @ _CORNERS.T).T
        faces, colors = [], []
        for name, idx in _FACES.items():
            faces.append([v[i] for i in idx])
            colors.append("crimson" if name == "front(+X)" else "#6fa8dc")
        poly = Poly3DCollection(faces, alpha=0.65, edgecolor="k", linewidths=0.5)
        poly.set_facecolor(colors)
        ax.add_collection3d(poly)

        # 机体坐标轴 (X前=红 Y左=绿 Z上=蓝)
        for k, c in ((0, "r"), (1, "g"), (2, "b")):
            ax.quiver(0, 0, 0, *(R[:, k] * 1.0), color=c, lw=2.5, arrow_length_ratio=0.12)

        r, p, yaw = quat_to_rpy_deg(*q)
        status = "" if got else "  [未收到 /imu]"
        ax.set_title(f"IMU 3D 姿态  roll={r:+.0f}  pitch={p:+.0f}  yaw={yaw:+.0f}{status}\n"
                     f"红面=前(+X)  红/绿/蓝轴=机体X/Y/Z")
        return []

    ani = FuncAnimation(fig, update, interval=60, blit=False, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
