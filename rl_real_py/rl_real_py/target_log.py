"""Policy target 角度 记录 + 出图 (独立节点, 单文件)。

订阅 policy 输出的目标关节角 (/dog_joint_pos, Float64MultiArray), 逐帧记录到 CSV;
Ctrl-C 停止时把每个关节的 target 角度随时间画成曲线 (每关节一个子图)。

关节数由消息长度自动判断。12 关节时默认按 dual_tron 顺序标注 L1..L6 / R1..R6。

用法:
  ros2 run rl_real_py target_log
  ros2 run rl_real_py target_log --topic /dog_joint_pos --outfile ~/target.csv
"""
import argparse
import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def joint_labels(n):
    """默认关节名: 12 关节用 dual_tron 顺序 (L1..L6, R1..R6), 否则 j0..j(n-1)。"""
    if n == 12:
        return [f"L{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 7)]
    return [f"j{i}" for i in range(n)]


class TargetLog(Node):
    def __init__(self, args):
        super().__init__("target_log")
        self.args = args
        self.f = open(args.outfile, "w", newline="")
        self.w = csv.writer(self.f)
        self.header_written = False
        self.rows = []
        self.t0 = None
        self.n = 0
        self.create_subscription(Float64MultiArray, args.topic, self.cb, 10)
        print(f"订阅 {args.topic}, 记录 policy target 角度 -> {args.outfile}  (Ctrl-C 停止并出图)")

    def cb(self, m):
        t = time.time()
        if self.t0 is None:
            self.t0 = t
        vals = list(m.data)
        if not self.header_written:
            self.w.writerow(["t"] + joint_labels(len(vals)))
            self.header_written = True
        row = [t - self.t0] + vals
        self.rows.append(row)
        self.w.writerow([f"{v:.6f}" for v in row])
        self.f.flush()
        self.n += 1
        if self.n % 50 == 0:
            print(f"\r已记录 {self.n} 帧  {len(vals)} 关节   ", end="")


def save_plot(rows, outfile):
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
    t = a[:, 0]
    njoint = a.shape[1] - 1
    labels = joint_labels(njoint)

    ncol = 2
    nrow = math.ceil(njoint / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.2 * nrow), sharex=True, squeeze=False)
    for j in range(njoint):
        ax = axes[j // ncol][j % ncol]
        ax.plot(t, a[:, 1 + j], lw=1.0)
        ax.set_ylabel(labels[j])
        ax.grid(True, alpha=0.3)
    for k in range(njoint, nrow * ncol):  # 关掉多余的空子图
        axes[k // ncol][k % ncol].axis("off")
    for c in range(ncol):
        axes[nrow - 1][c].set_xlabel("t (s)")
    fig.suptitle("policy target joint angles (rad)")
    fig.tight_layout()
    p = outfile.rsplit(".", 1)[0] + ".png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"图 -> {p}")


def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="/dog_joint_pos")
    p.add_argument("--outfile", default=os.path.join(os.getcwd(), time.strftime("target_%Y%m%d_%H%M%S.csv")))
    parsed, _ = p.parse_known_args()

    rclpy.init(args=args)
    node = None
    try:
        node = TargetLog(parsed)
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.f.close()
            print(f"\n停止, 共 {node.n} 帧 -> {parsed.outfile}")
            save_plot(node.rows, parsed.outfile)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
