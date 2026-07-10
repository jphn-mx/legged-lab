"""电机轨迹跟踪测试 (WOAN 双足, 12 关节)。

逐个关节给一条已知正弦轨迹, 其余关节保持 default; 同时从 /left_joint_states
记录实测角度/角速度, 评估每个电机的位置跟踪 (增益、相位滞后、RMS 误差)。

接口 (与 rl_real_tron 一致):
  发布 /dog_joint_pos   std_msgs/Float64MultiArray  12 维, 顺序 L0-L5,R0-R5
  订阅 /left_joint_states sensor_msgs/JointState    .position/.velocity, 同序

轨迹:  cmd_j(t) = default_j + env(t) * A_j * sin(2*pi*f*t)
  env(t): 0->1 (ramp 升余弦) -> 1 (保持) -> 1->0 (ramp)   起止都回到 default, 无跳变
  A_j 按关节限位自适应裁剪, 保证轨迹始终在 [lower_j, upper_j] 内。

用法:
  ros2 run rl_real_py motor_track_test                       # 全部 12 关节, 默认参数
  ros2 run rl_real_py motor_track_test --joints L0,L3 --amp 0.1 --freq 0.3
  python3 motor_tracking_test.py --dry-run                   # 无硬件, 合成数据自测分析
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from ament_index_python.packages import get_package_share_directory

# 关节标签 (实机 /left_joint_states 与 /dog_joint_pos 命名顺序)
JOINT_NAMES = [f"L{i}" for i in range(6)] + [f"R{i}" for i in range(6)]
NUM_JOINTS = 12


def load_config(config_file):
    """读取 default/limits。优先用源码树相对路径 (无需 colcon build),
    回退到 ament share 约定路径 (与 rl_real_tron 一致)。"""
    candidates = []
    # 1) 相对本文件: rl_real_py/rl_real_py/ -> ../configs/<file>
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, '..', 'configs', config_file))
    # 2) ament share + ../../../../src/.../configs/<file>
    try:
        package_path = get_package_share_directory('rl_real_py')
        candidates.append(os.path.join(
            package_path, '..', '..', '..', '..',
            'src', 'rl_real_py', 'configs', config_file))
    except Exception:
        pass

    config_path = next((p for p in candidates if os.path.isfile(p)), None)
    if config_path is None:
        raise FileNotFoundError(f"找不到配置 {config_file}, 尝试过: {candidates}")
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    default = np.array(cfg["default_angles"], dtype=np.float64)
    lower = np.array(cfg["joint_lower_limits"], dtype=np.float64)
    upper = np.array(cfg["joint_upper_limits"], dtype=np.float64)
    assert len(default) == NUM_JOINTS and len(lower) == NUM_JOINTS and len(upper) == NUM_JOINTS
    return default, lower, upper


def envelope(t_local, duration, ramp):
    """升余弦升降包络, 在 [0, duration] 内 0->1->1->0。"""
    if ramp <= 0.0:
        return 1.0
    ramp = min(ramp, duration / 2.0)
    if t_local < ramp:
        return 0.5 * (1.0 - math.cos(math.pi * t_local / ramp))
    if t_local > duration - ramp:
        return 0.5 * (1.0 - math.cos(math.pi * (duration - t_local) / ramp))
    return 1.0


class MotorTrackingTest(Node):
    def __init__(self, args):
        super().__init__("motor_tracking_test")
        self.args = args
        self.default, self.lower, self.upper = load_config(args.config)

        # 选择要测的关节
        if args.joints:
            wanted = [j.strip().upper() for j in args.joints.split(",") if j.strip()]
            self.joint_seq = [JOINT_NAMES.index(j) for j in wanted]
        else:
            self.joint_seq = list(range(NUM_JOINTS))

        # 每关节自适应幅度: 保证 default +/- A 在限位内 (留 margin)
        self.amp = np.zeros(NUM_JOINTS, dtype=np.float64)
        for j in range(NUM_JOINTS):
            head = min(self.default[j] - self.lower[j], self.upper[j] - self.default[j])
            self.amp[j] = min(args.amp, max(0.0, args.margin * head))

        self.dt = 1.0 / args.rate
        self.omega = 2.0 * math.pi * args.freq

        # ROS I/O
        qos = QoSProfile(depth=1)
        self.pub = self.create_publisher(Float64MultiArray, "/dog_joint_pos", qos)
        self.sub = self.create_subscription(JointState, "/left_joint_states", self.joint_cb, 10)

        self.meas_pos = np.array(self.default, dtype=np.float64)
        self.meas_vel = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.got_state = False

        # 记录: 每行 (t_global, joint_idx, t_local, cmd, meas_pos, meas_vel)
        self.rows = []
        self.t0 = None

        # 状态机
        self.phase = "settle"
        self.seq_i = 0
        self.phase_start = None
        self.done = False

        self._print_plan()
        if not args.no_confirm and not args.dry_run:
            try:
                input("\n按回车开始测试 (Ctrl-C 取消) ... ")
            except (EOFError, KeyboardInterrupt):
                raise SystemExit(0)

        self.timer = self.create_timer(self.dt, self.tick)

    # ------------------------------------------------------------------
    def _print_plan(self):
        print("=" * 70)
        print("电机轨迹跟踪测试 (sequential, sine)")
        print(f"  rate={self.args.rate}Hz  freq={self.args.freq}Hz  amp<={self.args.amp}rad  "
              f"duration={self.args.duration}s/关节  ramp={self.args.ramp}s")
        print(f"  settle={self.args.settle}s  rest={self.args.rest}s  margin={self.args.margin}")
        seq = "  ".join(f"{JOINT_NAMES[j]}(A={self.amp[j]:.3f})" for j in self.joint_seq)
        print(f"  关节顺序: {seq}")
        total = self.args.settle + len(self.joint_seq) * (self.args.duration + self.args.rest)
        print(f"  预计用时 ~{total:.0f}s")
        print("=" * 70)

    def joint_cb(self, msg):
        n = min(NUM_JOINTS, len(msg.position))
        self.meas_pos[:n] = msg.position[:n]
        if len(msg.velocity) >= n:
            self.meas_vel[:n] = msg.velocity[:n]
        self.got_state = True

    def _publish(self, cmd):
        # 安全兜底: 任何路径发出的指令都硬性裁剪到关节限位内
        cmd = np.clip(np.asarray(cmd, dtype=np.float64), self.lower, self.upper)
        m = Float64MultiArray()
        m.data = [float(x) for x in cmd]
        self.pub.publish(m)

    # ------------------------------------------------------------------
    def tick(self):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
            self.phase_start = now
        t_global = now - self.t0
        elapsed = now - self.phase_start
        cmd = np.array(self.default, dtype=np.float64)  # 默认: 全部 default

        if self.phase == "settle":
            self._publish(cmd)
            if elapsed >= self.args.settle:
                self._enter_run(now)
            return

        if self.phase == "run":
            j = self.joint_seq[self.seq_i]
            env = envelope(elapsed, self.args.duration, self.args.ramp)
            cmd[j] = self.default[j] + env * self.amp[j] * math.sin(self.omega * elapsed)
            cmd = np.clip(cmd, self.lower, self.upper)   # 记录与发布的指令一致 (兜底裁剪)
            self._publish(cmd)
            self.rows.append((t_global, j, elapsed, cmd[j],
                              self.meas_pos[j], self.meas_vel[j]))
            if elapsed >= self.args.duration:
                self.phase = "rest"
                self.phase_start = now
            return

        if self.phase == "rest":
            self._publish(cmd)
            if elapsed >= self.args.rest:
                self.seq_i += 1
                if self.seq_i >= len(self.joint_seq):
                    self._finish()
                else:
                    self._enter_run(now)
            return

    def _enter_run(self, now):
        self.phase = "run"
        self.phase_start = now
        j = self.joint_seq[self.seq_i]
        print(f"\n[{self.seq_i + 1}/{len(self.joint_seq)}] 测试关节 {JOINT_NAMES[j]} "
              f"A={self.amp[j]:.3f}rad f={self.args.freq}Hz ...")

    def _finish(self):
        self.done = True
        self.timer.cancel()
        self._publish(self.default)   # 安全: 回到 default
        analyze_and_save(self.rows, self.default, self.amp, self.omega,
                         self.args, self.joint_seq)
        rclpy.shutdown()


# ----------------------------------------------------------------------
# 分析 + 输出 (独立函数, 便于 --dry-run 复用)
# ----------------------------------------------------------------------
def fit_sine(t, y, omega):
    """最小二乘拟合 y ~ a*sin(wt)+b*cos(wt)+c, 返回 (a, b, c)。"""
    s = np.sin(omega * t)
    co = np.cos(omega * t)
    A = np.column_stack([s, co, np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0], coef[1], coef[2]


def analyze_joint(rows_j, default_j, amp_j, omega, duration, ramp):
    """对单关节稳态段 (env==1) 计算 gain/lag/rms/max。"""
    t = np.array([r[2] for r in rows_j])           # t_local
    cmd = np.array([r[3] for r in rows_j])
    meas = np.array([r[4] for r in rows_j])
    # 稳态段: env==1 且丢掉第一个周期
    period = 2.0 * math.pi / omega if omega > 0 else duration
    steady = (t >= max(ramp, period)) & (t <= duration - ramp)
    if steady.sum() < 8 or amp_j <= 1e-9:
        return None
    ts, cs, ms = t[steady], cmd[steady], meas[steady]
    cmd_dev = cs - default_j
    meas_dev = ms - default_j
    a, b, _ = fit_sine(ts, meas_dev, omega)
    gain = math.sqrt(a * a + b * b) / amp_j
    lag_rad = -math.atan2(b, a)                    # 实测滞后于指令为正
    lag_deg = math.degrees(lag_rad)
    lag_ms = (lag_rad / omega) * 1000.0 if omega > 0 else 0.0
    err = meas_dev - cmd_dev
    rms = float(np.sqrt(np.mean(err ** 2)))
    max_err = float(np.max(np.abs(ms - cs)))
    return dict(gain=gain, lag_deg=lag_deg, lag_ms=lag_ms, rms=rms, max_err=max_err,
                n=int(steady.sum()))


def analyze_and_save(rows, default, amp, omega, args, joint_seq):
    outdir = args.outdir or os.path.join(
        os.getcwd(), time.strftime("motor_tracking_%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)

    # CSV
    csv_path = os.path.join(outdir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_global", "joint_idx", "joint_name", "t_local",
                    "cmd", "meas_pos", "meas_vel"])
        for r in rows:
            w.writerow([f"{r[0]:.5f}", r[1], JOINT_NAMES[r[1]], f"{r[2]:.5f}",
                        f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:.6f}"])

    # 按关节分组分析
    lines = []
    header = (f"{'joint':>5} | {'A(rad)':>7} | {'f(Hz)':>5} | {'gain':>6} | "
              f"{'lag(deg)':>8} | {'lag(ms)':>8} | {'rms(rad)':>9} | {'max(rad)':>9} | {'n':>5}")
    sep = "-" * len(header)
    lines += [header, sep]
    by_joint = {}
    for r in rows:
        by_joint.setdefault(r[1], []).append(r)

    for j in joint_seq:
        rows_j = by_joint.get(j, [])
        m = analyze_joint(rows_j, default[j], amp[j], omega, args.duration, args.ramp) if rows_j else None
        if m is None:
            lines.append(f"{JOINT_NAMES[j]:>5} | {amp[j]:7.3f} |  (数据不足/幅度为 0, 跳过)")
        else:
            lines.append(f"{JOINT_NAMES[j]:>5} | {amp[j]:7.3f} | {args.freq:5.2f} | "
                         f"{m['gain']:6.3f} | {m['lag_deg']:8.2f} | {m['lag_ms']:8.2f} | "
                         f"{m['rms']:9.5f} | {m['max_err']:9.5f} | {m['n']:5d}")

    summary = "\n".join(lines)
    print("\n" + "=" * 70)
    print("跟踪测试汇总 (gain 理想≈1, lag/rms 越小越好):")
    print(summary)
    print("=" * 70)
    print(f"CSV : {csv_path}")

    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    # 绘图 (无 matplotlib 则跳过)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_dir = os.path.join(outdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        for j in joint_seq:
            rows_j = by_joint.get(j, [])
            if not rows_j:
                continue
            t = np.array([r[2] for r in rows_j])
            cmd = np.array([r[3] for r in rows_j])
            meas = np.array([r[4] for r in rows_j])
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            ax1.plot(t, cmd, label="cmd", lw=1.5)
            ax1.plot(t, meas, label="meas", lw=1.2)
            ax1.set_ylabel("pos (rad)")
            ax1.set_title(f"joint {JOINT_NAMES[j]}  tracking")
            ax1.legend(); ax1.grid(True, alpha=0.3)
            ax2.plot(t, meas - cmd, color="r", lw=1.0)
            ax2.set_ylabel("err (rad)"); ax2.set_xlabel("t_local (s)")
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"{JOINT_NAMES[j]}.png"), dpi=110)
            plt.close(fig)
        print(f"图  : {plot_dir}/<joint>.png")
    except ImportError:
        print("(未安装 matplotlib, 跳过绘图)")


# ----------------------------------------------------------------------
def _dry_run(args):
    """无硬件自测: 合成 meas = gain*cmd(滞后), 验证状态机外的分析链路。"""
    default, lower, upper = load_config(args.config)
    omega = 2.0 * math.pi * args.freq
    amp = np.zeros(NUM_JOINTS)
    for j in range(NUM_JOINTS):
        head = min(default[j] - lower[j], upper[j] - default[j])
        amp[j] = min(args.amp, max(0.0, args.margin * head))
    joint_seq = list(range(NUM_JOINTS))
    dt = 1.0 / args.rate
    true_gain, true_lag_s = 0.9, 0.020   # 已知: 增益 0.9, 滞后 20ms
    rows, t_global = [], 0.0
    for j in joint_seq:
        steps = int(args.duration / dt)
        for k in range(steps):
            tl = k * dt
            env = envelope(tl, args.duration, args.ramp)
            cmd = default[j] + env * amp[j] * math.sin(omega * tl)
            env2 = envelope(tl - true_lag_s, args.duration, args.ramp)
            meas = default[j] + true_gain * env2 * amp[j] * math.sin(omega * (tl - true_lag_s))
            rows.append((t_global, j, tl, cmd, meas, 0.0))
            t_global += dt
    print(f"[dry-run] 合成信号: 真值 gain={true_gain}, lag={true_lag_s*1000:.0f}ms "
          f"({math.degrees(omega*true_lag_s):.1f}deg). 期望下表 gain≈{true_gain}, lag 接近该值。")
    analyze_and_save(rows, default, amp, omega, args, joint_seq)


def build_argparser():
    p = argparse.ArgumentParser(description="WOAN 电机轨迹跟踪测试")
    p.add_argument("--rate", type=float, default=200.0, help="发布频率 Hz")
    p.add_argument("--freq", type=float, default=0.5, help="正弦频率 Hz")
    p.add_argument("--amp", type=float, default=0.15, help="幅度上限 rad (按限位再裁剪)")
    p.add_argument("--duration", type=float, default=6.0, help="每关节时长 s")
    p.add_argument("--ramp", type=float, default=1.0, help="升降包络时长 s")
    p.add_argument("--settle", type=float, default=2.0, help="开始前保持 default 时长 s")
    p.add_argument("--rest", type=float, default=1.5, help="关节间隔保持 default 时长 s")
    p.add_argument("--margin", type=float, default=0.9, help="限位安全裕度 (0~1)")
    p.add_argument("--joints", type=str, default="", help="子集, 逗号分隔, 如 L0,L3 (默认全 12)")
    p.add_argument("--config", type=str, default="dual_tron.yaml", help="配置文件名")
    p.add_argument("--outdir", type=str, default="", help="输出目录 (默认按时间戳新建)")
    p.add_argument("--no-confirm", action="store_true", help="跳过回车确认")
    p.add_argument("--dry-run", action="store_true", help="无硬件, 合成数据自测分析链路")
    return p


def main(args=None):
    parsed, _ = build_argparser().parse_known_args(args)

    # margin 是限位安全裕度 (0~1), 夹紧防止幅度超出限位 (>1 会让正弦峰被硬裁剪而失真)
    if parsed.margin > 1.0 or parsed.margin < 0.0:
        print(f"[warn] --margin={parsed.margin} 超出 [0,1], 已夹紧到 "
              f"{min(max(parsed.margin, 0.0), 1.0)}")
        parsed.margin = min(max(parsed.margin, 0.0), 1.0)

    if parsed.dry_run:
        _dry_run(parsed)
        return

    rclpy.init(args=args)
    node = None
    try:
        node = MotorTrackingTest(parsed)
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # 安全: 退出前把指令置回 default
        try:
            if node is not None and not node.done:
                node._publish(node.default)
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
