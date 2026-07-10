"""真机静态刚度实测 (WOAN 双足, 12 关节) — 反推每关节"有效刚度 kp_eff"。

目的: 动态承重踏步数据反推 kp_eff 的 R^2 太低、不可靠。本脚本用"静止 + 已知/稳定
负载"测每个关节的有效位置环刚度, 信噪比远高于踏步数据。

== 增益数据流的事实 (务必先看) ==
本节点只发布 /dog_joint_pos (std_msgs/Float64MultiArray, 12 维纯位置目标, 实机序
L1..L6,R1..R6)。**真机 kp/kd 不由本脚本下发**, 而是写死在下游 arm_control 驱动层
(robot 工作空间 armcontrol/arm_control_node + 达妙 CAN)。所以本脚本测到的 kp_eff
就是"驱动层实际生效的有效刚度", 正是要找的量。flat.yaml 里的标称命令增益
(hip/knee kp=126 kd=4.83, ankle kp=7 kd=0.27) 仅作为对比基准打印, 不参与下发。
  → kp_eff ≈ 标称 126/7  : 驱动层确实按训练增益跑, 真机=训练
  → kp_eff <<  标称       : 驱动层增益更低 / 力矩限 / 间隙 → sim2real 偏软

== 两种方案 ==
方案 A (默认, --mode A, 推荐) — 站立保持:
  机器人站在地面承重, 不跑策略、不踏步, 命令全部关节保持 default 姿态静止 hold 秒,
  记录每关节稳态 (目标=default, 实测 q, 实测 effort)。负载=机器人自重(真实工况)。
    kp_eff(关节 j) = 稳态 effort_j / 稳态位置误差_j,  误差 = default_j - q_j
  单点反推: 只给一个误差点 → 直接相除得 kp_eff, 无法判线性度(R^2)。
  若想要 R^2, 用 --mode A --offsets "..." 让某关节在几个偏置点保持(见下), 拟合斜率。

方案 A+ (--mode A --joints J --offsets "-0.1,-0.05,0,0.05,0.1") — 单关节多偏置保持:
  机器人仍站地面承重, 选 1 个关节, 在 default+Δ 的几个 Δ 上各保持 hold 秒, 记录
  (误差=cmd-q, effort)。对 (误差, effort) 做过原点最小二乘 → 斜率=kp_eff, 给出 R^2。
  注意: 改一个关节的偏置会改变整机姿态/受力, 偏置要小, 一次只动一个关节。

方案 B (--mode B, 可选, 更严格) — 重力加载单关节:
  悬挂机器人, 手动把单条腿摆成不同姿态 (让连杆自重对目标关节产生已知重力力矩
  tau_g = m*g*L*sin(theta)), 命令该关节保持各姿态目标。关节停在 kp_eff*误差=tau_g
  的平衡点。本脚本只负责"按你给的目标角序列逐个保持并记录 q/effort"; 重力力矩
  tau_g 需你按 URDF 的 m/L 自行算好, 用 --loads 传入与 --offsets 对应的 tau_g。
    kp_eff = 对 (误差, tau_g) 过原点最小二乘斜率 (Nm/rad), 给 R^2。
  比 A 干净(单关节、负载已知)但要手动摆姿态+算重力。

== 安全 ==
- 方案 A: 机器人必须先稳稳站在地面 (或低台), 周围无人/无障碍, 准备好随时断电。
- 方案 B: 机器人必须悬挂牢固, 腿可自由活动, 手动摆姿态时人手离开后再确认开始。
- 所有发布的指令硬性裁剪到 flat.yaml 的 joint_lower/upper_limits 内 (双保险)。
- 偏置/姿态幅度小; 一次只动一个关节; Ctrl-C 立即停止并把指令拉回 default。
- --dry-run 无硬件自测分析逻辑; 真机务必先 settle 再 hold。

== 用法 ==
  # 方案 A: 站立保持, 一次测全部 12 关节的单点 kp_eff
  ros2 run rl_real_py stiffness_test --mode A

  # 方案 A+: 单关节多偏置, 出斜率+R^2 (例: 测左膝 L4)
  ros2 run rl_real_py stiffness_test --mode A --joints L4 \
        --offsets "-0.10,-0.05,0,0.05,0.10" --hold 2.0

  # 方案 B: 重力加载, 目标角序列 + 对应已知重力力矩 (Nm)
  ros2 run rl_real_py stiffness_test --mode B --joints L4 \
        --abs-targets "0.0,0.3,0.6" --loads "0.0,1.8,3.4" --hold 2.0

  # 无硬件自测
  python3 stiffness_test.py --dry-run
"""
import argparse
import csv
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

# 关节标签 (实机 /left_joint_states 与 /dog_joint_pos 命名顺序, 实机序)
JOINT_NAMES = [f"L{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 7)]
NUM_JOINTS = 12


def load_config(config_file):
    """读取 default / limits / 标称命令增益 (kp/kd, 仅用于对比打印)。
    优先用源码树相对路径 (无需 colcon build), 回退到 ament share。"""
    candidates = []
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, '..', 'configs', config_file))
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
    default = np.array(cfg["default_angles"], dtype=np.float64)  # 注意: flat.yaml 的 default 是 Lab 序!
    lower = np.array(cfg["joint_lower_limits"], dtype=np.float64)  # 实机序
    upper = np.array(cfg["joint_upper_limits"], dtype=np.float64)  # 实机序
    # 标称命令增益 (实机序, 仅打印对比, 不下发)
    kp_nom = np.array(cfg.get("joint_kp", [np.nan] * NUM_JOINTS), dtype=np.float64)
    # default_angles 在 flat.yaml 是 Lab 序 (L1,R1,L2,R2,...), 需重排到实机序 L1..L6,R1..R6
    sim_names = cfg["joint_index_in_sim"]
    real_names = cfg["joint_index_in_real"]
    sim2real = [sim_names.index(n) for n in real_names]
    default_real = default[sim2real]
    assert len(default_real) == NUM_JOINTS and len(lower) == NUM_JOINTS and len(upper) == NUM_JOINTS
    return default_real, lower, upper, kp_nom


def fit_through_origin(err, eff):
    """过原点最小二乘 eff = k * err; 返回 (k, R^2)。需 >=2 个点且 err 有变化。"""
    err = np.asarray(err, dtype=np.float64)
    eff = np.asarray(eff, dtype=np.float64)
    if len(err) < 2 or np.allclose(err, 0.0):
        return np.nan, np.nan
    k = float(np.dot(err, eff) / np.dot(err, err))
    pred = k * err
    ss_res = float(np.sum((eff - pred) ** 2))
    ss_tot = float(np.sum((eff - np.mean(eff)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return k, r2


class StiffnessTest(Node):
    def __init__(self, args):
        super().__init__("stiffness_test")
        self.args = args
        self.default, self.lower, self.upper, self.kp_nom = load_config(args.config)

        # 选关节
        if args.joints:
            wanted = [j.strip().upper() for j in args.joints.split(",") if j.strip()]
            self.joint_seq = [JOINT_NAMES.index(j) for j in wanted]
        else:
            self.joint_seq = list(range(NUM_JOINTS))

        # 偏置序列 (方案 A+) / 绝对目标序列 (方案 B)
        self.offsets = self._parse_list(args.offsets) if args.offsets else [0.0]
        self.abs_targets = self._parse_list(args.abs_targets) if args.abs_targets else None
        self.loads = self._parse_list(args.loads) if args.loads else None

        if args.mode == "B":
            if not args.joints or len(self.joint_seq) != 1:
                raise SystemExit("方案 B 需 --joints 指定且仅 1 个关节")
            if self.abs_targets is None or self.loads is None:
                raise SystemExit("方案 B 需 --abs-targets 与 --loads (长度一致, 后者是已知重力力矩 Nm)")
            if len(self.abs_targets) != len(self.loads):
                raise SystemExit("--abs-targets 与 --loads 长度必须一致")

        self.dt = 1.0 / args.rate
        self.settle_frac = 0.5  # 每个保持段后半段算稳态

        qos = QoSProfile(depth=1)
        self.pub = self.create_publisher(Float64MultiArray, "/dog_joint_pos", qos)
        self.sub = self.create_subscription(JointState, "/left_joint_states", self.joint_cb, 10)

        self.meas_pos = np.array(self.default, dtype=np.float64)
        self.meas_eff = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.eff_has_data = False
        self.got_state = False

        # 记录: 每行 (t_global, phase, joint_idx, level_idx, cmd_target, meas_q, meas_eff, in_steady)
        self.rows = []
        self.t0 = None

        # 段计划: list of (joint_idx, target_abs, load) — phase 一段段保持
        self.plan = self._build_plan()
        self.seg_i = 0
        self.phase = "settle"
        self.phase_start = None
        self.done = False

        self._print_plan()
        if not args.no_confirm and not args.dry_run:
            try:
                input("\n确认机器人已就位 (方案 A=站稳承重 / 方案 B=悬挂腿空, 摆好姿态), 按回车开始 (Ctrl-C 取消) ... ")
            except (EOFError, KeyboardInterrupt):
                raise SystemExit(0)

        self.timer = self.create_timer(self.dt, self.tick)

    @staticmethod
    def _parse_list(s):
        return [float(x) for x in str(s).replace(" ", "").split(",") if x != ""]

    def _build_plan(self):
        """生成保持段序列。每段 = 一个关节在一个目标角处保持 hold 秒。"""
        plan = []
        if self.args.mode == "B":
            j = self.joint_seq[0]
            for k, (tgt, ld) in enumerate(zip(self.abs_targets, self.loads)):
                plan.append((j, float(tgt), float(ld)))
        else:  # 方案 A / A+
            for j in self.joint_seq:
                for lv in self.offsets:
                    plan.append((j, float(self.default[j] + lv), None))
        return plan

    def _print_plan(self):
        print("=" * 74)
        print(f"真机静态刚度测试  mode={self.args.mode}  rate={self.args.rate}Hz  "
              f"hold={self.args.hold}s/段  settle={self.args.settle}s")
        print("  [重要] 本脚本只发位置目标到 /dog_joint_pos; kp/kd 由 arm_control 驱动层决定。")
        print("  [安全] 方案 A 机器人站稳承重 / 方案 B 悬挂腿空; 准备随时断电; 指令已限位裁剪。")
        if self.args.mode == "B":
            j = self.joint_seq[0]
            print(f"  方案 B 单关节 {JOINT_NAMES[j]}: 目标角 {self.abs_targets} rad  对应重力力矩 {self.loads} Nm")
        else:
            seq = " ".join(JOINT_NAMES[j] for j in self.joint_seq)
            print(f"  方案 A 关节: {seq}   偏置 Δ(rad): {self.offsets}")
        total = self.args.settle + len(self.plan) * (self.args.hold + self.args.rest)
        print(f"  共 {len(self.plan)} 个保持段, 预计 ~{total:.0f}s")
        print("=" * 74)

    def joint_cb(self, msg):
        n = min(NUM_JOINTS, len(msg.position))
        self.meas_pos[:n] = msg.position[:n]
        if len(msg.effort) >= n:
            self.meas_eff[:n] = msg.effort[:n]
            self.eff_has_data = True
        self.got_state = True

    def _publish(self, cmd):
        cmd = np.clip(np.asarray(cmd, dtype=np.float64), self.lower, self.upper)  # 安全兜底
        m = Float64MultiArray()
        m.data = [float(x) for x in cmd]
        self.pub.publish(m)
        return cmd

    def tick(self):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
            self.phase_start = now
        t_global = now - self.t0
        elapsed = now - self.phase_start

        # 基础指令: 全部 default; 当前段把目标关节设到段目标
        cmd = np.array(self.default, dtype=np.float64)

        if self.phase == "settle":
            self._publish(cmd)
            if elapsed >= self.args.settle:
                self.seg_i = 0
                self.phase = "hold"
                self.phase_start = now
            return

        if self.phase == "hold":
            j, tgt, _ld = self.plan[self.seg_i]
            cmd[j] = tgt
            cmd = self._publish(cmd)
            in_steady = elapsed >= self.args.hold * (1.0 - self.settle_frac)
            self.rows.append((t_global, "hold", j, self.seg_i, float(cmd[j]),
                              float(self.meas_pos[j]), float(self.meas_eff[j]), int(in_steady)))
            if elapsed >= self.args.hold:
                self.phase = "rest"
                self.phase_start = now
            return

        if self.phase == "rest":
            self._publish(cmd)  # 回到全 default 缓冲
            if elapsed >= self.args.rest:
                self.seg_i += 1
                if self.seg_i >= len(self.plan):
                    self.phase = "done"
                    self.done = True
                else:
                    self.phase = "hold"
                self.phase_start = now
            return

    # ------------------------------------------------------------------
    def finalize(self):
        """回 default, 写 CSV, 计算并打印每关节 kp_eff。"""
        try:
            self._publish(np.array(self.default, dtype=np.float64))
        except Exception:
            pass
        if not self.rows:
            print("\n[结果] 无数据 (未收到 /left_joint_states 或被提前中断)")
            return
        path = os.path.join(os.getcwd(), time.strftime(f"stiffness_{self.args.mode}_%Y%m%d_%H%M%S.csv"))
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "phase", "joint", "joint_name", "level_idx",
                        "cmd_target_rad", "meas_q_rad", "meas_effort_Nm", "in_steady"])
            for (t, ph, j, lv, tgt, q, eff, st) in self.rows:
                w.writerow([f"{t:.4f}", ph, j, JOINT_NAMES[j], lv,
                            f"{tgt:.6f}", f"{q:.6f}", f"{eff:.6f}", st])
        print(f"\n[结果] 原始数据 -> {path}")
        if not self.eff_has_data:
            print("[结果] 警告: /left_joint_states 无 effort 字段, effort 全 0, kp_eff 无意义。")

        self._analyze(path)

    def _analyze(self, csv_path):
        """逐关节: 取每个保持段稳态 (in_steady) 的均值, 反推 kp_eff。"""
        arr = self.rows
        # 按 (joint, level_idx) 聚合稳态均值
        agg = {}
        for (t, ph, j, lv, tgt, q, eff, st) in arr:
            if ph != "hold" or st != 1:
                continue
            key = (j, lv)
            agg.setdefault(key, {"tgt": tgt, "q": [], "eff": []})
            agg[key]["q"].append(q)
            agg[key]["eff"].append(eff)

        print("\n" + "=" * 74)
        print("每关节有效刚度 kp_eff  (= 稳态 effort / 稳态位置误差;  误差 = cmd_target - q)")
        print(f"{'关节':<6}{'标称kp':>8}{'点数':>5}{'kp_eff':>12}{'R^2':>8}   说明")
        print("-" * 74)
        lines_csv = []
        for j in self.joint_seq:
            keys = sorted([k for k in agg if k[0] == j], key=lambda k: k[1])
            errs, effs = [], []
            for k in keys:
                d = agg[k]
                q_m = float(np.mean(d["q"]))
                eff_m = float(np.mean(d["eff"]))
                if self.args.mode == "B":
                    # 方案 B: 自变量=误差(cmd-q), 因变量=已知重力力矩 load
                    _, _, _, ld = next((p for p in self.plan if p[0] == j and abs(p[1] - d["tgt"]) < 1e-9),
                                       (j, d["tgt"], None, None)) if False else (None, None, None, None)
                    # load 从 plan 取
                    ld = next(p[2] for p in self.plan if p[0] == j and abs(p[1] - d["tgt"]) < 1e-6)
                    err = d["tgt"] - q_m
                    errs.append(err)
                    effs.append(ld)
                else:
                    err = d["tgt"] - q_m
                    errs.append(err)
                    effs.append(eff_m)

            kp_nom = self.kp_nom[j] if j < len(self.kp_nom) else float("nan")
            if len(errs) >= 2:
                k, r2 = fit_through_origin(errs, effs)
                note = self._note(k, kp_nom, r2)
                print(f"{JOINT_NAMES[j]:<6}{kp_nom:>8.1f}{len(errs):>5}{k:>12.2f}{r2:>8.3f}   {note}")
            elif len(errs) == 1:
                err, eff = errs[0], effs[0]
                k = eff / err if abs(err) > 1e-6 else float("nan")
                note = "单点(无R^2); 误差≈0时不可信, 加 --offsets 拟合" if abs(err) <= 1e-6 \
                    else self._note(k, kp_nom, float("nan"))
                print(f"{JOINT_NAMES[j]:<6}{kp_nom:>8.1f}{1:>5}{k:>12.2f}{'-':>8}   {note}")
                r2 = float("nan")
            else:
                print(f"{JOINT_NAMES[j]:<6}{kp_nom:>8.1f}{0:>5}{'-':>12}{'-':>8}   无稳态数据")
                continue
            lines_csv.append((JOINT_NAMES[j], kp_nom, len(errs), k, r2))
        print("=" * 74)
        print("解读: kp_eff≈标称→驱动层按训练增益跑(真机=训练); kp_eff<<标称→偏软(增益低/力矩限/间隙)。")
        print("      R^2 高=线性(纯PD,增益低); R^2 低/曲线=饱和(力矩/电流限)或死区(间隙)。")

        # 摘要 CSV
        sp = csv_path.rsplit(".", 1)[0] + "_kpeff.csv"
        with open(sp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["joint_name", "kp_nominal", "n_points", "kp_eff_Nm_per_rad", "R2"])
            for (nm, kpn, n, k, r2) in lines_csv:
                w.writerow([nm, f"{kpn:.3f}", n, f"{k:.4f}", f"{r2:.4f}"])
        print(f"[结果] kp_eff 摘要 -> {sp}")

    @staticmethod
    def _note(k, kp_nom, r2):
        if not np.isfinite(k):
            return ""
        if np.isfinite(kp_nom) and kp_nom > 0:
            ratio = k / kp_nom
            tag = "≈标称(真机≈训练)" if ratio > 0.8 else ("偏软" if ratio > 0.3 else "远低于标称!")
            note = f"{ratio*100:.0f}%标称 {tag}"
        else:
            note = ""
        if np.isfinite(r2) and r2 < 0.9:
            note += " | R^2低: 非线性(饱和/死区?)"
        return note


def make_dry_rows(node):
    """无硬件自测: 合成 kp_eff=80 (标称126) + 噪声的稳态数据, 验证分析逻辑。"""
    rng = np.random.default_rng(0)
    rows = []
    t = 0.0
    for seg, (j, tgt, ld) in enumerate(node.plan):
        q = tgt - 5.0 / 80.0 if node.args.mode != "B" else tgt - (ld / 80.0)  # 误差=eff/kp_eff
        eff = (tgt - q) * 80.0 + rng.normal(0, 0.05)
        for _ in range(5):
            rows.append((t, "hold", j, seg, float(tgt), float(q + rng.normal(0, 0.001)),
                         float(eff if node.args.mode != "B" else ld), 1))
            t += node.dt
    node.eff_has_data = True
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="真机静态刚度实测 (反推 kp_eff)")
    p.add_argument("--mode", choices=["A", "B"], default="A", help="A=站立保持(默认), B=重力加载单关节")
    p.add_argument("--config", default="flat.yaml")
    p.add_argument("--joints", default="", help="逗号分隔, 实机序 L1..L6,R1..R6; 空=全部(仅A)")
    p.add_argument("--offsets", default="", help="方案A+: default 上的偏置序列(rad), 如 '-0.1,-0.05,0,0.05,0.1'")
    p.add_argument("--abs-targets", default="", help="方案B: 绝对目标角序列(rad)")
    p.add_argument("--loads", default="", help="方案B: 与 abs-targets 对应的已知重力力矩(Nm)")
    p.add_argument("--rate", type=float, default=200.0, help="发布频率 Hz")
    p.add_argument("--settle", type=float, default=2.0, help="开始前保持 default 的稳定时间 s")
    p.add_argument("--hold", type=float, default=2.0, help="每个目标保持时长 s (后半段算稳态)")
    p.add_argument("--rest", type=float, default=1.0, help="段间回 default 缓冲 s")
    p.add_argument("--no-confirm", action="store_true", help="跳过回车确认 (谨慎)")
    p.add_argument("--dry-run", action="store_true", help="无硬件, 合成数据自测分析")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        # 无 ROS, 直接构造节点对象做分析自测
        class _Stub(StiffnessTest):
            def __init__(self, a):
                self.args = a
                self.default, self.lower, self.upper, self.kp_nom = load_config(a.config)
                if a.joints:
                    self.joint_seq = [JOINT_NAMES.index(j.strip().upper())
                                      for j in a.joints.split(",") if j.strip()]
                else:
                    self.joint_seq = list(range(NUM_JOINTS))
                self.offsets = self._parse_list(a.offsets) if a.offsets else [0.0]
                self.abs_targets = self._parse_list(a.abs_targets) if a.abs_targets else None
                self.loads = self._parse_list(a.loads) if a.loads else None
                self.dt = 1.0 / a.rate
                self.settle_frac = 0.5
                self.plan = self._build_plan()
                self.eff_has_data = False
        node = _Stub.__new__(_Stub)
        _Stub.__init__(node, args)
        node.rows = make_dry_rows(node)
        node._analyze(os.path.join(os.getcwd(), "stiffness_dryrun.csv"))
        return

    rclpy.init()
    node = None
    try:
        node = StiffnessTest(args)
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\n[中断] Ctrl-C, 拉回 default 并保存已采数据 ...")
    finally:
        if node is not None:
            node.finalize()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
