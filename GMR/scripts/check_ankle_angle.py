"""Report ankle-pitch (joint_R5 / joint_L5) angle stats from retargeting, to tune the IK config.

The ankle-pitch joint angle == 0 means the foot is neutral (perpendicular to the shank, ~90 deg).
Run this after editing the foot `rot_offset` quaternion of right_toe_link / left_toe_link in
  general_motion_retargeting/ik_configs/smplx_to_a1_legs_v2.json  (ik_match_table1 -- table1 sets the
  target orientation; table2 only sets its weight)
and adjust the quaternion until mean |joint_R5/L5| is close to 0 (no viewer, headless).

  conda activate env_isaaclab   # (or gmr)
  python scripts/check_ankle_angle.py --smplx_file <motion.npz>
"""
import argparse
import pathlib
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

# dof_pos (qpos[7:]) is in robot XML joint order R1..R6, L1..L6 -> ankle pitch indices:
R5 = 7 + 4   # qpos index of joint_R5
L5 = 7 + 10  # qpos index of joint_L5

if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--smplx_file", required=True, help="SMPLX motion .npz")
    ap.add_argument("--robot", default="a1_legs_v2")
    args = ap.parse_args()

    smplx_data, body_model, smplx_output, h = load_smplx_file(args.smplx_file, HERE / ".." / "assets" / "body_models")
    frames, _ = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=50)
    retarget = GMR(actual_human_height=h, src_human="smplx", tgt_robot=args.robot)

    r5, l5 = [], []
    for f in frames:
        q = retarget.retarget(f)
        r5.append(q[R5])
        l5.append(q[L5])
    r5, l5 = np.array(r5), np.array(l5)

    def stats(name, a):
        print(f"{name}: mean={np.degrees(a.mean()):+6.1f}  median={np.degrees(np.median(a)):+6.1f}  "
              f"min={np.degrees(a.min()):+6.1f}  max={np.degrees(a.max()):+6.1f}  (deg)")

    print(f"{len(frames)} frames | target ~ 0 deg (neutral ankle, foot ⊥ shank)")
    stats("joint_R5 (R ankle pitch)", r5)
    stats("joint_L5 (L ankle pitch)", l5)
    print(f"mean |ankle| = {np.degrees(np.abs(np.r_[r5, l5]).mean()):.1f} deg   "
          f"(>0 hooked-up/down; tune ik_match_table2 foot rot_offset to minimize)")
