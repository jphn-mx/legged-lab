"""Convert GMR output pkl to CSV format for csv_to_npz.py.

GMR pkl format:
    root_pos: (T, 3) - root position
    root_rot: (T, 4) - root rotation in xyzw
    dof_pos:  (T, 12) - joint positions in MuJoCo order (R1-R6, L1-L6)
    fps: int

CSV format (for csv_to_npz.py):
    columns: [x, y, z, qx, qy, qz, qw, dof_1, ..., dof_12]

Usage:
    python pkl_to_csv.py jumps1_subject1.pkl
    python pkl_to_csv.py jumps1_subject1.pkl --output jumps1.csv
"""
# python pkl_to_csv.py jumps1_subject1.pkl --output /some/path/jumps1.csv


import argparse
import os
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Convert GMR pkl to WBT csv.")
    parser.add_argument("input", type=str, help="Path to input pkl file.")
    parser.add_argument("--output", type=str, default=None, help="Output csv path. Default: same name with .csv.")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    root_pos = np.array(data["root_pos"])
    root_rot = np.array(data["root_rot"])
    dof_pos = np.array(data["dof_pos"])
    fps = data["fps"]

    print(f"Loaded: {args.input}")
    print(f"  fps={fps}, frames={root_pos.shape[0]}, dofs={dof_pos.shape[1]}")

    csv_data = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

    output_path = args.output
    if output_path is None:
        output_path = os.path.splitext(args.input)[0] + ".csv"

    np.savetxt(output_path, csv_data, delimiter=",")
    print(f"Saved: {output_path} (shape={csv_data.shape})")


if __name__ == "__main__":
    main()
