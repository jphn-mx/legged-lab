import argparse
import pickle
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing pickle files from GMR",
    )
    parser.add_argument(
        "--z_offset", type=float, default=0.0,
        help="Offset to subtract from root_pos Z. If not set, auto-computes so that min Z across all files equals --target_z.",
    )
    parser.add_argument(
        "--target_z", type=float, default=0.32,
        help="Target minimum root Z height (default: 0.32m, approximate A1 standing base height).",
    )
    args = parser.parse_args()

    out_folder = os.path.join(args.folder, "csv")
    os.makedirs(out_folder, exist_ok=True)

    pkl_files = [f for f in os.listdir(args.folder) if f.endswith(".pkl")]

    # Auto-compute z_offset: find global min Z across all files
    if args.z_offset is None:
        global_min_z = float("inf")
        for file in pkl_files:
            with open(os.path.join(args.folder, file), "rb") as f:
                motion_data = pickle.load(f)
            root_z = np.array(motion_data["root_pos"])[:, 2]
            global_min_z = min(global_min_z, root_z.min())
        z_offset = global_min_z - args.target_z
        print(f"Auto z_offset: {z_offset:.4f}m (global min Z={global_min_z:.4f}, target={args.target_z})")
    else:
        z_offset = args.z_offset
        print(f"Using manual z_offset: {z_offset:.4f}m")

    for i, file in enumerate(pkl_files):
        with open(os.path.join(args.folder, file), "rb") as f:
            motion_data = pickle.load(f)

        dof_pos = motion_data["dof_pos"]
        frame_rate = motion_data["fps"]
        motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
        motion[:, :3] = motion_data["root_pos"]
        motion[:, 2] -= z_offset
        motion[:, 3:7] = motion_data["root_rot"]
        motion[:, 7:] = dof_pos

        if frame_rate > 30:
            downsample_factor = frame_rate / 30.0
            indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
            old_length = motion.shape[0]
            motion = motion[indices]
            print(f"Downsampled from {old_length} to {motion.shape[0]} frames")

        np.savetxt(
            os.path.join(args.folder, "csv", file.replace(".pkl", ".csv")),
            motion,
            delimiter=",",
        )
        print(f"({i+1}/{len(pkl_files)}) Saved {file.replace('.pkl', '.csv')} (Z range: {motion[:,2].min():.4f} ~ {motion[:,2].max():.4f})")
