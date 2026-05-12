import argparse
import pathlib
import os
import multiprocessing as mp

import numpy as np
from natsort import natsorted
from rich import print
import torch
import pickle

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel
import gc
import time
import psutil


def check_memory(threshold_gb=30):
    mem = psutil.virtual_memory()
    available_memory_gb = mem.available / (1024 ** 3)
    if available_memory_gb < threshold_gb:
        used_memory_gb = (mem.total - mem.available) / (1024 ** 3)
        print(f"[WARNING] Memory usage:{used_memory_gb:.2f} GB, available:{available_memory_gb:.2f} GB")
        return True
    return False


HERE = pathlib.Path(__file__).parent
SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"


def process_file(smplx_file_path, tgt_file_path, tgt_folder, total_files):
    num_pause = 0
    while check_memory():
        print(f"[PAUSE] Paused processing {smplx_file_path} to prevent memory overflow. num_pause: {num_pause}")
        time.sleep(60 * 2)
        num_pause += 1
        if num_pause > 10:
            print(f"[ERROR] Memory usage is still high after 10 pauses. Exiting.")
            return

    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            smplx_file_path, SMPLX_FOLDER
        )
    except Exception as e:
        print(f"Error loading {smplx_file_path}: {e}")
        return

    tgt_fps = 30
    try:
        smplx_frame_data_list, aligned_fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
        )
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return

    retargeter = GMR(
        src_human="smplx",
        tgt_robot="a1_legs_v1",
        actual_human_height=actual_human_height,
    )

    qpos_list = []
    for smplx_frame_data in smplx_frame_data_list:
        qpos = retargeter.retarget(smplx_frame_data)
        qpos_list.append(qpos.copy())

    qpos_list = np.array(qpos_list)

    device = "cuda:0"
    kinematics_model = KinematicsModel(retargeter.xml_file, device=device)

    try:
        root_pos = qpos_list[:, :3]
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]

    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot,
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )

    body_names = kinematics_model.body_names

    # Height adjust to ensure the lowest part is on the ground
    body_pos, _ = kinematics_model.forward_kinematics(
        torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
        torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
    )
    lowest_height = torch.min(body_pos[..., 2]).item()
    root_pos[:, 2] = root_pos[:, 2] - lowest_height

    # Offset using the first frame so motion starts at origin
    root_pos[:, :2] -= root_pos[0, :2]

    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
    }

    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)

    done = 0
    for root, _, files in os.walk(tgt_folder):
        done += len([f for f in files if f.endswith('.pkl')])
    print(f"Processed {done}/{total_files}: {tgt_file_path}")

    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Retarget ACCAD SMPL-X motions to A1 robot")
    parser.add_argument(
        "--src_folder",
        type=str,
        default=str(HERE / ".." / "motion_data" / "ACCAD"),
        help="Path to ACCAD dataset folder containing SMPL-X npz files.",
    )
    parser.add_argument(
        "--tgt_folder",
        type=str,
        default=str(HERE / ".." / "motion_data" / "ACCAD_a1_gmr"),
        help="Path to save retargeted A1 motion pkl files.",
    )
    parser.add_argument("--override", default=False, action="store_true")
    parser.add_argument("--num_cpus", default=4, type=int)
    args = parser.parse_args()

    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")
    print(f"Source: {args.src_folder}")
    print(f"Target: {args.tgt_folder}")

    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    exclude_keywords = ["crawl", "_lie", "upstairs", "downstairs"]

    args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.npz"):
                continue
            if not filename.endswith(".npz"):
                continue
            motion_name = filename.split('.')[0]
            if any(kw in motion_name.lower() for kw in exclude_keywords):
                continue

            smplx_file_path = os.path.join(dirpath, filename)
            tgt_file_path = smplx_file_path.replace(src_folder, tgt_folder).replace(".npz", ".pkl")

            if not os.path.exists(tgt_file_path) or args.override:
                args_list.append((smplx_file_path, tgt_file_path, tgt_folder))

    total_files = len(args_list)
    print(f"Total files to process: {total_files}")

    if total_files == 0:
        print("No files to process. Done.")
        return

    with mp.Pool(args.num_cpus) as pool:
        pool.starmap(process_file, [a + (total_files,) for a in args_list])

    print(f"Done. Saved to {tgt_folder}")


if __name__ == "__main__":
    main()
