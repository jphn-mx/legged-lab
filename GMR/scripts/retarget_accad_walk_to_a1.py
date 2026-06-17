import pathlib
import os
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
ACCAD_ROOT = HERE / ".." / "motion_data" / "ACCAD"
TGT_ROOT = HERE / ".." / "motion_data" / "ACCAD_g1_gmr2"

walk_dirs = [
    "Male2Walking_c3d",
]

files = []
for d in walk_dirs:
    dir_path = ACCAD_ROOT / d
    if not dir_path.exists():
        continue
    for f in sorted(dir_path.glob("*_stageii.npz")):
        files.append((str(f), d))

print(f"Found {len(files)} walk motions to retarget.")

for smplx_file, subdir in files:
    filename = os.path.basename(smplx_file).replace("_stageii.npz", ".pkl")
    save_path = str(TGT_ROOT / subdir / filename)

    if os.path.exists(save_path):
        print(f"[SKIP] {save_path} already exists")
        continue

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"[RUN] {os.path.basename(smplx_file)} -> {save_path}")

    cmd = [
        sys.executable,
        str(HERE / "smplx_to_robot.py"),
        "--smplx_file", smplx_file,
        "--robot", "unitree_g1",#"a1_legs_v2",
        "--save_path", save_path,
        "--rate_limit",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr[-500:]}")
    else:
        print(f"[DONE] {save_path}")

print("All done.")
