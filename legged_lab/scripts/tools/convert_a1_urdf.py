"""Convert A1 URDF to USD format for Isaac Lab."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

def main():
    urdf_path = os.path.join(os.path.dirname(__file__), "..", "..", "source", "legged_lab", "legged_lab", "data", "Robots", "A1", "A1-legs_V1.urdf")
    urdf_path = os.path.abspath(urdf_path)

    output_dir = os.path.join(os.path.dirname(urdf_path), "usd")

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=output_dir,
        usd_file_name="A1-legs_V1.usd",
        fix_base=False,
        merge_fixed_joints=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=2.0,
            ),
        ),
    )

    converter = UrdfConverter(cfg)
    print(f"USD saved to: {converter.usd_path}")

    simulation_app.close()

if __name__ == "__main__":
    main()
