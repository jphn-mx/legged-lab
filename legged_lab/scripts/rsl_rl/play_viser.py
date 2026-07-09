# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--viser", action="store_true", default=False, help="Visualize the rollout in a browser via a viser web server."
)
parser.add_argument("--viser_port", type=int, default=8080, help="Port for the viser web viewer.")
parser.add_argument(
    "--viser_mjcf",
    type=str,
    default=None,
    help="Path to the robot MJCF used by the viser viewer. Defaults to the A1 V2 MJCF in the repo root.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# viser does its own browser-side rendering from robot state, so Isaac Sim can run fully headless
if args_cli.viser:
    args_cli.headless = True
    args_cli.real_time = True  # play at wall-clock speed so the browser view looks natural

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import legged_lab.tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        raise NotImplementedError("--use_pretrained_checkpoint is not supported in this Isaac Lab version.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # optional: web-based visualization via viser (renders robot state in the browser, no Isaac Sim GUI required)
    viser_viewer = None
    viser_cmd_term = None
    if args_cli.viser:
        from viser_viewer import create_viser_viewer

        robot = env.unwrapped.scene["robot"]
        dof_names = list(robot.data.joint_names)

        # resolve MJCF path (default: A1 V2 MJCF at the repo root)
        if args_cli.viser_mjcf is not None:
            mjcf_path = args_cli.viser_mjcf
        else:
            repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            mjcf_path = os.path.join(repo_root, "A1_legs_V2_mjcf", "A1_legs_V2_mjcf.xml")
        if not os.path.exists(mjcf_path):
            raise FileNotFoundError(f"[viser] MJCF not found: {mjcf_path}. Pass --viser_mjcf <path>.")

        # if the task has a velocity command, drive it from the web sliders
        cmd_ranges = None
        if "base_velocity" in env.unwrapped.command_manager.active_terms:
            viser_cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
            r = viser_cmd_term.cfg.ranges
            cmd_ranges = {
                "lin_vel_x": tuple(r.lin_vel_x),
                "lin_vel_y": tuple(r.lin_vel_y),
                "ang_vel_yaw": tuple(r.ang_vel_z),
            }
            # take manual control: stop auto-resampling and heading/standing overrides
            viser_cmd_term.cfg.resampling_time_range = (1.0e9, 1.0e9)
            viser_cmd_term.time_left[:] = 1.0e9
            viser_cmd_term.is_heading_env[:] = False
            viser_cmd_term.is_standing_env[:] = False

        viser_viewer = create_viser_viewer(mjcf_path, dof_names, port=args_cli.viser_port, cmd_ranges=cmd_ranges)
        print(f"[INFO] viser web viewer started at http://<server-ip>:{args_cli.viser_port}")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # inject the velocity command from the web sliders
        if viser_cmd_term is not None:
            cmd = viser_viewer.get_command()
            viser_cmd_term.vel_command_b[:, 0] = float(cmd[0])
            viser_cmd_term.vel_command_b[:, 1] = float(cmd[1])
            viser_cmd_term.vel_command_b[:, 2] = float(cmd[2])
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        # push robot state to the web viewer (env 0)
        if viser_viewer is not None:
            robot = env.unwrapped.scene["robot"]
            base_pos = (robot.data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]).cpu().numpy()
            base_quat = robot.data.root_quat_w[0].cpu().numpy()  # Isaac Lab uses wxyz
            joint_pos = robot.data.joint_pos[0].cpu().numpy()
            viser_viewer.update(base_pos, base_quat, joint_pos)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if viser_viewer is not None:
        viser_viewer.stop()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
