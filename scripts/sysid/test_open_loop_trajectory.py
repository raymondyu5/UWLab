# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher



# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--save_video", type=str, default=None, help="Path to save video (e.g. /tmp/out.mp4). Requires --enable_cameras.")
parser.add_argument("--num_steps", type=int, default=200, help="Number of steps when saving video.")

parser.add_argument("--trajectory_file", type=str, default=None, help="Path to the trajectory file.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import numpy as np


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")


    trajectory = np.load(args_cli.trajectory_file)
    traj_states = trajectory["states"]
    traj_actions = trajectory["actions"]

    # reset environment
    env.reset()

    frames = []
    step = 0

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

        if args_cli.save_video:
            # Read RGB from first env's camera sensor (shape: H x W x 3, uint8)
            camera = env.unwrapped.scene["camera"]
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
            frames.append(rgb)
            step += 1
            if step >= args_cli.num_steps:
                break

    if args_cli.save_video and frames:
        import imageio
        imageio.mimwrite(args_cli.save_video, frames, fps=30)
        print(f"[INFO]: Video saved to {args_cli.save_video}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
