# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quick script to check camera views from fixed_camera and train_camera in Franka Leap grasp env.

Captures RGB and depth from each camera after reset and saves side-by-side comparisons.
Use --num_resets to sample multiple random train_camera poses (train_camera is randomized on reset).

Usage (inside container):
    ./uwlab.sh -p scripts/sysid/check_camera_views.py
    ./uwlab.sh -p scripts/sysid/check_camera_views.py --num_resets 5 --output_dir logs/sysid/camera_check
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Check fixed_camera and train_camera views in Franka Leap grasp env."
)
parser.add_argument(
    "--task",
    type=str,
    default="UW-FrankaLeap-GraspPinkCup-JointAbs-v0",
    help="Task ID (must have both fixed_camera and train_camera, e.g. eval_mode grasp task).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/sysid/camera_check",
    help="Output directory for saved images.",
)
parser.add_argument(
    "--num_resets",
    type=int,
    default=1,
    help="Number of resets to run. Each reset randomizes train_camera; saves one comparison per reset.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    EVAL_MODE,
    parse_franka_leap_env_cfg,
)


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        run_mode=EVAL_MODE,
        device="cuda:0",
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    # Add depth to both cameras for this script

    env = gym.make(args_cli.task, cfg=env_cfg)

    scene = env.unwrapped.scene

    depth_key = "depth"
    for i in range(args_cli.num_resets):
        obs, _ = env.reset()

        fixed_rgb = scene["fixed_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
        train_rgb = scene["train_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
        fixed_depth = scene["fixed_camera"].data.output[depth_key][0].cpu().numpy()
        train_depth = scene["train_camera"].data.output[depth_key][0].cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(fixed_rgb)
        axes[0, 0].set_title("fixed_camera RGB")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(train_rgb)
        axes[0, 1].set_title("train_camera RGB")
        axes[0, 1].axis("off")
        im0 = axes[1, 0].imshow(fixed_depth, cmap="viridis")
        axes[1, 0].set_title("fixed_camera depth")
        axes[1, 0].axis("off")
        plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)
        im1 = axes[1, 1].imshow(train_depth, cmap="viridis")
        axes[1, 1].set_title("train_camera depth")
        axes[1, 1].axis("off")
        plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
        plt.suptitle(f"Reset {i + 1}/{args_cli.num_resets}")
        plt.tight_layout()

        suffix = f"_reset{i}" if args_cli.num_resets > 1 else ""
        path = os.path.join(args_cli.output_dir, f"camera_views{suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {path}")

    env.close()
    print(f"[INFO] Done. Images saved to {args_cli.output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
