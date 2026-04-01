"""
Visualise bottle + box spawn distribution for the Singulate task.

Resets the env N times across E parallel envs and scatter-plots the
(x, y) positions of both objects in the env-local frame.

Usage:
    ./uwlab.sh -p scripts/eval/plot_singulate_spawns.py \
        --num_envs 64 --num_resets 20 \
        --output spawn_distribution.png \
        --headless
"""

import argparse
import os
import sys

from uwlab.utils.paths import setup_third_party_paths

setup_third_party_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_resets", type=int, default=20, help="Number of resets per env.")
parser.add_argument(
    "--task_id",
    type=str,
    default="UW-FrankaLeap-SingulateBottle-JointAbs-v0",
)
parser.add_argument("--output", type=str, default="spawn_distribution.png")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import uwlab_tasks  # noqa: F401  registers gym envs
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    RL_MODE,
    parse_franka_leap_env_cfg,
)
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.tasks.bottle_singulate import (
    BOTTLE_WIDTH, BOTTLE_LENGTH,
    BOX_WIDTH, BOX_LENGTH,
    BOTTLE_X_RANGE, BOTTLE_Y_RANGE,
)


def collect_spawn_positions(env, num_resets: int):
    bottle_xy = []
    box_xy = []

    for _ in range(num_resets):
        env.reset()
        origins = env.unwrapped.scene.env_origins  # (E, 3)

        bottle_state = env.unwrapped.scene["grasp_object"].data.root_state_w[:, :3].clone()
        bottle_state -= origins
        bottle_xy.append(bottle_state[:, :2].cpu())

        box_state = env.unwrapped.scene["box"].data.root_state_w[:, :3].clone()
        box_state -= origins
        box_xy.append(box_state[:, :2].cpu())

    return (
        torch.cat(bottle_xy, dim=0).numpy(),
        torch.cat(box_xy, dim=0).numpy(),
    )


def draw_rect(ax, cx, cy, w, h, **kwargs):
    """Draw a rectangle centred at (cx, cy) with given width/height."""
    ax.add_patch(plt.Rectangle(
        (cx - w / 2, cy - h / 2), w, h, **kwargs
    ))


def main():
    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task_id,
        run_mode=RL_MODE,
        device="cuda:0",
        num_envs=args_cli.num_envs,
    )
    # Disable cameras / point clouds to keep startup fast.
    env_cfg.scene.train_camera = None
    env_cfg.scene.fixed_camera = None

    env = gym.make(args_cli.task_id, cfg=env_cfg)

    print(f"Collecting {args_cli.num_resets} resets × {args_cli.num_envs} envs "
          f"= {args_cli.num_resets * args_cli.num_envs} samples …")
    bottle_xy, box_xy = collect_spawn_positions(env, args_cli.num_resets)
    env.close()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(bottle_xy[:, 0], bottle_xy[:, 1],
               s=8, alpha=0.4, color="steelblue", label="bottle centre")
    ax.scatter(box_xy[:, 0], box_xy[:, 1],
               s=8, alpha=0.4, color="tomato", label="box centre")

    # Draw a representative bottle + box footprint at the mean positions
    b_mean = bottle_xy.mean(axis=0)
    bx_mean = box_xy.mean(axis=0)
    draw_rect(ax, b_mean[0], b_mean[1], BOTTLE_WIDTH, BOTTLE_LENGTH,
              linewidth=1.5, edgecolor="steelblue", facecolor="none",
              linestyle="--", label="bottle footprint (mean)")
    draw_rect(ax, bx_mean[0], bx_mean[1], BOX_WIDTH, BOX_LENGTH,
              linewidth=1.5, edgecolor="tomato", facecolor="none",
              linestyle="--", label="box footprint (mean)")

    # Draw the bottle randomisation bounding box
    bx_lo, bx_hi = BOTTLE_X_RANGE
    by_lo, by_hi = BOTTLE_Y_RANGE
    ax.add_patch(plt.Rectangle(
        (bx_lo, by_lo), bx_hi - bx_lo, by_hi - by_lo,
        linewidth=1, edgecolor="grey", facecolor="none",
        linestyle=":", label="bottle spawn bounds"
    ))

    ax.set_xlabel("x (m, env-local)")
    ax.set_ylabel("y (m, env-local)")
    ax.set_title(
        f"Singulate spawn distribution\n"
        f"({args_cli.num_resets} resets × {args_cli.num_envs} envs)"
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    out_path = os.path.abspath(args_cli.output)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()