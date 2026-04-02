"""
Visualise bottle + box spawn distribution for the Singulate task.

Resets the env N times across E parallel envs and scatter-plots the
(x, y) positions of both objects in the env-local frame.

Optionally saves a grid of camera images (one per reset of env 0)
via --num_images.  Images require --enable_cameras to be set by the
AppLauncher (passed automatically when --num_images > 0).

Usage:
    # Scatter plot only (fast, no cameras)
    ./uwlab.sh -p scripts/eval/plot_singulate_spawns.py \
        --num_envs 64 --num_resets 20 \
        --output spawn_distribution.png --headless

    # Scatter plot + image grid
    ./uwlab.sh -p scripts/eval/plot_singulate_spawns.py \
        --num_envs 4 --num_resets 1 \
        --num_images 16 --images_output spawn_images.png --headless
"""

import argparse
import os
from tqdm import tqdm

from uwlab.utils.paths import setup_third_party_paths

setup_third_party_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_resets", type=int, default=20, help="Resets for scatter plot.")
parser.add_argument(
    "--task_id",
    type=str,
    default="UW-FrankaLeap-SingulateBottle-JointAbs-v0",
)
parser.add_argument("--output", type=str, default="spawn_distribution.png")
parser.add_argument(
    "--num_images",
    type=int,
    default=0,
    help="If > 0, save a grid of this many images from fixed_camera (env 0).",
)
parser.add_argument("--images_output", type=str, default="logs/spawn_images.png")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.num_images > 0:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import uwlab_tasks  # noqa: F401  registers gym envs
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    RL_MODE, EVAL_MODE,
    parse_franka_leap_env_cfg,
)
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.tasks.bottle_singulate import (
    BOTTLE_WIDTH, BOTTLE_LENGTH,
    BOX_WIDTH, BOX_LENGTH,
    BOTTLE_X_RANGE, BOTTLE_Y_RANGE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_camera_rgb(env, camera_name: str = "fixed_camera") -> np.ndarray | None:
    """Return (H, W, 3) uint8 RGB for env index 0, or None if unavailable."""
    scene = env.unwrapped.scene
    if camera_name not in scene._sensors:
        return None
    cam = scene[camera_name]
    if "rgb" not in cam.data.output:
        return None
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb


def collect_spawn_positions(env, num_resets: int):
    bottle_xy, box_xy = [], []
    for _ in range(num_resets):
        env.reset()
        origins = env.unwrapped.scene.env_origins

        b = env.unwrapped.scene["grasp_object"].data.root_state_w[:, :3].clone() - origins
        box = env.unwrapped.scene["box"].data.root_state_w[:, :3].clone() - origins
        box[:, 1] += BOX_LENGTH / 2  # convert box origin from far-right edge to asset centre
        bottle_xy.append(b[:, :2].cpu())
        box_xy.append(box[:, :2].cpu())

    return (
        torch.cat(bottle_xy, dim=0).numpy(),
        torch.cat(box_xy, dim=0).numpy(),
    )

def collect_images(env, num_images: int):
    """Collect `num_images` RGB frames and matching top-down positions (env 0).

    Returns:
        images:    list of (H, W, 3) uint8 arrays
        positions: list of (bottle_xy, box_xy) where each is a (2,) numpy array
    """
    images, positions = [], []

    for _ in tqdm(range(num_images)):
        env.reset()
        # Step once with the task's warmup action so the camera buffer is
        # refreshed — zero action is wrong for JointAbs control.
        hold_action = env.unwrapped.cfg.warmup_action(env.unwrapped)
        env.unwrapped.step(hold_action)

        rgb = _read_camera_rgb(env)
        if rgb is None:
            continue

        origins = env.unwrapped.scene.env_origins
        b = (env.unwrapped.scene["grasp_object"].data.root_state_w[0, :3] - origins[0]).cpu().numpy()
        box = (env.unwrapped.scene["box"].data.root_state_w[0, :3] - origins[0]).cpu().numpy()
        box[1] += BOX_LENGTH / 2  # convert box origin from far-right edge to asset centre
        images.append(rgb)
        positions.append((b[:2], box[:2]))

    return images[:num_images], positions[:num_images]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def draw_rect(ax, cx, cy, w, h, **kwargs):
    ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, **kwargs))


def save_scatter(bottle_xy, box_xy, num_resets, num_envs, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(bottle_xy[:, 0], bottle_xy[:, 1],
               s=8, alpha=0.4, color="steelblue", label="bottle centre")
    ax.scatter(box_xy[:, 0], box_xy[:, 1],
               s=8, alpha=0.4, color="tomato", label="box centre")

    b_mean = bottle_xy.mean(axis=0)
    bx_mean = box_xy.mean(axis=0)
    draw_rect(ax, b_mean[0], b_mean[1], BOTTLE_WIDTH, BOTTLE_LENGTH,
              linewidth=1.5, edgecolor="steelblue", facecolor="none",
              linestyle="--", label="bottle footprint (mean)")
    draw_rect(ax, bx_mean[0], bx_mean[1], BOX_WIDTH, BOX_LENGTH,
              linewidth=1.5, edgecolor="tomato", facecolor="none",
              linestyle="--", label="box footprint (mean)")

    bx_lo, bx_hi = BOTTLE_X_RANGE
    by_lo, by_hi = BOTTLE_Y_RANGE
    ax.add_patch(plt.Rectangle(
        (bx_lo, by_lo), bx_hi - bx_lo, by_hi - by_lo,
        linewidth=1, edgecolor="grey", facecolor="none",
        linestyle=":", label="bottle spawn bounds",
    ))

    ax.set_xlabel("x (m, env-local)")
    ax.set_ylabel("y (m, env-local)")
    ax.set_title(f"Singulate spawn distribution\n({num_resets} resets × {num_envs} envs)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter saved → {out_path}")


def _grid_layout(n: int):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def save_image_grid(images: list[np.ndarray], out_path: str):
    n = len(images)
    rows, cols = _grid_layout(n)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
            ax.set_title(f"reset {i}", fontsize=7)
        ax.axis("off")
    fig.suptitle("fixed_camera view after reset (env 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Image grid saved → {out_path}")


def save_topdown_grid(positions: list[tuple], out_path: str):
    """Save a grid of top-down footprint diagrams matching the image grid layout."""
    n = len(positions)
    rows, cols = _grid_layout(n)

    # Axis limits: slightly outside the bottle spawn bounds + box clearance.
    x_pad, y_pad = 0.15, 0.10
    x_lo = BOTTLE_X_RANGE[0] - BOX_WIDTH - x_pad
    x_hi = BOTTLE_X_RANGE[1] + x_pad
    y_lo = BOTTLE_Y_RANGE[0] - BOTTLE_LENGTH / 2 - y_pad
    y_hi = BOTTLE_Y_RANGE[1] + BOTTLE_LENGTH / 2 + y_pad

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n:
            bottle_xy, box_xy = positions[i]
            draw_rect(ax, bottle_xy[0], bottle_xy[1], BOTTLE_WIDTH, BOTTLE_LENGTH,
                      facecolor="steelblue", edgecolor="navy", alpha=0.5, linewidth=1)
            ax.scatter(*bottle_xy, color="navy", s=20, zorder=3)

            draw_rect(ax, box_xy[0], box_xy[1], BOX_WIDTH, BOX_LENGTH,
                      facecolor="tomato", edgecolor="darkred", alpha=0.5, linewidth=1)
            ax.scatter(*box_xy, color="darkred", s=20, zorder=3)

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_aspect("equal")
            ax.set_title(f"reset {i}", fontsize=7)
            ax.tick_params(labelsize=5)
        else:
            ax.axis("off")

    fig.suptitle("top-down footprints after reset (env 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Top-down grid saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    want_images = args_cli.num_images > 0
    run_mode = EVAL_MODE if want_images else RL_MODE
    # For images we only need a small number of envs; scatter plot can use more.
    num_envs = 1 if want_images else args_cli.num_envs 

    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task_id,
        run_mode=run_mode,
        device="cuda:0",
        num_envs=num_envs,
    )

    if not want_images:
        # Disable cameras entirely to keep startup fast.
        env_cfg.scene.train_camera = None
        env_cfg.scene.fixed_camera = None

    env = gym.make(
        args_cli.task_id,
        cfg=env_cfg,
        render_mode="rgb_array" if want_images else None,
    )

    # --- Scatter plot ---
    print(f"Collecting {args_cli.num_resets} resets × {num_envs} envs …")
    bottle_xy, box_xy = collect_spawn_positions(env, args_cli.num_resets)
    save_scatter(bottle_xy, box_xy, args_cli.num_resets, num_envs,
                 os.path.abspath(args_cli.output))

    # --- Images ---
    if want_images:
        print(f"Capturing {args_cli.num_images} images from fixed_camera …")
        images, positions = collect_images(env, args_cli.num_images)
        if images:
            img_path = os.path.abspath(args_cli.images_output)
            save_image_grid(images, img_path)
            base, ext = os.path.splitext(img_path)
            save_topdown_grid(positions, base + "_topdown" + ext)
        else:
            print("Warning: no images captured (fixed_camera may not be available).")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()