# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal script to verify the depth observation term works in distill mode.

Creates the Franka Leap grasp env in DISTILL_MODE, resets, steps once, then:
  1. Reads raw depth from the distill camera
  2. Gets seg_pc from obs (RenderedSegPC output)
  3. Saves RGB, depth, and point cloud visualization
  4. Prints diagnostics (depth stats, valid points)

Usage (inside container):
    ./uwlab.sh -p scripts/sysid/verify_distill_depth.py --headless --enable_cameras
    ./uwlab.sh -p scripts/sysid/verify_distill_depth.py --headless --enable_cameras --camera fixed_camera
    ./uwlab.sh -p scripts/sysid/verify_distill_depth.py --headless --enable_cameras --no_crop  # if seg_pc has 0 valid (crop region filters all)
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Verify depth observation term works in distill mode."
)
parser.add_argument(
    "--task",
    type=str,
    default="UW-FrankaLeap-GraspPinkCup-JointAbs-v0",
    help="Task ID (must have seg_pc, e.g. GraspPinkCup, GraspBottle).",
)
parser.add_argument(
    "--camera",
    type=str,
    default="train_camera",
    choices=["fixed_camera", "train_camera"],
    help="Camera used for RenderedSegPC in distill mode.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/sysid/verify_distill_depth",
    help="Output directory for saved images.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric.",
)
parser.add_argument(
    "--no_crop",
    action="store_true",
    default=False,
    help="Use permissive pcd_crop_region to verify unprojection. Default crop may filter all points "
    "(calibrated for real camera CL8384200N1; sim camera pose can differ).",
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
    DISTILL_MODE,
    parse_franka_leap_env_cfg,
)


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        run_mode=DISTILL_MODE,
        device="cuda:0",
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.distill_camera_name = args_cli.camera
    if args_cli.no_crop:
        env_cfg.pcd_crop_region = [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0]
        print("[INFO] Using --no_crop: permissive pcd_crop_region for verification")

    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene

    # Reset and step once so camera buffers are populated
    obs, _ = env.reset()
    warmup = env.unwrapped.cfg.warmup_action(env.unwrapped)
    obs, _, _, _, _ = env.step(warmup)

    cam = scene[args_cli.camera]
    depth_key = "depth"
    if depth_key not in cam.data.output:
        depth_key = "distance_to_image_plane"
    if depth_key not in cam.data.output:
        print(f"[FAIL] Camera {args_cli.camera} has no depth. Available: {list(cam.data.output.keys())}")
        env.close()
        return

    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
    depth = cam.data.output[depth_key]
    if depth.dim() == 3:
        depth_np = depth[0].cpu().numpy()
    else:
        depth_np = depth[0, ..., 0].cpu().numpy() if depth.shape[-1] == 1 else depth[0].cpu().numpy()

    inf_count = np.isinf(depth_np).sum()
    finite = np.isfinite(depth_np)
    depth_min = float(np.min(depth_np[finite])) if np.any(finite) else float("nan")
    depth_max = float(np.max(depth_np[finite])) if np.any(finite) else float("nan")

    print(f"[Depth] camera={args_cli.camera}, key={depth_key}")
    print(f"        shape={depth_np.shape}, inf_count={inf_count}, finite_min={depth_min:.4f}, finite_max={depth_max:.4f}")

    seg_pc = obs["policy"]["seg_pc"][0].cpu().numpy()  # (3, N)
    seg_pc = seg_pc.T  # (N, 3)
    valid = np.any(np.abs(seg_pc) > 1e-6, axis=1)
    n_valid = np.sum(valid)
    print(f"[seg_pc] shape={seg_pc.shape}, valid_points={n_valid}")

    # Save outputs: RGB, depth, and multiple PC views (XY, XZ, YZ) to see coordinate layout
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{args_cli.camera} RGB")
    axes[0, 0].axis("off")

    d_plot = np.where(np.isfinite(depth_np), depth_np, np.nan)
    im = axes[0, 1].imshow(d_plot, cmap="viridis", vmin=depth_min if np.isfinite(depth_min) else 0, vmax=depth_max if np.isfinite(depth_max) else 2)
    axes[0, 1].set_title(f"{args_cli.camera} depth")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    pts = seg_pc[valid]
    if pts.shape[0] > 0:
        axes[0, 2].scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=1, cmap="viridis")
        axes[0, 2].set_title(f"seg_pc XY (n={n_valid})")
        axes[0, 2].set_xlabel("x")
        axes[0, 2].set_ylabel("y")
        axes[0, 2].set_aspect("equal")
        axes[0, 2].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[0, 2].axvline(0, color="gray", linestyle="--", alpha=0.5)

        axes[1, 0].scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], s=1, cmap="viridis")
        axes[1, 0].set_title("seg_pc XZ")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("z")
        axes[1, 0].set_aspect("equal")
        axes[1, 0].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[1, 0].axvline(0, color="gray", linestyle="--", alpha=0.5)

        axes[1, 1].scatter(pts[:, 1], pts[:, 2], c=pts[:, 0], s=1, cmap="viridis")
        axes[1, 1].set_title("seg_pc YZ")
        axes[1, 1].set_xlabel("y")
        axes[1, 1].set_ylabel("z")
        axes[1, 1].set_aspect("equal")
        axes[1, 1].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].axvline(0, color="gray", linestyle="--", alpha=0.5)

        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        axes[1, 2].text(0.1, 0.9, f"x: [{x_min:.3f}, {x_max:.3f}]", transform=axes[1, 2].transAxes, fontsize=10)
        axes[1, 2].text(0.1, 0.75, f"y: [{y_min:.3f}, {y_max:.3f}]", transform=axes[1, 2].transAxes, fontsize=10)
        axes[1, 2].text(0.1, 0.6, f"z: [{z_min:.3f}, {z_max:.3f}]", transform=axes[1, 2].transAxes, fontsize=10)
        axes[1, 2].text(0.1, 0.4, "Expected: ~(0,0,0) along (1,0,0)", transform=axes[1, 2].transAxes, fontsize=9)
    else:
        axes[0, 2].set_title(f"seg_pc XY (n=0)")
        axes[1, 0].set_title("seg_pc XZ (no points)")
        axes[1, 1].set_title("seg_pc YZ (no points)")
    axes[1, 2].set_title("seg_pc bounds")
    axes[1, 2].axis("off")

    plt.suptitle("Distill mode depth verification")
    plt.tight_layout()
    out_path = os.path.join(args_cli.output_dir, "verify_distill_depth.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {out_path}")

    if inf_count == depth_np.size:
        print("[FAIL] All depth values are inf. Check --enable_cameras and headless rendering.")
    elif n_valid < 10:
        print("[WARN] seg_pc has very few valid points. Try --no_crop (pcd_crop_region may filter all points).")
    else:
        print("[OK] Depth and seg_pc appear to be working.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
