"""
Visualize a collected distillation episode as an MP4.

Renders seg_pc (camera-rendered point cloud) frame-by-frame using matplotlib,
alongside reward and joint position plots. Uses imageio-ffmpeg — no system
ffmpeg required.

Usage (outside container):
    python scripts/imitation_learning/visualize_distill_episode.py \
        --episode_dir logs/distill_collection/pour_bottle_0327/episode_0 \
        --output video.mp4

    # Or pick a random episode from a collection dir:
    python scripts/imitation_learning/visualize_distill_episode.py \
        --collection_dir logs/distill_collection/pour_bottle_0327 \
        --episode_idx 0 \
        --output video.mp4
"""

import argparse
import os
import sys

import numpy as np

# imageio + imageio-ffmpeg are in third_party/pip_packages
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(_repo_root, "third_party", "pip_packages"))

import imageio
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_episode(episode_dir: str) -> dict:
    ep_idx = os.path.basename(episode_dir).replace("episode_", "")
    zarr_path = os.path.join(episode_dir, f"episode_{ep_idx}.zarr")
    store = zarr.open(zarr_path, mode="r")
    grp = store["data"]
    return {k: np.array(grp[k]) for k in grp.keys()}


def render_frame(data: dict, t: int, fig, ax_pc, ax_rew, ax_arm, ax_hand) -> np.ndarray:
    seg_pc   = data["seg_pc"]       # (T, N, 3)
    rewards  = data["rewards"]      # (T,)
    arm_pos  = data["arm_joint_pos"]  # (T, 7)
    hand_pos = data["hand_joint_pos"] # (T, 16)
    T = len(rewards)

    # --- Point cloud ---
    ax_pc.cla()
    pts = seg_pc[t]  # (N, 3)
    # Color by height (z)
    z = pts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    ax_pc.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                  c=z_norm, cmap="viridis", s=1, alpha=0.6)
    ax_pc.set_title(f"seg_pc  t={t}/{T-1}", fontsize=9)
    ax_pc.set_xlabel("x", fontsize=7); ax_pc.set_ylabel("y", fontsize=7); ax_pc.set_zlabel("z", fontsize=7)
    ax_pc.tick_params(labelsize=6)

    # --- Reward ---
    ax_rew.cla()
    ax_rew.plot(rewards[:t+1], color="#1F77B4", linewidth=1.2)
    ax_rew.axvline(t, color="red", linewidth=0.8, alpha=0.7)
    ax_rew.set_xlim(0, T - 1)
    ax_rew.set_title("reward", fontsize=9)
    ax_rew.tick_params(labelsize=6)

    # --- Arm joints ---
    ax_arm.cla()
    for j in range(arm_pos.shape[1]):
        ax_arm.plot(arm_pos[:t+1, j], linewidth=0.8, alpha=0.8)
    ax_arm.axvline(t, color="red", linewidth=0.8, alpha=0.7)
    ax_arm.set_xlim(0, T - 1)
    ax_arm.set_title("arm joint pos", fontsize=9)
    ax_arm.tick_params(labelsize=6)

    # --- Hand joints ---
    ax_hand.cla()
    for j in range(hand_pos.shape[1]):
        ax_hand.plot(hand_pos[:t+1, j], linewidth=0.8, alpha=0.6)
    ax_hand.axvline(t, color="red", linewidth=0.8, alpha=0.7)
    ax_hand.set_xlim(0, T - 1)
    ax_hand.set_title("hand joint pos", fontsize=9)
    ax_hand.tick_params(labelsize=6)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, default=None, help="Path to a single episode_N directory")
    parser.add_argument("--collection_dir", type=str, default=None, help="Path to collection dir")
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--output", type=str, default="episode_replay.mp4")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--stride", type=int, default=1,
                        help="Render every Nth frame (use >1 to speed up long episodes)")
    args = parser.parse_args()

    if args.episode_dir:
        episode_dir = args.episode_dir
    elif args.collection_dir:
        episode_dir = os.path.join(args.collection_dir, f"episode_{args.episode_idx}")
    else:
        parser.error("provide --episode_dir or --collection_dir")

    print(f"Loading: {episode_dir}")
    data = load_episode(episode_dir)
    T = len(data["rewards"])
    print(f"  Steps: {T}")
    print(f"  seg_pc shape: {data['seg_pc'].shape}")
    print(f"  Total reward: {data['rewards'].sum():.3f}")

    fig = plt.figure(figsize=(14, 7))
    ax_pc   = fig.add_subplot(1, 4, 1, projection="3d")
    ax_rew  = fig.add_subplot(1, 4, 2)
    ax_arm  = fig.add_subplot(1, 4, 3)
    ax_hand = fig.add_subplot(1, 4, 4)

    frames = []
    timesteps = range(0, T, args.stride)
    print(f"Rendering {len(timesteps)} frames...")
    for i, t in enumerate(timesteps):
        frame = render_frame(data, t, fig, ax_pc, ax_rew, ax_arm, ax_hand)
        frames.append(frame)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(timesteps)}")

    plt.close(fig)

    print(f"Writing {args.output} ...")
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264",
                                 output_params=["-crf", "23", "-pix_fmt", "yuv420p"])
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
