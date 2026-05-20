"""
Visualize collected RL rollout data (from collect_rl_rollouts.py).

Produces:
  1. Point cloud plots for a few episodes (a few sampled timesteps each),
     with a 3-D perspective view and a top-down XY view per timestep.
  2. Per-timestep scatter plots of arm/hand joint positions and actions:
     x-axis = episode timestep, one scatter point per episode per timestep.


Usage (no Isaac Sim needed — runs on the host):
./uwlab.sh -p scripts/reinforcement_learning/rialto/visualize_rl_rollouts.py \
    --data_dir data_storage/rialto_rl/rl/cube_grasp_rl_privileged/ \
    --n_episodes 6 \
    --n_dist_episodes 100 \
    --pc_timesteps 0 2 5 10 15 100 \
    --output_dir logs/viz_out
"""

import argparse
import os
import glob

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ── helpers ────────────────────────────────────────────────────────────────

def load_episodes(data_dir: str, n: int) -> list[dict]:
    pattern = os.path.join(data_dir, "episode_*", "episode_*.zarr")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No episode zarr files found in {data_dir!r}. "
                                f"Pattern tried: {pattern}")
    paths = paths[:n]
    episodes = []
    for p in paths:
        store = zarr.open(p, mode="r")
        ep = {k: store[f"data/{k}"][:] for k in store["data"].keys()}
        episodes.append(ep)
    print(f"Loaded {len(episodes)} episodes from {data_dir}")
    return episodes


# ── 1. Point cloud visualisation ──────────────────────────────────────────

def plot_point_clouds(episodes: list[dict], timesteps: list[int], output_dir: str):
    """
    For each episode, plot the seg_pc point cloud at the requested timesteps.
    Row 0: 3-D perspective view.  Row 1: top-down XY view.
    """
    for ep_idx, ep in enumerate(episodes):
        pc_seq = ep["seg_pc"]          # (T, N, 3) or (T, 3, N)
        T = pc_seq.shape[0]

        # Normalise to (T, N, 3)
        if pc_seq.ndim == 3 and pc_seq.shape[-1] != 3 and pc_seq.shape[1] == 3:
            pc_seq = pc_seq.transpose(0, 2, 1)

        valid_ts = [t for t in timesteps if t < T]
        if not valid_ts:
            valid_ts = [0]

        n_cols = len(valid_ts)
        fig = plt.figure(figsize=(4 * n_cols, 8))
        fig.suptitle(f"Episode {ep_idx}  (T={T})", fontsize=11)

        for col, t in enumerate(valid_ts):
            pts = pc_seq[t]  # (N, 3)
            z = pts[:, 2]
            z_norm = (z - z.min()) / (z.ptp() + 1e-8)

            # Row 0: 3-D perspective
            ax3d = fig.add_subplot(2, n_cols, col + 1, projection="3d")
            ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=z_norm, cmap="viridis", s=1, alpha=0.6)
            ax3d.set_title(f"t={t}  3D", fontsize=9)
            ax3d.set_xlabel("x", fontsize=7)
            ax3d.set_ylabel("y", fontsize=7)
            ax3d.set_zlabel("z", fontsize=7)
            ax3d.tick_params(labelsize=6)

            # Row 1: top-down XY
            ax2d = fig.add_subplot(2, n_cols, n_cols + col + 1)
            ax2d.scatter(pts[:, 0], pts[:, 1],
                         c=z_norm, cmap="viridis", s=1, alpha=0.6)
            ax2d.set_title(f"t={t}  top-down", fontsize=9)
            ax2d.set_xlabel("x", fontsize=7)
            ax2d.set_ylabel("y", fontsize=7)
            ax2d.set_aspect("equal")
            ax2d.tick_params(labelsize=6)

        plt.tight_layout()
        out = os.path.join(output_dir, f"pc_episode_{ep_idx}.png")
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  Point cloud plot -> {out}")


# ── 2. Per-timestep scatter plots ─────────────────────────────────────────

def plot_timestep_scatter(episodes: list[dict], output_dir: str):
    """
    For each key (arm_joint_pos, hand_joint_pos, actions), plot one figure
    with one subplot per dimension.  X-axis = episode timestep; each episode
    contributes one scatter point per timestep.  Episodes with different
    lengths are plotted as-is (no truncation needed).
    """
    keys_info = [
        ("arm_joint_pos",  "Arm joint pos per timestep"),
        ("hand_joint_pos", "Hand joint pos per timestep"),
        ("actions",        "Actions per timestep"),
        ("rewards",        "Rewards per timestep"),
    ]

    for key, title in keys_info:
        arrays = [ep[key] for ep in episodes if key in ep]
        if not arrays:
            print(f"  Key {key!r} not found in any episode, skipping.")
            continue

        # rewards are saved as (T,); expand to (T, 1) for uniform handling
        arrays = [a[:, None] if a.ndim == 1 else a for a in arrays]
        D = arrays[0].shape[1]
        n_cols = min(D, 8)
        n_rows = (D + n_cols - 1) // n_cols

        # Pre-build flat (timestep, value) arrays per dimension across all episodes.
        ts_all = [np.concatenate([np.arange(a.shape[0]) for a in arrays])]
        vals_all = [np.concatenate([a[:, d] for a in arrays]) for d in range(D)]
        ts_flat = ts_all[0]

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2.8 * n_cols, 2.2 * n_rows),
                                 squeeze=False)
        fig.suptitle(f"{title}  (E={len(arrays)})", fontsize=11)

        for d in range(D):
            r, c = divmod(d, n_cols)
            ax = axes[r][c]
            ax.scatter(ts_flat, vals_all[d], s=1, alpha=0.15, linewidths=0,
                       color="steelblue", rasterized=True)
            ax.set_title(f"dim {d}", fontsize=8)
            ax.set_xlabel("timestep", fontsize=6)
            ax.tick_params(labelsize=6)

        for d in range(D, n_rows * n_cols):
            r, c = divmod(d, n_cols)
            axes[r][c].set_visible(False)

        plt.tight_layout()
        out = os.path.join(output_dir, f"scatter_{key}.png")
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  Timestep scatter -> {out}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize collected RL rollout zarr data.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory written by collect_rl_rollouts.py.")
    parser.add_argument("--n_episodes", type=int, default=6,
                        help="Number of episodes to use for point cloud plots.")
    parser.add_argument("--n_dist_episodes", type=int, default=100,
                        help="Number of episodes to use for per-timestep scatter plots.")
    parser.add_argument("--pc_timesteps", type=int, nargs="+", default=[0, 25, 50, 75, 100],
                        help="Timestep indices at which to visualise the point cloud.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save PNG figures. Default: <data_dir>/viz/")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.data_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)

    print("Plotting point clouds...")
    pc_episodes = load_episodes(args.data_dir, args.n_episodes)
    plot_point_clouds(pc_episodes, args.pc_timesteps, out_dir)

    print("Plotting per-timestep scatter distributions...")
    dist_episodes = load_episodes(args.data_dir, args.n_dist_episodes)
    plot_timestep_scatter(dist_episodes, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
