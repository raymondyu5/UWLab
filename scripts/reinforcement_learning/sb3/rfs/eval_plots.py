"""Debug plots for RFSEvalCallback (pose / reward / bottle-motion traces)."""

import os

import matplotlib.pyplot as plt
import numpy as np


def _plot_debug_poses(
    ee_traj: np.ndarray,
    bottle_traj: np.ndarray,
    tip_traj: np.ndarray,
    cup_traj: np.ndarray,
    output_dir: str,
    episode_idx: int,
    done_steps: list[int | None] | None = None,
):
    """Save per-env debug plots for XY trajectories and Z-over-time."""
    num_envs = bottle_traj.shape[1]
    n_cols = min(4, num_envs)
    n_rows = int(np.ceil(num_envs / n_cols))

    bottle_color = "#8B4513"  # saddle brown
    tip_color = "#000000"  # black
    cup_color = "#FFC0CB"  # pink
    ee_color = "#2E8B57"  # sea green

    t_total = bottle_traj.shape[0]

    # XY trajectories
    fig_xy, axes_xy = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows), squeeze=False)
    for env_id in range(num_envs):
        t_end = t_total
        if done_steps is not None and done_steps[env_id] is not None:
            t_end = int(done_steps[env_id]) + 1

        r, c = divmod(env_id, n_cols)
        ax = axes_xy[r][c]

        b_xy = bottle_traj[:t_end, env_id, :2]
        t_xy = tip_traj[:t_end, env_id, :2]
        c_xy = cup_traj[:t_end, env_id, :2]
        ee_xy = ee_traj[:t_end, env_id, :2]

        ax.plot(ee_xy[:, 0], ee_xy[:, 1], color=ee_color, linewidth=1.6, alpha=0.9)
        ax.scatter(ee_xy[0, 0], ee_xy[0, 1], color=ee_color, s=30, marker="o")
        ax.scatter(ee_xy[-1, 0], ee_xy[-1, 1], color=ee_color, s=40, marker="x")

        ax.plot(b_xy[:, 0], b_xy[:, 1], color=bottle_color, linewidth=1.6, alpha=0.9)
        ax.scatter(b_xy[0, 0], b_xy[0, 1], color=bottle_color, s=30, marker="o")
        ax.scatter(b_xy[-1, 0], b_xy[-1, 1], color=bottle_color, s=40, marker="x")

        ax.plot(t_xy[:, 0], t_xy[:, 1], color=tip_color, linewidth=1.6, alpha=0.9)
        ax.scatter(t_xy[0, 0], t_xy[0, 1], color=tip_color, s=30, marker="o")
        ax.scatter(t_xy[-1, 0], t_xy[-1, 1], color=tip_color, s=40, marker="x")

        ax.scatter(c_xy[:, 0], c_xy[:, 1], color=cup_color, s=12, alpha=0.45)
        ax.scatter(c_xy[0, 0], c_xy[0, 1], color=cup_color, s=30, marker="o")
        ax.scatter(c_xy[-1, 0], c_xy[-1, 1], color=cup_color, s=40, marker="x")

        ax.set_title(f"env {env_id}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.25)
        ax.axis("equal")

    for env_id in range(num_envs, n_rows * n_cols):
        r, c = divmod(env_id, n_cols)
        axes_xy[r][c].axis("off")

    fig_xy.tight_layout()
    xy_path = os.path.join(output_dir, f"debug_pose_xy_ep{episode_idx + 1:03d}.png")
    fig_xy.savefig(xy_path, dpi=180)
    plt.close(fig_xy)

    # Z over time
    fig_z, axes_z = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.6 * n_rows), squeeze=False)
    for env_id in range(num_envs):
        t_end = t_total
        if done_steps is not None and done_steps[env_id] is not None:
            t_end = int(done_steps[env_id]) + 1

        r, c = divmod(env_id, n_cols)
        ax = axes_z[r][c]

        b_z = tip_traj[:t_end, env_id, 2]
        c_z = cup_traj[:t_end, env_id, 2]
        step_idx = np.arange(b_z.shape[0])

        ax.plot(step_idx, b_z, color=bottle_color, linewidth=1.8, label="bottle z")
        ax.plot(step_idx, c_z, color=cup_color, linewidth=1.8, label="cup z")

        ax.set_title(f"env {env_id}")
        ax.set_xlabel("step")
        ax.set_ylabel("z")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    for env_id in range(num_envs, n_rows * n_cols):
        r, c = divmod(env_id, n_cols)
        axes_z[r][c].axis("off")

    fig_z.tight_layout()
    z_path = os.path.join(output_dir, f"debug_pose_z_ep{episode_idx + 1:03d}.png")
    fig_z.savefig(z_path, dpi=180)
    plt.close(fig_z)


def _plot_debug_rewards(
    reward_traj: np.ndarray,
    output_dir: str,
    episode_idx: int,
    done_steps: list[int | None] | None = None,
):
    """Save per-env reward-over-time debug plots."""
    num_envs = reward_traj.shape[1]
    n_cols = min(4, num_envs)
    n_rows = int(np.ceil(num_envs / n_cols))

    fig_rew, axes_rew = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    t_total = reward_traj.shape[0]
    for env_id in range(num_envs):
        t_end = t_total
        if done_steps is not None and done_steps[env_id] is not None:
            t_end = int(done_steps[env_id]) + 1
        r, c = divmod(env_id, n_cols)
        ax = axes_rew[r][c]

        rew = reward_traj[:t_end, env_id]
        step_idx = np.arange(rew.shape[0])
        ax.plot(step_idx, rew, color="#1F77B4", linewidth=1.8)
        ax.set_title(f"env {env_id}")
        ax.set_xlabel("step")
        ax.set_ylabel("reward")
        ax.grid(alpha=0.25)

    for env_id in range(num_envs, n_rows * n_cols):
        r, c = divmod(env_id, n_cols)
        axes_rew[r][c].axis("off")

    fig_rew.tight_layout()
    rew_path = os.path.join(output_dir, f"debug_reward_ep{episode_idx + 1:03d}.png")
    fig_rew.savefig(rew_path, dpi=180)
    plt.close(fig_rew)
