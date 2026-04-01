"""
RFS evaluation callback for SB3 PPO training.

Runs deterministic rollouts at fixed intervals with specific spawn poses,
logs success rate, and video to wandb.
"""

import collections
import glob
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import SpawnCfg, SpawnPose
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.tasks.rewards.pour_rewards import compute_tip_pos


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


def _plot_debug_bottle_motion(
    bottle_traj: np.ndarray,
    output_dir: str,
    episode_idx: int,
    done_steps: list[int | None] | None = None,
):
    """
    Save per-env debug plots for bottle XY trajectories.

    Mirrors `scripts/eval/play_bc.py` style (n_rows/n_cols, markers, dpi/fig sizes).
    """
    num_envs = bottle_traj.shape[1]
    n_cols = min(4, num_envs)
    n_rows = int(np.ceil(num_envs / n_cols))

    bottle_color = "#8B4513"  # saddle brown

    fig_xy, axes_xy = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )
    t_total = bottle_traj.shape[0]
    for env_id in range(num_envs):
        t_end = t_total
        if done_steps is not None and done_steps[env_id] is not None:
            t_end = int(done_steps[env_id]) + 1
        r, c = divmod(env_id, n_cols)
        ax = axes_xy[r][c]

        b_xy = bottle_traj[:t_end, env_id, :2]  # (t_end, 2)
        ax.plot(b_xy[:, 0], b_xy[:, 1], color=bottle_color, linewidth=1.6, alpha=0.9)
        ax.scatter(b_xy[0, 0], b_xy[0, 1], color=bottle_color, s=30, marker="o")
        ax.scatter(b_xy[-1, 0], b_xy[-1, 1], color=bottle_color, s=40, marker="x")

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


def _plot_debug_object_z(
    bottle_traj: np.ndarray,
    output_dir: str,
    episode_idx: int,
    done_steps: list[int | None] | None = None,
):
    """Save per-env debug plots for bottle/object z over time."""
    num_envs = bottle_traj.shape[1]
    n_cols = min(4, num_envs)
    n_rows = int(np.ceil(num_envs / n_cols))

    bottle_color = "#8B4513"  # saddle brown

    fig_z, axes_z = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    t_total = bottle_traj.shape[0]
    for env_id in range(num_envs):
        t_end = t_total
        if done_steps is not None and done_steps[env_id] is not None:
            t_end = int(done_steps[env_id]) + 1
        r, c = divmod(env_id, n_cols)
        ax = axes_z[r][c]

        b_z = bottle_traj[:t_end, env_id, 2]
        step_idx = np.arange(b_z.shape[0])
        ax.plot(step_idx, b_z, color=bottle_color, linewidth=1.8, label="object z")

        ax.set_title(f"env {env_id}")
        ax.set_xlabel("step")
        ax.set_ylabel("z")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    for env_id in range(num_envs, n_rows * n_cols):
        r, c = divmod(env_id, n_cols)
        axes_z[r][c].axis("off")

    fig_z.tight_layout()
    z_path = os.path.join(output_dir, f"debug_object_z_ep{episode_idx + 1:03d}.png")
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


def _sb3_process_obs(obs_dict: dict) -> dict:
    """
    Convert Isaac env policy obs dict to the format SB3 model.predict() expects.

    Mirrors Sb3VecEnvWrapper._process_obs: returns dict[str, np.ndarray] with
    seg_pc/rgb keys excluded (removed from single_observation_space by RFSWrapper).
    """
    policy_obs = obs_dict.get("policy", obs_dict)
    skip_keys = {"seg_pc", "rgb"}
    result = {}
    for key, val in policy_obs.items():
        if key in skip_keys:
            continue
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().float().numpy()
        result[key] = val
    return result


class RFSEvalCallback(BaseCallback):
    """
    SB3 callback that periodically runs deterministic RFS rollouts.

    Args:
        rfs_env: RFSWrapper instance (direct access, bypasses SB3 layer).
        spawn_cfg: SpawnCfg defining eval poses. All num_envs are used per pose.
        episode_steps: Max steps per episode (from env cfg.horizon).
        log_dir: Root log dir; eval output goes to log_dir/eval/step_XXXXXXXX/.
        eval_interval: Fire every this many rollouts.
        record_video: Save per-episode MP4s via EvalLogger.
        record_scatter: Save scatter_success/scatter_lifted via EvalLogger.
        record_debug_plots: Save per-step debug pose/reward plots.
        verbose: SB3 verbosity.
    """

    def __init__(
        self,
        rfs_env,
        spawn_cfg: SpawnCfg,
        log_dir: str,
        eval_interval: int = 200,
        record_video: bool = True,
        record_scatter: bool = True,
        record_debug_plots: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.rfs_env = rfs_env
        self.spawn_cfg = spawn_cfg

        self.log_dir = log_dir
        self.eval_interval = eval_interval
        self.record_video = record_video
        self.record_scatter = record_scatter
        self.record_debug_plots = record_debug_plots
        # Rolling buffer of episode successes for training-time success rate.
        # Matches IsaacLab rl_cfm_pcd_wrapper.py success_buffer pattern.
        self._success_buffer = collections.deque(maxlen=400)
        self._grasped_buffer = collections.deque(maxlen=400)
        # Extra metric buffers keyed by metric name — populated lazily from env's metrics_spec.
        self._extra_buffers: dict[str, collections.deque] = {}
        self._last_extra: dict[str, list] = {}
        self._last_success = [False] * rfs_env.num_envs
        self._last_grasped = [False] * rfs_env.num_envs
        self._cache_initialized = [False] * rfs_env.num_envs
        self._rollout_count = 0

    def _episode_steps_after_reset(self) -> int:
        """Steps remaining in the episode after all warmups have run.

        Reads directly from the env's live episode counter so this is correct
        regardless of how episode_length_s is extended in train.py or the wrapper.
        Using cfg.horizon is wrong because it doesn't account for warmup steps
        that have already been consumed from the episode budget.
        """
        ep_buf = int(self.rfs_env.unwrapped.episode_length_buf[0].item())
        max_ep = int(self.rfs_env.unwrapped.max_episode_length)
        return max(1, (max_ep - ep_buf) // self.rfs_env.residual_step)

    def _on_training_start(self) -> None:
        self._run_eval()

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.eval_interval == 0:
            ckpt_path = os.path.join(self.log_dir, f"model_{self._rollout_count:06d}")
            self.model.save(ckpt_path)
            if self.verbose:
                print(f"[RFSEvalCallback] Saved checkpoint: {ckpt_path}.zip")
            self._run_eval()

    def _on_step(self) -> bool:
        # Log training success rate on episode completions.
        # We cache is_success every non-terminal step (scene not yet reset).
        # When done fires, the scene is already reset, so we use the cached value.
        dones = self.locals.get("dones")
        if dones is not None and wandb.run is not None:
            metrics_seen = getattr(self.rfs_env, "_metrics_seen", None)
            if metrics_seen is None:
                return True
            # Lazily create extra buffers for any metrics beyond is_success / is_grasped.
            for key in metrics_seen:
                if key not in ("is_success", "is_grasped") and key not in self._extra_buffers:
                    self._extra_buffers[key] = collections.deque(maxlen=400)
                    self._last_extra[key] = [False] * self.rfs_env.num_envs
            for i, done in enumerate(dones):
                if done:
                    self._success_buffer.append(float(metrics_seen["is_success"][i]))
                    self._grasped_buffer.append(float(metrics_seen["is_grasped"][i]))
                    for key, buf in self._extra_buffers.items():
                        buf.append(float(metrics_seen[key][i]))
                        self._last_extra[key][i] = False
                    self._last_success[i] = False
                    self._last_grasped[i] = False
                    self._cache_initialized[i] = False
                else:
                    self._last_success[i] = float(metrics_seen["is_success"][i])
                    self._last_grasped[i] = float(metrics_seen["is_grasped"][i])
                    for key in self._extra_buffers:
                        self._last_extra[key][i] = float(metrics_seen[key][i])
                    self._cache_initialized[i] = True
            if len(self._success_buffer) == self._success_buffer.maxlen:
                log_dict = {
                    "train/success_rate": sum(self._success_buffer) / len(self._success_buffer),
                    "train/grasped_rate": sum(self._grasped_buffer) / len(self._grasped_buffer),
                }
                for key, buf in self._extra_buffers.items():
                    if len(buf) == buf.maxlen:
                        log_dict[f"train/{key}_rate"] = sum(buf) / len(buf)
                wandb.log(log_dict, step=self.num_timesteps)
        return True

    def _run_eval(self):
        step_tag = f"step_{self.num_timesteps:010d}"
        output_dir = os.path.join(self.log_dir, "eval", step_tag)
        isaac_env = self.rfs_env.env.unwrapped
        cfg = isaac_env.cfg
        video_fps = 1.0 / (cfg.sim.dt * cfg.decimation)
        logger = EvalLogger(
            output_dir,
            record_video=self.record_video,
            record_plots=self.record_scatter,
            video_fps=video_fps,
        )
        device = self.rfs_env.device
        num_envs = self.rfs_env.num_envs

        prev_cache = getattr(self.rfs_env, "enable_metrics_cache", False)
        if hasattr(self.rfs_env, "enable_metrics_cache"):
            self.rfs_env.enable_metrics_cache = True
        try:
            if not self.spawn_cfg.poses:
                self._run_random_eval(logger, isaac_env, device, num_envs, output_dir)
            else:
                self._run_fixed_pose_eval(logger, isaac_env, device, num_envs)
        finally:
            if hasattr(self.rfs_env, "enable_metrics_cache"):
                self.rfs_env.enable_metrics_cache = prev_cache
            self.rfs_env._collect_substep_frames = False

        results = logger.finalize()

        if self.verbose:
            n_tot = results.get("n_total", results["n_episodes"])
            print(f"[RFSEvalCallback] step={self.num_timesteps}: "
                  f"success={results['n_success']}/{n_tot} "
                  f"({100*results['success_rate']:.1f}%)")

        self._log_to_wandb(results, output_dir)

        # Eval modified env state, formatter buffers, and PPO history.
        # Reset everything and sync SB3's cached obs so the next training
        # rollout doesn't start from a stale observation.
        reset_out = self.model.env.reset()
        sb3_obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        self.model._last_obs = sb3_obs
        self.model._last_episode_starts = np.ones((num_envs,), dtype=bool)

    def _run_fixed_pose_eval(self, logger, isaac_env, device, num_envs):
        poses = self.spawn_cfg.poses
        for pose_idx, pose in enumerate(poses):
            if self.verbose:
                print(
                    f"[RFSEvalCallback] eval pose {pose_idx+1}/{len(poses)}: "
                    f"record_video={self.record_video and pose_idx == 0}",
                    flush=True,
                )
            record_this = self.record_video and pose_idx == 0
            self.rfs_env._collect_substep_frames = record_this
            obs_dict, _ = self.rfs_env.reset_to_spawn(pose)
            obs_np = _sb3_process_obs(obs_dict)

            episode_steps = self._episode_steps_after_reset()
            progress_every = max(1, episode_steps // 10)

            logger.begin_episode(pose.name, {"x": pose.x, "y": pose.y, "yaw": pose.yaw})

            episode_successes = []
            episode_success_ever = []
            episode_grasps = []
            episode_extra: dict[str, list[float]] = {}  # keyed by metric name
            recorded = [False] * num_envs

            last_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_extra: dict[str, torch.Tensor] = {}  # lazily initialized

            for step_idx in range(episode_steps):
                # Pre-step: read metrics before env.step() so we never read the
                # post-auto-reset state for envs whose episode just ended.
                # last_success: overwrite each step (end-of-episode criterion).
                # ever_*: OR-accumulate (anytime criterion).
                active = torch.tensor([not recorded[i] for i in range(num_envs)], dtype=torch.bool, device=device)
                if active.any():
                    metrics_np = isaac_env.metrics.get_metrics()
                    m_success = torch.from_numpy(metrics_np["is_success"]).to(device).bool()
                    m_grasped = torch.from_numpy(metrics_np["is_grasped"]).to(device).bool()
                    last_success[active]  = m_success[active]
                    ever_success[active] |= m_success[active]
                    ever_grasped[active] |= m_grasped[active]
                    for key, arr in metrics_np.items():
                        if key in ("is_success", "is_grasped"):
                            continue
                        if key not in ever_extra:
                            ever_extra[key] = torch.zeros(num_envs, dtype=torch.bool, device=device)
                            episode_extra[key] = []
                        m = torch.from_numpy(arr).to(device).bool()
                        ever_extra[key][active] |= m[active]

                action, _ = self.model.predict(obs_np, deterministic=False)
                action_t = torch.tensor(action, dtype=torch.float32, device=device)
                obs_dict, _, terminated, truncated, _ = self.rfs_env.step(action_t)
                obs_np = _sb3_process_obs(obs_dict)

                for i in range(num_envs):
                    if (terminated[i] or truncated[i]) and not recorded[i]:
                        episode_successes.append(float(last_success[i]))
                        episode_success_ever.append(float(ever_success[i]))
                        episode_grasps.append(float(ever_grasped[i]))
                        for key in ever_extra:
                            episode_extra[key].append(float(ever_extra[key][i]))
                        recorded[i] = True

                ee_pose = obs_dict["policy"].get("ee_pose", obs_dict["policy"].get("right_ee_pose"))
                obj_pos = isaac_env.scene["grasp_object"].data.root_pos_w[0].cpu().numpy()
                ee_pose_np = ee_pose[0].cpu().numpy() if ee_pose is not None else np.zeros(7)
                for frame in (self.rfs_env.last_substep_frames or [None]):
                    logger.record_step(ee_pose=ee_pose_np, object_pose=obj_pos, action=action[0], frame=frame)

                if self.verbose and (step_idx % progress_every == 0 or step_idx == episode_steps - 1):
                    print(
                        f"[RFSEvalCallback] eval pose {pose_idx+1}/{len(poses)} "
                        f"step {step_idx+1}/{episode_steps}",
                        flush=True,
                    )

                if all(recorded):
                    break

            n_success = sum(episode_successes)
            n_success_ever = sum(episode_success_ever)
            n_total = len(episode_successes) if episode_successes else num_envs
            extra_metrics = {"n_grasped": sum(episode_grasps)}
            extra_metrics["n_success_ever"] = n_success_ever
            for key, vals in episode_extra.items():
                extra_metrics[key] = int(sum(vals))
            logger.end_episode(
                n_success / n_total if n_total > 0 else False,
                n_success=n_success, n_total=n_total,
                n_success_ever=n_success_ever,
                extra_metrics=extra_metrics,
            )

    def _run_random_eval(self, logger, isaac_env, device, num_envs, output_dir):
        num_trials = self.spawn_cfg.num_trials if self.spawn_cfg.num_trials > 0 else 1
        record_first = self.record_video

        for trial_idx in range(num_trials):
            if self.verbose:
                print(
                    f"[RFSEvalCallback] eval random trial {trial_idx+1}/{num_trials}: "
                    f"record_video={record_first and trial_idx == 0}",
                    flush=True,
                )
            record_this = record_first and trial_idx == 0
            self.rfs_env._collect_substep_frames = record_this
            obs_dict, _ = self.rfs_env.reset()
            obs_np = _sb3_process_obs(obs_dict)

            episode_steps = self._episode_steps_after_reset()
            progress_every = max(1, episode_steps // 10)

            # Capture per-env initial object positions (local, relative to env origins).
            init_pos_w = isaac_env.scene["grasp_object"].data.root_pos_w.clone()  # (N, 3)
            init_pos_local = (init_pos_w - isaac_env.scene.env_origins).cpu().numpy()  # (N, 3)

            logger.begin_episode(f"random_trial_{trial_idx}", None)

            per_env_success = [False] * num_envs
            per_env_success_ever = [False] * num_envs
            per_env_grasped = [False] * num_envs
            per_env_extra: dict[str, list] = {}  # keyed by metric name
            recorded = [False] * num_envs
            done_steps: list[int | None] = [None] * num_envs

            # last_success: overwritten each step — end-of-episode criterion.
            # ever_*: OR-accumulated — anytime criterion.
            last_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_extra: dict[str, torch.Tensor] = {}  # lazily initialized

            ee_pos_traj = [] if self.record_debug_plots else None
            bottle_pos_traj = [] if self.record_debug_plots else None
            tip_pos_traj = [] if self.record_debug_plots else None
            cup_pos_traj = [] if self.record_debug_plots else None
            reward_traj = [] if self.record_debug_plots else None

            for step_idx in range(episode_steps):
                # Pre-step: read metrics before env.step() for all still-active envs.
                # Active = not yet terminated. Frozen envs keep their last OR-accumulated value.
                active = torch.tensor([not recorded[i] for i in range(num_envs)], dtype=torch.bool, device=device)
                if active.any():
                    metrics_np = isaac_env.metrics.get_metrics()
                    m_success = torch.from_numpy(metrics_np["is_success"]).to(device).bool()
                    m_grasped = torch.from_numpy(metrics_np["is_grasped"]).to(device).bool()
                    last_success[active]  = m_success[active]
                    ever_success[active] |= m_success[active]
                    ever_grasped[active] |= m_grasped[active]
                    for key, arr in metrics_np.items():
                        if key in ("is_success", "is_grasped"):
                            continue
                        if key not in ever_extra:
                            ever_extra[key] = torch.zeros(num_envs, dtype=torch.bool, device=device)
                            per_env_extra[key] = [False] * num_envs
                        m = torch.from_numpy(arr).to(device).bool()
                        ever_extra[key][active] |= m[active]

                if self.record_debug_plots:
                    ee_pose = obs_dict["policy"].get("ee_pose", obs_dict["policy"].get("right_ee_pose"))
                    if ee_pose is None:
                        ee_pos_local_pre = np.zeros((num_envs, 3), dtype=np.float32)
                    else:
                        ee_pos_local_pre = ee_pose[:, :3].detach().cpu().numpy()

                    bottle_pos_local_pre, tip_pos_local_pre, cup_pos_local_pre = compute_tip_pos(isaac_env)
                    ee_pos_traj.append(ee_pos_local_pre)
                    bottle_pos_traj.append(bottle_pos_local_pre.detach().cpu().numpy())
                    tip_pos_traj.append(tip_pos_local_pre.detach().cpu().numpy())
                    cup_pos_traj.append(cup_pos_local_pre.detach().cpu().numpy())

                action, _ = self.model.predict(obs_np, deterministic=False)
                action_t = torch.tensor(action, dtype=torch.float32, device=device)
                obs_dict, reward, terminated, truncated, _ = self.rfs_env.step(action_t)
                obs_np = _sb3_process_obs(obs_dict)

                if self.record_debug_plots:
                    reward_np = (
                        reward.detach().cpu().numpy() if isinstance(reward, torch.Tensor) else reward
                    )
                    reward_traj.append(np.asarray(reward_np, dtype=np.float32))

                for i in range(num_envs):
                    if (terminated[i] or truncated[i]) and not recorded[i]:
                        per_env_success[i]      = bool(last_success[i])
                        per_env_success_ever[i] = bool(ever_success[i])
                        per_env_grasped[i]      = bool(ever_grasped[i])
                        for key in ever_extra:
                            per_env_extra[key][i] = bool(ever_extra[key][i])
                        recorded[i] = True
                        done_steps[i] = step_idx

                ee_pose = obs_dict["policy"].get("ee_pose", obs_dict["policy"].get("right_ee_pose"))
                obj_pos = isaac_env.scene["grasp_object"].data.root_pos_w[0].cpu().numpy()
                ee_pose_np = ee_pose[0].cpu().numpy() if ee_pose is not None else np.zeros(7)
                for frame in (self.rfs_env.last_substep_frames or [None]):
                    logger.record_step(ee_pose=ee_pose_np, object_pose=obj_pos, action=action[0], frame=frame)

                if self.verbose and (step_idx % progress_every == 0 or step_idx == episode_steps - 1):
                    print(
                        f"[RFSEvalCallback] eval random trial {trial_idx+1}/{num_trials} "
                        f"step {step_idx+1}/{episode_steps}",
                        flush=True,
                    )

                if all(recorded):
                    break

            n_success      = sum(per_env_success)
            n_success_ever = sum(per_env_success_ever)
            extra_metrics = {"n_grasped": sum(per_env_grasped), "n_success_ever": n_success_ever}
            for key, vals in per_env_extra.items():
                extra_metrics[key] = int(sum(vals))
            logger.end_episode(
                n_success / num_envs,
                n_success=n_success, n_total=num_envs,
                n_success_ever=n_success_ever,
                extra_metrics=extra_metrics,
            )
            # Use the first extra metric (e.g. is_grasped or task-specific) as scatter secondary.
            secondary_key = next(iter(per_env_extra), None)
            logger.record_scatter_points(
                xs=init_pos_local[:, 0],
                ys=init_pos_local[:, 1],
                successes=per_env_success,
                secondary=per_env_extra.get(secondary_key) if secondary_key else None,
                secondary_name=secondary_key or "secondary",
            )

            if self.record_debug_plots and bottle_pos_traj:
                _plot_debug_poses(
                    ee_traj=np.stack(ee_pos_traj, axis=0),
                    bottle_traj=np.stack(bottle_pos_traj, axis=0),
                    tip_traj=np.stack(tip_pos_traj, axis=0),
                    cup_traj=np.stack(cup_pos_traj, axis=0),
                    output_dir=output_dir,
                    episode_idx=trial_idx,
                    done_steps=done_steps,
                )
            if self.record_debug_plots and reward_traj:
                _plot_debug_rewards(
                    reward_traj=np.stack(reward_traj, axis=0),
                    output_dir=output_dir,
                    episode_idx=trial_idx,
                    done_steps=done_steps,
                )

    def _log_to_wandb(self, results: dict, output_dir: str):
        if wandb.run is None:
            return

        log_dict = {
            "eval/success_rate_end": results["success_rate"],
            "eval/success_rate_ever": results.get("success_rate_ever", 0.0),
            "eval/n_success": results["n_success"],
        }
        for key, rate in results.get("extra_metric_rates", {}).items():
            log_dict[f"eval/{key}_rate"] = rate

        scatter_success_path = os.path.join(output_dir, "scatter_success.png")
        if os.path.isfile(scatter_success_path):
            log_dict["eval/scatter_success"] = wandb.Image(scatter_success_path)

        scatter_lifted_path = os.path.join(output_dir, "scatter_lifted.png")
        if os.path.isfile(scatter_lifted_path):
            log_dict["eval/scatter_lifted"] = wandb.Image(scatter_lifted_path)

        video_dir = os.path.join(output_dir, "videos")
        if os.path.isdir(video_dir):
            videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
            if videos:
                log_dict["eval/video"] = wandb.Video(videos[0])

        wandb.log(log_dict, step=self.num_timesteps)
