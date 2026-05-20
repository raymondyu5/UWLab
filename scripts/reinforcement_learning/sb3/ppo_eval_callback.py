"""
PPO evaluation callback for vanilla SB3 PPO training.

Mirrors RFSEvalCallback but works directly against a FrankaLeapGraspEnv
(no RFSWrapper). Runs deterministic rollouts at fixed intervals, logs
success rate and extra task metrics, saves scatter plots, and syncs to wandb.

Video is not supported in rl_mode (cameras are disabled). Use --video for
training-rollout video recording via gym.wrappers.RecordVideo.
"""

import collections
import glob
import os

import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import SpawnCfg


class WandbRewardTermCallback(BaseCallback):
    """Logs individual Isaac reward terms to wandb at the end of each rollout."""

    def __init__(self, isaac_env, verbose=0):
        super().__init__(verbose)
        self._isaac_env = isaac_env

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if wandb.run is None:
            return
        mgr = self._isaac_env.reward_manager
        term_means = {name: val.mean().item() for name, val in mgr._episode_sums.items()}
        log_dict = {f"rewards/{name}": mean_val for name, mean_val in term_means.items()}
        log_dict["rewards/total"] = sum(term_means.values())
        wandb.log(log_dict, step=self.num_timesteps)


class PPOEvalCallback(BaseCallback):
    """
    SB3 callback that periodically runs deterministic PPO rollouts for evaluation.

    Args:
        isaac_env: FrankaLeapGraspEnv (raw Isaac env) for direct metrics access.
        gym_env: Gym-wrapped Isaac env (pre-SB3). Required for render() when record_video=True.
        spawn_cfg: SpawnCfg defining eval poses (random if no poses defined).
        log_dir: Root log dir; eval output goes to log_dir/eval/step_XXXXXXXXXX/.
        eval_interval: Fire every this many rollouts (PPO updates).
        record_scatter: Save scatter_success PNG via EvalLogger.
        record_video: Save per-episode MP4s via viewport render (works in rl_mode; requires render_mode="rgb_array" in gym.make).
        verbose: SB3 verbosity.
    """

    def __init__(
        self,
        isaac_env,
        spawn_cfg: SpawnCfg,
        log_dir: str,
        gym_env=None,
        eval_interval: int = 50,
        record_scatter: bool = True,
        record_video: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self._isaac_env = isaac_env
        self._gym_env = gym_env
        self.spawn_cfg = spawn_cfg
        self.log_dir = log_dir
        self.eval_interval = eval_interval
        self.record_scatter = record_scatter
        self.record_video = record_video

        self._rollout_count = 0

        # Rolling buffers for training-time success rate (filled from _on_step).
        self._success_buffer = collections.deque(maxlen=400)
        self._extra_buffers: dict[str, collections.deque] = {}

        # Per-env metric cache: updated every step for active envs.
        # When done fires (post auto-reset), we read from this cache
        # (T-1 approximation — one step before the reset).
        self._prev_metrics: dict[int, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # SB3 callback hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self._run_eval()

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.eval_interval == 0:
            ckpt_path = os.path.join(self.log_dir, f"model_{self._rollout_count:06d}")
            self.model.save(ckpt_path)
            if self.verbose:
                print(f"[PPOEvalCallback] Saved checkpoint: {ckpt_path}.zip")
            self._run_eval()

    def _has_metrics(self) -> bool:
        """Return True if the env has any metrics_spec configured."""
        return bool(getattr(self._isaac_env, "metrics", None) and self._isaac_env.metrics._specs)

    def _on_step(self) -> bool:
        """Update rolling training success rate on every env step."""
        dones = self.locals.get("dones")
        if dones is None:
            return True

        if not self._has_metrics():
            return True

        # Current metrics: correct for active envs; wrong for just-reset envs.
        current_metrics = self._isaac_env.metrics.get_metrics()
        if not current_metrics:
            return True

        for i, done in enumerate(dones):
            if done:
                # Use the cached T-1 metrics (last active state before auto-reset).
                cached = self._prev_metrics.pop(i, None)
                if cached is not None:
                    success_val = cached.get("is_success", 0.0)
                    self._success_buffer.append(float(success_val))
                    for key, val in cached.items():
                        if key == "is_success":
                            continue
                        if key not in self._extra_buffers:
                            self._extra_buffers[key] = collections.deque(maxlen=400)
                        self._extra_buffers[key].append(float(val))
            else:
                # Update cache for active envs with current post-step metrics.
                self._prev_metrics[i] = {k: float(v[i]) for k, v in current_metrics.items()}

        # Log rolling rates once the buffer is sufficiently filled.
        if wandb.run is not None and len(self._success_buffer) >= 100:
            log_dict = {
                "train/success_rate": sum(self._success_buffer) / len(self._success_buffer),
            }
            for key, buf in self._extra_buffers.items():
                if len(buf) >= 100:
                    log_dict[f"train/{key}_rate"] = sum(buf) / len(buf)
            wandb.log(log_dict, step=self.num_timesteps)

        return True

    # ------------------------------------------------------------------
    # Eval rollout
    # ------------------------------------------------------------------

    def _run_eval(self):
        step_tag = f"step_{self.num_timesteps:010d}"
        output_dir = os.path.join(self.log_dir, "eval", step_tag)
        cfg = self._isaac_env.cfg
        video_fps = 1.0 / (cfg.sim.dt * cfg.decimation)

        logger = EvalLogger(
            output_dir,
            record_video=self.record_video,
            record_plots=self.record_scatter,
            video_fps=video_fps,
        )

        device = self._isaac_env.device
        num_envs = self._isaac_env.num_envs

        if not self.spawn_cfg.poses:
            self._run_random_eval(logger, device, num_envs)
        else:
            self._run_fixed_pose_eval(logger, device, num_envs)

        results = logger.finalize()

        if self.verbose:
            n_tot = results.get("n_total", results["n_episodes"])
            print(
                f"[PPOEvalCallback] step={self.num_timesteps}: "
                f"success={results['n_success']}/{n_tot} "
                f"({100 * results['success_rate']:.1f}%)"
            )

        self._log_to_wandb(results, output_dir)

        # Reset training env and sync SB3's cached obs so the next rollout
        # starts from a valid observation.
        if hasattr(self.model, "env") and self.model.env is not None:
            reset_out = self.model.env.reset()
            sb3_obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
            self.model._last_obs = sb3_obs
            self.model._last_episode_starts = np.ones((num_envs,), dtype=bool)
        self._prev_metrics.clear()

    def _on_episode_reset(self) -> None:
        """Hook called after each episode reset. Override in subclasses to reset stateful models."""
        pass

    def _episode_horizon(self) -> int:
        """Steps per episode after warmup (from live episode counter)."""
        ep_buf = int(self._isaac_env.episode_length_buf[0].item())
        max_ep = int(self._isaac_env.max_episode_length)
        return max(1, max_ep - ep_buf)

    def _run_fixed_pose_eval(self, logger, device, num_envs):
        from uwlab.eval.spawn import SpawnPose
        poses = self.spawn_cfg.poses
        for pose_idx, pose in enumerate(poses):
            if self.verbose:
                print(
                    f"[PPOEvalCallback] eval pose {pose_idx + 1}/{len(poses)}: "
                    f"x={pose.x:.2f} y={pose.y:.2f} yaw={pose.yaw:.2f}",
                    flush=True,
                )

            # Reset and optionally set object/robot to spawn pose.
            obs_dict, _ = self._isaac_env.reset()
            self._on_episode_reset()
            obs_np = self._extract_obs(obs_dict)

            episode_steps = self._episode_horizon()
            logger.begin_episode(pose.name, {"x": pose.x, "y": pose.y, "yaw": pose.yaw})

            per_env_success = [False] * num_envs
            per_env_success_ever = [False] * num_envs
            per_env_extra: dict[str, list] = {}
            recorded = [False] * num_envs

            last_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_extra: dict[str, torch.Tensor] = {}

            for step_idx in range(episode_steps):
                # Read metrics before step (T-1 approximation for terminated envs).
                active = [not recorded[i] for i in range(num_envs)]
                if any(active) and self._has_metrics():
                    metrics = self._isaac_env.metrics.get_metrics()
                    m_success = torch.tensor(metrics.get("is_success", np.zeros(num_envs)), device=device).bool()
                    active_t = torch.tensor(active, device=device)
                    last_success[active_t] = m_success[active_t]
                    ever_success[active_t] |= m_success[active_t]
                    for key, arr in metrics.items():
                        if key == "is_success":
                            continue
                        if key not in ever_extra:
                            ever_extra[key] = torch.zeros(num_envs, dtype=torch.bool, device=device)
                            per_env_extra[key] = [False] * num_envs
                        m = torch.tensor(arr, device=device).bool()
                        ever_extra[key][active_t] |= m[active_t]

                action_np, _ = self.model.predict(obs_np, deterministic=False)
                action_t = torch.tensor(action_np, dtype=torch.float32, device=device)

                obs_dict, _, terminated, truncated, _ = self._isaac_env.step(action_t)
                obs_np = self._extract_obs(obs_dict)
                done = terminated | truncated

                for i in range(num_envs):
                    if done[i] and not recorded[i]:
                        per_env_success[i] = bool(last_success[i])
                        per_env_success_ever[i] = bool(ever_success[i])
                        for key in ever_extra:
                            per_env_extra[key][i] = bool(ever_extra[key][i])
                        recorded[i] = True

                # Log object pose for first env (trajectory tracking).
                obj_pos = self._isaac_env.scene["grasp_object"].data.root_pos_w[0].cpu().numpy()
                ee_pose_np = self._get_ee_pose_np(obs_dict)
                frame = self._gym_env.render() if self.record_video and self._gym_env is not None else None
                logger.record_step(ee_pose=ee_pose_np, object_pose=obj_pos, action=action_np[0], frame=frame)

                if all(recorded):
                    break

                if self.verbose and step_idx % max(1, episode_steps // 5) == 0:
                    print(f"[PPOEvalCallback]   step {step_idx + 1}/{episode_steps}", flush=True)

            # Catch any envs that didn't terminate early.
            for i in range(num_envs):
                if not recorded[i]:
                    per_env_success[i] = bool(last_success[i])
                    per_env_success_ever[i] = bool(ever_success[i])
                    for key in ever_extra:
                        per_env_extra[key][i] = bool(ever_extra[key][i])

            n_success = sum(per_env_success)
            n_success_ever = sum(per_env_success_ever)
            extra_metrics = {"n_success_ever": n_success_ever}
            for key, vals in per_env_extra.items():
                extra_metrics[key] = int(sum(vals))
            logger.end_episode(
                n_success / num_envs,
                n_success=n_success, n_total=num_envs,
                n_success_ever=n_success_ever,
                extra_metrics=extra_metrics,
            )

    def _run_random_eval(self, logger, device, num_envs):
        num_trials = max(1, self.spawn_cfg.num_trials)
        for trial_idx in range(num_trials):
            if self.verbose:
                print(f"[PPOEvalCallback] eval random trial {trial_idx + 1}/{num_trials}", flush=True)

            obs_dict, _ = self._isaac_env.reset()
            self._on_episode_reset()
            obs_np = self._extract_obs(obs_dict)
            episode_steps = self._episode_horizon()

            # Capture per-env initial object positions for scatter plot.
            init_pos_w = self._isaac_env.scene["grasp_object"].data.root_pos_w.clone()
            init_pos_local = (init_pos_w - self._isaac_env.scene.env_origins).cpu().numpy()

            logger.begin_episode(f"random_trial_{trial_idx}", None)

            per_env_success = [False] * num_envs
            per_env_success_ever = [False] * num_envs
            per_env_extra: dict[str, list] = {}
            recorded = [False] * num_envs
            done_steps: list[int | None] = [None] * num_envs

            last_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_extra: dict[str, torch.Tensor] = {}

            for step_idx in range(episode_steps):
                # Read metrics before step.
                active_t = torch.tensor([not recorded[i] for i in range(num_envs)], device=device)
                if active_t.any() and self._has_metrics():
                    metrics = self._isaac_env.metrics.get_metrics()
                    m_success = torch.tensor(metrics.get("is_success", np.zeros(num_envs)), device=device).bool()
                    last_success[active_t] = m_success[active_t]
                    ever_success[active_t] |= m_success[active_t]
                    for key, arr in metrics.items():
                        if key == "is_success":
                            continue
                        if key not in ever_extra:
                            ever_extra[key] = torch.zeros(num_envs, dtype=torch.bool, device=device)
                            per_env_extra[key] = [False] * num_envs
                        m = torch.tensor(arr, device=device).bool()
                        ever_extra[key][active_t] |= m[active_t]

                action_np, _ = self.model.predict(obs_np, deterministic=False)
                action_t = torch.tensor(action_np, dtype=torch.float32, device=device)

                obs_dict, _, terminated, truncated, _ = self._isaac_env.step(action_t)
                obs_np = self._extract_obs(obs_dict)
                done = terminated | truncated

                for i in range(num_envs):
                    if done[i] and not recorded[i]:
                        per_env_success[i] = bool(last_success[i])
                        per_env_success_ever[i] = bool(ever_success[i])
                        for key in ever_extra:
                            per_env_extra[key][i] = bool(ever_extra[key][i])
                        recorded[i] = True
                        done_steps[i] = step_idx

                obj_pos = self._isaac_env.scene["grasp_object"].data.root_pos_w[0].cpu().numpy()
                ee_pose_np = self._get_ee_pose_np(obs_dict)
                frame = self._gym_env.render() if self.record_video and self._gym_env is not None else None
                logger.record_step(ee_pose=ee_pose_np, object_pose=obj_pos, action=action_np[0], frame=frame)

                if all(recorded):
                    break

                if self.verbose and step_idx % max(1, episode_steps // 5) == 0:
                    print(f"[PPOEvalCallback]   step {step_idx + 1}/{episode_steps}", flush=True)

            # Catch any envs that hit time_out without triggering done.
            for i in range(num_envs):
                if not recorded[i]:
                    per_env_success[i] = bool(last_success[i])
                    per_env_success_ever[i] = bool(ever_success[i])
                    for key in ever_extra:
                        per_env_extra[key][i] = bool(ever_extra[key][i])

            n_success = sum(per_env_success)
            n_success_ever = sum(per_env_success_ever)
            extra_metrics = {"n_success_ever": n_success_ever}
            for key, vals in per_env_extra.items():
                extra_metrics[key] = int(sum(vals))

            logger.end_episode(
                n_success / num_envs,
                n_success=n_success, n_total=num_envs,
                n_success_ever=n_success_ever,
                extra_metrics=extra_metrics,
            )

            # Scatter plot: initial object XY vs success.
            secondary_key = next(iter(per_env_extra), None)
            logger.record_scatter_points(
                xs=init_pos_local[:, 0],
                ys=init_pos_local[:, 1],
                successes=per_env_success,
                secondary=per_env_extra.get(secondary_key) if secondary_key else None,
                secondary_name=secondary_key or "secondary",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_obs(self, obs_dict: dict) -> np.ndarray:
        """Convert Isaac env obs dict to numpy array for model.predict.

        With concatenate_terms=True, obs_dict['policy'] is a flat tensor (num_envs, obs_dim).
        """
        policy_obs = obs_dict.get("policy", obs_dict)
        if isinstance(policy_obs, torch.Tensor):
            return policy_obs.detach().cpu().float().numpy()
        # Dict of tensors: concatenate along last dim.
        arrays = []
        for key, val in policy_obs.items():
            if isinstance(val, torch.Tensor):
                arr = val.detach().cpu().float().numpy()
            else:
                arr = np.asarray(val, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            arrays.append(arr)
        return np.concatenate(arrays, axis=-1)

    def _get_ee_pose_np(self, obs_dict: dict) -> np.ndarray:
        """Extract ee_pose from obs dict for EvalLogger trajectory recording."""
        policy_obs = obs_dict.get("policy", obs_dict)
        if isinstance(policy_obs, dict):
            ee = policy_obs.get("ee_pose")
            if ee is not None:
                return ee[0].detach().cpu().numpy()
        return np.zeros(7)

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

        for filename in ("scatter_success.png", "scatter_lifted.png"):
            path = os.path.join(output_dir, filename)
            if os.path.isfile(path):
                log_dict[f"eval/{filename[:-4]}"] = wandb.Image(path)

        wandb.log(log_dict, step=self.num_timesteps)
