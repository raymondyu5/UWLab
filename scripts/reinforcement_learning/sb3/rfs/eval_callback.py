"""
RFS evaluation callback for SB3 PPO training.

Runs deterministic rollouts at fixed intervals with specific spawn poses,
logs success rate, heatmap, and video to wandb.
"""

import collections
import glob
import os

import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import SpawnCfg, SpawnPose


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
        record_plots: Save heatmap.png and trajectories.png via EvalLogger.
        verbose: SB3 verbosity.
    """

    def __init__(
        self,
        rfs_env,
        spawn_cfg: SpawnCfg,
        episode_steps: int,
        log_dir: str,
        eval_interval: int = 200,
        record_video: bool = True,
        record_plots: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.rfs_env = rfs_env
        self.spawn_cfg = spawn_cfg
        self.episode_steps = episode_steps
        self.log_dir = log_dir
        self.eval_interval = eval_interval
        self.record_video = record_video
        self.record_plots = record_plots
        # Rolling buffer of episode successes for training-time success rate.
        # Matches IsaacLab rl_cfm_pcd_wrapper.py success_buffer pattern.
        self._success_buffer = collections.deque(maxlen=400)
        self._grasped_buffer = collections.deque(maxlen=400)
        self._lifted_buffer = collections.deque(maxlen=400)
        self._near_miss_buffer = collections.deque(maxlen=400)
        self._last_success = [False] * rfs_env.num_envs
        self._last_grasped = [False] * rfs_env.num_envs
        self._last_lifted = [False] * rfs_env.num_envs
        self._last_near_miss = [False] * rfs_env.num_envs
        self._cache_initialized = [False] * rfs_env.num_envs
        self._rollout_count = 0

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
            isaac_env = self.rfs_env.env.unwrapped
            success = isaac_env.cfg.is_success(isaac_env)
            grasped = isaac_env.cfg.is_grasped(isaac_env) if hasattr(isaac_env.cfg, "is_grasped") else None
            lifted = isaac_env.cfg.is_lifted(isaac_env) if hasattr(isaac_env.cfg, "is_lifted") else None
            near_miss = isaac_env.cfg.is_near_miss(isaac_env) if hasattr(isaac_env.cfg, "is_near_miss") else None
            for i, done in enumerate(dones):
                if done:
                    if self._cache_initialized[i]:
                        self._success_buffer.append(float(self._last_success[i]))
                        self._grasped_buffer.append(float(self._last_grasped[i]))
                        self._lifted_buffer.append(float(self._last_lifted[i]))
                        self._near_miss_buffer.append(float(self._last_near_miss[i]))
                    self._last_success[i] = False
                    self._last_grasped[i] = False
                    self._last_lifted[i] = False
                    self._last_near_miss[i] = False
                    self._cache_initialized[i] = False
                else:
                    self._last_success[i] = float(success[i].item())
                    self._last_grasped[i] = float(grasped[i].item()) if grasped is not None else 0.0
                    self._last_lifted[i] = float(lifted[i].item()) if lifted is not None else 0.0
                    self._last_near_miss[i] = float(near_miss[i].item()) if near_miss is not None else 0.0
                    self._cache_initialized[i] = True
            if len(self._success_buffer) == self._success_buffer.maxlen:
                wandb.log(
                    {
                        "train/success_rate": sum(self._success_buffer) / len(self._success_buffer),
                        "train/grasped_rate": sum(self._grasped_buffer) / len(self._grasped_buffer),
                        "train/lifted_rate": sum(self._lifted_buffer) / len(self._lifted_buffer),
                        "train/near_miss_rate": sum(self._near_miss_buffer) / len(self._near_miss_buffer),
                    },
                    step=self.num_timesteps,
                )
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
            record_plots=self.record_plots,
            video_fps=video_fps,
        )
        device = self.rfs_env.device
        num_envs = self.rfs_env.num_envs

        if not self.spawn_cfg.poses:
            self._run_random_eval(logger, isaac_env, device, num_envs, output_dir)
        else:
            self._run_fixed_pose_eval(logger, isaac_env, device, num_envs)

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
        sb3_obs, _ = self.model.env.reset()
        self.model._last_obs = sb3_obs
        self.model._last_episode_starts = np.ones((num_envs,), dtype=bool)

    def _run_fixed_pose_eval(self, logger, isaac_env, device, num_envs):
        poses = self.spawn_cfg.poses
        for pose_idx, pose in enumerate(poses):
            record_this = self.record_video and pose_idx == 0
            obs_dict, _ = self.rfs_env.reset_to_spawn(pose)
            obs_np = _sb3_process_obs(obs_dict)
            logger.begin_episode(pose.name, {"x": pose.x, "y": pose.y, "yaw": pose.yaw})

            episode_successes = []
            episode_grasps = []
            episode_lifted = []
            episode_near_miss = []
            _has_grasped = hasattr(isaac_env.cfg, "is_grasped")
            _has_lifted = hasattr(isaac_env.cfg, "is_lifted")
            _has_near_miss = hasattr(isaac_env.cfg, "is_near_miss")
            recorded = [False] * num_envs
            for _ in range(self.episode_steps):
                success_before = isaac_env.cfg.is_success(isaac_env)
                grasped_before = isaac_env.cfg.is_grasped(isaac_env) if _has_grasped else None
                lifted_before = isaac_env.cfg.is_lifted(isaac_env) if _has_lifted else None
                near_miss_before = isaac_env.cfg.is_near_miss(isaac_env) if _has_near_miss else None
                action, _ = self.model.predict(obs_np, deterministic=False)
                action_t = torch.tensor(action, dtype=torch.float32, device=device)
                obs_dict, _, terminated, truncated, _ = self.rfs_env.step(action_t)
                obs_np = _sb3_process_obs(obs_dict)

                for i in range(num_envs):
                    if (terminated[i] or truncated[i]) and not recorded[i]:
                        episode_successes.append(success_before[i].item())
                        if _has_grasped:
                            episode_grasps.append(grasped_before[i].item())
                        if _has_lifted:
                            episode_lifted.append(lifted_before[i].item())
                        if _has_near_miss:
                            episode_near_miss.append(near_miss_before[i].item())
                        recorded[i] = True

                ee_pose = obs_dict["policy"].get("ee_pose", obs_dict["policy"].get("right_ee_pose"))
                obj_pos = isaac_env.scene["grasp_object"].data.root_pos_w[0].cpu().numpy()
                frame = self.rfs_env.render() if record_this else None

                logger.record_step(
                    ee_pose=ee_pose[0].cpu().numpy() if ee_pose is not None else np.zeros(7),
                    object_pose=obj_pos,
                    action=action[0],
                    frame=frame,
                )

                if all(recorded):
                    break

            n_success = sum(episode_successes)
            n_total = len(episode_successes) if episode_successes else num_envs
            n_grasped = sum(episode_grasps) if episode_grasps else None
            n_lifted = sum(episode_lifted) if episode_lifted else None
            n_near_miss = sum(episode_near_miss) if episode_near_miss else None
            logger.end_episode(n_success / n_total if n_total > 0 else False, n_success=n_success, n_total=n_total,
                               n_grasped=n_grasped, n_lifted=n_lifted, n_near_miss=n_near_miss)

    def _run_random_eval(self, logger, isaac_env, device, num_envs, output_dir):
        num_trials = self.spawn_cfg.num_trials if self.spawn_cfg.num_trials > 0 else 1
        record_first = self.record_video

        for trial_idx in range(num_trials):
            record_this = record_first and trial_idx == 0
            obs_dict, _ = self.rfs_env.reset()
            obs_np = _sb3_process_obs(obs_dict)

            # Capture per-env initial object positions (local, relative to env origins).
            init_pos_w = isaac_env.scene["grasp_object"].data.root_pos_w.clone()  # (N, 3)
            init_pos_local = (init_pos_w - isaac_env.scene.env_origins).cpu().numpy()  # (N, 3)

            logger.begin_episode(f"random_trial_{trial_idx}", None)

            per_env_success = [False] * num_envs
            per_env_grasped = [False] * num_envs
            per_env_lifted = [False] * num_envs
            per_env_near_miss = [False] * num_envs
            _has_grasped = hasattr(isaac_env.cfg, "is_grasped")
            _has_lifted = hasattr(isaac_env.cfg, "is_lifted")
            _has_near_miss = hasattr(isaac_env.cfg, "is_near_miss")
            recorded = [False] * num_envs
            for _ in range(self.episode_steps):
                success_before = isaac_env.cfg.is_success(isaac_env)
                grasped_before = isaac_env.cfg.is_grasped(isaac_env) if _has_grasped else None
                lifted_before = isaac_env.cfg.is_lifted(isaac_env) if _has_lifted else None
                near_miss_before = isaac_env.cfg.is_near_miss(isaac_env) if _has_near_miss else None
                action, _ = self.model.predict(obs_np, deterministic=False)
                action_t = torch.tensor(action, dtype=torch.float32, device=device)
                obs_dict, _, terminated, truncated, _ = self.rfs_env.step(action_t)
                obs_np = _sb3_process_obs(obs_dict)

                for i in range(num_envs):
                    if (terminated[i] or truncated[i]) and not recorded[i]:
                        per_env_success[i] = bool(success_before[i].item())
                        if _has_grasped:
                            per_env_grasped[i] = bool(grasped_before[i].item())
                        if _has_lifted:
                            per_env_lifted[i] = bool(lifted_before[i].item())
                        if _has_near_miss:
                            per_env_near_miss[i] = bool(near_miss_before[i].item())
                        recorded[i] = True

                ee_pose = obs_dict["policy"].get("ee_pose", obs_dict["policy"].get("right_ee_pose"))
                obj_pos = isaac_env.scene["grasp_object"].data.root_pos_w[0].cpu().numpy()
                frame = self.rfs_env.render() if record_this else None

                logger.record_step(
                    ee_pose=ee_pose[0].cpu().numpy() if ee_pose is not None else np.zeros(7),
                    object_pose=obj_pos,
                    action=action[0],
                    frame=frame,
                )

                if all(recorded):
                    break

            n_success = sum(per_env_success)
            n_grasped = sum(per_env_grasped) if _has_grasped else None
            n_lifted = sum(per_env_lifted) if _has_lifted else None
            n_near_miss = sum(per_env_near_miss) if _has_near_miss else None
            logger.end_episode(n_success / num_envs, n_success=n_success, n_total=num_envs,
                               n_grasped=n_grasped, n_lifted=n_lifted, n_near_miss=n_near_miss)
            logger.record_scatter_points(
                xs=init_pos_local[:, 0],
                ys=init_pos_local[:, 1],
                successes=per_env_success,
            )

    def _log_to_wandb(self, results: dict, output_dir: str):
        if wandb.run is None:
            return

        log_dict = {
            "eval/success_rate": results["success_rate"],
            "eval/n_success": results["n_success"],
            "eval/near_miss_rate": results.get("near_miss_rate", 0.0),
            "eval/lifted_rate": results.get("lifted_rate", 0.0),
            "eval/grasped_rate": results.get("grasped_rate", 0.0),
        }

        scatter_path = os.path.join(output_dir, "scatter.png")
        if os.path.isfile(scatter_path):
            log_dict["eval/scatter"] = wandb.Image(scatter_path)

        heatmap_path = os.path.join(output_dir, "heatmap.png")
        if os.path.isfile(heatmap_path):
            log_dict["eval/heatmap"] = wandb.Image(heatmap_path)

        video_dir = os.path.join(output_dir, "videos")
        if os.path.isdir(video_dir):
            videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
            if videos:
                log_dict["eval/video"] = wandb.Video(videos[0])

        wandb.log(log_dict, step=self.num_timesteps)
