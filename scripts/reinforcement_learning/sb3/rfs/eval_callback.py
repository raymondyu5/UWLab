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
        spawn_cfg: SpawnCfg defining eval poses and num_trials per pose.
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
        self._success_buffer = collections.deque(maxlen=200)
        self._rollout_count = 0

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1

        if wandb.run is not None:
            metrics = {k: v for k, v in self.logger.name_to_value.items() if v is not None}
            if metrics:
                wandb.log(metrics, step=self.num_timesteps)

        if self._rollout_count % self.eval_interval == 0:
            self._run_eval()

    def _on_step(self) -> bool:
        # Log training success rate on episode completions.
        # dones[i] is True when env i just finished an episode.
        dones = self.locals.get("dones")
        if dones is not None and wandb.run is not None:
            isaac_env = self.rfs_env.env.unwrapped
            success = isaac_env.cfg.is_success(isaac_env)  # (num_envs,) bool tensor
            for i, done in enumerate(dones):
                if done:
                    self._success_buffer.append(float(success[i].item()))
            if self._success_buffer:
                wandb.log(
                    {"train/success_rate": sum(self._success_buffer) / len(self._success_buffer)},
                    step=self.num_timesteps,
                )
        return True

    def _run_eval(self):
        step_tag = f"step_{self.num_timesteps:010d}"
        output_dir = os.path.join(self.log_dir, "eval", step_tag)
        logger = EvalLogger(output_dir, record_video=self.record_video, record_plots=self.record_plots)

        isaac_env = self.rfs_env.env.unwrapped
        device = self.rfs_env.device

        poses = self.spawn_cfg.poses if self.spawn_cfg.poses else [SpawnPose()]
        num_trials = self.spawn_cfg.num_trials
        episode_idx = 0

        for pose in poses:
            for _ in range(num_trials):
                # Only record video for the first episode of each eval.
                record_this = self.record_video and episode_idx == 0
                obs_dict, _ = self.rfs_env.reset_to_spawn(pose)
                obs_np = _sb3_process_obs(obs_dict)
                logger.begin_episode(pose.name, {"x": pose.x, "y": pose.y, "yaw": pose.yaw})

                for _ in range(self.episode_steps):
                    action, _ = self.model.predict(obs_np, deterministic=True)
                    action_t = torch.tensor(action, dtype=torch.float32, device=device)
                    obs_dict, _, terminated, truncated, _ = self.rfs_env.step(action_t)
                    obs_np = _sb3_process_obs(obs_dict)

                    ee_pose = obs_dict["policy"].get("ee_pose", obs_dict["policy"].get("right_ee_pose"))
                    obj_pos = isaac_env.scene["grasp_object"].data.root_pos_w[0].cpu().numpy()
                    frame = self.rfs_env.render() if record_this else None

                    logger.record_step(
                        ee_pose=ee_pose[0].cpu().numpy() if ee_pose is not None else np.zeros(7),
                        object_pose=obj_pos,
                        action=action[0],
                        frame=frame,
                    )

                    if terminated[0] or truncated[0]:
                        break

                success = bool(isaac_env.cfg.is_success(isaac_env)[0].item())
                logger.end_episode(success)
                episode_idx += 1

        results = logger.finalize()

        if self.verbose:
            print(f"[RFSEvalCallback] step={self.num_timesteps}: "
                  f"success={results['n_success']}/{results['n_episodes']} "
                  f"({100*results['success_rate']:.1f}%)")

        self._log_to_wandb(results, output_dir)

    def _log_to_wandb(self, results: dict, output_dir: str):
        if wandb.run is None:
            return

        log_dict = {
            "eval/success_rate": results["success_rate"],
            "eval/n_success": results["n_success"],
            "eval/n_episodes": results["n_episodes"],
        }

        heatmap_path = os.path.join(output_dir, "heatmap.png")
        if os.path.isfile(heatmap_path):
            log_dict["eval/heatmap"] = wandb.Image(heatmap_path)

        video_dir = os.path.join(output_dir, "videos")
        if os.path.isdir(video_dir):
            videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
            if videos:
                log_dict["eval/video"] = wandb.Video(videos[0])

        wandb.log(log_dict, step=self.num_timesteps)
