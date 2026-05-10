"""
RialTo Stage 1: BC pretraining on zarr demos followed by PPO + BC cotraining in sim.

Trains a state-based MLP teacher policy conditioned on proprioception + privileged object pose (ee_pose + hand_joint_pos + manipulated_object_pose → joint targets).
The teacher is later used by Stage 3 DAgger.

Recommended pipeline:
    Step 0 (collect):
        ./uwlab.sh -p scripts/reinforcement_learning/rialto/collect_rialto_rollouts.py \\
            --task UW-FrankaLeap-GraspBottleRandomResets-JointAbs-v0 \\
            --checkpoint_dir logs/bc/cfm_pcd_bottle_... \\
            --output_dir /path/to/rialto_rollouts \\
            --target_episodes 200 --num_envs 16 --headless

    Step 1 (this script):
        ./uwlab.sh -p scripts/reinforcement_learning/rialto/train_stage2_ppo.py --task UW-FrankaLeap-GraspBottleRandomResets-JointAbs-PPO-v0 \
            --bc_checkpoint  logs/rialto/bc/bc_0508_1320/bc_pretrained.pt \
            --num_envs 128 --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RialTo Stage 1: BC pretrain + PPO + BC cotraining.")
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name.")
parser.add_argument("--data_path", type=str, default=None,
                    help="Path to zarr demo directory. Required unless --bc_checkpoint or --checkpoint is provided.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument(
    "--obs_keys", nargs="+", default=["arm_joint_pos", "hand_joint_pos", "manipulated_object_pose"],
    help="Zarr obs keys to concatenate into state (must match env policy obs group). "
         "Default includes object pose as privileged info; use data from collect_rialto_rollouts.py.",
)
parser.add_argument("--action_key", type=str, default="actions")
# BC pretraining
parser.add_argument("--bc_epochs", type=int, default=500, help="BC pretraining epochs.")
parser.add_argument("--bc_lr", type=float, default=3e-4)
parser.add_argument("--bc_batch_size", type=int, default=256)
# PPO + BC cotraining
parser.add_argument("--bc_coef", type=float, default=0.5, help="BC loss weight during PPO phase.")
parser.add_argument("--bc_demo_batch_size", type=int, default=256)
parser.add_argument("--n_timesteps", type=int, default=400_000_000)
parser.add_argument("--n_steps", type=int, default=200, help="SB3 PPO n_steps per env.")
parser.add_argument("--ppo_batch_size", type=int, default=4096)
parser.add_argument("--n_epochs", type=int, default=10, help="PPO gradient epochs per rollout.")
parser.add_argument("--ppo_lr", type=float, default=3e-4)
parser.add_argument("--vf_coef", type=float, default=0.001)
parser.add_argument("--ent_coef", type=float, default=0.0)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--target_kl", type=float, default=6.0,
                    help="KL divergence threshold for early stopping. Set to 0 to disable.")
parser.add_argument("--warmup_rollouts", type=int, default=10,
                    help="Number of rollouts to collect deterministically (std≈0) before "
                         "enabling exploration. Useful to let the BC-pretrained policy seed "
                         "the replay buffer without noise.")
parser.add_argument("--critic_warmup_rollouts", type=int, default=0,
                    help="Train only the value head for this many rollouts after behavioral warmup "
                         "before allowing policy gradient updates. Lets the critic bootstrap "
                         "from the BC-initialized policy's returns before the actor moves.")
parser.add_argument("--log_std_init", type=float, default=-2.0,
                    help="Initial log_std for the PPO Gaussian policy. "
                         "std = exp(log_std_init). Default -1.0 → std≈0.37. "
                         "SB3 default is 0.0 (std=1.0) which is too noisy post-BC.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume PPO from this .zip checkpoint (skips BC pretraining).")
parser.add_argument("--bc_checkpoint", type=str, default=None,
                    help="Path to bc_pretrained.pt from train_stage1_bc.py. "
                         "Loads BC weights and skips in-script BC training.")
parser.add_argument("--log_dir", type=str, default="logs/rialto/stage1")
parser.add_argument("--wandb_project", type=str, default="rialto_uwlab")
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--eval_interval", type=int, default=50,
                    help="Evaluate policy every N PPO rollouts.")
parser.add_argument("--eval_spawn", type=str, default=None,
                    help="Spawn config name from configs/eval/spawns/ (default: random).")
parser.add_argument("--no_eval_video", action="store_true", default=False,
                    help="Disable eval video recording (video is on by default).")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── all Isaac-dependent imports after AppLauncher ──────────────────────────

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import zarr
import gymnasium as gym

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3.common.vec_env import VecEnvWrapper
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import sys as _sys, os as _os
_RIALTO_DIR = _os.path.dirname(_os.path.abspath(__file__))
_SB3_DIR    = _os.path.abspath(_os.path.join(_RIALTO_DIR, "../../sb3"))
_RFS_DIR    = _os.path.join(_SB3_DIR, "rfs")
for _p in [_RIALTO_DIR, _SB3_DIR, _RFS_DIR]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from bc_ppo import BcPPO

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap import (
    RL_MODE,
    parse_franka_leap_env_cfg,
)
from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import SpawnCfg, load_spawn_cfg


# ── Logging helpers (inlined to avoid fragile sys.path dependencies) ───────

import wandb as _wandb_module
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter


class WandbOutputFormat(KVWriter):
    """Forwards SB3 logger flushes to wandb."""

    def write(self, key_values, key_excluded, step=0):
        if _wandb_module.run is None:
            return
        log_dict = {}
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "wandb" in excluded:
                continue
            if isinstance(value, (int, float, np.floating, np.integer)):
                log_dict[key] = float(value)
        if log_dict:
            _wandb_module.log(log_dict, step=step)

    def close(self):
        pass


class RialtoMetricsCallback(BaseCallback):
    """Accumulates env success metrics across each rollout and logs via SB3 logger.

    Reads metrics at every step (instantaneous per-env state) and averages across
    all steps in the rollout. Reports 'fraction of env-steps where condition held',
    which rises as the policy improves and the object stays lifted/grasped longer.
    Also logs per-term reward sums from the Isaac reward manager.
    """

    def __init__(self, isaac_env, eval_interval: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self._isaac_env = isaac_env
        self._eval_interval = eval_interval
        self._rollout_count = 0
        self._step_accum: dict = {}   # metric_key -> list[float], cleared each rollout

    def _on_step(self) -> bool:
        try:
            metrics = self._isaac_env.metrics.get_metrics()
        except Exception:
            return True
        for k, arr in metrics.items():
            # arr is a bool numpy array of shape (num_envs,)
            self._step_accum.setdefault(k, []).append(float(arr.mean()))
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1

        # Log reward terms every rollout
        try:
            mgr = self._isaac_env.reward_manager
            for name, val in mgr._episode_sums.items():
                v = val.mean().item()
                self.model.logger.record(f"rewards/{name}", v)
        except Exception as e:
            print(f"[RialtoMetricsCallback] reward_manager error: {e}", flush=True)

        if self._rollout_count % self._eval_interval != 0:
            self._step_accum.clear()
        else:
            # Log averaged success/grasp metrics at eval interval
            for k, vals in self._step_accum.items():
                if vals:
                    v = float(np.mean(vals))
                    self.model.logger.record(f"metrics/{k}", v)
            self._step_accum.clear()



class RialtoEvalCallback(BaseCallback):
    """
    Periodically runs deterministic rollouts on the live Isaac env and logs
    success rate + scatter plots to wandb, mirroring RFSEvalCallback.

    Args:
        gym_env:       Raw gym.Env (before SB3 wrapping) shared with training.
        isaac_env:     gym_env.unwrapped — direct Isaac env access for metrics/teleport.
        obs_keys:      Observation keys to concatenate into flat policy obs (same as ProprioVecWrapper).
        eval_interval: Fire every N PPO rollouts.
        log_dir:       Root log dir; eval output goes to log_dir/eval/step_XXXXXXXXXX/.
        spawn_cfg:     SpawnCfg with fixed poses, or None for random reset eval.
        record_video:  Save episode MP4s via EvalLogger.
        verbose:       SB3 verbosity.
    """

    def __init__(self, gym_env, isaac_env, obs_keys, eval_interval, log_dir,
                 spawn_cfg=None, record_video=False, verbose=0):
        super().__init__(verbose=verbose)
        self._gym_env = gym_env
        self._isaac_env = isaac_env
        self._obs_keys = obs_keys
        self.eval_interval = eval_interval
        self.log_dir = log_dir
        self.spawn_cfg = spawn_cfg
        self.record_video = record_video
        self._rollout_count = 0

    def _obs_to_flat(self, obs_dict):
        """Extract obs from policy obs dict and return as flat numpy array.

        Handles two cases:
          - concatenate_terms=True: policy obs is already a flat tensor, return directly.
          - concatenate_terms=False: policy obs is a dict, extract obs_keys and concatenate.
        """
        policy = obs_dict.get("policy", obs_dict)
        if isinstance(policy, (torch.Tensor, np.ndarray)):
            val = policy.detach().cpu().float().numpy() if isinstance(policy, torch.Tensor) else policy
            return val.astype(np.float32)
        parts = []
        for key in self._obs_keys:
            val = policy[key]
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().float().numpy()
            parts.append(val.reshape(val.shape[0], -1) if val.ndim > 1 else val)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _render_frame(self):
        """Return a uint8 HxWx3 frame from the env, or None if rendering fails."""
        frame = self._gym_env.render()
        if frame is None:
            return None
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]  # (num_envs, H, W, C) → (H, W, C)
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        return frame

    def _find_obj_key(self):
        scene_keys = list(self._isaac_env.scene.keys())
        if "grasp_object" in scene_keys:
            return "grasp_object"
        return next(
            k for k in scene_keys
            if k not in ("terrain", "robot", "table", "ground", "dome_light")
            and not k.endswith("_contact")
        )

    def _reset_to_spawn(self, pose, obj_key):
        """Reset env then teleport object to the given spawn pose."""
        obs_dict, _ = self._gym_env.reset()
        device = self._isaac_env.device
        defaults = getattr(self._isaac_env.cfg, "object_spawn_defaults", None)
        if defaults is None:
            return obs_dict
        obj = self._isaac_env.scene[obj_key]
        default_pos = torch.tensor(defaults["default_pos"], dtype=torch.float32, device=device)
        default_rot = torch.tensor(defaults["default_rot"], dtype=torch.float32, device=device)
        new_pos = default_pos.clone()
        new_pos[0] += pose.x
        new_pos[1] += pose.y
        root_state = obj.data.default_root_state.clone()
        root_state[:, :3] = new_pos.unsqueeze(0) + self._isaac_env.scene.env_origins
        root_state[:, 3:7] = default_rot.unsqueeze(0)
        root_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(root_state)
        self._isaac_env.sim.step()
        obs_dict = {"policy": self._isaac_env.observation_manager.compute()["policy"]}
        return obs_dict

    def _on_training_start(self):
        self._run_eval()

    def _on_rollout_end(self):
        self._rollout_count += 1
        if self._rollout_count % self.eval_interval == 0:
            ckpt_path = os.path.join(self.log_dir, f"model_{self._rollout_count:06d}")
            self.model.save(ckpt_path)
            if self.verbose:
                print(f"[RialtoEvalCallback] Saved checkpoint: {ckpt_path}.zip")
            self._run_eval()

    def _on_step(self):
        return True

    def _run_eval(self):
        step_tag = f"step_{self.num_timesteps:010d}"
        output_dir = os.path.join(self.log_dir, "eval", step_tag)
        isaac_env = self._isaac_env
        cfg = isaac_env.cfg
        video_fps = 1.0 / (cfg.sim.dt * cfg.decimation)
        num_envs = isaac_env.num_envs
        device = isaac_env.device
        obj_key = self._find_obj_key()

        logger = EvalLogger(output_dir, record_video=self.record_video,
                            record_plots=True, video_fps=video_fps)

        use_fixed = self.spawn_cfg is not None and self.spawn_cfg.poses
        if use_fixed:
            self._run_fixed_eval(logger, isaac_env, num_envs, device, obj_key)
        else:
            num_trials = self.spawn_cfg.num_trials if self.spawn_cfg is not None else 1
            self._run_random_eval(logger, isaac_env, num_envs, device, obj_key,
                                  num_trials, output_dir)

        results = logger.finalize()

        if self.verbose:
            n_tot = results.get("n_total", results["n_episodes"])
            print(f"[RialtoEvalCallback] step={self.num_timesteps}: "
                  f"success={results['n_success']}/{n_tot} "
                  f"({100 * results['success_rate']:.1f}%)")

        self._log_to_wandb(results, output_dir)

        # Restore env to clean training state after eval modified it.
        reset_out = self.model.env.reset()
        sb3_obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        self.model._last_obs = sb3_obs
        self.model._last_episode_starts = np.ones((num_envs,), dtype=bool)

    def _run_fixed_eval(self, logger, isaac_env, num_envs, device, obj_key):
        for pose_idx, pose in enumerate(self.spawn_cfg.poses):
            record_this = self.record_video and pose_idx == 0
            if self.verbose:
                print(f"[RialtoEvalCallback] eval pose {pose_idx + 1}/{len(self.spawn_cfg.poses)}: "
                      f"x={pose.x:.3f} y={pose.y:.3f} record_video={record_this}", flush=True)
            obs_dict = self._reset_to_spawn(pose, obj_key)
            obs_flat = self._obs_to_flat(obs_dict)
            episode_steps = int(isaac_env.max_episode_length)
            progress_every = max(1, episode_steps // 10)

            logger.begin_episode(pose.name or f"pose_{pose_idx}",
                                 {"x": pose.x, "y": pose.y, "yaw": pose.yaw})

            last_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
            recorded = [False] * num_envs

            for step_idx in range(episode_steps):
                active = torch.tensor([not recorded[i] for i in range(num_envs)],
                                      dtype=torch.bool, device=device)
                if active.any():
                    metrics_np = isaac_env.metrics.get_metrics()
                    m_success = torch.from_numpy(metrics_np["is_success"]).to(device).bool()
                    m_grasped = torch.from_numpy(
                        metrics_np.get("is_grasped", np.zeros(num_envs, dtype=bool))
                    ).to(device).bool()
                    last_success[active] = m_success[active]
                    ever_success[active] |= m_success[active]
                    ever_grasped[active] |= m_grasped[active]

                action, _ = self.model.predict(obs_flat, deterministic=True)
                action_t = torch.tensor(action, dtype=torch.float32, device=device)
                obs_dict, _, terminated, truncated, _ = self._gym_env.step(action_t)
                obs_flat = self._obs_to_flat(obs_dict)

                for i in range(num_envs):
                    if (terminated[i] or truncated[i]) and not recorded[i]:
                        recorded[i] = True

                frame = self._render_frame() if record_this else None
                obj_pos = isaac_env.scene[obj_key].data.root_pos_w[0].cpu().numpy()
                logger.record_step(ee_pose=np.zeros(7), object_pose=obj_pos, action=action[0],
                                   frame=frame)

                if self.verbose and (step_idx % progress_every == 0 or step_idx == episode_steps - 1):
                    print(f"[RialtoEvalCallback] pose {pose_idx + 1} step {step_idx + 1}/{episode_steps}",
                          flush=True)

                if all(recorded):
                    break

            n_success_end = int(last_success.sum().item())
            n_success_ever = int(ever_success.sum().item())
            n_grasped = int(ever_grasped.sum().item())
            logger.end_episode(
                n_success_end / num_envs,
                n_success=n_success_end, n_total=num_envs,
                n_success_ever=n_success_ever,
                extra_metrics={"n_grasped": n_grasped, "n_success_ever": n_success_ever},
            )

    def _run_random_eval(self, logger, isaac_env, num_envs, device, obj_key,
                         num_trials, output_dir):
        for trial_idx in range(num_trials):
            record_this = self.record_video and trial_idx == 0
            if self.verbose:
                print(f"[RialtoEvalCallback] eval random trial {trial_idx + 1}/{num_trials} "
                      f"record_video={record_this}", flush=True)
            obs_dict, _ = self._gym_env.reset()
            obs_flat = self._obs_to_flat(obs_dict)

            init_pos_w = isaac_env.scene[obj_key].data.root_pos_w.clone()
            init_pos_local = (init_pos_w - isaac_env.scene.env_origins).cpu().numpy()

            episode_steps = int(isaac_env.max_episode_length)
            progress_every = max(1, episode_steps // 10)

            logger.begin_episode(f"random_trial_{trial_idx}", None)

            last_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
            recorded = [False] * num_envs

            for step_idx in range(episode_steps):
                active = torch.tensor([not recorded[i] for i in range(num_envs)],
                                      dtype=torch.bool, device=device)
                if active.any():
                    metrics_np = isaac_env.metrics.get_metrics()
                    m_success = torch.from_numpy(metrics_np["is_success"]).to(device).bool()
                    m_grasped = torch.from_numpy(
                        metrics_np.get("is_grasped", np.zeros(num_envs, dtype=bool))
                    ).to(device).bool()
                    last_success[active] = m_success[active]
                    ever_success[active] |= m_success[active]
                    ever_grasped[active] |= m_grasped[active]

                action, _ = self.model.predict(obs_flat, deterministic=True)
                action_t = torch.tensor(action, dtype=torch.float32, device=device)
                obs_dict, _, terminated, truncated, _ = self._gym_env.step(action_t)
                obs_flat = self._obs_to_flat(obs_dict)

                for i in range(num_envs):
                    if (terminated[i] or truncated[i]) and not recorded[i]:
                        recorded[i] = True

                frame = self._render_frame() if record_this else None
                obj_pos = isaac_env.scene[obj_key].data.root_pos_w[0].cpu().numpy()
                logger.record_step(ee_pose=np.zeros(7), object_pose=obj_pos, action=action[0],
                                   frame=frame)

                if self.verbose and (step_idx % progress_every == 0 or step_idx == episode_steps - 1):
                    print(f"[RialtoEvalCallback] trial {trial_idx + 1} step {step_idx + 1}/{episode_steps}",
                          flush=True)

                if all(recorded):
                    break

            n_success_end = int(last_success.sum().item())
            n_success_ever = int(ever_success.sum().item())
            n_grasped = int(ever_grasped.sum().item())
            logger.end_episode(
                n_success_end / num_envs,
                n_success=n_success_end, n_total=num_envs,
                n_success_ever=n_success_ever,
                extra_metrics={"n_grasped": n_grasped, "n_success_ever": n_success_ever},
            )
            logger.record_scatter_points(
                xs=init_pos_local[:, 0],
                ys=init_pos_local[:, 1],
                successes=ever_success.cpu().tolist(),
                secondary=ever_grasped.cpu().tolist(),
                secondary_name="is_grasped",
            )

    def _log_to_wandb(self, results, output_dir):
        import wandb
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
        scatter_grasped_path = os.path.join(output_dir, "scatter_is_grasped.png")
        if os.path.isfile(scatter_grasped_path):
            log_dict["eval/scatter_grasped"] = wandb.Image(scatter_grasped_path)
        import glob as _glob
        video_dir = os.path.join(output_dir, "videos")
        if os.path.isdir(video_dir):
            videos = sorted(_glob.glob(os.path.join(video_dir, "*.mp4")))
            if videos:
                log_dict["eval/video"] = wandb.Video(videos[0])
        wandb.log(log_dict, step=self.num_timesteps)


# ── Data loading ───────────────────────────────────────────────────────────

def load_zarr_demos(data_path: str, obs_keys: list, action_key: str):
    """Load zarr episodes; return flat (obs, actions) float32 numpy arrays.

    Episodes are discovered as episode_*/episode_*.zarr or top-level *.zarr.
    obs = concatenation of obs_keys along last axis.
    """
    root = Path(data_path)
    zarr_files = sorted(root.glob("episode_*/episode_*.zarr"))
    if not zarr_files:
        zarr_files = sorted(root.glob("*.zarr"))
    if not zarr_files:
        raise FileNotFoundError(f"No zarr files found under {root}")

    print(f"[Stage1] Loading {len(zarr_files)} episodes from {root}")
    obs_list, act_list = [], []

    for zpath in zarr_files:
        z = zarr.open(str(zpath), mode="r")

        obs_parts = []
        for key in obs_keys:
            arr = None
            for candidate in (f"data/{key}", key):
                if candidate in z:
                    arr = z[candidate][:].astype(np.float32)
                    break
            if arr is None:
                print(f"  [warn] '{key}' missing in {zpath.name}, skipping episode.")
                break
            obs_parts.append(arr)
        else:
            act = None
            for candidate in (f"data/{action_key}", action_key):
                if candidate in z:
                    act = z[candidate][:].astype(np.float32)
                    break
            if act is None:
                print(f"  [warn] '{action_key}' missing in {zpath.name}, skipping episode.")
                continue
            obs = np.concatenate(obs_parts, axis=-1)
            T = min(len(obs), len(act))
            obs_list.append(obs[:T])
            act_list.append(act[:T])

    if not obs_list:
        raise RuntimeError("No valid episodes loaded.")

    all_obs = np.concatenate(obs_list, axis=0)
    all_acts = np.concatenate(act_list, axis=0)
    print(f"[Stage1] {len(all_obs)} timesteps — obs {all_obs.shape}, acts {all_acts.shape}")
    return all_obs, all_acts


# ── Policy network (matches SB3 MlpPolicy net_arch=[256, 256]) ─────────────

class SimpleMLP(nn.Module):
    """Shared network + action head matching SB3 MlpPolicy(net_arch=[256,256]).

    Attribute names mirror SB3 (modern API):
      shared_net  ↔ policy.mlp_extractor.policy_net
      action_net  ↔ policy.action_net
    so weights can be transferred directly via load_state_dict().
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        layers, in_dim = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.shared_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(in_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.shared_net(x))


# ── Env observation wrapper ────────────────────────────────────────────────

class ProprioVecWrapper(VecEnvWrapper):
    """Selects obs_keys from the Dict obs returned by Sb3VecEnvWrapper and
    concatenates them into a flat (num_envs, obs_dim) float32 numpy array.

    Sb3VecEnvWrapper reads observation_space from the unwrapped Isaac env
    (bypassing any gym.Wrapper), so a post-hoc VecEnvWrapper is required.
    It returns obs as dict[str, np.ndarray] with per-env shapes already set.
    """

    def __init__(self, venv, obs_keys: list):
        # venv.observation_space is a Dict space from Sb3VecEnvWrapper
        # with per-env shapes (no batch dim).
        spaces_dict = venv.observation_space.spaces
        obs_dim = 0
        for key in obs_keys:
            if key not in spaces_dict:
                raise KeyError(f"obs key '{key}' not in Sb3VecEnvWrapper observation_space: {list(spaces_dict.keys())}")
            obs_dim += int(np.prod(spaces_dict[key].shape))
        flat_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        super().__init__(venv, observation_space=flat_space)
        self._obs_keys = obs_keys
        print(f"[ProprioVecWrapper] obs_dim={obs_dim}  keys={obs_keys}")

    def _extract(self, obs: dict) -> np.ndarray:
        parts = [obs[k].reshape(len(obs[k]), -1) if obs[k].ndim > 1 else obs[k]
                 for k in self._obs_keys]
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def reset(self) -> np.ndarray:
        return self._extract(self.venv.reset())

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        # Sb3VecEnvWrapper stores the raw Dict obs as terminal_observation on episode end.
        # Extract it so SB3 sees a flat array matching our Box obs space.
        for info in infos:
            if "terminal_observation" in info:
                info["terminal_observation"] = self._extract(info["terminal_observation"])
        return self._extract(obs), rews, dones, infos


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    device = args_cli.device or "cuda:0"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.join(args_cli.log_dir, f"stage1_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[Stage1] Logging to {log_dir}")

    # ── 1. Optionally load zarr demonstrations ───────────────────────────
    demo_obs_np = demo_acts_np = None
    demo_obs_t = demo_acts_t = None
    if args_cli.data_path is not None:
        demo_obs_np, demo_acts_np = load_zarr_demos(
            args_cli.data_path, args_cli.obs_keys, args_cli.action_key
        )
        demo_obs_t = torch.tensor(demo_obs_np, dtype=torch.float32)
        demo_acts_t = torch.tensor(demo_acts_np, dtype=torch.float32)

    # ── 2. Build / load BC model (skipped when resuming from PPO zip) ────
    net_arch = [512, 512, 512]
    bc_model = None  # only needed when transferring weights into a fresh BcPPO

    if args_cli.checkpoint is None:
        if args_cli.bc_checkpoint is not None:
            print(f"[Stage2] Loading BC weights from {args_cli.bc_checkpoint} ...")
            ckpt = torch.load(args_cli.bc_checkpoint, map_location=device)
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            # Prefer metadata dims; fall back to inferring from weight shapes.
            if isinstance(ckpt, dict) and "obs_dim" in ckpt:
                obs_dim, action_dim = ckpt["obs_dim"], ckpt["action_dim"]
                net_arch = ckpt.get("net_arch", net_arch)
            else:
                obs_dim = state_dict["shared_net.0.weight"].shape[1]
                action_dim = state_dict["action_net.weight"].shape[0]
                i = 0
                net_arch = []
                while f"shared_net.{i}.weight" in state_dict:
                    net_arch.append(state_dict[f"shared_net.{i}.weight"].shape[0])
                    i += 2
            bc_model = SimpleMLP(obs_dim, action_dim, hidden=tuple(net_arch)).to(device)
            bc_model.load_state_dict(state_dict)
            print(f"[Stage2] BC weights loaded (obs_dim={obs_dim} action_dim={action_dim} net_arch={net_arch}).")
        else:
            if demo_obs_np is None:
                raise ValueError(
                    "--data_path is required when neither --bc_checkpoint nor --checkpoint is provided."
                )
            obs_dim = demo_obs_np.shape[-1]
            action_dim = demo_acts_np.shape[-1]
            bc_model = SimpleMLP(obs_dim, action_dim, hidden=tuple(net_arch)).to(device)
            optimizer = torch.optim.Adam(bc_model.parameters(), lr=args_cli.bc_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args_cli.bc_epochs, eta_min=1e-5
            )
            loader = DataLoader(
                TensorDataset(demo_obs_t, demo_acts_t),
                batch_size=args_cli.bc_batch_size, shuffle=True, drop_last=True,
            )
            print(f"[Stage2] BC pretraining for {args_cli.bc_epochs} epochs ...")
            bc_model.train()
            for epoch in range(1, args_cli.bc_epochs + 1):
                total_loss = 0.0
                for obs_b, act_b in loader:
                    obs_b, act_b = obs_b.to(device), act_b.to(device)
                    loss = F.mse_loss(bc_model(obs_b), act_b)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()
                if epoch % 50 == 0:
                    print(f"  epoch {epoch:4d}/{args_cli.bc_epochs}  loss={total_loss/len(loader):.5f}")
            bc_ckpt = os.path.join(log_dir, "bc_pretrained.pt")
            torch.save(bc_model.state_dict(), bc_ckpt)
            print(f"[Stage2] BC weights saved → {bc_ckpt}")
    else:
        print(f"[Stage2] --checkpoint supplied; skipping BC pretraining.")

    # ── 3. Create Isaac Sim environment ──────────────────────────────────
    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        RL_MODE,
        device=device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    gym_env = gym.make(args_cli.task, cfg=env_cfg,
                       render_mode="rgb_array" if not args_cli.no_eval_video else None)
    isaac_env = gym_env.unwrapped   # keep reference for metrics + eval callback
    env = Sb3VecEnvWrapper(gym_env)
    if hasattr(env.observation_space, "spaces"):
        env = ProprioVecWrapper(env, args_cli.obs_keys)
    else:
        print(f"[Stage2] Observation space is already flat Box(dim={env.observation_space.shape[0]}), skipping ProprioVecWrapper.")

    # ── 4. Create BcPPO agent (BC loss fused into every minibatch) ───────
    agent = BcPPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args_cli.seed,
        device=device,
        tensorboard_log=log_dir,
        policy_kwargs={"net_arch": net_arch, "log_std_init": args_cli.log_std_init},
        n_steps=args_cli.n_steps,
        batch_size=args_cli.ppo_batch_size,
        n_epochs=args_cli.n_epochs,
        learning_rate=args_cli.ppo_lr,
        ent_coef=args_cli.ent_coef,
        vf_coef=args_cli.vf_coef,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=args_cli.max_grad_norm,
        target_kl=args_cli.target_kl if args_cli.target_kl > 0 else None,
        bc_obs=demo_obs_t,
        bc_acts=demo_acts_t,
        bc_coef=args_cli.bc_coef,
        bc_batch_size=args_cli.bc_demo_batch_size,
        warmup_rollouts=args_cli.warmup_rollouts,
        critic_warmup_rollouts=args_cli.critic_warmup_rollouts,
    )

    if args_cli.checkpoint:
        agent = BcPPO.load(args_cli.checkpoint, env=env, device=device)
        print(f"[Stage2] Resumed from checkpoint: {args_cli.checkpoint}")
    else:
        print("[Stage2] Transferring BC weights into BcPPO policy ...")
        agent.policy.mlp_extractor.policy_net.load_state_dict(bc_model.shared_net.state_dict())
        agent.policy.action_net.load_state_dict(bc_model.action_net.state_dict())
        print("[Stage2] Weight transfer done.")

    # ── 5. Wandb + SB3 logger ────────────────────────────────────────────
    import wandb
    run_name = args_cli.wandb_run_name or f"rialto-stage1-{timestamp}"
    if args_cli.wandb_project:
        wandb.init(
            project=args_cli.wandb_project,
            name=run_name,
            config={k: v for k, v in vars(args_cli).items() if k not in ("device",)},
        )
        print(f"[Stage1] wandb run: {wandb.run.get_url()}")

    # Route all SB3 logger flushes (losses, ep_rew_mean, bc_loss, …) to wandb.
    sb3_logger = sb3_configure(folder=log_dir, format_strings=["stdout", "tensorboard"])
    if args_cli.wandb_project:
        sb3_logger.output_formats.append(WandbOutputFormat())
    agent.set_logger(sb3_logger)

    # ── 6. Callbacks ─────────────────────────────────────────────────────
    metrics_cb = RialtoMetricsCallback(
        isaac_env=isaac_env,
        eval_interval=args_cli.eval_interval,
        verbose=0,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // args_cli.num_envs, 1),
        save_path=log_dir,
        name_prefix="stage1",
        verbose=1,
    )

    spawn_cfg = None
    if args_cli.eval_spawn is not None:
        spawn_cfg = load_spawn_cfg(args_cli.eval_spawn, "configs/eval/spawns")
    else:
        spawn_cfg = SpawnCfg(poses=[], num_trials=1)

    eval_cb = RialtoEvalCallback(
        gym_env=gym_env,
        isaac_env=isaac_env,
        obs_keys=args_cli.obs_keys,
        eval_interval=args_cli.eval_interval,
        log_dir=log_dir,
        spawn_cfg=spawn_cfg,
        record_video=not args_cli.no_eval_video,
        verbose=1,
    )

    print(f"[Stage1] PPO + BC cotraining for {args_cli.n_timesteps:,} timesteps ...")
    agent.learn(
        total_timesteps=args_cli.n_timesteps,
        callback=[ckpt_cb, metrics_cb, eval_cb],
        progress_bar=True,
        log_interval=1,
    )

    final_path = os.path.join(log_dir, "stage1_final")
    agent.save(final_path)
    print(f"[Stage1] Final model → {final_path}.zip")

    if wandb.run is not None:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
