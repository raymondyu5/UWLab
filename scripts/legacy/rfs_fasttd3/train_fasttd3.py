"""
Train an RFS (Residual Flow-matching RL) policy with FastTD3.

All algorithm parameters are in configs/rl/rfs_fasttd3_cfg.yaml.
CLI handles env setup and infrastructure only.

Usage (inside container):
    # Grasp task
    ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train_fasttd3.py \\
        --task UW-FrankaLeap-GraspPinkCup-IkRel-v0 \\
        --num_envs 1024 \\
        --diffusion_path logs/real/mini_18/cfm/pcd_cfm/horizon_4_nobs_1 \\
        --headless

    # Pour task (narrow eval grid)
    ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train_fasttd3.py \\
        --task UW-FrankaLeap-PourBottle-IkRel-v0 \\
        --num_envs 1024 \\
        --diffusion_path logs/real/.../cfm/pcd_cfm/horizon_4_nobs_1 \\
        --eval_spawn random_1_trial \\
        --headless
"""

import argparse
import contextlib
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train RFS policy with FastTD3.")

# Env / infra
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from .pt checkpoint.")

# RFS — override individual config values from the YAML if needed
parser.add_argument("--diffusion_path", type=str, required=True)
parser.add_argument("--cfg", type=str, default="configs/rl/rfs_fasttd3_cfg.yaml",
                    help="Path to RFS+FastTD3 config YAML.")
parser.add_argument("--noise_dims", type=str, default=None,
                    help="Override rfs.noise_dims. Format: 'start:end'.")
parser.add_argument("--residual_dims", type=str, default=None,
                    help="Override rfs.residual_dims. Format: 'start:end'.")
parser.add_argument("--eval_interval", type=int, default=None,
                    help="Override eval.interval (in rollout-log units).")
parser.add_argument("--eval_spawn", type=str, default=None,
                    help="Override eval.spawn config name.")
parser.add_argument("--no_eval_video", action="store_true", default=False)
parser.add_argument("--eval_plots", action="store_true", default=False,
                    help="Legacy alias for --eval_debug_plots.")
parser.add_argument("--eval_debug_plots", action="store_true", default=False)
parser.add_argument("--asymmetric_ac", action="store_true", default=False,
                    help="Asymmetric AC: actor sees CFM embedding; critic sees privileged sim state.")

# Wandb
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

# Ensure rfs/ dir is on sys.path so sibling modules are importable.
_RFS_DIR = os.path.dirname(os.path.abspath(__file__))
_SB3_DIR = os.path.dirname(_RFS_DIR)
_UWLAB_DIR = os.path.abspath(os.path.join(_RFS_DIR, "../../../../"))
_PIP_PACKAGES = os.path.join(_UWLAB_DIR, "third_party", "pip_packages")
for _p in [_RFS_DIR, _SB3_DIR, _PIP_PACKAGES]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _cleanup_pbar(*args):
    import gc
    for obj in gc.get_objects():
        if "tqdm_rich" in type(obj).__name__:
            obj.close()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _cleanup_pbar)

"""Rest everything follows."""

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import wandb
import yaml
from tensordict import TensorDict

from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import uwlab_tasks  # noqa: F401

from wrapper import RFSWrapper
from eval_callback import RFSEvalCallback
from callbacks import WandbNoisePredCallback, WandbRewardTermCallback
from fast_td3 import Actor, Critic
from fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    mark_step,
)
from uwlab.eval.spawn import load_spawn_cfg


# ---------------------------------------------------------------------------
# Config / arg helpers (identical to train.py)
# ---------------------------------------------------------------------------

def _parse_dims(s: str):
    s = s.strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected 'start:end', got: {s!r}")
    return (int(parts[0]), int(parts[1]))


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _short_task(task: str) -> str:
    """'UW-FrankaLeap-GraspBottle-IkRel-v0' -> 'GraspBottle'"""
    import re
    name = task
    if name.startswith("UW-FrankaLeap-"):
        name = name[len("UW-FrankaLeap-"):]
    name = re.sub(r"-(IkRel-)?v\d+$", "", name)
    return name


def _format_ckpt_meta(meta: dict | None) -> str:
    if not meta:
        return "unavailable"
    parts = []
    for key in ("epoch", "global_step", "best_val_mse", "best_model_score"):
        if key in meta:
            value = meta[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.6g}")
            else:
                parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "unavailable"


# ---------------------------------------------------------------------------
# Obs-space helpers
# ---------------------------------------------------------------------------

def _actor_keys(obs_space, asymmetric: bool) -> list[str]:
    """Sorted list of obs keys that go to the actor."""
    keys = sorted(obs_space.spaces.keys())
    if asymmetric:
        return [k for k in keys if k.startswith("actor_")]
    return keys


def _critic_keys(obs_space, asymmetric: bool) -> list[str]:
    """Sorted list of obs keys that go to the critic."""
    keys = sorted(obs_space.spaces.keys())
    if asymmetric:
        return [k for k in keys if k.startswith("critic_")]
    return keys


def _flat_dim(obs_space, keys: list[str]) -> int:
    return sum(int(np.prod(obs_space[k].shape)) for k in keys)


def _flatten_obs(obs_dict: dict, keys: list[str], device: torch.device) -> torch.Tensor:
    """Flatten a subset of keys from a dict obs into a single (B, D) tensor."""
    parts = []
    for k in keys:
        v = obs_dict[k]
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=torch.float32, device=device)
        parts.append(v.to(device=device, dtype=torch.float32).reshape(v.shape[0], -1))
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# FastTD3Adapter  — thin shim so RFSEvalCallback can use the TD3 actor
# ---------------------------------------------------------------------------

class FastTD3Adapter:
    """Minimal interface that satisfies RFSEvalCallback's ``self.model`` expectations.

    RFSEvalCallback needs:
      - ``.predict(obs_np, deterministic)``  →  (np.ndarray, None)
      - ``.save(path)``
      - ``.num_timesteps`` (int, updated externally)
      - ``.env`` with ``.reset()``  (optional, used to sync obs after eval)
      - ``._last_obs`` / ``._last_episode_starts``  (set by this class after reset)
    """

    def __init__(
        self,
        actor: Actor,
        rfs_env: RFSWrapper,
        actor_keys: list[str],
        device: torch.device,
        # References needed for save_params
        qnet: Critic,
        qnet_target: Critic,
        obs_normalizer: nn.Module,
        critic_obs_normalizer: nn.Module,
        log_dir: str,
    ):
        self.actor = actor
        self.env = rfs_env
        self._actor_keys = actor_keys
        self._device = device
        self._qnet = qnet
        self._qnet_target = qnet_target
        self._obs_norm = obs_normalizer
        self._critic_obs_norm = critic_obs_normalizer
        self._log_dir = log_dir

        self.num_timesteps: int = 0
        self._last_obs = None
        self._last_episode_starts = None

    def predict(self, obs_np: dict, deterministic: bool = False):
        """Convert numpy obs dict → flat tensor, run actor, return numpy actions."""
        obs_tensors = {
            k: torch.from_numpy(v).to(self._device)
            for k, v in obs_np.items()
            if k in self._actor_keys
        }
        flat = _flatten_obs(obs_tensors, self._actor_keys, self._device)
        with torch.no_grad():
            # Always deterministic during eval — noise is not meaningful for RFS eval.
            action = self.actor.explore(flat, dones=None, deterministic=True)
        return action.cpu().numpy(), None

    def save(self, path: str) -> None:
        save_params(
            global_step=self.num_timesteps,
            actor=self.actor,
            qnet=self._qnet,
            qnet_target=self._qnet_target,
            obs_normalizer=self._obs_norm,
            critic_obs_normalizer=self._critic_obs_norm,
            args=None,
            path=path + ".pt",
        )
        print(f"[FastTD3Adapter] Saved checkpoint: {path}.pt")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    cfg = _load_cfg(args_cli.cfg)
    td3_cfg = cfg["td3"]
    rfs_cfg = cfg["rfs"]
    eval_cfg = cfg["eval"]

    # CLI overrides
    noise_dims = _parse_dims(args_cli.noise_dims or rfs_cfg["noise_dims"])
    residual_dims = _parse_dims(args_cli.residual_dims or rfs_cfg["residual_dims"])
    eval_interval = args_cli.eval_interval or eval_cfg["interval"]
    eval_spawn = args_cli.eval_spawn or eval_cfg["spawn"]

    # Training settings
    gamma = td3_cfg["gamma"]
    learning_starts = td3_cfg["learning_starts"]
    num_updates = td3_cfg["num_updates"]
    policy_frequency = td3_cfg["policy_frequency"]
    policy_noise = td3_cfg["policy_noise"]
    noise_clip = td3_cfg["noise_clip"]
    tau = td3_cfg["tau"]
    use_cdq = td3_cfg["use_cdq"]
    batch_size = td3_cfg["batch_size"]
    rollout_log_interval = td3_cfg.get("rollout_log_interval", 200)
    amp_enabled = td3_cfg.get("amp", False)
    amp_dtype_str = td3_cfg.get("amp_dtype", "bf16")
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = args_cli.wandb_run_name or f"{_short_task(args_cli.task)}_td3_{timestamp}_{uuid4().hex[:6]}"
    log_dir = os.path.abspath(os.path.join("logs", "rfs", run_name))
    print(f"[INFO] Run: {run_name}")
    print(f"[INFO] Logging to: {log_dir}")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # ------------------------------------------------------------------
    # Environment setup (identical to train.py)
    # ------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device else "cuda:0",
        num_envs=args_cli.num_envs,
    )
    env_cfg.run_mode = "rl_mode"
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg, "table_z_range"):
        env_cfg.table_z_range = (0.0, 0.0)

    internal_warmup = int(getattr(env_cfg, "num_warmup_steps", 0))
    external_warmup = rfs_cfg.get("num_warmup_steps", 0)
    env_cfg.episode_length_s += external_warmup * env_cfg.decimation * env_cfg.sim.dt

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    rfs_env = RFSWrapper(
        env,
        diffusion_path=args_cli.diffusion_path,
        residual_step=rfs_cfg["residual_step"],
        noise_dims=noise_dims,
        residual_dims=residual_dims,
        residual_scale=rfs_cfg["residual_scale"],
        clip_actions=rfs_cfg["clip_actions"],
        finger_smooth_alpha=rfs_cfg["finger_smooth_alpha"],
        finger_start_dim=rfs_cfg.get("finger_start_dim", 6),
        num_warmup_steps=rfs_cfg.get("num_warmup_steps", 0),
        asymmetric_ac=args_cli.asymmetric_ac,
        gamma=gamma,
        ppo_history=rfs_cfg.get("ppo_history", False),
    )
    ckpt_path = getattr(rfs_env, "diffusion_ckpt_path", None)
    ckpt_resolved_path = getattr(rfs_env, "diffusion_ckpt_resolved_path", None)
    ckpt_meta = getattr(rfs_env, "diffusion_ckpt_meta", {})
    ckpt_meta_str = _format_ckpt_meta(ckpt_meta)
    print(f"[INFO] Diffusion checkpoint: {ckpt_path}")
    if ckpt_resolved_path and ckpt_resolved_path != ckpt_path:
        print(f"[INFO] Diffusion checkpoint resolved path: {ckpt_resolved_path}")
    print(f"[INFO] Diffusion checkpoint metadata: {ckpt_meta_str}")

    device = rfs_env.device
    num_envs = rfs_env.num_envs

    # ------------------------------------------------------------------
    # Obs-space analysis
    # ------------------------------------------------------------------
    # RFSWrapper patches env.unwrapped.single_observation_space / single_action_space
    # rather than exposing its own observation_space attribute.
    policy_obs_space = rfs_env.env.unwrapped.single_observation_space["policy"]
    a_keys = _actor_keys(policy_obs_space, args_cli.asymmetric_ac)
    c_keys = _critic_keys(policy_obs_space, args_cli.asymmetric_ac)
    n_obs = _flat_dim(policy_obs_space, a_keys)
    n_critic_obs = _flat_dim(policy_obs_space, c_keys)
    n_act = int(np.prod(rfs_env.env.unwrapped.single_action_space.shape))

    print(f"[INFO] Actor obs keys : {a_keys}  →  dim={n_obs}")
    print(f"[INFO] Critic obs keys: {c_keys}  →  dim={n_critic_obs}")
    print(f"[INFO] Action dim     : {n_act}")

    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), {
        "actor_obs_keys": a_keys,
        "critic_obs_keys": c_keys,
        "n_obs": n_obs,
        "n_critic_obs": n_critic_obs,
        "n_act": n_act,
        **td3_cfg,
    })

    # ------------------------------------------------------------------
    # Networks
    # ------------------------------------------------------------------
    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=num_envs,
        init_scale=td3_cfg["init_scale"],
        hidden_dim=td3_cfg["actor_hidden_dim"],
        std_min=td3_cfg["std_min"],
        std_max=td3_cfg["std_max"],
        device=device,
    )
    qnet = Critic(
        n_obs=n_critic_obs,
        n_act=n_act,
        num_atoms=td3_cfg["num_atoms"],
        v_min=td3_cfg["v_min"],
        v_max=td3_cfg["v_max"],
        hidden_dim=td3_cfg["critic_hidden_dim"],
        device=device,
    )
    qnet_target = Critic(
        n_obs=n_critic_obs,
        n_act=n_act,
        num_atoms=td3_cfg["num_atoms"],
        v_min=td3_cfg["v_min"],
        v_max=td3_cfg["v_max"],
        hidden_dim=td3_cfg["critic_hidden_dim"],
        device=device,
    )
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=td3_cfg["critic_learning_rate"],
        weight_decay=td3_cfg["weight_decay"],
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=td3_cfg["actor_learning_rate"],
        weight_decay=td3_cfg["weight_decay"],
    )
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=200_000_000 // num_envs,
        eta_min=td3_cfg["critic_learning_rate_end"],
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=200_000_000 // num_envs,
        eta_min=td3_cfg["actor_learning_rate_end"],
    )

    # ------------------------------------------------------------------
    # Normalizers
    # ------------------------------------------------------------------
    if td3_cfg["obs_normalization"]:
        obs_normalizer = EmpiricalNormalization(n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(n_critic_obs, device=device)
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    if td3_cfg["reward_normalization"]:
        g_max = min(abs(td3_cfg["v_min"]), abs(td3_cfg["v_max"]))
        reward_normalizer = RewardNormalizer(gamma=gamma, device=device, g_max=g_max)
    else:
        reward_normalizer = nn.Identity()

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------
    rb = SimpleReplayBuffer(
        n_env=num_envs,
        buffer_size=td3_cfg["buffer_size"],
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=args_cli.asymmetric_ac,
        n_steps=td3_cfg.get("num_steps", 1),
        gamma=gamma,
        device=device,
    )

    # ------------------------------------------------------------------
    # AMP
    # ------------------------------------------------------------------
    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    global_step = 0
    if args_cli.checkpoint is not None:
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        qnet.load_state_dict(ckpt["qnet_state_dict"])
        qnet_target.load_state_dict(ckpt["qnet_target_state_dict"])
        if td3_cfg["obs_normalization"] and hasattr(obs_normalizer, "load_state_dict"):
            obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])
            critic_obs_normalizer.load_state_dict(ckpt["critic_obs_normalizer_state"])
        global_step = ckpt.get("global_step", 0)
        print(f"[INFO] Resumed from {args_cli.checkpoint}, global_step={global_step}")

    # ------------------------------------------------------------------
    # Adapter + callbacks
    # ------------------------------------------------------------------
    adapter = FastTD3Adapter(
        actor=actor,
        rfs_env=rfs_env,
        actor_keys=a_keys,
        device=device,
        qnet=qnet,
        qnet_target=qnet_target,
        obs_normalizer=obs_normalizer,
        critic_obs_normalizer=critic_obs_normalizer,
        log_dir=log_dir,
    )

    spawn_cfg = load_spawn_cfg(eval_spawn, "configs/eval/spawns")
    eval_cb = RFSEvalCallback(
        rfs_env=rfs_env,
        spawn_cfg=spawn_cfg,
        log_dir=log_dir,
        eval_interval=eval_interval,
        record_video=not args_cli.no_eval_video and eval_cfg["record_video"],
        record_scatter=True,
        record_debug_plots=(args_cli.eval_debug_plots or args_cli.eval_plots),
        verbose=1,
    )
    eval_cb.model = adapter
    eval_cb.num_timesteps = global_step * num_envs

    reward_term_cb = WandbRewardTermCallback(env)
    noise_pred_cb = WandbNoisePredCallback(rfs_env)
    # Wire up model so callbacks can read num_timesteps
    reward_term_cb.model = adapter
    noise_pred_cb.model = adapter

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    if args_cli.wandb_project:
        ckpt_note_lines = [
            f"diffusion_ckpt={ckpt_path}",
            f"diffusion_ckpt_resolved={ckpt_resolved_path or ckpt_path}",
            f"diffusion_ckpt_meta={ckpt_meta_str}",
        ]
        wandb.init(
            project=args_cli.wandb_project,
            name=run_name,
            notes=f"{command}\n" + "\n".join(ckpt_note_lines),
            config={
                **vars(args_cli),
                **cfg,
                "diffusion_checkpoint_path": ckpt_path,
                "diffusion_checkpoint_resolved_path": ckpt_resolved_path or ckpt_path,
                "diffusion_checkpoint_meta": ckpt_meta,
            },
        )

    # ------------------------------------------------------------------
    # Update functions (adapted from train_fast_td3.py)
    # ------------------------------------------------------------------

    def normalize_obs(x: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_normalizer, nn.Identity):
            return x
        return obs_normalizer(x, update=True)

    def normalize_critic_obs(x: torch.Tensor) -> torch.Tensor:
        if isinstance(critic_obs_normalizer, nn.Identity):
            return x
        return critic_obs_normalizer(x, update=True)

    def normalize_reward(r: torch.Tensor) -> torch.Tensor:
        if isinstance(reward_normalizer, nn.Identity):
            return r
        return reward_normalizer(r)

    def update_main(data: TensorDict, logs_dict: TensorDict) -> TensorDict:
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if args_cli.asymmetric_ac:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()

            if td3_cfg.get("disable_bootstrap", False):
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions).mul(policy_noise).clamp(-noise_clip, noise_clip)
            next_state_actions = (actor(next_observations) + clipped_noise).clamp(-1.0, 1.0)
            discount = gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                qf1_next_target_proj, qf2_next_target_proj = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards,
                    bootstrap,
                    discount,
                )
                qf1_next_val = qnet_target.get_value(qf1_next_target_proj)
                qf2_next_val = qnet_target.get_value(qf2_next_target_proj)
                if use_cdq:
                    qf_next_dist = torch.where(
                        qf1_next_val.unsqueeze(1) < qf2_next_val.unsqueeze(1),
                        qf1_next_target_proj,
                        qf2_next_target_proj,
                    )
                    qf1_next_target_proj = qf2_next_target_proj = qf_next_dist
                # else: use separate targets per head (less aggressive)

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_proj * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_proj * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if td3_cfg["use_grad_norm_clipping"]:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=td3_cfg["max_grad_norm"] if td3_cfg["max_grad_norm"] > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)

        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qnet_target.get_value(qf1_next_target_proj).max().detach()
        logs_dict["qf_min"] = qnet_target.get_value(qf1_next_target_proj).min().detach()
        return logs_dict

    def update_pol(data: TensorDict, logs_dict: TensorDict) -> TensorDict:
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            critic_observations = (
                data["critic_observations"] if args_cli.asymmetric_ac
                else data["observations"]
            )
            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_val = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_val = qnet.get_value(F.softmax(qf2, dim=1))
            qf_val = torch.minimum(qf1_val, qf2_val) if use_cdq else (qf1_val + qf2_val) / 2.0
            actor_loss = -qf_val.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)

        if td3_cfg["use_grad_norm_clipping"]:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=td3_cfg["max_grad_norm"] if td3_cfg["max_grad_norm"] > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)

        scaler.step(actor_optimizer)
        scaler.update()

        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    # ------------------------------------------------------------------
    # Initial eval + obs cache
    # ------------------------------------------------------------------
    obs_dict, _ = rfs_env.reset()
    flat_obs = _flatten_obs(obs_dict["policy"], a_keys, device)
    flat_critic_obs = _flatten_obs(obs_dict["policy"], c_keys, device) if args_cli.asymmetric_ac else flat_obs
    dones = torch.zeros(num_envs, device=device, dtype=torch.bool)

    eval_cb._on_training_start()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    import tqdm as tqdm_module
    total_timesteps = 200_000_000
    pbar = tqdm_module.tqdm(total=total_timesteps, initial=global_step * num_envs, unit="ts")
    logs_dict = TensorDict()
    update_counter = 0  # counts critic updates for policy_frequency

    with contextlib.suppress(KeyboardInterrupt):
        while global_step * num_envs < total_timesteps:
            mark_step()

            # --- Collection ---
            with torch.no_grad():
                norm_obs = normalize_obs(flat_obs)
                actions = actor.explore(norm_obs, dones=dones.float())

            next_obs_dict, rewards, terminated, truncated, info = rfs_env.step(actions)

            # RFSWrapper collapses terminated|truncated into terminated; truncated=0
            # Use any_reset as dones for replay buffer.
            dones = terminated  # (n_env,) bool tensor

            next_flat_obs = _flatten_obs(next_obs_dict["policy"], a_keys, device)
            next_flat_critic = (
                _flatten_obs(next_obs_dict["policy"], c_keys, device)
                if args_cli.asymmetric_ac else next_flat_obs
            )

            if td3_cfg["reward_normalization"] and not isinstance(reward_normalizer, nn.Identity):
                reward_normalizer.update_stats(rewards, dones.float())

            transition = TensorDict(
                {
                    "observations": flat_obs,
                    "actions": actions,
                    "next": {
                        "observations": next_flat_obs,
                        "rewards": rewards,
                        "dones": dones.float(),
                        "truncations": torch.zeros_like(rewards),  # wrapper collapses into dones
                    },
                },
                batch_size=(num_envs,),
                device=device,
            )
            if args_cli.asymmetric_ac:
                transition["critic_observations"] = flat_critic_obs
                transition["next"]["critic_observations"] = next_flat_critic

            rb.extend(transition)

            flat_obs = next_flat_obs
            flat_critic_obs = next_flat_critic

            # --- Updates ---
            if global_step >= learning_starts:
                per_env_batch = max(1, batch_size // num_envs)
                for i in range(num_updates):
                    data = rb.sample(per_env_batch)
                    # Normalize obs in sampled batch
                    data["observations"] = normalize_obs(data["observations"])
                    data["next"]["observations"] = normalize_obs(data["next"]["observations"])
                    if args_cli.asymmetric_ac:
                        data["critic_observations"] = normalize_critic_obs(data["critic_observations"])
                        data["next"]["critic_observations"] = normalize_critic_obs(
                            data["next"]["critic_observations"]
                        )
                    raw_rewards = data["next"]["rewards"]
                    data["next"]["rewards"] = normalize_reward(raw_rewards)

                    logs_dict = update_main(data, logs_dict)
                    update_counter += 1
                    if update_counter % policy_frequency == 0:
                        logs_dict = update_pol(data, logs_dict)
                    soft_update(qnet, qnet_target, tau)

            # --- Callbacks: every step ---
            adapter.num_timesteps = global_step * num_envs
            eval_cb.num_timesteps = global_step * num_envs
            eval_cb.locals = {"dones": dones.cpu().numpy()}
            eval_cb._on_step()

            # --- Callbacks: every rollout_log_interval steps ---
            if (global_step + 1) % rollout_log_interval == 0 and global_step >= learning_starts:
                noise_pred_cb.num_timesteps = global_step * num_envs
                reward_term_cb.num_timesteps = global_step * num_envs
                noise_pred_cb._on_rollout_end()
                reward_term_cb._on_rollout_end()
                eval_cb._on_rollout_end()  # internally checks rollout_count % eval_interval

                if wandb.run is not None:
                    log_dict = {"env_rewards": rewards.mean().item()}
                    if "actor_loss" in logs_dict.keys():
                        log_dict.update({
                            "actor_loss": logs_dict["actor_loss"].mean().item(),
                            "qf_loss": logs_dict["qf_loss"].mean().item(),
                            "qf_max": logs_dict["qf_max"].mean().item(),
                            "qf_min": logs_dict["qf_min"].mean().item(),
                            "actor_grad_norm": logs_dict["actor_grad_norm"].mean().item(),
                            "critic_grad_norm": logs_dict["critic_grad_norm"].mean().item(),
                            "actor_lr": actor_scheduler.get_last_lr()[0],
                            "critic_lr": q_scheduler.get_last_lr()[0],
                            "buffer_size": rb.size,
                        })
                    wandb.log(log_dict, step=global_step * num_envs)

            global_step += 1
            actor_scheduler.step()
            q_scheduler.step()
            pbar.update(num_envs)

    pbar.close()

    # Final checkpoint
    final_path = os.path.join(log_dir, "model_final.pt")
    save_params(
        global_step=global_step,
        actor=actor,
        qnet=qnet,
        qnet_target=qnet_target,
        obs_normalizer=obs_normalizer,
        critic_obs_normalizer=critic_obs_normalizer,
        args=None,
        path=final_path,
    )
    print(f"[INFO] Final model saved to {final_path}")

    if wandb.run is not None:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
