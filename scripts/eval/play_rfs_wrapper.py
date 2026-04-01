"""
Run RFSEvalCallback against an Isaac env wrapped in RFSWrapper.

Loads the frozen CFM from ``diffusion_path`` and either:
  - a trained PPO noise policy (``--rfs_checkpoint`` .zip), or
  - a Gaussian policy over the full PPO action vector (no ``--rfs_checkpoint``).

Usage:
    ./uwlab.sh -p scripts/eval/play_rfs_wrapper.py \\
        --eval_cfg configs/eval/rfs_eval_example.yaml \\
        --diffusion_path logs/bc_cfm_pcd/... \\
        --rfs_cfg configs/rl/rfs_cfg.yaml \\
        --rfs_checkpoint logs/rfs/MyRun/model_000050.zip \\
        --headless

Omit ``--rfs_checkpoint`` to evaluate with Gaussian noise (full dim: residual + noise).
"""

import argparse
import os
import sys

from uwlab.utils.paths import setup_third_party_paths

setup_third_party_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RFS eval via RFSEvalCallback (PPO or Gaussian noise).")
parser.add_argument("--eval_cfg", type=str, required=True, help="YAML with task_id, spawn, device, etc.")
parser.add_argument(
    "--diffusion_path",
    type=str,
    default=None,
    help="BC / CFM checkpoint directory (checkpoints/*.ckpt). Overrides eval_cfg.diffusion_path.",
)
parser.add_argument(
    "--rfs_cfg",
    type=str,
    default="configs/rl/arm_rfs_joint_cfg.yaml",
    help="RFS hyperparams (noise_dims, residual_dims, residual_step, ...).",
)
parser.add_argument(
    "--rfs_checkpoint",
    type=str,
    default=None,
    help="SB3 PPO .zip; if omitted, use Gaussian noise policy.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Override eval_cfg num_envs.")
parser.add_argument(
    "--sim_type",
    type=str,
    choices=["eval", "distill", "rl"],
    default="rl",
    help="eval/distill/rl → env parse mode.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default=None, help="Eval logs / videos root.")
parser.add_argument(
    "--asymmetric_ac",
    action="store_true",
    default=False,
    help="Force asymmetric actor/critic obs layout. If omitted and --rfs_checkpoint is set, "
    "inferred from the zip (actor_* / critic_* keys).",
)
parser.add_argument("--noise_dims", type=str, default=None, help="Override rfs_cfg, format start:end.")
parser.add_argument("--residual_dims", type=str, default=None, help="Override rfs_cfg, format start:end.")
parser.add_argument("overrides", nargs="*", help="Key=value overrides for eval_cfg YAML.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

import yaml as _yaml

_record_video = False
if args_cli.eval_cfg and os.path.isfile(args_cli.eval_cfg):
    with open(args_cli.eval_cfg) as _f:
        _cfg = _yaml.safe_load(_f)
    _record_video = bool(_cfg.get("record_video", False))
    for kv in args_cli.overrides:
        key, _, val = kv.partition("=")
        if key == "record_video":
            _record_video = str(val).lower() == "true"
            break
if _record_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from uwlab.eval.spawn import load_spawn_cfg, SpawnCfg

from scripts.reinforcement_learning.sb3.rfs.eval_callback import RFSEvalCallback

from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    DISTILL_MODE,
    EVAL_MODE,
    RL_MODE,
    parse_franka_leap_env_cfg,
)
import uwlab_tasks  # noqa: F401  registers gym envs

_UWLAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_SPAWNS_DIR = os.path.join(_UWLAB_DIR, "configs/eval/spawns")
_RFS_DIR = os.path.join(_UWLAB_DIR, "scripts/reinforcement_learning/sb3/rfs")
if _RFS_DIR not in sys.path:
    sys.path.insert(0, _RFS_DIR)

from asymmetric_policy import AsymmetricActorCriticPolicy, resolve_asymmetric_ac
from wrapper import RFSWrapper


def _parse_dims(s: str | None):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected 'start:end', got: {s!r}")
    return (int(parts[0]), int(parts[1]))


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_eval_cfg(path: str, overrides: list) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        key, _, val = kv.partition("=")
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.lstrip("-").replace(".", "").isdigit():
            val = float(val) if "." in val else int(val)
        cfg[key] = val
    return cfg


class GaussianNoisePolicy:
    """SB3-like policy: full PPO action dim = residual_flat + noise * horizon."""

    def __init__(self, rfs_env: RFSWrapper):
        self.rfs_env = rfs_env
        self.num_timesteps = 0

    def predict(self, obs, deterministic: bool = False):
        b = self.rfs_env.num_envs
        d = self.rfs_env.n_residual_flat + self.rfs_env.n_noise * self.rfs_env.policy_horizon
        if deterministic:
            return np.zeros((b, d), dtype=np.float32), {}
        return np.random.randn(b, d).astype(np.float32), {}


def main():
    eval_cfg = _load_eval_cfg(args_cli.eval_cfg, args_cli.overrides)

    diffusion_path = args_cli.diffusion_path or eval_cfg.get("diffusion_path") or eval_cfg.get("checkpoint")
    if diffusion_path is None:
        raise ValueError("Set --diffusion_path or eval_cfg diffusion_path / checkpoint (BC checkpoint dir).")
    diffusion_path = os.path.expanduser(diffusion_path)

    rfs_yaml_path = args_cli.rfs_cfg
    if not os.path.isabs(rfs_yaml_path):
        rfs_yaml_path = os.path.join(_UWLAB_DIR, rfs_yaml_path)
    cfg = _load_yaml(rfs_yaml_path)
    ppo_cfg = cfg["ppo"]
    rfs_cfg = cfg["rfs"]

    noise_dims = _parse_dims(args_cli.noise_dims or rfs_cfg["noise_dims"])
    residual_dims = _parse_dims(args_cli.residual_dims or rfs_cfg.get("residual_dims") or "")

    device = torch.device(eval_cfg.get("device", "cuda:0"))
    num_envs = args_cli.num_envs or eval_cfg.get("num_envs", 1)
    record_video = bool(eval_cfg.get("record_video", False))

    task_id = eval_cfg["task_id"]
    run_mode = EVAL_MODE if args_cli.sim_type == "eval" else RL_MODE if args_cli.sim_type == "rl" else DISTILL_MODE
    env_cfg = parse_franka_leap_env_cfg(
        task_id,
        run_mode=run_mode,
        device=str(device),
        num_envs=num_envs,
    )
    env_cfg.seed = args_cli.seed
    if args_cli.sim_type == "eval":
        env_cfg.scene.train_camera = None
        if hasattr(env_cfg.events, "reset_camera"):
            env_cfg.events.reset_camera = None
    if not record_video:
        env_cfg.scene.fixed_camera = None
        if hasattr(env_cfg.events, "reset_fixed_camera"):
            env_cfg.events.reset_fixed_camera = None

    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if record_video else None)

    asymmetric_ac = resolve_asymmetric_ac(args_cli.asymmetric_ac, rfs_cfg, args_cli.rfs_checkpoint)

    rfs_env = RFSWrapper(
        env,
        diffusion_path=diffusion_path,
        residual_step=rfs_cfg["residual_step"],
        noise_dims=noise_dims,
        residual_dims=residual_dims,
        residual_scale=rfs_cfg["residual_scale"],
        clip_actions=rfs_cfg["clip_actions"],
        finger_smooth_alpha=rfs_cfg["finger_smooth_alpha"],
        finger_start_dim=rfs_cfg.get("finger_start_dim", 6),
        num_warmup_steps=rfs_cfg.get("num_warmup_steps", 0),
        asymmetric_ac=asymmetric_ac,
        gamma=float(ppo_cfg["gamma"]),
        ppo_history=rfs_cfg.get("ppo_history", False),
    )

    sb3_env = Sb3VecEnvWrapper(rfs_env)

    if args_cli.rfs_checkpoint:
        ckpt = os.path.expanduser(args_cli.rfs_checkpoint)
        custom_objects = {"policy_class": AsymmetricActorCriticPolicy} if asymmetric_ac else None
        model = PPO.load(ckpt, env=sb3_env, custom_objects=custom_objects, print_system_info=False)
    else:
        model = GaussianNoisePolicy(rfs_env)

    eval_spawn = eval_cfg.get("spawn", None)
    spawn_cfg = load_spawn_cfg(eval_spawn, _SPAWNS_DIR) if eval_spawn else SpawnCfg(poses=[], num_trials=1)

    if args_cli.output_dir:
        log_dir = args_cli.output_dir
    else:
        log_dir = eval_cfg.get("log_dir", os.path.join("logs", "rfs_eval", task_id.replace(":", "_")))
    os.makedirs(log_dir, exist_ok=True)

    eval_cb = RFSEvalCallback(
        rfs_env=rfs_env,
        spawn_cfg=spawn_cfg,
        log_dir=log_dir,
        eval_interval=int(eval_cfg.get("eval_interval", 1000)),
        record_video=bool(eval_cfg.get("record_video", True)),
        record_scatter=bool(eval_cfg.get("record_scatter", eval_cfg.get("record_plots", True))),
        record_debug_plots=bool(eval_cfg.get("record_debug_plots", False)),
        verbose=1,
    )
    eval_cb.model = model
    eval_cb.num_timesteps = int(eval_cfg.get("num_timesteps", 0))

    eval_cb._run_eval()

    rfs_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
