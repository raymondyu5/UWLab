"""
Evaluate a BC (CFM) policy inside the Isaac Sim environment.

Usage (inside container):
    ./isaaclab.sh -p scripts/play_bc.py \\
        --eval_cfg configs/eval/pink_cup_bc_joint_abs.yaml \\
        checkpoint=/path/to/outputs/2026-03-04/12-00-00 \\
        record_video=true

./uwlab.sh -p scripts/eval/play_bc.py --eval_cfg configs/eval/bottle_pour_bc.yaml --enable_cameras --record_video --checkpoint logs/bc_cfm_pcd_bourbon_0312 --headless

The checkpoint dir must contain:
    .hydra/config.yaml   (saved by Hydra at train time)
    checkpoints/best.ckpt (or latest.ckpt)

obs_keys can be overridden in eval config; if not set, uses the checkpoint's training config.
image_keys, downsample_points, and policy architecture are loaded from the training config.
"""

import argparse
import os
import sys

from uwlab.utils.paths import setup_third_party_paths
setup_third_party_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a BC policy in Isaac Sim.")
parser.add_argument("--eval_cfg", type=str, required=True,
                    help="Path to eval config YAML (configs/eval/*.yaml)")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Override num_envs from eval config")
parser.add_argument("--sim_type", type=str, choices=["eval", "distill", "rl"], default="rl",
                    help="eval: use eval mode. distill: use distill mode.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default=None,
                    help="Override output directory for results")
parser.add_argument("overrides", nargs="*",
                    help="Key=value overrides for eval config (e.g. checkpoint=/path record_video=true)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Enable cameras when record_video is requested (needed for env.render())
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
import matplotlib.pyplot as plt
import isaaclab.utils.math as math_utils
from stable_baselines3 import PPO

from uwlab.eval.spawn import load_spawn_cfg, SpawnCfg

from scripts.reinforcement_learning.sb3.rfs.eval_callback import RFSEvalCallback

from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    EVAL_MODE,
    RL_MODE,
    DISTILL_MODE,
    parse_franka_leap_env_cfg,
)
import uwlab_tasks  # noqa: F401  registers gym envs

_UWLAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_RFS_DIR = os.path.join(_UWLAB_DIR, "scripts/reinforcement_learning/sb3/rfs")
if _RFS_DIR not in sys.path:
    sys.path.insert(0, _RFS_DIR)

from wrapper import RFSWrapper


class Gaussian01Policy:
    """Minimal SB3-like model interface for standalone callback eval."""

    def __init__(self, env):
        self.env = env
        self.num_timesteps = self.env.unwrapped.cfg.horizon

    def predict(self, obs, deterministic: bool = False):
        batch_size = self.env.num_envs
        action_dim =  self.env.n_noise * self.env.policy_horizon
        return np.random.randn(batch_size, action_dim), {}


def _load_eval_cfg(path: str, overrides: list) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        key, _, val = kv.partition("=")
        # simple type coercion
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.lstrip("-").replace(".", "").isdigit():
            val = float(val) if "." in val else int(val)
        cfg[key] = val
    return cfg


def main():
    eval_cfg = _load_eval_cfg(args_cli.eval_cfg, args_cli.overrides)

    checkpoint_dir = eval_cfg.get("checkpoint")
    if checkpoint_dir is None:
        raise ValueError("eval_cfg must specify 'checkpoint'")
    checkpoint_dir = os.path.expanduser(checkpoint_dir)

    device = torch.device(eval_cfg.get("device", "cuda:0"))
    num_envs = args_cli.num_envs or eval_cfg.get("num_envs", 1)
    action_horizon = int(eval_cfg.get("action_horizon", 1))
    record_video = bool(eval_cfg.get("record_video", False))
    record_plots = bool(eval_cfg.get("record_plots", True))

    # Create env
    task_id = eval_cfg["task_id"]

    run_mode = EVAL_MODE if args_cli.sim_type == "eval" else RL_MODE if args_cli.sim_type == "rl" else DISTILL_MODE
    env_cfg = parse_franka_leap_env_cfg(
        task_id,
        run_mode=run_mode,
        device=str(device),
        num_envs=num_envs,
    )
    env_cfg.seed = args_cli.seed
    # In distill mode, train_camera is needed for RenderedSegPC — keep it alive.
    # In eval mode (synthetic PCD), cameras are not needed so disable them to save overhead.
    if args_cli.sim_type == "eval":
        env_cfg.scene.train_camera = None
        if hasattr(env_cfg.events, "reset_camera"):
            env_cfg.events.reset_camera = None
    if not record_video:
        env_cfg.scene.fixed_camera = None
        if hasattr(env_cfg.events, "reset_fixed_camera"):
            env_cfg.events.reset_fixed_camera = None
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if record_video else None)

    env = RFSWrapper(
        env,
        diffusion_path=checkpoint_dir,
        noise_dims=(0, 23),
        residual_step=1,
        residual_dims=None,
        residual_scale=0.0,
        clip_actions=False,
        finger_smooth_alpha=1.0,
    )

    eval_spawn = eval_cfg.get("spawn", None)
    spawn_cfg = load_spawn_cfg(eval_spawn, "configs/eval/spawns") 

    log_dir = 'logs/rfs_eval/PourBottle_0326'
    eval_cb = RFSEvalCallback(
        rfs_env=env,
        spawn_cfg=spawn_cfg,
        log_dir=log_dir,
        eval_interval=1000, #
        record_video=True,
        record_plots=True,
        verbose=1,
    )

    model = Gaussian01Policy(env=env)
    eval_cb.model = model
    eval_cb.num_timesteps = 0

    eval_cb._run_eval()
            
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()