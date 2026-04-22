"""
Train an FPO (Flow Policy Optimization) policy.

Loads a BC CFMPCDPolicy checkpoint, freezes the PointNet obs encoder, and
fine-tunes the UNet denoising network end-to-end with FPO policy gradients.

Usage (inside container):
    ./uwlab.sh -p scripts/reinforcement_learning/fpo/train.py \\
        --task UW-FrankaLeap-GraspCube-JointAbs-v0 \\
        --num_envs 2048 \\
        --diffusion_path logs/bc_cfm_pcd_cube_grasp_0416_absjoint_h16_hist4_fast \\
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

parser = argparse.ArgumentParser(description="Train FPO policy.")

# Env / infra
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--seed", type=int, default=42)

# FPO
parser.add_argument("--diffusion_path", type=str, required=True)
parser.add_argument("--cfg", type=str, default="configs/rl/arm_fpo_cfg.yaml")
parser.add_argument("--eval_interval", type=int, default=None)
parser.add_argument("--eval_spawn", type=str, default=None)
parser.add_argument("--no_eval_video", action="store_true", default=False)

# Wandb
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)

# BC regularization
parser.add_argument("--bc_dataset_path", type=str, default=None,
                    help="Path to real robot zarr dataset for BC regularization.")
parser.add_argument("--bc_coef", type=float, default=0.0,
                    help="Weight on BC regularization loss (0 = disabled).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.no_eval_video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

_FPO_DIR = os.path.dirname(os.path.abspath(__file__))
_UWLAB_DIR = os.path.abspath(os.path.join(_FPO_DIR, "../../../"))
_PIP_PACKAGES = os.path.join(_UWLAB_DIR, "third_party", "pip_packages")
_RFS_DIR = os.path.join(_UWLAB_DIR, "scripts", "reinforcement_learning", "sb3", "rfs")
for _p in [_FPO_DIR, _PIP_PACKAGES, _RFS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    import gc
    for obj in gc.get_objects():
        if "tqdm_rich" in type(obj).__name__:
            obj.close()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, cleanup_pbar)

"""Everything after AppLauncher."""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml
import wandb

from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import uwlab_tasks  # noqa: F401

from fpo_wrapper import FPOWrapper
from fpo_trainer import FPOTrainer
from fpo_eval import run_fpo_eval
from real_dataset import RealDatasetLoader
from uwlab.utils.checkpoint import extract_ckpt_metadata, format_ckpt_metadata
from uwlab.eval.spawn import load_spawn_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_ACT_FNS = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU}


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _short_task(task: str) -> str:
    import re
    name = task
    if name.startswith("UW-FrankaLeap-"):
        name = name[len("UW-FrankaLeap-"):]
    name = re.sub(r"-(JointAbs-|IkRel-)?v\d+$", "", name)
    return name


def build_critic(obs_dim: int, net_arch: list, activation_fn: type) -> nn.Module:
    layers = []
    in_dim = obs_dim
    for out_dim in net_arch:
        layers += [nn.Linear(in_dim, out_dim), activation_fn()]
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def main():
    cfg     = _load_cfg(args_cli.cfg)
    fpo_cfg = cfg.get("fpo", {})
    ppo_cfg = cfg["ppo"]
    eval_cfg = cfg.get("eval", {})

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name  = args_cli.wandb_run_name or f"{_short_task(args_cli.task)}_{timestamp}_{uuid4().hex[:6]}"
    log_dir   = os.path.abspath(os.path.join("logs", "fpo", run_name))
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)
    print(f"[INFO] Run: {run_name}")
    print(f"[INFO] Logging to: {log_dir}")

    device = args_cli.device if args_cli.device else "cuda:0"

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.run_mode = "rl_mode"
    env_cfg.seed     = args_cli.seed
    if hasattr(env_cfg, "table_z_range"):
        env_cfg.table_z_range = (0.0, 0.0)

    num_warmup = fpo_cfg.get("num_warmup_steps", 0)
    env_cfg.episode_length_s += num_warmup * env_cfg.decimation * env_cfg.sim.dt

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    need_render = not args_cli.no_eval_video and eval_cfg.get("record_video", True)
    env     = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if need_render else None)
    fpo_env = FPOWrapper(
        env,
        diffusion_path=args_cli.diffusion_path,
        num_warmup_steps=num_warmup,
        gamma=ppo_cfg["gamma"],
        n_cfm_samples=fpo_cfg.get("n_cfm_samples", 1),
    )

    ckpt_meta_str = format_ckpt_metadata(fpo_env.diffusion_ckpt_meta)
    print(f"[INFO] Diffusion checkpoint: {fpo_env.diffusion_ckpt_path}")
    print(f"[INFO] Diffusion checkpoint metadata: {ckpt_meta_str}")

    # Build critic MLP for privileged obs.
    act_fn  = _ACT_FNS[ppo_cfg.get("activation_fn", "elu")]
    net_arch = ppo_cfg["net_arch"]
    if isinstance(net_arch, dict):
        vf_arch = net_arch.get("vf", net_arch.get("pi", [256, 128, 64]))
    else:
        vf_arch = net_arch

    critic = build_critic(fpo_env.critic_obs_dim, vf_arch, act_fn).to(device)
    print(f"[INFO] Critic: obs_dim={fpo_env.critic_obs_dim}, "
          f"arch={vf_arch}, params={sum(p.numel() for p in critic.parameters()):,}")

    dump_yaml(os.path.join(log_dir, "params", "cfg.yaml"), cfg)

    # BC regularization: precompute real data pool.
    bc_loader = None
    bc_coef = args_cli.bc_coef or float(fpo_cfg.get("bc_coef", 0.0))
    if args_cli.bc_dataset_path and bc_coef > 0.0:
        bc_loader = RealDatasetLoader(
            dataset_path=args_cli.bc_dataset_path,
            n_obs_steps=fpo_env._ppo_n_obs_steps,
            action_dim=fpo_env.cfm_action_dim,
            downsample_points=fpo_env.formatter.downsample_points,
            device=torch.device(device),
            horizon=fpo_env.policy_horizon,
        )
        bc_loader.precompute_pool(fpo_env)
        print(f"[INFO] BC regularization: coef={bc_coef}, dataset={args_cli.bc_dataset_path}")

    # Wandb setup.
    if args_cli.wandb_project:
        wandb.init(
            project=args_cli.wandb_project,
            name=run_name,
            notes=command,
            config={**vars(args_cli), **cfg,
                    "diffusion_ckpt_path": fpo_env.diffusion_ckpt_path,
                    "diffusion_ckpt_meta": fpo_env.diffusion_ckpt_meta},
        )

    def log_fn(metrics: dict):
        if wandb.run is not None:
            wandb.log(metrics, step=metrics.get("timesteps"))

    eval_interval = args_cli.eval_interval or eval_cfg.get("interval", 50)
    eval_spawn    = args_cli.eval_spawn    or eval_cfg.get("spawn", "random_1_trial")
    record_video  = not args_cli.no_eval_video and eval_cfg.get("record_video", True)

    spawn_cfg = load_spawn_cfg(eval_spawn, "configs/eval/spawns")
    print(f"[INFO] Eval: every {eval_interval} iters, spawn={eval_spawn}, "
          f"poses={len(spawn_cfg.poses) or 'random'}, trials={spawn_cfg.num_trials}, "
          f"record_video={record_video}")

    if bc_coef > 0.0:
        cfg.setdefault("fpo", {})["bc_coef"] = bc_coef

    trainer = FPOTrainer(
        env=fpo_env,
        critic=critic,
        cfg=cfg,
        device=torch.device(device),
        log_fn=log_fn,
        bc_loader=bc_loader,
    )

    def eval_fn(t_so_far, iteration):
        return run_fpo_eval(
            fpo_env=fpo_env,
            spawn_cfg=spawn_cfg,
            log_dir=log_dir,
            t_so_far=t_so_far,
            record_video=record_video,
        )

    with contextlib.suppress(KeyboardInterrupt):
        trainer.train(
            total_timesteps=600_000_000,
            eval_fn=eval_fn,
            eval_interval=eval_interval,
            log_dir=log_dir,
            save_interval=50,
        )

    trainer._save(log_dir, trainer._iteration, trainer._t_so_far)
    print(f"[INFO] Final checkpoint saved to {log_dir}/checkpoints/best.ckpt")

    if wandb.run is not None:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
