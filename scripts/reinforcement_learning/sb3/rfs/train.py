"""
Train an RFS (Residual Flow-matching RL) policy with PPO.

All algorithm parameters are in configs/rl/rfs_cfg.yaml.
CLI handles env setup and infrastructure only.

Usage (inside container):
    # Grasp task (default eval spawn: cardinal_3x3 from rfs_cfg.yaml)
    ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py \\
        --task UW-FrankaLeap-GraspPinkCup-IkRel-v0 \\
        --num_envs 1024 \\
        --diffusion_path logs/real/mini_18/cfm/pcd_cfm/horizon_4_nobs_1 \\
        --headless

    # Pour task (use narrow ±5cm eval grid)
    ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py \\
        --task UW-FrankaLeap-PourBottle-IkRel-v0 \\
        --num_envs 1024 \\
        --diffusion_path logs/real/.../cfm/pcd_cfm/horizon_4_nobs_1 \\
        --eval_spawn bottle_pour_narrow \\
        --headless
"""

import argparse
import contextlib
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train RFS policy with PPO.")

# Env / infra
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--checkpoint", type=str, default=None, help="Resume PPO from .zip file.")

# RFS — override individual config values from the YAML if needed
parser.add_argument("--diffusion_path", type=str, required=True)
parser.add_argument("--cfg", type=str, default="configs/rl/rfs_cfg.yaml",
                    help="Path to RFS config YAML.")
parser.add_argument("--noise_dims", type=str, default=None,
                    help="Override rfs.noise_dims. Format: 'start:end'.")
parser.add_argument("--residual_dims", type=str, default=None,
                    help="Override rfs.residual_dims. Format: 'start:end'.")
parser.add_argument("--eval_interval", type=int, default=None,
                    help="Override eval.interval.")
parser.add_argument("--eval_spawn", type=str, default=None,
                    help="Override eval.spawn config name.")
parser.add_argument("--no_eval_video", action="store_true", default=False)
parser.add_argument("--no_eval_plots", action="store_true", default=False)
parser.add_argument("--asymmetric_ac", action="store_true", default=False,
                    help="Asymmetric AC: actor sees CFM embedding only; critic sees privileged sim state.")

# Wandb
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

#TODO cleanup this so we don't have to do it anymore
# Ensure rfs/ dir is on sys.path so sibling modules are importable.
# isaac-sim's python.sh resets PYTHONPATH, so this is necessary.
_RFS_DIR = os.path.dirname(os.path.abspath(__file__))
_SB3_DIR = os.path.dirname(_RFS_DIR)
_UWLAB_DIR = os.path.abspath(os.path.join(_RFS_DIR, "../../../../"))
_PIP_PACKAGES = os.path.join(_UWLAB_DIR, "third_party", "pip_packages")
for _p in [_RFS_DIR, _SB3_DIR, _PIP_PACKAGES]:
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

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from stable_baselines3 import PPO

from stable_baselines3.common.logger import configure as sb3_configure

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import uwlab_tasks  # noqa: F401

from wrapper import RFSWrapper
from eval_callback import RFSEvalCallback
from callbacks import WandbNoisePredCallback, WandbRewardTermCallback, WandbOutputFormat
from asymmetric_policy import AsymmetricActorCriticPolicy
from uwlab.eval.spawn import load_spawn_cfg


_ACT_FNS = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU}


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


def main():
    cfg = _load_cfg(args_cli.cfg)
    ppo_cfg = cfg["ppo"]
    rfs_cfg = cfg["rfs"]
    eval_cfg = cfg["eval"]

    # CLI overrides
    noise_dims = _parse_dims(args_cli.noise_dims or rfs_cfg["noise_dims"])
    residual_dims = _parse_dims(args_cli.residual_dims or rfs_cfg["residual_dims"])
    eval_interval = args_cli.eval_interval or eval_cfg["interval"]
    eval_spawn = args_cli.eval_spawn or eval_cfg["spawn"]

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = args_cli.wandb_run_name or f"{_short_task(args_cli.task)}_{timestamp}"
    log_dir = os.path.abspath(os.path.join("logs", "rfs", run_name))
    print(f"[INFO] Run: {run_name}")
    print(f"[INFO] Logging to: {log_dir}")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

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
    env_cfg.episode_length_s += (internal_warmup + external_warmup) * env_cfg.decimation * env_cfg.sim.dt

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    need_render = args_cli.video or (eval_cfg["record_video"] and not args_cli.no_eval_video)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if need_render else None)


    control_hz = 1 / (env_cfg.sim.dt * env_cfg.decimation) 
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, **{
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "fps": control_hz
        })

    rfs_env = RFSWrapper(
        env,
        diffusion_path=args_cli.diffusion_path,
        residual_step=rfs_cfg["residual_step"],
        noise_dims=noise_dims,
        residual_dims=residual_dims,
        residual_scale=rfs_cfg["residual_scale"],
        clip_actions=rfs_cfg["clip_actions"],
        finger_smooth_alpha=rfs_cfg["finger_smooth_alpha"],
        num_warmup_steps=rfs_cfg.get("num_warmup_steps", 0),
        asymmetric_ac=args_cli.asymmetric_ac,
    )

    sb3_env = Sb3VecEnvWrapper(rfs_env)

    print(f"[INFO] PPO observation space: {sb3_env.observation_space}")
    print(f"[INFO] PPO action space: {sb3_env.action_space}")

    if args_cli.wandb_project:
        wandb.init(
            project=args_cli.wandb_project,
            name=run_name,
            notes=command,
            config={**vars(args_cli), **cfg},
        )

    act_fn = _ACT_FNS[ppo_cfg.get("activation_fn", "elu")]
    agent_kwargs = dict(
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        learning_rate=ppo_cfg["learning_rate"],
        vf_coef=ppo_cfg["vf_coef"],
        ent_coef=ppo_cfg["ent_coef"],
        target_kl=ppo_cfg["target_kl"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs={
            "net_arch": ppo_cfg["net_arch"],
            "activation_fn": act_fn,
        },
        seed=args_cli.seed,
        verbose=1,
    )
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_kwargs)

    policy_cls = AsymmetricActorCriticPolicy if args_cli.asymmetric_ac else "MultiInputPolicy"
    agent = PPO(policy_cls, sb3_env, **agent_kwargs)
    if args_cli.checkpoint is not None:
        agent = PPO.load(args_cli.checkpoint, env=sb3_env, print_system_info=True)

    sb3_logger = sb3_configure(folder=None, format_strings=[])
    sb3_logger.output_formats.append(WandbOutputFormat())
    agent.set_logger(sb3_logger)

    spawn_cfg = load_spawn_cfg(eval_spawn, "configs/eval/spawns")
    horizon = getattr(env_cfg, "horizon", 200)
    eval_cb = RFSEvalCallback(
        rfs_env=rfs_env,
        spawn_cfg=spawn_cfg,
        episode_steps=horizon,
        log_dir=log_dir,
        eval_interval=eval_interval,
        record_video=not args_cli.no_eval_video and eval_cfg["record_video"],
        record_plots=not args_cli.no_eval_plots and eval_cfg["record_plots"],
        verbose=1,
    )
    reward_term_cb = WandbRewardTermCallback(env)
    noise_pred_cb = WandbNoisePredCallback(rfs_env)

    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=200_000_000,
            callback=[eval_cb, reward_term_cb, noise_pred_cb],
            progress_bar=True,
            log_interval=1,
        )

    agent.save(os.path.join(log_dir, "model"))
    print(f"[INFO] Model saved to {os.path.join(log_dir, 'model.zip')}")

    if wandb.run is not None:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
