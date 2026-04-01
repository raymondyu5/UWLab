"""
Evaluate a trained RFS PPO checkpoint (synthetic/mesh PCD, rl_mode).

Loads the checkpoint, runs one eval round via RFSEvalCallback, and exits.
No training occurs. Supports multiple envs for fast evaluation.

Usage (inside container):
    ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/eval_rl.py \\
        --task UW-FrankaLeap-PourBottle-JointAbs-v0 \\
        --num_envs 16 \\
        --diffusion_path /path/to/diffusion_ckpt \\
        --checkpoint /path/to/ppo/model_000600.zip \\
        --cfg configs/rl/arm_rfs_joint_cfg.yaml \\
        --asymmetric_ac \\
        --eval_spawn random_1_trial \\
        --headless
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained RFS PPO checkpoint.")

parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO .zip checkpoint.")
parser.add_argument("--diffusion_path", type=str, required=True)
parser.add_argument("--cfg", type=str, default="configs/rl/rfs_cfg.yaml")
parser.add_argument("--residual_dims", type=str, default=None)
parser.add_argument("--eval_spawn", type=str, default=None)
parser.add_argument("--no_eval_video", action="store_true", default=False)
parser.add_argument("--eval_debug_plots", action="store_true", default=False)
parser.add_argument("--asymmetric_ac", action="store_true", default=False)
parser.add_argument("--output_dir", type=str, default=None,
                    help="Where to write eval results. Defaults to <checkpoint_dir>/eval/")
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

_RFS_DIR = os.path.dirname(os.path.abspath(__file__))
_SB3_DIR = os.path.dirname(_RFS_DIR)
_UWLAB_DIR = os.path.abspath(os.path.join(_RFS_DIR, "../../../../"))
_PIP_PACKAGES = os.path.join(_UWLAB_DIR, "third_party", "pip_packages")
for _p in [_RFS_DIR, _SB3_DIR, _PIP_PACKAGES]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import yaml
import wandb

from stable_baselines3.common.logger import configure as sb3_configure

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import uwlab_tasks  # noqa: F401

from wrapper import RFSWrapper
from gpu_vec_env import GpuSb3VecEnvWrapper
from eval_callback import RFSEvalCallback
from callbacks import WandbOutputFormat
from asymmetric_policy import AsymmetricActorCriticPolicy
from regularized_ppo import RegularizedPPO
from buffers import GpuDictRolloutBuffer
from uwlab.eval.spawn import load_spawn_cfg


def _parse_dims(s: str):
    s = s.strip()
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected 'start:end', got: {s!r}")
    return (int(parts[0]), int(parts[1]))


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = _load_cfg(args_cli.cfg)
    rfs_cfg = cfg["rfs"]
    eval_cfg = cfg["eval"]
    use_gpu_buffer = cfg.get("gpu_buffer", True)

    residual_dims = _parse_dims(args_cli.residual_dims or rfs_cfg["residual_dims"])
    eval_spawn = args_cli.eval_spawn or eval_cfg["spawn"]

    ckpt_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    log_dir = args_cli.output_dir or os.path.join(ckpt_dir, "eval")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Eval output: {log_dir}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device else "cuda:0",
        num_envs=args_cli.num_envs,
    )
    env_cfg.run_mode = "rl_mode"
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg, "table_z_range"):
        env_cfg.table_z_range = (0.0, 0.0)

    external_warmup = rfs_cfg.get("num_warmup_steps", 0)
    env_cfg.episode_length_s += external_warmup * env_cfg.decimation * env_cfg.sim.dt

    need_render = eval_cfg["record_video"] and not args_cli.no_eval_video
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if need_render else None)

    rfs_env = RFSWrapper(
        env,
        diffusion_path=args_cli.diffusion_path,
        residual_step=rfs_cfg["residual_step"],
        noise_dims=_parse_dims(rfs_cfg["noise_dims"]),
        residual_dims=residual_dims,
        residual_scale=rfs_cfg["residual_scale"],
        clip_actions=rfs_cfg["clip_actions"],
        finger_smooth_alpha=rfs_cfg["finger_smooth_alpha"],
        finger_start_dim=rfs_cfg.get("finger_start_dim", 6),
        num_warmup_steps=rfs_cfg.get("num_warmup_steps", 0),
        asymmetric_ac=args_cli.asymmetric_ac,
        gamma=cfg["ppo"]["gamma"],
        ppo_history=rfs_cfg.get("ppo_history", False),
    )

    sb3_env = GpuSb3VecEnvWrapper(rfs_env) if use_gpu_buffer else Sb3VecEnvWrapper(rfs_env)

    if args_cli.wandb_project:
        run_name = args_cli.wandb_run_name or f"eval_{os.path.basename(args_cli.checkpoint)}"
        wandb.init(project=args_cli.wandb_project, name=run_name, config=vars(args_cli))

    sb3_logger = sb3_configure(folder=None, format_strings=[])
    sb3_logger.output_formats.append(WandbOutputFormat())

    agent = RegularizedPPO.load(args_cli.checkpoint, env=sb3_env, print_system_info=True)
    agent.set_logger(sb3_logger)

    spawn_cfg = load_spawn_cfg(eval_spawn, "configs/eval/spawns")

    eval_cb = RFSEvalCallback(
        rfs_env=rfs_env,
        spawn_cfg=spawn_cfg,
        log_dir=log_dir,
        eval_interval=999_999_999,  # never fires during training; we call manually
        record_video=need_render,
        record_scatter=True,
        record_debug_plots=args_cli.eval_debug_plots,
        verbose=1,
    )
    eval_cb.init_callback(agent)
    eval_cb._run_eval()

    if wandb.run is not None:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
