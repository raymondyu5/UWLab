"""
Evaluate a trained RFS/DSRL noise-space PPO policy.

Flow (per step):
  1. RFSWrapper runs PointNet on current obs -> pcd_feat embedding
  2. Actor (PPO) sees [pcd_emb | ee_pose | hand_joint_pos] -> outputs noise
  3. Noise is injected as CFM starting trajectory -> denoised base action
  4. Base action sent to env

Usage (inside container):
    ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/play_rfs.py \\
        --task UW-FrankaLeap-PourBottle-IkRel-v0 \\
        --diffusion_path logs/bc_cfm_pcd_bourbon_0312 \\
        --checkpoint logs/rfs/PourBottle_0314_1609/model_000300.zip \\
        --cfg configs/rl/dsrl_cfg.yaml \\
        --asymmetric_ac \\
        --num_envs 16 \\
        --num_episodes 5 \\
        --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval RFS/DSRL noise-space PPO.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--diffusion_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to SB3 PPO .zip checkpoint.")
parser.add_argument("--cfg", type=str, default="configs/rl/dsrl_cfg.yaml")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--eval_spawn", type=str, default=None,
                    help="Spawn config name in configs/eval/spawns/ (e.g. bottle_pour_narrow). "
                         "If omitted, uses random resets.")
parser.add_argument("--asymmetric_ac", action="store_true", default=False)
parser.add_argument("--record_video", action="store_true", default=False)
parser.add_argument("--output_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.record_video:
    args_cli.enable_cameras = True

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
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import uwlab_tasks  # noqa: F401

from wrapper import RFSWrapper
from asymmetric_policy import AsymmetricActorCriticPolicy, resolve_asymmetric_ac
from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import load_spawn_cfg, SpawnCfg


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_dims(s: str):
    s = s.strip()
    if not s:
        return None
    start, _, end = s.partition(":")
    return (int(start), int(end))


def _to_numpy(obs_dict: dict) -> dict:
    policy_obs = obs_dict.get("policy", obs_dict)
    return {
        k: v.detach().cpu().float().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in policy_obs.items()
    }


def main():
    cfg = _load_cfg(args_cli.cfg)
    rfs_cfg = cfg["rfs"]

    noise_dims = _parse_dims(rfs_cfg["noise_dims"])
    residual_dims = _parse_dims(rfs_cfg.get("residual_dims", "") or "")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device else "cuda:0",
        num_envs=args_cli.num_envs,
    )
    env_cfg.run_mode = "rl_mode"
    if hasattr(env_cfg, "table_z_range"):
        env_cfg.table_z_range = (0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.record_video else None)

    asymmetric_ac = resolve_asymmetric_ac(args_cli.asymmetric_ac, rfs_cfg, args_cli.checkpoint)

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
        asymmetric_ac=asymmetric_ac,
    )
    # Enable expensive metric caching only for evaluation success counting.
    rfs_env.enable_metrics_cache = True

    sb3_env = Sb3VecEnvWrapper(rfs_env)

    checkpoint_path = os.path.expanduser(args_cli.checkpoint)
    custom_objects = {"policy_class": AsymmetricActorCriticPolicy} if asymmetric_ac else None
    agent = PPO.load(checkpoint_path, env=sb3_env, custom_objects=custom_objects,
                     print_system_info=False)

    output_dir = args_cli.output_dir or os.path.join(
        os.path.dirname(checkpoint_path),
        "eval",
        os.path.splitext(os.path.basename(checkpoint_path))[0],
    )
    os.makedirs(output_dir, exist_ok=True)
    video_fps = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    logger = EvalLogger(
        output_dir,
        record_video=args_cli.record_video,
        record_plots=True,
        video_fps=video_fps,
    )

    isaac_env = env.unwrapped
    horizon = getattr(env_cfg, "horizon", 200)

    if args_cli.eval_spawn:
        spawn_cfg = load_spawn_cfg(args_cli.eval_spawn, "configs/eval/spawns")
    else:
        spawn_cfg = SpawnCfg(poses=[], num_trials=args_cli.num_episodes)

    # Build episode list: fixed poses or random trials
    if spawn_cfg.poses:
        episodes = [
            (pose, trial_idx)
            for pose in spawn_cfg.poses
            for trial_idx in range(spawn_cfg.num_trials)
        ]
    else:
        episodes = [(None, i) for i in range(spawn_cfg.num_trials)]

    with torch.inference_mode():
        for ep_idx, (pose, trial_idx) in enumerate(episodes):
            if pose is not None:
                obs_dict, _ = rfs_env.reset_to_spawn(pose)
                ep_name = f"{pose.name}_trial{trial_idx}"
                spawn_info = {"x": pose.x, "y": pose.y, "yaw": pose.yaw}
            else:
                obs_dict, _ = rfs_env.reset()
                ep_name = f"random_{trial_idx}"
                spawn_info = None

            obs_np = _to_numpy(obs_dict)
            logger.begin_episode(ep_name, spawn_info)
            per_env_success = [False] * args_cli.num_envs
            recorded = [False] * args_cli.num_envs

            for _ in range(horizon):
                action, _ = agent.predict(obs_np, deterministic=True)
                action_t = torch.tensor(action, dtype=torch.float32, device=rfs_env.device)
                obs_dict, _, terminated, truncated, _ = rfs_env.step(action_t)
                obs_np = _to_numpy(obs_dict)
                metrics_seen = getattr(rfs_env, "_metrics_seen", None)
                if metrics_seen is None:
                    raise RuntimeError(
                        "RFSWrapper did not populate `_metrics_seen`. "
                        "Ensure metric caching is enabled in RFSWrapper.step()."
                    )

                for i in range(args_cli.num_envs):
                    if (terminated[i] or truncated[i]) and not recorded[i]:
                        per_env_success[i] = bool(metrics_seen["is_success"][i])
                        recorded[i] = True

                frame = rfs_env.render() if args_cli.record_video else None
                obj_pos = isaac_env.scene["grasp_object"].data.root_pos_w[0].cpu().numpy()
                logger.record_step(
                    ee_pose=np.zeros(7),
                    object_pose=obj_pos,
                    action=action[0],
                    frame=frame,
                )

                if all(recorded):
                    break

            n_success = sum(per_env_success)
            logger.end_episode(
                n_success / args_cli.num_envs,
                n_success=n_success,
                n_total=args_cli.num_envs,
            )
            print(f"[play_rfs] [{ep_idx + 1}/{len(episodes)}] {ep_name}: "
                  f"{n_success}/{args_cli.num_envs} envs succeeded")

    results = logger.finalize()
    print(f"\n[play_rfs] Success rate: {results['success_rate']:.1%} "
          f"({results['n_success']}/{results.get('n_total', results['n_episodes'])})")
    print(f"[play_rfs] Results -> {output_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
