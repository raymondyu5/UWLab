"""
Diagnostic: run the BC diffusion policy in rl_mode env with random Gaussian noise — no PPO.

Identical to play_bc.py except:
  - env uses run_mode="rl_mode" (disables cameras in scene, like RL training)
  - obs formatted via RFSWrapper._get_diffusion_obs (same path as RL training)
  - NO PPO noise: policy.predict_action(..., noise=None) → torch.randn (random Gaussian)
  - NO PPO residual: residual_dims=None, residual_scale=0.0
  - action clipping disabled (clip_actions=False) to match play_bc.py

If success rate matches play_bc.py  → PPO noise is the problem.
If success rate is also bad          → something in the RL env stack is broken.

Usage (inside container):
    ./uwlab.sh -p scripts/eval/play_bc.py \\
        --eval_cfg configs/eval/bottle_pour_bc.yaml \\
        checkpoint=logs/bc_cfm_pcd_bourbon_0312 \\
        record_video=true --num_envs 4 --seed 42 \\
        --output_dir logs/diagnostic/bc_vs_rl_mode/play_bc \\
        --enable_cameras --headless

    ./uwlab.sh -p scripts/eval/play_bc_rl_mode.py \\
        --diffusion_path logs/bc_cfm_pcd_bourbon_0312 \\
        --spawn bottle_pour_narrow \\
        --num_envs 4 --seed 42 \\
        --output_dir logs/diagnostic/bc_vs_rl_mode/play_bc_rl_mode \\
        --record_video --enable_cameras --headless
"""

import argparse
import os
import sys

_EXTRA_PATHS = [
    "/workspace/uwlab/third_party/diffusion_policy",
    "/workspace/uwlab/third_party/pip_packages",
]
for _p in _EXTRA_PATHS:
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--diffusion_path", type=str, required=True)
parser.add_argument("--task", type=str, default="UW-FrankaLeap-PourBottle-IkRel-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_warmup_steps", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--spawn", type=str, default="bottle_pour_narrow",
                    help="Spawn config name in configs/eval/spawns/")
parser.add_argument("--output_dir", type=str, default="logs/diagnostic/bc_vs_rl_mode/play_bc_rl_mode")
parser.add_argument("--record_video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.record_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import isaaclab.utils.math as math_utils

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import uwlab_tasks  # noqa: F401

_UWLAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_RFS_DIR = os.path.join(_UWLAB_DIR, "scripts/reinforcement_learning/sb3/rfs")
if _RFS_DIR not in sys.path:
    sys.path.insert(0, _RFS_DIR)

from wrapper import RFSWrapper
from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import load_spawn_cfg


def _set_object_pose(env, x_offset, y_offset, yaw_offset, default_pos, default_rot_quat, reset_height):
    """Identical to play_bc.py: set object pose after env.reset()."""
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    obj = env.unwrapped.scene["grasp_object"]

    pos = torch.tensor(
        [[default_pos[0] + x_offset, default_pos[1] + y_offset, reset_height]],
        device=device, dtype=torch.float32,
    ).repeat(num_envs, 1)

    base_quat = torch.tensor([list(default_rot_quat)], device=device, dtype=torch.float32).repeat(num_envs, 1)
    delta_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(num_envs, device=device),
        torch.zeros(num_envs, device=device),
        torch.full((num_envs,), yaw_offset, device=device),
    )
    quat = math_utils.quat_mul(delta_quat, base_quat)

    pos_world = pos + env.unwrapped.scene.env_origins
    env_ids = torch.arange(num_envs, device=device)
    obj.write_root_pose_to_sim(torch.cat([pos_world, quat], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device), env_ids=env_ids)


def main():
    device = torch.device("cuda:0")
    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(args_cli.task, device=str(device), num_envs=num_envs)
    env_cfg.run_mode = "rl_mode"
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg, "table_z_range"):
        env_cfg.table_z_range = (0.0, 0.0)

    internal_warmup = int(getattr(env_cfg, "num_warmup_steps", 0))
    external_warmup = args_cli.num_warmup_steps
    env_cfg.episode_length_s += (internal_warmup + external_warmup) * env_cfg.decimation * env_cfg.sim.dt

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.record_video else None)

    # NO PPO noise: noise=None → torch.randn (random Gaussian) inside predict_action
    # NO PPO residual: residual_dims=None, residual_scale=0.0
    # clip_actions=True to test whether clipping is what hurts RL performance
    rfs_env = RFSWrapper(
        env,
        diffusion_path=args_cli.diffusion_path,
        noise_dims=(0, 22),
        residual_dims=None,
        residual_scale=0.0,
        clip_actions=True,
        finger_smooth_alpha=1.0,
    )

    isaac_env = env.unwrapped
    episode_steps = int(isaac_env.cfg.horizon)
    warmup_act = isaac_env.cfg.warmup_action(isaac_env)

    spawn_cfg = load_spawn_cfg(args_cli.spawn, os.path.join(_UWLAB_DIR, "configs/eval/spawns"))

    _spawn_defaults = isaac_env.cfg.object_spawn_defaults
    default_pos = tuple(_spawn_defaults["default_pos"])
    default_rot = tuple(_spawn_defaults["default_rot"])
    reset_height = float(_spawn_defaults["reset_height"])

    if spawn_cfg.poses:
        episodes = [(pose.name, pose) for pose in spawn_cfg.poses for _ in range(spawn_cfg.num_trials)]
    else:
        episodes = [(f"random_{i}", None) for i in range(spawn_cfg.num_trials)]

    cfg = isaac_env.cfg
    video_fps = 1.0 / (cfg.sim.dt * cfg.decimation)
    logger = EvalLogger(
        args_cli.output_dir,
        record_video=args_cli.record_video,
        record_plots=True,
        video_fps=video_fps,
    )

    with torch.inference_mode():
        for ep_idx, (pose_name, pose) in enumerate(episodes):
            obs_raw, _ = env.reset()
            rfs_env.last_obs = obs_raw

            if pose is not None:
                _set_object_pose(env, pose.x, pose.y, pose.yaw,
                                 default_pos, default_rot, reset_height)
                spawn_info = {"x": pose.x, "y": pose.y, "yaw": pose.yaw}
            else:
                spawn_info = None

            # Match play_bc.py: extra warmup on top of env's built-in warmup
            for _ in range(args_cli.num_warmup_steps):
                obs_raw, _, _, _, _ = env.step(warmup_act)
            rfs_env.last_obs = obs_raw

            logger.begin_episode(pose_name, spawn_info)

            per_env_done = [False] * num_envs
            per_env_success = [False] * num_envs
            per_env_partial = [False] * num_envs

            for step in range(episode_steps):
                diffusion_obs = rfs_env._get_diffusion_obs(rfs_env.last_obs["policy"])
                # noise=None → torch.randn inside policy (random Gaussian, not PPO)
                base_action = rfs_env.policy.predict_action(diffusion_obs, noise=None)["action_pred"][:, 0]

                # Check success/partial on current state BEFORE stepping to avoid
                # reading the auto-reset state when the episode truncates.
                pre_step_success = isaac_env.cfg.is_success(isaac_env).cpu().numpy()
                pre_step_partial = isaac_env.cfg.is_partial_success(isaac_env).cpu().numpy()
                for i in range(num_envs):
                    if not per_env_done[i]:
                        if pre_step_success[i]:
                            per_env_success[i] = True
                        if pre_step_partial[i]:
                            per_env_partial[i] = True

                obs_raw, _, terminated, truncated, _ = env.step(base_action)
                rfs_env.last_obs = obs_raw

                frame = rfs_env.render() if args_cli.record_video else None
                obj = isaac_env.scene["grasp_object"]
                obj_pose = (obj.data.root_pos_w - isaac_env.scene.env_origins).cpu().numpy()[0]
                logger.record_step(np.zeros(7), obj_pose, base_action.cpu().numpy()[0], frame=frame)

                for i in range(num_envs):
                    if (terminated[i] or truncated[i]) and not per_env_done[i]:
                        per_env_done[i] = True

                if all(per_env_done):
                    break

            n_success = sum(per_env_success)
            n_partial = sum(per_env_partial)
            logger.end_episode(n_success / num_envs, n_success=n_success, n_total=num_envs,
                               partial_success=n_partial / num_envs)
            print(f"[play_bc_rl_mode] Episode {ep_idx+1}/{len(episodes)} "
                  f"(spawn={pose_name}): {n_success}/{num_envs} success, {n_partial}/{num_envs} partial")

    results = logger.finalize()
    print(f"\n[play_bc_rl_mode] Success rate: {results['success_rate']:.1%} "
          f"({results['n_success']}/{results['n_episodes']})")
    print(f"[play_bc_rl_mode] Results -> {args_cli.output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
