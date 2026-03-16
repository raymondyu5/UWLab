"""
Evaluate an SB3 RL policy inside the Isaac Sim environment.

Usage (inside container):
    ./isaaclab.sh -p scripts/eval/play_rl.py \\
        --eval_cfg configs/eval/pink_cup_rl_joint_abs.yaml \\
        checkpoint=/path/to/logs/sb3/task/model.zip

The RL agent reads obs directly from the env (flat vector) — no obs formatting needed.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an SB3 RL policy in Isaac Sim.")
parser.add_argument("--eval_cfg", type=str, required=True,
                    help="Path to eval config YAML (configs/eval/*.yaml)")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("overrides", nargs="*",
                    help="Key=value overrides, e.g. checkpoint=/path record_video=true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import yaml
import isaaclab.utils.math as math_utils
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper

from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import load_spawn_cfg, SpawnCfg

import uwlab_tasks  # noqa: F401


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


def _set_object_pose(env, x_offset, y_offset, yaw_offset,
                     default_pos, default_rot_quat, reset_height):
    # unwrap through gym wrappers to get the Isaac env
    isaac_env = env.unwrapped
    device = isaac_env.device
    num_envs = isaac_env.num_envs
    obj = isaac_env.scene["object"]

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
    pos_world = pos + isaac_env.scene.env_origins
    env_ids = torch.arange(num_envs, device=device)
    obj.write_root_pose_to_sim(torch.cat([pos_world, quat], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device), env_ids=env_ids)


def _check_success(isaac_env) -> np.ndarray:
    return isaac_env.cfg.is_success(isaac_env).cpu().numpy()


def main():
    eval_cfg = _load_eval_cfg(args_cli.eval_cfg, args_cli.overrides)

    checkpoint_path = eval_cfg.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("eval_cfg must specify 'checkpoint'")
    checkpoint_path = os.path.expanduser(checkpoint_path)

    device = eval_cfg.get("device", "cuda:0")
    num_envs = args_cli.num_envs or eval_cfg.get("num_envs", 1)
    record_video = bool(eval_cfg.get("record_video", False))
    record_plots = bool(eval_cfg.get("record_plots", True))

    task_id = eval_cfg["task_id"]
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if record_video else None)
    isaac_env = env.unwrapped

    # Wrap for SB3
    env_sb3 = Sb3VecEnvWrapper(env)
    if os.path.isfile(checkpoint_path.replace(".zip", "_vecnormalize.pkl")):
        env_sb3 = VecNormalize.load(
            checkpoint_path.replace(".zip", "_vecnormalize.pkl"), env_sb3)
        env_sb3.training = False
        env_sb3.norm_reward = False

    agent = PPO.load(checkpoint_path, env_sb3, print_system_info=False)

    # Spawn config
    spawn_name = eval_cfg.get("spawn", None)
    spawn_dir = os.path.join(os.path.dirname(args_cli.eval_cfg), "spawns")
    if spawn_name is not None and os.path.isdir(spawn_dir):
        spawn_cfg = load_spawn_cfg(spawn_name, spawn_dir)
    else:
        spawn_cfg = SpawnCfg(poses=[], num_trials=int(eval_cfg.get("num_episodes", 10)))

    _spawn_defaults = isaac_env.cfg.object_spawn_defaults
    default_pos = tuple(eval_cfg.get("object_default_pos", _spawn_defaults["default_pos"]))
    default_rot = tuple(eval_cfg.get("object_default_rot", _spawn_defaults["default_rot"]))
    reset_height = float(eval_cfg.get("object_reset_height", _spawn_defaults["reset_height"]))

    output_dir = os.path.join(
        os.path.dirname(checkpoint_path), "eval",
        os.path.splitext(os.path.basename(args_cli.eval_cfg))[0],
    )
    os.makedirs(output_dir, exist_ok=True)
    cfg = isaac_env.cfg
    video_fps = 1.0 / (cfg.sim.dt * cfg.decimation)
    logger = EvalLogger(
        output_dir,
        record_video=record_video,
        record_plots=record_plots,
        video_fps=video_fps,
    )

    if spawn_cfg.poses:
        episodes = [(pose.name, pose) for pose in spawn_cfg.poses for _ in range(spawn_cfg.num_trials)]
    else:
        episodes = [(None, None)] * spawn_cfg.num_trials

    horizon = int(eval_cfg.get("episode_steps", 200))

    num_warmup_steps = int(eval_cfg.get("num_warmup_steps", 5))

    with torch.inference_mode():
        for ep_idx, (spawn_name_ep, spawn_pose) in enumerate(episodes):
            obs = env_sb3.reset()

            if spawn_pose is not None:
                _set_object_pose(isaac_env, spawn_pose.x, spawn_pose.y, spawn_pose.yaw,
                                 default_pos, default_rot, reset_height)

            # Warm-up: hold position using task-defined safe action
            warmup_act = isaac_env.cfg.warmup_action(isaac_env)
            assert warmup_act.shape == (isaac_env.num_envs, isaac_env.action_manager.total_action_dim), \
                f"warmup_action shape {warmup_act.shape} doesn't match action dim {isaac_env.action_manager.total_action_dim}"
            warmup_act_np = warmup_act.cpu().numpy()
            for _ in range(num_warmup_steps):
                obs, _, _, _ = env_sb3.step(warmup_act_np)

            spawn_pose_dict = (
                {"x": spawn_pose.x, "y": spawn_pose.y, "yaw": spawn_pose.yaw}
                if spawn_pose is not None else None
            )
            logger.begin_episode(spawn_name_ep, spawn_pose_dict)

            success = False
            for step in range(horizon):
                actions, _ = agent.predict(obs, deterministic=True)
                obs, _, dones, infos = env_sb3.step(actions)

                ee_pose_local = isaac_env.cfg.observations.policy.ee_pose.func(
                    isaac_env,
                    **isaac_env.cfg.observations.policy.ee_pose.params,
                )
                obj = isaac_env.scene["object"]
                obj_pose = obj.data.root_pos_w - isaac_env.scene.env_origins

                logger.record_step(
                    ee_pose_local.cpu().numpy()[0],
                    obj_pose.cpu().numpy()[0],
                    actions[0] if actions.ndim > 1 else actions,
                )

                success = bool(_check_success(isaac_env)[0])

            logger.end_episode(success)
            print(f"[play_rl] Episode {ep_idx+1}/{len(episodes)}: {'SUCCESS' if success else 'fail'}"
                  + (f" (spawn={spawn_name_ep})" if spawn_name_ep else ""))

    results = logger.finalize()
    print(f"\n[play_rl] Done. Success rate: {results['success_rate']:.1%}")
    print(f"[play_rl] Results saved to: {output_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()