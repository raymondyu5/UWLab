"""
Eval a trained BC policy (.pt checkpoint from train_stage1_bc.py) in Isaac Sim.

Usage:
    ./uwlab.sh -p scripts/reinforcement_learning/rialto/eval_bc_policy.py \
        --task UW-FrankaLeap-GraspBottleRandomResets-JointAbs-v0 \
        --checkpoint logs/rialto/bc/bc_20241201_1234/bc_pretrained.pt \
        --num_envs 16 --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval a BC policy in Isaac Sim.")
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to bc_pretrained.pt (from train_stage1_bc.py).")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument(
    "--obs_keys", nargs="+", default=None,
    help="Override obs keys (default: read from checkpoint metadata).",
)
parser.add_argument("--num_episodes", type=int, default=3,
                    help="Number of random-reset eval episodes to run.")
parser.add_argument("--log_dir", type=str, default="logs/rialto/eval_bc")
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--eval_spawn", type=str, default=None,
                    help="Spawn config name from configs/eval/spawns/ (default: random).")
parser.add_argument("--no_eval_video", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── all Isaac-dependent imports after AppLauncher ──────────────────────────

import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap import (
    RL_MODE,
    parse_franka_leap_env_cfg,
)
from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import SpawnCfg, load_spawn_cfg


# ── Policy network (must match train_stage1_bc.py) ─────────────────────────

class SimpleMLP(nn.Module):
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


class BCPolicyWrapper:
    """Wraps SimpleMLP with a .predict() interface compatible with SB3 callbacks."""

    def __init__(self, net: SimpleMLP, device: str):
        self.net = net.to(device)
        self.device = device
        self.net.eval()

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = self.net(obs_t).cpu().numpy()
        return actions, None


# ── Checkpoint loading ─────────────────────────────────────────────────────

def load_bc_checkpoint(path: str, device: str):
    """Load a SimpleMLP from a .pt file, inferring dims from metadata or weight shapes."""
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        obs_dim = ckpt["obs_dim"]
        action_dim = ckpt["action_dim"]
        net_arch = ckpt.get("net_arch", [256, 256])
        obs_keys = ckpt.get("obs_keys", None)
        print(f"[eval_bc] Loaded checkpoint metadata: obs_dim={obs_dim} action_dim={action_dim} "
              f"net_arch={net_arch} obs_keys={obs_keys}")
    else:
        # Legacy: plain state_dict — infer everything from weight shapes.
        state_dict = ckpt
        obs_dim = state_dict["shared_net.0.weight"].shape[1]
        action_dim = state_dict["action_net.weight"].shape[0]
        net_arch = []
        i = 0
        while f"shared_net.{i}.weight" in state_dict:
            net_arch.append(state_dict[f"shared_net.{i}.weight"].shape[0])
            i += 2
        obs_keys = None
        print(f"[eval_bc] Legacy checkpoint — inferred: obs_dim={obs_dim} action_dim={action_dim} "
              f"net_arch={net_arch}")

    net = SimpleMLP(obs_dim, action_dim, hidden=tuple(net_arch))
    net.load_state_dict(state_dict)
    return BCPolicyWrapper(net, device), obs_keys


# ── Obs extraction ─────────────────────────────────────────────────────────

def obs_to_flat(obs_dict: dict, obs_keys: list) -> np.ndarray:
    policy = obs_dict.get("policy", obs_dict)
    parts = []
    for key in obs_keys:
        val = policy[key]
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().float().numpy()
        parts.append(val.reshape(val.shape[0], -1) if val.ndim > 1 else val)
    return np.concatenate(parts, axis=-1).astype(np.float32)


# ── Eval loop ──────────────────────────────────────────────────────────────

def run_eval(policy, gym_env, isaac_env, obs_keys, num_episodes, log_dir,
             spawn_cfg=None, record_video=False):
    """Run evaluation episodes and return aggregated results."""
    device = isaac_env.device
    num_envs = isaac_env.num_envs
    cfg = isaac_env.cfg
    video_fps = 1.0 / (cfg.sim.dt * cfg.decimation)

    # Find the manipulated object key in the scene.
    scene_keys = list(isaac_env.scene.keys())
    if "grasp_object" in scene_keys:
        obj_key = "grasp_object"
    else:
        obj_key = next(
            k for k in scene_keys
            if k not in ("terrain", "robot", "table", "ground", "dome_light")
            and not k.endswith("_contact")
        )

    logger = EvalLogger(log_dir, record_video=record_video, record_plots=True, video_fps=video_fps)
    use_fixed = spawn_cfg is not None and spawn_cfg.poses

    if use_fixed:
        _run_fixed_eval(policy, gym_env, isaac_env, obs_keys, obj_key, device, num_envs,
                        spawn_cfg, logger, record_video)
    else:
        _run_random_eval(policy, gym_env, isaac_env, obs_keys, obj_key, device, num_envs,
                         num_episodes, log_dir, logger, record_video)

    return logger.finalize()


def _run_random_eval(policy, gym_env, isaac_env, obs_keys, obj_key, device, num_envs,
                     num_episodes, log_dir, logger, record_video):
    episode_steps = int(isaac_env.max_episode_length)
    progress_every = max(1, episode_steps // 10)

    for trial_idx in range(num_episodes):
        record_this = record_video and trial_idx == 0
        print(f"[eval_bc] Random trial {trial_idx + 1}/{num_episodes}  record_video={record_this}",
              flush=True)

        obs_dict, _ = gym_env.reset()
        obs_flat = obs_to_flat(obs_dict, obs_keys)

        init_pos_w = isaac_env.scene[obj_key].data.root_pos_w.clone()
        init_pos_local = (init_pos_w - isaac_env.scene.env_origins).cpu().numpy()

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

            action, _ = policy.predict(obs_flat, deterministic=True)
            action_t = torch.tensor(action, dtype=torch.float32, device=device)
            obs_dict, _, terminated, truncated, _ = gym_env.step(action_t)
            obs_flat = obs_to_flat(obs_dict, obs_keys)

            for i in range(num_envs):
                if (terminated[i] or truncated[i]) and not recorded[i]:
                    recorded[i] = True

            frame = None
            if record_this:
                frame = gym_env.render()
                if frame is not None:
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    frame = np.asarray(frame)
                    if frame.ndim == 4:
                        frame = frame[0]
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

            obj_pos = isaac_env.scene[obj_key].data.root_pos_w[0].cpu().numpy()
            logger.record_step(ee_pose=np.zeros(7), object_pose=obj_pos,
                               action=action[0], frame=frame)

            if step_idx % progress_every == 0 or step_idx == episode_steps - 1:
                n_succ = int(ever_success.sum().item())
                print(f"  step {step_idx + 1}/{episode_steps}  ever_success={n_succ}/{num_envs}",
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
        print(f"[eval_bc] Trial {trial_idx + 1}: success={n_success_end}/{num_envs} "
              f"({100 * n_success_end / num_envs:.1f}%)  ever={n_success_ever}")


def _run_fixed_eval(policy, gym_env, isaac_env, obs_keys, obj_key, device, num_envs,
                    spawn_cfg, logger, record_video):
    episode_steps = int(isaac_env.max_episode_length)

    for pose_idx, pose in enumerate(spawn_cfg.poses):
        record_this = record_video and pose_idx == 0
        print(f"[eval_bc] Fixed pose {pose_idx + 1}/{len(spawn_cfg.poses)}: "
              f"x={pose.x:.3f} y={pose.y:.3f}", flush=True)

        # Reset and teleport object.
        obs_dict, _ = gym_env.reset()
        defaults = getattr(isaac_env.cfg, "object_spawn_defaults", None)
        if defaults is not None:
            obj = isaac_env.scene[obj_key]
            default_pos = torch.tensor(defaults["default_pos"], dtype=torch.float32, device=device)
            default_rot = torch.tensor(defaults["default_rot"], dtype=torch.float32, device=device)
            new_pos = default_pos.clone()
            new_pos[0] += pose.x
            new_pos[1] += pose.y
            root_state = obj.data.default_root_state.clone()
            root_state[:, :3] = new_pos.unsqueeze(0) + isaac_env.scene.env_origins
            root_state[:, 3:7] = default_rot.unsqueeze(0)
            root_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(root_state)
            isaac_env.sim.step()
            obs_dict = {"policy": isaac_env.observation_manager.compute()["policy"]}

        obs_flat = obs_to_flat(obs_dict, obs_keys)
        logger.begin_episode(pose.name or f"pose_{pose_idx}", {"x": pose.x, "y": pose.y})

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

            action, _ = policy.predict(obs_flat, deterministic=True)
            action_t = torch.tensor(action, dtype=torch.float32, device=device)
            obs_dict, _, terminated, truncated, _ = gym_env.step(action_t)
            obs_flat = obs_to_flat(obs_dict, obs_keys)

            for i in range(num_envs):
                if (terminated[i] or truncated[i]) and not recorded[i]:
                    recorded[i] = True

            frame = None
            if record_this:
                frame = gym_env.render()
                if frame is not None:
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    frame = np.asarray(frame)
                    if frame.ndim == 4:
                        frame = frame[0]
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

            obj_pos = isaac_env.scene[obj_key].data.root_pos_w[0].cpu().numpy()
            logger.record_step(ee_pose=np.zeros(7), object_pose=obj_pos,
                               action=action[0], frame=frame)

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
        print(f"[eval_bc] Pose {pose_idx + 1}: success={n_success_end}/{num_envs} "
              f"({100 * n_success_end / num_envs:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    device = args_cli.device or "cuda:0"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.join(args_cli.log_dir, f"eval_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[eval_bc] Logging to {log_dir}")

    # Load BC policy.
    print(f"[eval_bc] Loading checkpoint: {args_cli.checkpoint}")
    policy, ckpt_obs_keys = load_bc_checkpoint(args_cli.checkpoint, device)
    obs_keys = args_cli.obs_keys or ckpt_obs_keys
    if obs_keys is None:
        raise ValueError("obs_keys not in checkpoint and not supplied via --obs_keys")
    print(f"[eval_bc] Using obs_keys: {obs_keys}")

    # Create Isaac Sim environment.
    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task, RL_MODE, device=device, num_envs=args_cli.num_envs
    )
    env_cfg.seed = args_cli.seed
    gym_env = gym.make(
        args_cli.task, cfg=env_cfg,
        render_mode="rgb_array" if not args_cli.no_eval_video else None,
    )
    isaac_env = gym_env.unwrapped

    # Wandb.
    import wandb
    if args_cli.wandb_project:
        run_name = args_cli.wandb_run_name or f"rialto-eval-bc-{timestamp}"
        wandb.init(
            project=args_cli.wandb_project,
            name=run_name,
            config={k: v for k, v in vars(args_cli).items() if k not in ("device",)},
        )
        print(f"[eval_bc] wandb run: {wandb.run.get_url()}")

    spawn_cfg = None
    if args_cli.eval_spawn is not None:
        spawn_cfg = load_spawn_cfg(args_cli.eval_spawn, "configs/eval/spawns")

    results = run_eval(
        policy=policy,
        gym_env=gym_env,
        isaac_env=isaac_env,
        obs_keys=obs_keys,
        num_episodes=args_cli.num_episodes,
        log_dir=log_dir,
        spawn_cfg=spawn_cfg,
        record_video=not args_cli.no_eval_video,
    )

    print(f"\n[eval_bc] === Results ===")
    print(f"  success_rate (end):  {100 * results['success_rate']:.1f}%")
    print(f"  success_rate (ever): {100 * results.get('success_rate_ever', 0):.1f}%")
    print(f"  n_success:           {results['n_success']}")

    if args_cli.wandb_project and wandb.run is not None:
        import glob as _glob
        log_dict = {
            "eval/success_rate_end": results["success_rate"],
            "eval/success_rate_ever": results.get("success_rate_ever", 0.0),
            "eval/n_success": results["n_success"],
        }
        scatter_path = os.path.join(log_dir, "scatter_success.png")
        if os.path.isfile(scatter_path):
            log_dict["eval/scatter_success"] = wandb.Image(scatter_path)
        video_dir = os.path.join(log_dir, "videos")
        if os.path.isdir(video_dir):
            videos = sorted(_glob.glob(os.path.join(video_dir, "*.mp4")))
            if videos:
                log_dict["eval/video"] = wandb.Video(videos[0])
        wandb.log(log_dict)
        wandb.finish()

    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
