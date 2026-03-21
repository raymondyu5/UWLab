"""
Evaluate a distilled noise policy (NoiseMLP + CFM diffusion base) in a UWLab env.

Architecture:
  - Frozen CFM PCD diffusion base policy (BC checkpoint format: ema_model + config.yaml)
  - Distilled NoiseMLP checkpoint: maps (PCD embedding, ee_pose, hand_joint_pos)
      -> noise (horizon * action_dim flat) used as diffusion starting trajectory.
  - No residual: env action = diffusion base action only.

Both the noise policy and diffusion base receive rendered seg_pc.

Usage (inside container):
    ./uwlab.sh -p scripts/eval/play_noise_policy.py --headless \\
        --diffusion_checkpoint logs/bc_cfm_pcd_bourbon_0312 \\
        --noise_policy_checkpoint logs/noise_policy_0317/checkpoints/best.ckpt
"""

import argparse
import os
import sys

_EXTRA_PATHS = [
    "/workspace/uwlab/third_party/diffusion_policy",
    "/workspace/uwlab/third_party/pip_packages",
]
_RFS_PATH = "/workspace/uwlab/scripts/reinforcement_learning/sb3/rfs"
for _p in _EXTRA_PATHS + [_RFS_PATH]:
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

_DIFFUSION_CKPT_DEFAULT = "logs/bc_cfm_pcd_bourbon_0312"
_NOISE_CKPT_DEFAULT = "logs/noise_policy_0317/checkpoints/best.ckpt"

parser = argparse.ArgumentParser()
parser.add_argument("--diffusion_checkpoint", type=str, default=_DIFFUSION_CKPT_DEFAULT,
                    help="Dir with checkpoints/ and config.yaml (BC training format)")
parser.add_argument("--noise_policy_checkpoint", type=str, default=_NOISE_CKPT_DEFAULT,
                    help="Path to NoiseMLP .ckpt file")
parser.add_argument("--task", type=str, default="UW-FrankaLeap-PourBottle-IkRel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--num_warmup_steps", type=int, default=10)
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="Use NoiseMLP mean directly (no sampling from log_std)")
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--output_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.record_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from wrapper import _load_cfm_checkpoint
from uwlab.eval.eval_logger import EvalLogger

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import uwlab_tasks  # noqa: F401


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NoiseMLP(nn.Module):
    _ACT_FNS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}

    def __init__(self, input_dim: int, output_dim: int, net_arch: list, activation_fn: str):
        super().__init__()
        act_cls = self._ACT_FNS[activation_fn]
        layers = []
        in_dim = input_dim
        for out_dim in net_arch:
            layers += [nn.Linear(in_dim, out_dim), act_cls()]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_noise_policy(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mlp = NoiseMLP(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        net_arch=ckpt["net_arch"],
        activation_fn=ckpt["activation_fn"],
    )
    mlp.load_state_dict(ckpt["ema_model"])
    mlp.to(device)
    mlp.eval()
    log_std = ckpt["log_std"].to(device)
    diffusion_horizon = ckpt["diffusion_horizon"]
    action_dim = ckpt["action_dim"]
    return mlp, log_std, diffusion_horizon, action_dim


# ---------------------------------------------------------------------------
# Obs formatting
# ---------------------------------------------------------------------------

def _encode_noise_obs(policy_obs: dict, diffusion, downsample_points: int,
                      device: torch.device) -> torch.Tensor:
    """Encode rendered seg_pc with frozen PointNet, concat with ee_pose + hand joints."""
    pcd = policy_obs["seg_pc"].float().permute(0, 2, 1)  # (B, 3, N)
    N = pcd.shape[-1]
    if N > downsample_points:
        perm = torch.randperm(N, device=device)[:downsample_points]
        pcd = pcd[:, :, perm]
    nobs = diffusion.normalizer.normalize({"seg_pc": pcd.unsqueeze(1)})
    pcd_normalized = nobs["seg_pc"][:, 0]  # (B, 3, downsample_points)
    pcd_emb = diffusion.obs_encoder.encode_pcd_only({"seg_pc": pcd_normalized})  # (B, feat_dim)

    ee_pose = policy_obs["ee_pose"].float()               # (B, 7)
    hand_joints = policy_obs["joint_pos"][:, 7:].float()  # (B, 16)
    return torch.cat([pcd_emb, ee_pose, hand_joints], dim=-1)


def _format_diffusion_obs(policy_obs: dict, downsample_points: int,
                           device: torch.device) -> dict:
    """Build diffusion obs dict: ee_pose(7) + hand_joints(16) = 23D agent_pos."""
    ee_pose = policy_obs["ee_pose"].float()
    hand_joints = policy_obs["joint_pos"][:, 7:].float()
    agent_pos = torch.cat([ee_pose, hand_joints], dim=-1).unsqueeze(1)  # (B, 1, 23)

    pcd = policy_obs["seg_pc"].float()  # (B, 3, N)
    N = pcd.shape[-1]
    if N > downsample_points:
        perm = torch.randperm(N, device=device)[:downsample_points]
        pcd = pcd[:, :, perm]
    pcd = pcd.unsqueeze(1)  # (B, 1, 3, downsample_points)

    return {"agent_pos": agent_pos, "seg_pc": pcd}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda:0")

    diffusion, _ = _load_cfm_checkpoint(args_cli.diffusion_checkpoint, device)
    downsample_points = diffusion.obs_encoder.shape_meta["obs"]["seg_pc"]["shape"][-1]

    noise_mlp, log_std, diffusion_horizon, action_dim = _load_noise_policy(
        args_cli.noise_policy_checkpoint, device)

    env_cfg = parse_env_cfg(args_cli.task, device=str(device), num_envs=args_cli.num_envs)
    env_cfg.table_z_range = (0.0, 0.0)
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.record_video else None)
    isaac_env = env.unwrapped
    episode_steps = int(isaac_env.max_episode_length)

    output_dir = args_cli.output_dir or os.path.join(
        os.path.dirname(args_cli.noise_policy_checkpoint), "eval_uwlab")
    logger = EvalLogger(output_dir, record_video=args_cli.record_video, record_plots=False)

    with torch.inference_mode():
        for ep_idx in range(args_cli.num_episodes):
            obs_raw, _ = env.reset()
            warmup_act = isaac_env.cfg.warmup_action(isaac_env)
            for _ in range(args_cli.num_warmup_steps):
                obs_raw, _, _, _, _ = env.step(warmup_act)

            logger.begin_episode(None, None)
            success = False

            for step in range(episode_steps):
                policy_obs = obs_raw["policy"]

                # --- NoiseMLP predict ---
                noise_input = _encode_noise_obs(policy_obs, diffusion, downsample_points, device)
                noise_flat = noise_mlp(noise_input)  # (B, horizon * action_dim)
                if not args_cli.deterministic:
                    noise_flat = noise_flat + torch.randn_like(noise_flat) * log_std.exp()
                noise = noise_flat.reshape(-1, diffusion_horizon, action_dim)  # (B, 4, 22)

                # --- Diffusion base action ---
                diff_obs = _format_diffusion_obs(policy_obs, downsample_points, device)
                env_action = diffusion.predict_action(diff_obs, noise)["action_pred"][:, 0]
                # (B, 22) = [arm_delta(6) | hand_joints(16)]

                # Clip arm delta
                env_action[:, :3] = env_action[:, :3].clamp(-0.03, 0.03)
                env_action[:, 3:6] = env_action[:, 3:6].clamp(-0.05, 0.05)

                obs_raw, _, terminated, truncated, _ = env.step(env_action)

                if args_cli.record_video:
                    frame = env.render()
                    if frame is not None:
                        logger.record_step(np.zeros(3), np.zeros(3), np.zeros(3), frame=frame)

                success = bool(isaac_env.cfg.is_success(isaac_env).cpu()[0])
                if success or terminated.any() or truncated.any():
                    break

            logger.end_episode(success)
            print(f"[play_noise_policy] Episode {ep_idx+1}/{args_cli.num_episodes}: "
                  f"{'SUCCESS' if success else 'fail'}")

    results = logger.finalize()
    print(f"\n[play_noise_policy] Success rate: {results['success_rate']:.1%} "
          f"({results['n_success']}/{results['n_episodes']})")
    print(f"[play_noise_policy] Results -> {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
