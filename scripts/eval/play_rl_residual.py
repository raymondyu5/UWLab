"""
Evaluate a residual RL policy (PPO noise-space + CFM diffusion base) in a UWLab env.

Architecture:
  - Frozen CFM PCD diffusion base policy (IsaacLab checkpoint format)
  - PPO residual policy (SB3 .zip): outputs 110D = [residual(22D) | noise(88D)]
      residual: arm(6D) + hand(16D), in [-1,1], scaled to physical bounds
      noise:    (4, 22) starting trajectory for diffusion inference

Diffusion checkpoint inspected — obs_key: ['cartesian_position', 'gripper_position']
  -> agent_pos = ee_pose(7D) + joint_pos[:,7:](16D) = 23D
  -> action shape: (22,) = arm_delta(6D) + hand_joints(16D)
  -> horizon: 4, n_obs_steps: 1

PPO checkpoint inspected — obs: 55D, action: 110D bounded [-1,1]
  55D obs order (must match IsaacLab Sb3VecEnvWrapper exactly):
    joint_pos(23) + ee_pose[:3](3) + target_object_pose(7) +
    contact_obs(5) + object_in_tip[:7](7) + manipulated_object_pose(7) + cup_pose[:3](3)

Usage (inside container):
    ./uwlab.sh -p scripts/eval/play_rl_residual.py --headless
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

_DIFFUSION_CKPT_DEFAULT = (
    "/workspace/uwlab/logs/real/mar3"
    "/pcd_delta_ee_action_abs_ee_obs_h4_pour"
    "/cfm/pcd_cfm/horizon_4_nobs_1"
)
_PPO_CKPT_DEFAULT = (
    "/workspace/uwlab/logs/real/bourbon_pour"
    "/bourbon_pour_force_pert_mar4_050326_093157/ppo/model_1300.zip"
)

parser = argparse.ArgumentParser()
parser.add_argument("--diffusion_checkpoint", type=str, default=_DIFFUSION_CKPT_DEFAULT,
                    help="Dir containing checkpoints/latest.ckpt (IsaacLab BC format)")
parser.add_argument("--ppo_checkpoint", type=str, default=_PPO_CKPT_DEFAULT,
                    help="Path to SB3 PPO .zip file")
parser.add_argument("--diffusion_ckpt_name", type=str, default="latest.ckpt")
parser.add_argument("--task", type=str, default="UW-FrankaLeap-PourBottle-IkRel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--num_warmup_steps", type=int, default=10)
# Residual bounds — must match rl_env_bourbon_pour_pink_cup_synthetic_pc_force_pert.yaml
#   arm_residual_range: [0.005, 0.01]  (xyz, rpy)
#   hand residual: action_range[3] = 0.1
parser.add_argument("--arm_residual_xyz", type=float, default=0.005)
parser.add_argument("--arm_residual_rpy", type=float, default=0.01)
parser.add_argument("--hand_residual", type=float, default=0.1)
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
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer


class _RolloutBuffer(RolloutBuffer):
    def __init__(self, *args, gpu_buffer=False, **kwargs):
        super().__init__(*args, **kwargs)
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy
from uwlab.eval.eval_logger import EvalLogger

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import uwlab_tasks  # noqa: F401


def _load_diffusion(checkpoint_dir: str, ckpt_name: str, device: torch.device):
    path = os.path.join(checkpoint_dir, "checkpoints", ckpt_name)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dicts"]["ema_model"]
    cfg = ckpt["cfg"]

    shape_meta = {
        "action": {"shape": list(cfg.shape_meta.action.shape)},
        "obs": {
            "agent_pos": {"shape": list(cfg.shape_meta.obs.agent_pos.shape), "type": "low_dim"},
            "seg_pc": {"shape": list(cfg.shape_meta.obs.seg_pc.shape), "type": "pcd"},
        },
    }
    pcd_model = PointNet(
        in_channels=3,
        local_channels=(64, 64, 64, 128, 1024),
        global_channels=(512, 256),
        use_bn=False,
    )
    obs_encoder = MultiPCDObsEncoder(shape_meta=shape_meta, pcd_model=pcd_model)
    noise_scheduler = ConditionalFlowMatcher(sigma=float(cfg.policy.noise_scheduler.sigma))
    policy = CFMPCDPolicy(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        noise_scheduler=noise_scheduler,
        horizon=int(cfg.horizon),
        n_action_steps=int(cfg.n_action_steps) + int(cfg.n_latency_steps),
        n_obs_steps=int(cfg.n_obs_steps),
        num_inference_steps=5,
        diffusion_step_embed_dim=256,
        down_dims=tuple(cfg.policy.down_dims),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()
    return policy, cfg


def _format_diffusion_obs(policy_obs: dict, downsample_points: int,
                           device: torch.device) -> dict:
    """Build diffusion obs dict matching training: cartesian_position(7) + gripper_position(16) = 23D."""
    ee_pose = policy_obs["ee_pose"].float()            # (B, 7)
    hand_joints = policy_obs["joint_pos"][:, 7:].float()  # (B, 16) — hand portion of joint_pos
    agent_pos = torch.cat([ee_pose, hand_joints], dim=-1).unsqueeze(1)  # (B, 1, 23)

    pcd = policy_obs["seg_pc"].float()  # (B, 3, N)
    N = pcd.shape[-1]
    if N > downsample_points:
        perm = torch.randperm(N, device=device)[:downsample_points]
        pcd = pcd[:, :, perm]
    pcd = pcd.unsqueeze(1)  # (B, 1, 3, downsample_points)

    return {"agent_pos": agent_pos, "seg_pc": pcd}


def _format_ppo_obs(policy_obs: dict) -> np.ndarray:
    """Assemble the 55D obs that the PPO was trained on (IsaacLab Sb3VecEnvWrapper order).

    joint_pos(23) + ee_pose[:3](3) + target_object_pose(7) +
    contact_obs(5) + object_in_tip[:7](7) + manipulated_object_pose(7) + cup_pose[:3](3) = 55D
    """
    obs = torch.cat([
        policy_obs["joint_pos"],                    # (B, 23)
        policy_obs["ee_pose"][:, :3],               # (B, 3)  xyz only (IsaacLab stripped right_ee_pose)
        policy_obs["target_object_pose"],           # (B, 7)
        policy_obs["contact_obs"].float(),          # (B, 5)
        policy_obs["object_in_tip"][:, :7],         # (B, 7)  IsaacLab truncated to 7
        policy_obs["manipulated_object_pose"],      # (B, 7)
        policy_obs["cup_pose"][:, :3],              # (B, 3)  xyz only
    ], dim=-1)  # (B, 55)
    return obs.cpu().numpy()


def main():
    device = torch.device("cuda:0")

    diffusion, diff_cfg = _load_diffusion(
        args_cli.diffusion_checkpoint, args_cli.diffusion_ckpt_name, device)
    downsample_points = int(diff_cfg.downsample_points)
    diffusion_horizon = diffusion.horizon    # 4
    action_dim = diffusion.action_dim        # 22

    # Load PPO without env — avoids obs/act space mismatch at load time.
    agent = PPO.load(args_cli.ppo_checkpoint, device=device,
                     custom_objects={"rollout_buffer_class": _RolloutBuffer})

    # Residual scaling: [-1,1] -> physical bounds from training config
    residual_lower = torch.tensor(
        [-args_cli.arm_residual_xyz] * 3 + [-args_cli.arm_residual_rpy] * 3
        + [-args_cli.hand_residual] * 16, device=device)
    residual_upper = torch.tensor(
        [+args_cli.arm_residual_xyz] * 3 + [+args_cli.arm_residual_rpy] * 3
        + [+args_cli.hand_residual] * 16, device=device)

    env_cfg = parse_env_cfg(args_cli.task, device=str(device), num_envs=args_cli.num_envs)
    env_cfg.table_z_range = (0.0, 0.0)
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.record_video else None)
    isaac_env = env.unwrapped
    episode_steps = int(isaac_env.max_episode_length)

    output_dir = args_cli.output_dir or os.path.join(
        os.path.dirname(args_cli.ppo_checkpoint), "eval_uwlab")
    logger = EvalLogger(output_dir, record_video=args_cli.record_video, record_plots=True)

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

                # --- PPO predict (55D obs) ---
                ppo_obs_np = _format_ppo_obs(policy_obs)
                if args_cli.num_envs == 1:
                    ppo_action_np, _ = agent.predict(ppo_obs_np[0], deterministic=True)
                    ppo_action = torch.as_tensor(
                        ppo_action_np, device=device).unsqueeze(0)  # (1, 110)
                else:
                    ppo_action_np, _ = agent.predict(ppo_obs_np, deterministic=True)
                    ppo_action = torch.as_tensor(ppo_action_np, device=device)  # (B, 110)

                # --- Split: [residual(22) | noise(88)] ---
                residual_dim = 6 + 16  # arm + hand
                residual_raw = ppo_action[:, :residual_dim]          # (B, 22) in [-1, 1]
                noise = ppo_action[:, residual_dim:].reshape(
                    -1, diffusion_horizon, action_dim)                # (B, 4, 22)

                # Scale residual from [-1,1] to physical bounds
                residual = (residual_raw + 1.0) / 2.0 * (
                    residual_upper - residual_lower) + residual_lower  # (B, 22)

                # --- Diffusion base action ---
                diff_obs = _format_diffusion_obs(policy_obs, downsample_points, device)
                base = diffusion.predict_action(diff_obs, noise)["action_pred"][:, 0]
                # base: (B, 22) = [arm_delta(6) | hand_joints(16)]

                # --- Compose base + residual ---
                env_action = base.clone()
                env_action[:, :6] += residual[:, :6]   # arm delta
                env_action[:, 6:] += residual[:, 6:]   # hand joints

                # Clip arm delta (matches IsaacLab clip_real_world_actions)
                env_action[:, :3] = env_action[:, :3].clamp(-0.03, 0.03)   # xyz
                env_action[:, 3:6] = env_action[:, 3:6].clamp(-0.05, 0.05)  # rpy

                obs_raw, _, terminated, truncated, _ = env.step(env_action)

                if args_cli.record_video:
                    frame = env.render()
                    if frame is not None:
                        logger.record_step(np.zeros(3), np.zeros(3), np.zeros(3), frame=frame)

                success = bool(isaac_env.cfg.is_success(isaac_env).cpu()[0])
                if success or terminated.any() or truncated.any():
                    break

            logger.end_episode(success)
            print(f"[play_rl_residual] Episode {ep_idx+1}/{args_cli.num_episodes}: "
                  f"{'SUCCESS' if success else 'fail'}")

    results = logger.finalize()
    print(f"\n[play_rl_residual] Success rate: {results['success_rate']:.1%} "
          f"({results['n_success']}/{results['n_episodes']})")
    print(f"[play_rl_residual] Results -> {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
