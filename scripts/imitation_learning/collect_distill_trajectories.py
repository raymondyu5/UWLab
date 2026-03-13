"""
Collect trajectories from an RFS policy in distill_mode for BC/distillation training.

Uses depth-rendered point clouds from randomly sampled camera views (DISTILL_MODE).
Saves episodes in zarr format compatible with ZarrDataset and load_real_episode_zarr.

RFS policy: diffusion base + PPO residual (same as play_rl_residual.py).
Requires --enable_cameras for depth rendering.

Usage (inside container):
    ./uwlab.sh -p scripts/imitation_learning/collect_distill_trajectories.py \\
        --headless \\
        --enable_cameras \\
        --task UW-FrankaLeap-GraspPinkCup-IkRel-v0 \\
        --diffusion_checkpoint /path/to/cfm/checkpoint_dir \\
        --ppo_checkpoint /path/to/ppo/model.zip \\
        --output_dir logs/distill_collection \\
        --num_episodes 100
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

parser = argparse.ArgumentParser(
    description="Collect RFS policy trajectories in distill_mode (depth-rendered PC) for BC training."
)
parser.add_argument("--diffusion_checkpoint", type=str, required=True,
                    help="Dir containing checkpoints/latest.ckpt (IsaacLab BC format)")
parser.add_argument("--ppo_checkpoint", type=str, required=True,
                    help="Path to SB3 PPO .zip file")
parser.add_argument("--diffusion_ckpt_name", type=str, default="latest.ckpt")
parser.add_argument("--task", type=str, default="UW-FrankaLeap-GraspPinkCup-IkRel-v0",
                    help="Task with seg_pc (GraspPinkCup, PourBottle, GraspBottle). Must use IkRel-v0.")
parser.add_argument("--output_dir", type=str, default="logs/distill_collection")
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--num_warmup_steps", type=int, default=10)
parser.add_argument("--arm_residual_xyz", type=float, default=0.005)
parser.add_argument("--arm_residual_rpy", type=float, default=0.01)
parser.add_argument("--hand_residual", type=float, default=0.1)
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import zarr
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    parse_franka_leap_env_cfg,
    DISTILL_MODE,
)


class _RolloutBuffer(RolloutBuffer):
    def __init__(self, *args, gpu_buffer=False, **kwargs):
        super().__init__(*args, **kwargs)


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


def _format_diffusion_obs(policy_obs: dict, downsample_points: int, device: torch.device) -> dict:
    ee_pose = policy_obs["ee_pose"].float()
    hand_joints = policy_obs["joint_pos"][:, 7:].float()
    agent_pos = torch.cat([ee_pose, hand_joints], dim=-1).unsqueeze(1)

    pcd = policy_obs["seg_pc"].float()
    N = pcd.shape[-1]
    if N > downsample_points:
        perm = torch.randperm(N, device=device)[:downsample_points]
        pcd = pcd[:, :, perm]
    pcd = pcd.unsqueeze(1)
    return {"agent_pos": agent_pos, "seg_pc": pcd}


def _format_ppo_obs(policy_obs: dict) -> np.ndarray:
    """Assemble PPO obs. Uses zeros for cup_pose if not present (Grasp tasks)."""
    cup_pose = policy_obs.get("cup_pose")
    if cup_pose is None:
        B = policy_obs["joint_pos"].shape[0]
        cup_pose = torch.zeros(B, 3, device=policy_obs["joint_pos"].device, dtype=torch.float32)
    else:
        cup_pose = cup_pose[:, :3]

    obs = torch.cat([
        policy_obs["joint_pos"],
        policy_obs["ee_pose"][:, :3],
        policy_obs["target_object_pose"],
        policy_obs["contact_obs"].float(),
        policy_obs["object_in_tip"][:, :7],
        policy_obs["manipulated_object_pose"],
        cup_pose,
    ], dim=-1)
    return obs.cpu().numpy()


def _save_episode_zarr(output_dir: str, episode_idx: int, data: dict) -> str:
    ep_dir = os.path.join(output_dir, f"episode_{episode_idx}")
    os.makedirs(ep_dir, exist_ok=True)
    zarr_path = os.path.join(ep_dir, f"episode_{episode_idx}.zarr")
    store = zarr.open(zarr_path, mode="w")
    grp = store.create_group("data")
    for key, arr in data.items():
        arr_np = np.asarray(arr)
        grp.create_dataset(key, data=arr_np)
    return zarr_path


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    diffusion, diff_cfg = _load_diffusion(
        args_cli.diffusion_checkpoint, args_cli.diffusion_ckpt_name, device
    )
    ds = getattr(diff_cfg, "dataset", diff_cfg)
    downsample_points = int(getattr(ds, "downsample_points", 2048))
    diffusion_horizon = diffusion.horizon
    action_dim = diffusion.action_dim

    agent = PPO.load(
        args_cli.ppo_checkpoint,
        device=device,
        custom_objects={"rollout_buffer_class": _RolloutBuffer},
    )

    residual_lower = torch.tensor(
        [-args_cli.arm_residual_xyz] * 3 + [-args_cli.arm_residual_rpy] * 3
        + [-args_cli.hand_residual] * 16,
        device=device,
    )
    residual_upper = torch.tensor(
        [+args_cli.arm_residual_xyz] * 3 + [+args_cli.arm_residual_rpy] * 3
        + [+args_cli.hand_residual] * 16,
        device=device,
    )

    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        DISTILL_MODE,
        device=str(device),
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    isaac_env = env.unwrapped
    episode_steps = int(isaac_env.max_episode_length)

    os.makedirs(args_cli.output_dir, exist_ok=True)
    print(f"[collect_distill] Output: {args_cli.output_dir}")
    print(f"[collect_distill] Task: {args_cli.task}, episodes: {args_cli.num_episodes}")

    n_success = 0
    with torch.inference_mode():
        for ep_idx in tqdm(range(args_cli.num_episodes), desc="collect_distill"):
            obs_raw, _ = env.reset()
            warmup_act = isaac_env.cfg.warmup_action(isaac_env)
            for _ in range(args_cli.num_warmup_steps):
                obs_raw, _, _, _, _ = env.step(warmup_act)

            arm_joint_pos_list = []
            hand_joint_pos_list = []
            ee_pose_list = []
            actions_list = []
            seg_pc_list = []
            arm_joint_pos_target_list = []
            ee_pose_cmd_list = []
            rewards_list = []
            dones_list = []

            for step in range(episode_steps):
                policy_obs = obs_raw["policy"]

                ppo_obs_np = _format_ppo_obs(policy_obs)
                ppo_action_np, _ = agent.predict(ppo_obs_np[0], deterministic=True)
                ppo_action = torch.as_tensor(ppo_action_np, device=device).unsqueeze(0)

                residual_raw = ppo_action[:, :22]
                noise = ppo_action[:, 22:].reshape(-1, diffusion_horizon, action_dim)

                residual = (residual_raw + 1.0) / 2.0 * (
                    residual_upper - residual_lower
                ) + residual_lower

                diff_obs = _format_diffusion_obs(policy_obs, downsample_points, device)
                base = diffusion.predict_action(diff_obs, noise)["action_pred"][:, 0]

                env_action = base.clone()
                env_action[:, :6] += residual[:, :6]
                env_action[:, 6:] += residual[:, 6:]
                env_action[:, :3] = env_action[:, :3].clamp(-0.03, 0.03)
                env_action[:, 3:6] = env_action[:, 3:6].clamp(-0.05, 0.05)

                arm_joint_pos_list.append(policy_obs["joint_pos"][0, :7].cpu().numpy())
                hand_joint_pos_list.append(policy_obs["joint_pos"][0, 7:23].cpu().numpy())
                ee_pose_list.append(policy_obs["ee_pose"][0].cpu().numpy())
                actions_list.append(env_action[0].cpu().numpy())
                seg_pc_list.append(policy_obs["seg_pc"][0].T.cpu().numpy())
                ee_pose_cmd_list.append(policy_obs["ee_pose"][0].cpu().numpy())

                obs_raw, reward, terminated, truncated, _ = env.step(env_action)

                next_joint_pos = obs_raw["policy"]["joint_pos"][0, :7].cpu().numpy()
                arm_joint_pos_target_list.append(next_joint_pos)

                r = reward[0]
                rewards_list.append(float(r.item() if hasattr(r, "item") else r))
                done = bool(terminated[0].cpu().numpy() or truncated[0].cpu().numpy())
                dones_list.append(done)

                if terminated.any() or truncated.any():
                    break

            success = bool(isaac_env.cfg.is_success(isaac_env).cpu()[0])
            if success:
                n_success += 1

            episode_data = {
                "arm_joint_pos": np.array(arm_joint_pos_list),
                "hand_joint_pos": np.array(hand_joint_pos_list),
                "ee_pose": np.array(ee_pose_list),
                "actions": np.array(actions_list),
                "seg_pc": np.array(seg_pc_list),
                "arm_joint_pos_target": np.array(arm_joint_pos_target_list),
                "ee_pose_cmd": np.array(ee_pose_cmd_list),
                "rewards": np.array(rewards_list),
                "dones": np.array(dones_list),
            }
            zarr_path = _save_episode_zarr(args_cli.output_dir, ep_idx, episode_data)
            pbar.set_postfix(success=n_success, steps=len(actions_list))

    rate = 100 * n_success / args_cli.num_episodes if args_cli.num_episodes else 0
    print(f"\n[collect_distill] Done. Success rate: {n_success}/{args_cli.num_episodes} ({rate:.1f}%)")
    print(f"[collect_distill] Episodes saved to {args_cli.output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
