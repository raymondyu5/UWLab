"""
RialTo Pre-Stage-1: Roll out a pretrained CFM-PCD policy in sim (RL_MODE) and
collect successful trajectories with privileged information for Stage 1 teacher training.

Unlike training data (human demos), these rollouts are captured from the CFM policy
running in simulation with mesh-based point clouds, and include object pose as
privileged state that the state-based MLP teacher can condition on.

Saves zarr episodes compatible with train_stage1_bc_ppo.py and ZarrDataset:
    episode_N/episode_N.zarr
        data/ee_pose                 (T, 7)
        data/hand_joint_pos          (T, 16)
        data/arm_joint_pos           (T, 7)
        data/manipulated_object_pose (T, 7)   ← privileged info
        data/actions                 (T, A)
        data/rewards                 (T,)
        data/dones                   (T,)

Usage:
    ./uwlab.sh -p scripts/reinforcement_learning/rialto/collect_rialto_rollouts.py --task UW-FrankaLeap-GraspBottleRandomResets-JointAbs-v0 --checkpoint_dir logs/final_bc_checkpoints/bc_cfm_pcd_bottle_grasp_mixed_0412_absjoint_h16_hist4_noextnoise_fast --output_dir data_storage/rialto/bottle_grasp --target_episodes 100 --num_envs 1024 --headless
"""

import argparse
from uwlab.utils.paths import setup_third_party_paths

setup_third_party_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Collect sim rollouts from a CFM-PCD policy for RialTo Stage 1."
)
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name.")
parser.add_argument("--checkpoint_dir", type=str, required=True,
                    help="Path to CFM training run dir containing checkpoints/best.ckpt "
                         "and .hydra/config.yaml (or config.yaml).")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory to write zarr episodes into.")
parser.add_argument("--target_episodes", type=int, default=200,
                    help="Number of successful episodes to collect.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--action_horizon", type=int, default=1,
                    help="Steps between policy re-inferences. Defaults to policy n_action_steps.")
parser.add_argument("--privileged_keys", nargs="+",
                    default=["manipulated_object_pose"],
                    help="Extra env obs keys to save as privileged info.")
parser.add_argument("--success_key", type=str, default="is_lifted",
                    help="Metric key from env metrics to use as the save condition. "
                         "Episode is saved if this metric was ever True during the episode. "
                         "Common values: is_lifted, is_grasped, is_success, is_healthy_z.")
parser.add_argument("--checkpoint_name", type=str, default=None,
                    help="Checkpoint file inside checkpoints/ (default: best.ckpt or latest.ckpt).")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── all Isaac-dependent imports after AppLauncher ──────────────────────────

import os
import random
from collections import defaultdict

import numpy as np
import torch
import yaml
import zarr
import gymnasium as gym
from tqdm import tqdm
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy
from uwlab.eval.bc_obs_formatter import BCObsFormatter

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap import (
    RL_MODE,
    parse_franka_leap_env_cfg,
)


# ── Policy loading (mirrors play_bc.py::load_cfm_policy) ──────────────────

def _find_checkpoint(checkpoint_dir: str, ckpt_name: str | None = None) -> str:
    if ckpt_name is not None:
        path = os.path.join(checkpoint_dir, "checkpoints", ckpt_name)
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    for name in ("best.ckpt", "latest.ckpt"):
        path = os.path.join(checkpoint_dir, "checkpoints", name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No checkpoint in {checkpoint_dir}/checkpoints/")


def load_cfm_policy(checkpoint_dir: str, device: torch.device,
                    ckpt_name: str | None = None) -> tuple:
    """Load a CFMPCDPolicy and return (policy, train_cfg).

    Replicates the loading logic from scripts/eval/play_bc.py.
    The checkpoint embeds the training config; the normalizer is restored via
    policy.load_state_dict() since it was set on the model during training.
    """
    ckpt_path = _find_checkpoint(checkpoint_dir, ckpt_name)
    print(f"[collect] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Config: prefer embedded (new checkpoints), fall back to .hydra/config.yaml
    train_cfg = ckpt.get("cfg")
    if train_cfg is None:
        for candidate in (
            os.path.join(checkpoint_dir, ".hydra", "config.yaml"),
            os.path.join(checkpoint_dir, "config.yaml"),
        ):
            if os.path.isfile(candidate):
                with open(candidate) as f:
                    train_cfg = yaml.safe_load(f)
                break
    if train_cfg is None:
        raise FileNotFoundError(
            f"No training config found in checkpoint or {checkpoint_dir}."
        )

    ds_cfg  = train_cfg["dataset"]
    pol_cfg = dict(train_cfg["policy"])
    pn_cfg  = train_cfg["pointnet"]

    sd = ckpt["ema_model"]
    low_obs_dim = sd["normalizer.params_dict.agent_pos.scale"].shape[0]
    action_dim  = sd["normalizer.params_dict.action.scale"].shape[0]
    use_action_history = "normalizer.params_dict.past_actions.scale" in sd

    shape_meta = {
        "action": {"shape": [action_dim]},
        "obs": {"agent_pos": {"shape": [low_obs_dim], "type": "low_dim"}},
    }
    for key in list(ds_cfg["image_keys"]):
        shape_meta["obs"][key] = {"shape": [3, int(ds_cfg["downsample_points"])], "type": "pcd"}

    pcd_model = PointNet(
        in_channels=int(pn_cfg["in_channels"]),
        local_channels=tuple(pn_cfg["local_channels"]),
        global_channels=tuple(pn_cfg["global_channels"]),
        use_bn=bool(pn_cfg["use_bn"]),
    )
    obs_encoder = MultiPCDObsEncoder(shape_meta=shape_meta, pcd_model=pcd_model)

    policy = CFMPCDPolicy(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        noise_scheduler=ConditionalFlowMatcher(sigma=float(pol_cfg["sigma"])),
        horizon=int(train_cfg["horizon"]),
        n_action_steps=int(train_cfg["n_action_steps"]),
        n_obs_steps=int(train_cfg["n_obs_steps"]),
        num_inference_steps=int(pol_cfg["num_inference_steps"]),
        diffusion_step_embed_dim=int(pol_cfg["diffusion_step_embed_dim"]),
        down_dims=tuple(pol_cfg["down_dims"]),
        kernel_size=int(pol_cfg["kernel_size"]),
        n_groups=int(pol_cfg["n_groups"]),
        cond_predict_scale=bool(pol_cfg["cond_predict_scale"]),
        use_action_history=use_action_history,
    )
    policy.load_state_dict(sd)
    policy.to(device).eval()
    return policy, train_cfg


# ── Zarr saving ────────────────────────────────────────────────────────────

def _save_episode_zarr(output_dir: str, episode_idx: int, data: dict) -> str:
    ep_dir = os.path.join(output_dir, f"episode_{episode_idx}")
    os.makedirs(ep_dir, exist_ok=True)
    zarr_path = os.path.join(ep_dir, f"episode_{episode_idx}.zarr")
    store = zarr.open(zarr_path, mode="w")
    for key, arr in data.items():
        store[f"data/{key}"] = arr
    return zarr_path


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    device = torch.device(args_cli.device or "cuda:0")
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # ── 1. Load CFM-PCD policy ────────────────────────────────────────────
    policy, train_cfg = load_cfm_policy(
        args_cli.checkpoint_dir, device, ckpt_name=args_cli.checkpoint_name
    )
    ds_cfg = train_cfg["dataset"]
    obs_keys      = list(ds_cfg["obs_keys"])    # e.g. ["ee_pose", "hand_joint_pos"]
    image_keys    = list(ds_cfg["image_keys"])  # e.g. ["seg_pc"]
    downsample_pts = int(ds_cfg["downsample_points"])
    n_obs_steps   = int(train_cfg["n_obs_steps"])
    n_action_steps = int(train_cfg["n_action_steps"])
    chunk_relative = bool((ds_cfg or {}).get("chunk_relative", False))
    action_horizon = args_cli.action_horizon or n_action_steps
    action_dim = policy.action_dim

    print(f"[collect] obs_keys={obs_keys}  image_keys={image_keys}")
    print(f"[collect] n_action_steps={n_action_steps}  action_horizon={action_horizon}"
          f"  chunk_relative={chunk_relative}")
    print(f"[collect] privileged_keys={args_cli.privileged_keys}")

    # ── 2. Create Isaac Sim env (RL_MODE: mesh PCs, parallel envs) ───────
    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        RL_MODE,
        device=str(device),
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    isaac_env = env.unwrapped
    episode_steps = isaac_env.max_episode_length

    # ── 3. BCObsFormatter (handles obs history + PCD downsampling) ────────
    formatter = BCObsFormatter(
        obs_keys=obs_keys,
        image_keys=image_keys,
        downsample_points=downsample_pts,
        device=device,
        n_obs_steps=n_obs_steps,
        action_dim=action_dim if policy.use_action_history else 0,
    )

    # ── 4. Collection loop ────────────────────────────────────────────────
    num_envs = args_cli.num_envs
    n_success = 0
    n_attempts = 0
    success_key = args_cli.success_key
    ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    print(f"[collect] success_key={success_key!r}")

    env_data = [defaultdict(list) for _ in range(num_envs)]

    obs_raw, _ = env.reset()
    policy_obs = obs_raw["policy"]
    formatter.reset()

    warmup_act = isaac_env.cfg.warmup_action(isaac_env)

    pbar = tqdm(total=args_cli.target_episodes, desc="Successful episodes")

    with torch.inference_mode():
        action_seq = None
        chunk_start_obs = None

        step_in_episode = 0

        while n_success < args_cli.target_episodes:
            # ── Record pre-step data for all envs ────────────────────────
            for i in range(num_envs):
                env_data[i]["ee_pose"].append(policy_obs["ee_pose"][i].cpu().numpy())
                env_data[i]["hand_joint_pos"].append(policy_obs["hand_joint_pos"][i].cpu().numpy())
                env_data[i]["arm_joint_pos"].append(policy_obs["arm_joint_pos"][i].cpu().numpy())
                for key in args_cli.privileged_keys:
                    if key in policy_obs:
                        env_data[i][key].append(policy_obs[key][i].cpu().numpy())

            # ── CFM inference (re-infer every action_horizon steps) ───────
            if step_in_episode % action_horizon == 0:
                obs_dict = formatter.format(policy_obs)
                result = policy.predict_action(obs_dict)
                action_seq = result["action_pred"]  # (B, horizon, A)

                if chunk_relative:
                    chunk_start_obs = torch.cat(
                        [policy_obs[k].float() for k in obs_keys], dim=-1
                    )

            action_idx = min(step_in_episode % action_horizon, action_seq.shape[1] - 1)

            if chunk_relative:
                current_obs = torch.cat([policy_obs[k].float() for k in obs_keys], dim=-1)
                action_step = (chunk_start_obs + action_seq[:, action_idx] - current_obs).clone()
            else:
                action_step = action_seq[:, action_idx].clone()

            # Record action before step
            for i in range(num_envs):
                env_data[i]["actions"].append(action_step[i].cpu().numpy())

            # ── Read metrics BEFORE env.step (auto-reset clears them) ─────
            metrics = isaac_env.metrics.get_metrics()
            _zero = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_success |= torch.as_tensor(
                metrics.get(success_key, _zero), device=device
            ).bool()

            # ── Step ──────────────────────────────────────────────────────
            obs_raw, reward, terminated, truncated, _ = env.step(action_step)
            formatter.update_action(action_step)
            new_policy_obs = obs_raw["policy"]

            # Record post-step data
            for i in range(num_envs):
                r = reward[i]
                env_data[i]["rewards"].append(float(r.item() if hasattr(r, "item") else r))
                env_data[i]["dones"].append(
                    bool(terminated[i].cpu() or truncated[i].cpu())
                )

            # ── Handle done envs ──────────────────────────────────────────
            terminated_cpu = terminated.cpu().bool()
            truncated_cpu  = truncated.cpu().bool()
            done_mask = terminated_cpu | truncated_cpu

            if done_mask.any():
                for i in done_mask.nonzero(as_tuple=True)[0].tolist():
                    n_attempts += 1
                    is_success = bool(ever_success[i])

                    if is_success and n_success < args_cli.target_episodes:
                        episode_data = {k: np.array(v) for k, v in env_data[i].items()}
                        zarr_path = _save_episode_zarr(
                            args_cli.output_dir, n_success, episode_data
                        )
                        n_success += 1
                        pbar.update(1)

                    pbar.set_postfix(attempts=n_attempts, rate=f"{n_success/n_attempts:.1%}")

                    # Reset per-env buffers
                    env_data[i] = defaultdict(list)
                    ever_success[i] = False

                # Reset formatter history for done envs (fills frames with post-reset obs)
                formatter.reset_envs(done_mask.to(device), new_policy_obs)
                step_in_episode = -1  # will be incremented to 0 below

            policy_obs = new_policy_obs
            step_in_episode += 1

    pbar.close()
    env.close()

    total = sum(
        1 for ep_i in range(n_success)
        for _ in [os.path.join(args_cli.output_dir, f"episode_{ep_i}")]
        if os.path.isdir(_)
    )
    print(f"\n[collect] Done. {n_success} successful episodes out of {n_attempts} attempts"
          f"  ({n_success/max(n_attempts,1):.1%} success rate)")
    print(f"[collect] Data saved to: {args_cli.output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
