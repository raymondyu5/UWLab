"""
Collect trajectories from an RFS/DSRL policy in distill_mode for BC/distillation training.

Uses depth-rendered point clouds from randomly sampled camera views (DISTILL_MODE).
Saves episodes in zarr format compatible with ZarrDataset and load_real_episode_zarr.

RFS/DSRL policy: diffusion base + PPO noise (DSRL: no residual, full output is noise).
Requires --enable_cameras for depth rendering.

Obs formatting mirrors RFSWrapper exactly so the PPO checkpoint runs under the same
conditions it was trained in:
  - BCObsFormatter maintains the n_obs_steps rolling history for diffusion inference.
  - Asymmetric actor obs uses PPO history buffers (actor_pcd_emb + actor_agent_pos_history
    + actor_past_actions_history), not a hardcoded subset of keys.
  - Diffusion base policy receives mesh_pc (mapped to the "seg_pc" key) to match RL_MODE,
    where cameras are disabled and policy_obs["seg_pc"] is mesh-based.
  - Camera-rendered seg_pc is stored in zarr as the BC student's training observation.

Usage (inside container):
    ./uwlab.sh -p scripts/imitation_learning/collect_distill_trajectories.py \\
        --headless \\
        --enable_cameras \\
        --task UW-FrankaLeap-PourBottle-IkRel-v0 \\
        --diffusion_checkpoint /path/to/cfm/checkpoint_dir \\
        --ppo_checkpoint /path/to/ppo/model.zip \\
        --output_dir logs/distill_collection \\
        --num_episodes 100

    # Asymmetric AC (actor sees PCD embedding + agent_pos history + past_actions history):
    ./uwlab.sh -p scripts/imitation_learning/collect_distill_trajectories.py \\
        --headless \\
        --enable_cameras \\
        --asymmetric_ac \\
        --task UW-FrankaLeap-PourBottle-IkRel-v0 \\
        --diffusion_checkpoint /path/to/cfm/checkpoint_dir \\
        --ppo_checkpoint /path/to/ppo/model.zip \\
        --output_dir logs/distill_collection \\
        --num_episodes 100
"""

import argparse
import os
import sys

from uwlab.utils.paths import setup_third_party_paths
setup_third_party_paths()

_RFS_PATH = "/workspace/uwlab/scripts/reinforcement_learning/sb3/rfs"
if _RFS_PATH not in sys.path:
    sys.path.insert(0, _RFS_PATH)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Collect RFS/DSRL policy trajectories in distill_mode (depth-rendered PC) for BC training."
)
parser.add_argument("--diffusion_checkpoint", type=str, required=True,
                    help="Dir containing checkpoints/latest.ckpt (IsaacLab BC format)")
parser.add_argument("--ppo_checkpoint", type=str, required=True,
                    help="Path to SB3 PPO .zip file")
parser.add_argument("--task", type=str, default="UW-FrankaLeap-GraspPinkCup-IkRel-v0",
                    help="Task with seg_pc (GraspPinkCup, PourBottle, GraspBottle). Must use IkRel-v0.")
parser.add_argument("--output_dir", type=str, default="logs/distill_collection")
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel envs. Only used in mesh mode (RL_MODE, no cameras). Forced to 1 in rendered mode.",
)
parser.add_argument("--num_warmup_steps", type=int, default=10)
parser.add_argument(
    "--stored_seg_pc_source",
    type=str,
    default="rendered",
    choices=("rendered", "mesh", "both"),
    help=(
        "Which point cloud to store under dataset key 'seg_pc': "
        "'rendered' (camera, default), 'mesh' (synthetic mesh), or 'both' "
        "(store rendered in 'seg_pc' and mesh in extra key 'seg_pc_mesh')."
    ),
)
parser.add_argument("--n_residual", type=int, default=0,
                    help="Number of leading PPO output dims that are residual (0 = pure noise). "
                         "Must match rfs.residual_dims end value from training config.")
parser.add_argument("--residual_scale", type=float, default=0.01,
                    help="Scale applied to PPO residual output before adding to base action. "
                         "Must match rfs.residual_scale from training config.")

parser.add_argument("--asymmetric_ac", action="store_true", default=False,
                    help="Load AsymmetricActorCriticPolicy; actor sees PCD embedding + agent_pos history.")
parser.add_argument(
    "--actor_pcd_key",
    type=str,
    default="mesh",
    choices=("mesh", "rendered"),
    help="PCD source for the actor embedding: 'mesh' (mesh_pc, default) or 'rendered' (seg_pc from cameras).",
)
parser.add_argument("--finger_smooth_alpha", type=float, default=0.7,
                    help="LPFilter alpha for finger dims (matches dsrl_cfg.yaml). 1.0 = disabled.")
parser.add_argument("--horizon", type=int, default=None,
                    help="Override episode horizon (max steps). Defaults to task config value.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from collections import deque, defaultdict

import gymnasium as gym
import numpy as np
import torch
import zarr
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    parse_franka_leap_env_cfg,
    DISTILL_MODE,
    RL_MODE,
)

from wrapper import _load_cfm_checkpoint, LPFilter  # handles both old and new checkpoint formats
from uwlab.eval.bc_obs_formatter import BCObsFormatter

if args_cli.asymmetric_ac:
    from asymmetric_policy import AsymmetricActorCriticPolicy


class _RolloutBuffer(DictRolloutBuffer):
    def __init__(self, *args, gpu_buffer=False, **kwargs):
        super().__init__(*args, **kwargs)


def _compute_pcd_embedding(
    policy_obs: dict, diffusion, downsample_points: int, device: torch.device,
    src_key: str = "mesh_pc",
) -> torch.Tensor:
    """Encode PCD for the asymmetric actor obs.

    Mirrors RFSWrapper._compute_pcd_embedding.  In DISTILL_MODE, policy_obs["seg_pc"]
    is camera-rendered, but the base policy was trained on mesh-based PCD.  We read
    mesh_pc and normalise it under the "seg_pc" key — exactly what happens in RL_MODE
    where cameras are disabled and policy_obs["seg_pc"] IS the mesh PC.

    src_key: "mesh_pc" (default, matches training) or "seg_pc" (rendered, for gap eval).

    Returns (B, pcd_feat_dim).
    """
    pcd_key = diffusion.obs_encoder.pcd_keys[0]    # "seg_pc"
    pcd = policy_obs[src_key].float()               # (B, 3, N)
    N = pcd.shape[-1]
    if N > downsample_points:
        perm = torch.randperm(N, device=device)[:downsample_points]
        pcd = pcd[:, :, perm]
    nobs_pcd = diffusion.normalizer[pcd_key].normalize(pcd.unsqueeze(1))  # (B, 1, 3, N')
    pcd_obs = {pcd_key: nobs_pcd[:, 0]}                                   # (B, 3, N')
    return diffusion.obs_encoder.encode_pcd_only(pcd_obs).detach()         # (B, pcd_feat_dim)


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
    grp = store.require_group("data")
    for key, arr in data.items():
        grp[key] = np.asarray(arr)
    return zarr_path


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    diffusion, metadata = _load_cfm_checkpoint(args_cli.diffusion_checkpoint, device)
    downsample_points = diffusion.obs_encoder.shape_meta["obs"]["seg_pc"]["shape"][-1]
    diffusion_horizon = diffusion.horizon
    action_dim = diffusion.action_dim
    n_obs_steps = metadata["n_obs_steps"]
    use_action_history = metadata["use_action_history"]

    # BCObsFormatter mirrors what RFSWrapper uses for diffusion inference.
    # obs_keys and image_keys come from the checkpoint so we always match training.
    formatter = BCObsFormatter(
        obs_keys=metadata["obs_keys"],
        image_keys=metadata["image_keys"],
        downsample_points=metadata["downsample_points"],
        device=device,
        n_obs_steps=n_obs_steps,
        action_dim=action_dim if use_action_history else 0,
    )

    custom_objects = {"rollout_buffer_class": _RolloutBuffer}
    if args_cli.asymmetric_ac:
        custom_objects["policy_class"] = AsymmetricActorCriticPolicy
    agent = PPO.load(args_cli.ppo_checkpoint, device=device, custom_objects=custom_objects)

    # Mode selection:
    #   mesh mode   → RL_MODE (no cameras, seg_pc IS mesh_pc) → supports num_envs > 1
    #   rendered mode → DISTILL_MODE (cameras required)       → forces num_envs=1
    use_rl_mode = (args_cli.actor_pcd_key == "mesh")
    num_envs = args_cli.num_envs if use_rl_mode else 1
    if not use_rl_mode and args_cli.num_envs > 1:
        print(f"[collect_distill] WARNING: rendered mode requires cameras; forcing num_envs=1 (requested {args_cli.num_envs})")
    run_mode = RL_MODE if use_rl_mode else DISTILL_MODE

    # In RL_MODE, seg_pc is already mesh — no separate mesh_pc key exists.
    # In DISTILL_MODE rendered mode, actor and diffusion both see rendered seg_pc.
    # In DISTILL_MODE mesh mode (single env), actor and diffusion see mesh_pc via override.
    _actor_src_key = "seg_pc" if use_rl_mode else ("seg_pc" if args_cli.actor_pcd_key == "rendered" else "mesh_pc")

    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        run_mode,
        device=str(device),
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.horizon is not None:
        total_steps = (args_cli.horizon
                       + env_cfg.num_warmup_steps
                       + args_cli.num_warmup_steps)
        env_cfg.horizon = total_steps
        env_cfg.episode_length_s = total_steps * env_cfg.decimation * env_cfg.sim.dt
    env = gym.make(args_cli.task, cfg=env_cfg)
    isaac_env = env.unwrapped

    os.makedirs(args_cli.output_dir, exist_ok=True)
    print(f"[collect_distill] Output: {args_cli.output_dir}")
    print(f"[collect_distill] Task: {args_cli.task}, episodes: {args_cli.num_episodes}, num_envs: {num_envs}, mode: {run_mode}")
    print(f"[collect_distill] asymmetric_ac: {args_cli.asymmetric_ac}, actor_pcd_key: {args_cli.actor_pcd_key}")
    print(f"[collect_distill] stored_seg_pc_source: {args_cli.stored_seg_pc_source}")
    print(f"[collect_distill] obs_keys: {metadata['obs_keys']}, n_obs_steps: {n_obs_steps}, "
          f"use_action_history: {use_action_history}")

    finger_filter = LPFilter(alpha=args_cli.finger_smooth_alpha).to(device) \
        if args_cli.finger_smooth_alpha < 1.0 else None

    # --- Initial reset and warmup (applied to all envs together at startup) ---
    # When envs auto-reset mid-collection, they receive the env's internal warmup
    # (env_cfg.num_warmup_steps) but not this external warmup. Acceptable tradeoff.
    obs_raw, _ = env.reset()
    formatter.reset()
    if finger_filter is not None:
        finger_filter.reset()
        finger_filter(isaac_env.scene["robot"].data.joint_pos[:, 7:])

    warmup_act = isaac_env.cfg.warmup_action(isaac_env)  # (num_envs, action_dim)
    for _ in range(args_cli.num_warmup_steps):
        obs_raw, _, _, _, _ = env.step(warmup_act)

    # --- Initialize batch PPO history buffers from post-warmup obs ---
    post_warmup_policy_obs = obs_raw["policy"]
    post_warmup_agent_pos = torch.cat(
        [post_warmup_policy_obs[k].float() for k in metadata["obs_keys"]], dim=-1
    )  # (num_envs, D)
    ppo_history_buf = deque(
        [post_warmup_agent_pos.clone() for _ in range(n_obs_steps)],
        maxlen=n_obs_steps,
    )
    if use_action_history and n_obs_steps > 1:
        n_past = n_obs_steps - 1
        ppo_past_action_buf = deque(
            [torch.zeros(num_envs, action_dim, device=device) for _ in range(n_past)],
            maxlen=n_past,
        )
    else:
        ppo_past_action_buf = None

    if args_cli.asymmetric_ac:
        pcd_emb = _compute_pcd_embedding(
            post_warmup_policy_obs, diffusion, downsample_points, device, src_key=_actor_src_key
        )  # (num_envs, feat_dim)

    # --- Per-env episode state ---
    # Note: PourBottle has no success termination — only failure terminations.
    # terminated=True always means failure; success is checked via is_success BEFORE
    # env.step() because IsaacLab auto-resets on termination.
    env_data = [defaultdict(list) for _ in range(num_envs)]
    ever_grasped   = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ever_lifted    = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ever_near_miss = torch.zeros(num_envs, dtype=torch.bool, device=device)

    n_success = 0
    n_attempts = 0
    n_ever_grasped = 0
    n_ever_lifted = 0
    n_ever_near_miss = 0

    with torch.inference_mode():
        pbar = tqdm(total=args_cli.num_episodes, desc="collect_distill")
        while n_success < args_cli.num_episodes:
            policy_obs = obs_raw["policy"]

            # Build obs for diffusion base policy.
            # In RL_MODE, seg_pc is already mesh — no override needed.
            # In DISTILL_MODE rendered mode, diffusion also sees rendered seg_pc.
            # In DISTILL_MODE mesh mode, override seg_pc with mesh_pc to match training.
            if use_rl_mode:
                mesh_obs = policy_obs
            else:
                mesh_obs = dict(policy_obs)
                if _actor_src_key != "seg_pc":
                    mesh_obs["seg_pc"] = policy_obs["mesh_pc"]

            diff_obs = formatter.format(mesh_obs)

            # Build PPO obs (batch: num_envs).
            if args_cli.asymmetric_ac:
                actor_obs = {
                    "actor_pcd_emb": pcd_emb.cpu().numpy(),  # (num_envs, feat_dim)
                    "actor_agent_pos_history": torch.stack(
                        list(ppo_history_buf), dim=1
                    ).flatten(1).cpu().numpy(),  # (num_envs, n_obs_steps * D)
                }
                if use_action_history and n_obs_steps > 1 and ppo_past_action_buf is not None:
                    actor_obs["actor_past_actions_history"] = torch.stack(
                        list(ppo_past_action_buf), dim=1
                    ).flatten(1).cpu().numpy()
                obs_tensor, _ = agent.policy.obs_to_tensor(actor_obs)
            else:
                ppo_obs_np = _format_ppo_obs(policy_obs)  # (num_envs, D)
                obs_tensor, _ = agent.policy.obs_to_tensor(ppo_obs_np)

            dist = agent.policy.get_distribution(obs_tensor)
            ppo_mean = dist.distribution.loc  # (num_envs, output_dim)
            ppo_action_np = dist.get_actions(deterministic=False).detach().cpu().numpy()
            ppo_action_np = np.clip(ppo_action_np, agent.action_space.low, agent.action_space.high)
            ppo_action = torch.as_tensor(ppo_action_np, device=device)  # (num_envs, output_dim)

            residual_raw = ppo_action[:, :args_cli.n_residual]
            noise      = ppo_action[:, args_cli.n_residual:].reshape(num_envs, diffusion_horizon, action_dim)
            noise_mean = ppo_mean[:, args_cli.n_residual:].reshape(num_envs, diffusion_horizon, action_dim)

            base = diffusion.predict_action(diff_obs, noise)["action_pred"][:, 0]  # (num_envs, action_dim)
            env_action = base.clone()
            if finger_filter is not None:
                env_action[:, 6:] = finger_filter(env_action[:, 6:])
            if args_cli.n_residual > 0:
                env_action[:, :args_cli.n_residual] += residual_raw * args_cli.residual_scale

            # Collect pre-step data for each env.
            for i in range(num_envs):
                env_data[i]["arm_joint_pos"].append(policy_obs["arm_joint_pos"][i].cpu().numpy())
                env_data[i]["hand_joint_pos"].append(policy_obs["hand_joint_pos"][i].cpu().numpy())
                env_data[i]["ee_pose"].append(policy_obs["ee_pose"][i].cpu().numpy())
                env_data[i]["actions"].append(env_action[i].cpu().numpy())
                env_data[i]["noise"].append(noise[i].cpu().numpy())
                env_data[i]["noise_mean"].append(noise_mean[i].detach().cpu().numpy())
                env_data[i]["ee_pose_cmd"].append(policy_obs["ee_pose"][i].cpu().numpy())
                if use_rl_mode:
                    env_data[i]["seg_pc"].append(policy_obs["seg_pc"][i].T.cpu().numpy())
                else:
                    rendered_pc = policy_obs["seg_pc"][i].T.cpu().numpy()
                    mesh_pc     = policy_obs["mesh_pc"][i].T.cpu().numpy()
                    env_data[i]["seg_pc"].append(mesh_pc if args_cli.stored_seg_pc_source == "mesh" else rendered_pc)
                    if args_cli.stored_seg_pc_source == "both":
                        env_data[i]["seg_pc_mesh"].append(mesh_pc)

            # Get metrics before step (IsaacLab auto-resets on termination, so
            # querying after would return reset-state metrics).
            metrics = isaac_env.metrics.get_metrics()
            success_before  = metrics["is_success"]
            ever_grasped   |= torch.as_tensor(metrics["is_grasped"],   device=device).bool()
            ever_lifted    |= torch.as_tensor(metrics["is_healthy_z"],  device=device).bool()
            ever_near_miss |= torch.as_tensor(metrics["is_near_miss"],  device=device).bool()

            obs_raw, reward, terminated, truncated, _ = env.step(env_action)

            # Update histories for all envs (done-env reset handling corrects them below).
            formatter.update_action(env_action)
            new_policy_obs = obs_raw["policy"]
            new_agent_pos = torch.cat(
                [new_policy_obs[k].float() for k in metadata["obs_keys"]], dim=-1
            )  # (num_envs, D)
            ppo_history_buf.append(new_agent_pos)
            if ppo_past_action_buf is not None:
                ppo_past_action_buf.append(env_action.clone())
            if args_cli.asymmetric_ac:
                pcd_emb = _compute_pcd_embedding(
                    new_policy_obs, diffusion, downsample_points, device, src_key=_actor_src_key
                )

            # Collect post-step data for each env.
            for i in range(num_envs):
                env_data[i]["arm_joint_pos_target"].append(new_policy_obs["arm_joint_pos"][i].cpu().numpy())
                r = reward[i]
                env_data[i]["rewards"].append(float(r.item() if hasattr(r, "item") else r))
                env_data[i]["dones"].append(bool(terminated[i].cpu() or truncated[i].cpu()))

            # Detect done envs and handle their episodes.
            terminated_cpu = terminated.cpu().bool() if torch.is_tensor(terminated) else torch.as_tensor(terminated, dtype=torch.bool)
            truncated_cpu  = truncated.cpu().bool()  if torch.is_tensor(truncated)  else torch.as_tensor(truncated,  dtype=torch.bool)
            done_mask = terminated_cpu | truncated_cpu  # (num_envs,)

            if done_mask.any():
                for i in done_mask.nonzero(as_tuple=True)[0].tolist():
                    n_attempts += 1
                    episode_failed = bool(terminated_cpu[i])
                    success = (not episode_failed) and bool(success_before[i])

                    n_ever_grasped   += int(ever_grasped[i])
                    n_ever_lifted    += int(ever_lifted[i])
                    n_ever_near_miss += int(ever_near_miss[i])

                    if success and n_success < args_cli.num_episodes:
                        episode_data = {k: np.array(v) for k, v in env_data[i].items()}
                        _save_episode_zarr(args_cli.output_dir, n_success, episode_data)
                        n_success += 1
                        pbar.update(1)

                    # Reset per-env episode state.
                    env_data[i] = defaultdict(list)
                    ever_grasped[i]   = False
                    ever_lifted[i]    = False
                    ever_near_miss[i] = False

                # Reset formatter history for done envs (fills all frames with post-reset obs).
                reset_mask = done_mask.to(device)
                formatter.reset_envs(reset_mask, new_policy_obs)

                # Reset PPO history buffers: fill all frames for done envs with post-reset obs.
                current_agent_pos = torch.cat(
                    [new_policy_obs[k].float() for k in metadata["obs_keys"]], dim=-1
                )
                for frame in ppo_history_buf:
                    frame[reset_mask] = current_agent_pos[reset_mask]
                if ppo_past_action_buf is not None:
                    for frame in ppo_past_action_buf:
                        frame[reset_mask] = 0.0

                # Reset finger filter state for done envs.
                if finger_filter is not None and finger_filter.y is not None:
                    current_fingers = isaac_env.scene["robot"].data.joint_pos[:, 7:]
                    finger_filter.y[reset_mask] = current_fingers[reset_mask]

                # pcd_emb was already recomputed from new_policy_obs above (which contains
                # post-reset obs for done envs), so no additional handling needed.

            rate       = 100 * n_success        / max(n_attempts, 1)
            grasp_rate = 100 * n_ever_grasped   / max(n_attempts, 1)
            lift_rate  = 100 * n_ever_lifted    / max(n_attempts, 1)
            miss_rate  = 100 * n_ever_near_miss / max(n_attempts, 1)
            pbar.set_postfix(
                saved=n_success, attempts=n_attempts,
                rate=f"{rate:.1f}%", grasp=f"{grasp_rate:.1f}%",
                lift=f"{lift_rate:.1f}%", miss=f"{miss_rate:.1f}%",
            )
        pbar.close()

    rate = 100 * n_success / max(n_attempts, 1)
    print(f"\n[collect_distill] Done. Saved {n_success} successful episodes from {n_attempts} attempts ({rate:.1f}%)")
    print(f"[collect_distill] Episodes saved to {args_cli.output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
