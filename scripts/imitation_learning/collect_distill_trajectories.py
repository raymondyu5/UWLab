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

from collections import deque

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

    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        DISTILL_MODE,
        device=str(device),
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.horizon is not None:
        # horizon = policy steps only; add both warmup budgets so the episode
        # isn't cut short before the policy gets args_cli.horizon full steps.
        total_steps = (args_cli.horizon
                       + env_cfg.num_warmup_steps        # internal env warmup
                       + args_cli.num_warmup_steps)      # external script warmup
        env_cfg.horizon = total_steps
        env_cfg.episode_length_s = total_steps * env_cfg.decimation * env_cfg.sim.dt
    env = gym.make(args_cli.task, cfg=env_cfg)
    isaac_env = env.unwrapped
    episode_steps = int(isaac_env.max_episode_length)

    os.makedirs(args_cli.output_dir, exist_ok=True)
    print(f"[collect_distill] Output: {args_cli.output_dir}")
    print(f"[collect_distill] Task: {args_cli.task}, episodes: {args_cli.num_episodes}")
    print(f"[collect_distill] asymmetric_ac: {args_cli.asymmetric_ac}")
    print(f"[collect_distill] stored_seg_pc_source: {args_cli.stored_seg_pc_source}")
    print(f"[collect_distill] obs_keys: {metadata['obs_keys']}, n_obs_steps: {n_obs_steps}, "
          f"use_action_history: {use_action_history}")

    finger_filter = LPFilter(alpha=args_cli.finger_smooth_alpha).to(device) \
        if args_cli.finger_smooth_alpha < 1.0 else None

    # num_episodes = target number of SUCCESSFUL episodes to save.
    # Loop until that many successes are collected; failures are discarded.
    # Note: PourBottle has no success termination — only failure terminations
    # (bottle_dropped, bottle_too_far, cup_toppled). terminated=True always means
    # failure; success is only possible at truncated=True (timeout). We check
    # is_success() BEFORE env.step() because IsaacLab auto-resets on termination,
    # so querying after would return the reset state.
    n_success = 0
    n_attempts = 0
    with torch.inference_mode():
        pbar = tqdm(total=args_cli.num_episodes, desc="collect_distill")
        while n_success < args_cli.num_episodes:
            n_attempts += 1
            obs_raw, _ = env.reset()

            # Reset formatter and PPO history buffers — mirrors RFSWrapper.reset().
            formatter.reset()
            ppo_history_buf = None
            ppo_past_action_buf = None

            if finger_filter is not None:
                finger_filter.reset()
                current_fingers = isaac_env.scene["robot"].data.joint_pos[:, 7:]
                finger_filter(current_fingers)

            # External warmup — mirrors RFSWrapper._do_warmup().
            # Formatter is NOT updated during warmup (same as RFSWrapper).
            warmup_act = isaac_env.cfg.warmup_action(isaac_env)
            for _ in range(args_cli.num_warmup_steps):
                obs_raw, _, _, _, _ = env.step(warmup_act)

            # Initialize PPO history buffers from post-warmup obs.
            # Mirrors RFSWrapper.reset() after _do_warmup(): fills all n_obs_steps
            # frames with the current obs and initialises past_action with zeros.
            post_warmup_policy_obs = obs_raw["policy"]
            post_warmup_agent_pos = torch.cat(
                [post_warmup_policy_obs[k].float() for k in metadata["obs_keys"]], dim=-1
            )  # (1, D)
            ppo_history_buf = deque(
                [post_warmup_agent_pos.clone() for _ in range(n_obs_steps)],
                maxlen=n_obs_steps,
            )
            if use_action_history and n_obs_steps > 1:
                n_past = n_obs_steps - 1
                ppo_past_action_buf = deque(
                    [torch.zeros(1, action_dim, device=device) for _ in range(n_past)],
                    maxlen=n_past,
                )
            _actor_src_key = "seg_pc" if args_cli.actor_pcd_key == "rendered" else "mesh_pc"
            if args_cli.asymmetric_ac:
                pcd_emb = _compute_pcd_embedding(
                    post_warmup_policy_obs, diffusion, downsample_points, device, src_key=_actor_src_key
                )

            arm_joint_pos_list = []
            hand_joint_pos_list = []
            ee_pose_list = []
            actions_list = []
            noise_list = []
            noise_mean_list = []
            seg_pc_list = []
            seg_pc_mesh_list = []
            arm_joint_pos_target_list = []
            ee_pose_cmd_list = []
            rewards_list = []
            dones_list = []

            for step in range(episode_steps):
                policy_obs = obs_raw["policy"]

                # Build mesh-domain obs for diffusion base policy.
                # In DISTILL_MODE, policy_obs["seg_pc"] is camera-rendered, but the base
                # policy was trained on mesh-based PCD.  Replace seg_pc with mesh_pc so
                # the formatter and diffusion see mesh-domain PCD — identical to what they
                # see in RL_MODE where cameras are disabled and seg_pc IS mesh_pc.
                # All other keys (arm_joint_pos, hand_joint_pos, …) are unchanged.
                mesh_obs = dict(policy_obs)
                mesh_obs["seg_pc"] = policy_obs["mesh_pc"]

                # Format for diffusion inference (maintains n_obs_steps rolling history).
                # Mirrors RFSWrapper.step(): formatter.format(self.last_obs["policy"]).
                diff_obs = formatter.format(mesh_obs)

                # Build PPO obs.
                if args_cli.asymmetric_ac:
                    # Actor obs mirrors RFSWrapper._asymmetric_ppo_obs():
                    #   actor_pcd_emb            — PointNet embedding of current mesh PCD
                    #   actor_agent_pos_history  — flattened n_obs_steps frames of obs_keys
                    #   actor_past_actions_history — flattened (n_obs_steps-1) past actions
                    # Only actor keys are needed here; critic keys are not required for
                    # get_distribution (pi_features_extractor ignores critic_* keys).
                    actor_obs = {
                        "actor_pcd_emb": pcd_emb[0].cpu().numpy(),
                        "actor_agent_pos_history": torch.stack(
                            list(ppo_history_buf), dim=1
                        ).flatten(1)[0].cpu().numpy(),
                    }
                    if use_action_history and n_obs_steps > 1 and ppo_past_action_buf is not None:
                        actor_obs["actor_past_actions_history"] = torch.stack(
                            list(ppo_past_action_buf), dim=1
                        ).flatten(1)[0].cpu().numpy()
                    obs_tensor, _ = agent.policy.obs_to_tensor(actor_obs)
                else:
                    ppo_obs_np = _format_ppo_obs(policy_obs)
                    obs_tensor, _ = agent.policy.obs_to_tensor(ppo_obs_np[0])

                dist = agent.policy.get_distribution(obs_tensor)
                # mean: deterministic prediction — stored as supervision target for BC
                ppo_mean = dist.distribution.loc.squeeze(0)           # (output_dim,)
                # sample: used for env stepping (preserves stochastic behavior)
                ppo_action_np = dist.get_actions(deterministic=False).squeeze(0).detach().cpu().numpy()
                ppo_action_np = np.clip(ppo_action_np, agent.action_space.low, agent.action_space.high)

                ppo_action = torch.as_tensor(ppo_action_np, device=device).unsqueeze(0)

                residual_raw = ppo_action[:, :args_cli.n_residual]
                noise = ppo_action[:, args_cli.n_residual:].reshape(-1, diffusion_horizon, action_dim)
                noise_mean = ppo_mean[args_cli.n_residual:].reshape(diffusion_horizon, action_dim)

                base = diffusion.predict_action(diff_obs, noise)["action_pred"][:, 0]

                env_action = base.clone()
                if finger_filter is not None:
                    env_action[:, 6:] = finger_filter(env_action[:, 6:])
                if args_cli.n_residual > 0:
                    # Mirrors RFSWrapper.step(): action[:, residual_slice] += residual * residual_scale
                    env_action[:, :args_cli.n_residual] += residual_raw * args_cli.residual_scale

                arm_joint_pos_list.append(policy_obs["arm_joint_pos"][0].cpu().numpy())
                hand_joint_pos_list.append(policy_obs["hand_joint_pos"][0].cpu().numpy())
                ee_pose_list.append(policy_obs["ee_pose"][0].cpu().numpy())
                actions_list.append(env_action[0].cpu().numpy())
                noise_list.append(noise[0].cpu().numpy())
                noise_mean_list.append(noise_mean.detach().cpu().numpy())
                rendered_seg_pc = policy_obs["seg_pc"][0].T.cpu().numpy()
                mesh_seg_pc = policy_obs["mesh_pc"][0].T.cpu().numpy()
                if args_cli.stored_seg_pc_source == "mesh":
                    seg_pc_list.append(mesh_seg_pc)
                else:
                    seg_pc_list.append(rendered_seg_pc)
                if args_cli.stored_seg_pc_source == "both":
                    seg_pc_mesh_list.append(mesh_seg_pc)
                ee_pose_cmd_list.append(policy_obs["ee_pose"][0].cpu().numpy())

                success_before = isaac_env.metrics.get_metrics()["is_success"]
                obs_raw, reward, terminated, truncated, _ = env.step(env_action)

                # Advance all histories with post-step obs.
                # Mirrors RFSWrapper.step() cleanup (single-env: no reset_envs needed).
                new_policy_obs = obs_raw["policy"]

                # Formatter past-action history (feeds diff_obs["past_actions"] next step).
                formatter.update_action(env_action)

                # PPO agent-pos history (feeds actor_agent_pos_history next step).
                new_agent_pos = torch.cat(
                    [new_policy_obs[k].float() for k in metadata["obs_keys"]], dim=-1
                )
                ppo_history_buf.append(new_agent_pos)

                # PPO past-action history (feeds actor_past_actions_history next step).
                if ppo_past_action_buf is not None:
                    ppo_past_action_buf.append(env_action.clone())

                # PCD embedding from post-step obs (used as actor_pcd_emb next step).
                if args_cli.asymmetric_ac:
                    pcd_emb = _compute_pcd_embedding(
                        new_policy_obs, diffusion, downsample_points, device, src_key=_actor_src_key
                    )

                arm_joint_pos_target_list.append(new_policy_obs["arm_joint_pos"][0].cpu().numpy())

                r = reward[0]
                rewards_list.append(float(r.item() if hasattr(r, "item") else r))
                done = bool(terminated[0].cpu().numpy() or truncated[0].cpu().numpy())
                dones_list.append(done)

                if terminated.any() or truncated.any():
                    break

            # terminated=True means a failure condition fired — discard regardless of success_before.
            # truncated=True means timeout — success_before determines the outcome.
            episode_failed = bool(terminated[0].cpu().numpy())
            success = (not episode_failed) and bool(success_before[0])

            if success:
                episode_data = {
                    "arm_joint_pos":        np.array(arm_joint_pos_list),
                    "hand_joint_pos":       np.array(hand_joint_pos_list),
                    "ee_pose":              np.array(ee_pose_list),
                    "actions":              np.array(actions_list),
                    "noise":                np.array(noise_list),
                    "noise_mean":           np.array(noise_mean_list),
                    "seg_pc":               np.array(seg_pc_list),
                    "arm_joint_pos_target": np.array(arm_joint_pos_target_list),
                    "ee_pose_cmd":          np.array(ee_pose_cmd_list),
                    "rewards":              np.array(rewards_list),
                    "dones":                np.array(dones_list),
                }
                if args_cli.stored_seg_pc_source == "both":
                    episode_data["seg_pc_mesh"] = np.array(seg_pc_mesh_list)
                _save_episode_zarr(args_cli.output_dir, n_success, episode_data)
                n_success += 1
                pbar.update(1)

            rate = 100 * n_success / n_attempts
            pbar.set_postfix(saved=n_success, attempts=n_attempts, rate=f"{rate:.1f}%", steps=len(actions_list))
        pbar.close()

    rate = 100 * n_success / n_attempts
    print(f"\n[collect_distill] Done. Saved {n_success} successful episodes from {n_attempts} attempts ({rate:.1f}%)")
    print(f"[collect_distill] Episodes saved to {args_cli.output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
