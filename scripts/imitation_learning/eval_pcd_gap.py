"""
Measure the performance gap when the noise policy and/or diffusion base receive
rendered (camera-depth) vs synthetic (mesh-based) point clouds.

Five modes
----------
Mode            | Diffusion base PCD  | Actor (noise policy) PCD embedding
synthetic_both  | mesh_pc (synthetic) | mesh_pc (synthetic)  <- baseline
rendered_actor  | mesh_pc (synthetic) | seg_pc (rendered)    <- isolates noise policy gap
rendered_base   | seg_pc (rendered)   | mesh_pc (synthetic)  <- isolates base policy gap
rendered_both   | seg_pc (rendered)   | seg_pc (rendered)    <- full gap
no_noise        | seg_pc (rendered)   | (no noise policy)    <- diffusion base only, rendered

All modes run with DISTILL_MODE (cameras on) so video is always recorded.
In DISTILL_MODE: seg_pc = camera-depth rendered PC, mesh_pc = synthetic mesh PC.

Usage (inside container)
------------------------
# Baseline — synthetic PCD everywhere, no cameras needed:
./uwlab.sh -p scripts/imitation_learning/eval_pcd_gap.py \\
    --task UW-FrankaLeap-PourBottle-IkRel-v0 \\
    --diffusion_checkpoint logs/bc_cfm_pcd_bourbon_0312 \\
    --ppo_checkpoint logs/rfs/PourBottle_0315_1939/model_000350.zip \\
    --asymmetric_ac \\
    --mode synthetic_both \\
    --num_episodes 50

# Isolate the noise policy — base stays synthetic, actor gets rendered PCD:
./uwlab.sh -p scripts/imitation_learning/eval_pcd_gap.py \\
    --task UW-FrankaLeap-PourBottle-IkRel-v0 \\
    --diffusion_checkpoint logs/bc_cfm_pcd_bourbon_0312 \\
    --ppo_checkpoint logs/rfs/PourBottle_0315_1939/model_000350.zip \\
    --asymmetric_ac \\
    --mode rendered_actor \\
    --num_episodes 50

# Full gap — everything gets rendered PCD:
./uwlab.sh -p scripts/imitation_learning/eval_pcd_gap.py \\
    --task UW-FrankaLeap-PourBottle-IkRel-v0 \\
    --diffusion_checkpoint logs/bc_cfm_pcd_bourbon_0312 \\
    --ppo_checkpoint logs/rfs/PourBottle_0315_1939/model_000350.zip \\
    --asymmetric_ac \\
    --mode rendered_both \\
    --num_episodes 50
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

# -------------------------------------------------------------------
# Mode config: what PCD key each component reads, and whether
# DISTILL_MODE (cameras on, seg_pc = rendered) is needed.
# -------------------------------------------------------------------
MODE_CFG = {
    #                     distill  base_pcd_key    actor_pcd_key
    # DISTILL_MODE is on for all modes so cameras are always available for video.
    # In DISTILL_MODE: seg_pc = rendered, mesh_pc = synthetic.
    # actor_pcd_key=None means no noise policy (zero noise passed to diffusion base).
    "synthetic_both":  dict(distill=True,  base_pcd_key="mesh_pc", actor_pcd_key="mesh_pc"),
    "rendered_actor":  dict(distill=True,  base_pcd_key="mesh_pc", actor_pcd_key="seg_pc"),
    "rendered_base":   dict(distill=True,  base_pcd_key="seg_pc",  actor_pcd_key="mesh_pc"),
    "rendered_both":   dict(distill=True,  base_pcd_key="seg_pc",  actor_pcd_key="seg_pc"),
    "no_noise":        dict(distill=True,  base_pcd_key="seg_pc",  actor_pcd_key=None),
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--diffusion_checkpoint", type=str, required=True,
                    help="Dir containing checkpoints/ (tries best.ckpt then latest.ckpt)")
parser.add_argument("--ppo_checkpoint", type=str, required=False, default=None,
                    help="Path to SB3 PPO .zip file (not required for no_noise mode)")
parser.add_argument("--task", type=str, default="UW-FrankaLeap-PourBottle-IkRel-v0")
parser.add_argument("--mode", type=str, choices=list(MODE_CFG), required=True,
                    help="Which PCD mode to evaluate (see module docstring for details).")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Where to write results.json and videos/. "
                         "Defaults to <ppo_checkpoint_dir>/eval_pcd_gap/<mode>/")
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--num_warmup_steps", type=int, default=10)
parser.add_argument("--asymmetric_ac", action="store_true", default=False,
                    help="Use AsymmetricActorCriticPolicy (required for 1939-style checkpoints).")
parser.add_argument("--n_residual", type=int, default=0,
                    help="Leading PPO output dims that are residual (0 = DSRL/pure noise, 22 = RFS).")
parser.add_argument("--arm_residual_xyz", type=float, default=0.005)
parser.add_argument("--arm_residual_rpy", type=float, default=0.01)
parser.add_argument("--hand_residual", type=float, default=0.1)
parser.add_argument("--finger_smooth_alpha", type=float, default=0.7,
                    help="LPFilter alpha for finger dims (matches dsrl_cfg.yaml). 1.0 = disabled.")
parser.add_argument("--video_fps", type=int, default=10,
                    help="FPS for saved videos. Should match real-world control rate (default 10Hz).")
parser.add_argument("--horizon", type=int, default=None,
                    help="Override episode horizon (max steps). Defaults to task config value.")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if MODE_CFG[args_cli.mode]["distill"]:
    args_cli.enable_cameras = True  # cameras required for rendered PCD

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- everything below requires the sim to be running ---

import json
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    parse_franka_leap_env_cfg,
    DISTILL_MODE,
    EVAL_MODE,
)

from wrapper import _load_cfm_checkpoint, LPFilter  # handles both old and new checkpoint formats

if args_cli.asymmetric_ac:
    from asymmetric_policy import AsymmetricActorCriticPolicy


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class _RolloutBuffer(DictRolloutBuffer):
    def __init__(self, *args, gpu_buffer=False, **kwargs):
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def _format_diffusion_obs(
    policy_obs: dict,
    pcd_key: str,           # "seg_pc" (synthetic in normal mode, rendered in DISTILL) or "mesh_pc" (always synthetic)
    downsample_points: int,
    device: torch.device,
) -> dict:
    """Build the obs dict that CFMPCDPolicy.predict_action expects.

    pcd_key selects which point cloud from policy_obs to feed to the diffusion base.
    In DISTILL_MODE both 'seg_pc' (rendered) and 'mesh_pc' (synthetic) are available.
    In normal mode only 'seg_pc' exists (it is synthetic).
    """
    ee_pose    = policy_obs["ee_pose"].float()
    hand_joints = policy_obs["joint_pos"][:, 7:].float()
    agent_pos  = torch.cat([ee_pose, hand_joints], dim=-1).unsqueeze(1)

    pcd = policy_obs[pcd_key].float()
    N = pcd.shape[-1]
    if N > downsample_points:
        perm = torch.randperm(N, device=device)[:downsample_points]
        pcd = pcd[:, :, perm]
    pcd = pcd.unsqueeze(1)
    return {"agent_pos": agent_pos, "seg_pc": pcd}


def _compute_pcd_embedding(
    policy_obs: dict,
    diffusion,
    pcd_key: str,           # which point cloud to embed
    downsample_points: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the frozen CFM PointNet on the chosen PCD and return the embedding (B, pcd_feat_dim)."""
    diff_obs = _format_diffusion_obs(policy_obs, pcd_key, downsample_points, device)
    nobs = diffusion.normalizer.normalize(diff_obs)
    pcd_obs = {k: nobs[k][:, 0] for k in diffusion.obs_encoder.pcd_keys}
    return diffusion.obs_encoder.encode_pcd_only(pcd_obs)


def _format_actor_obs_asymmetric(
    policy_obs: dict,
    diffusion,
    actor_pcd_key: str,
    downsample_points: int,
    device: torch.device,
) -> dict:
    """Build the actor obs dict for AsymmetricActorCriticPolicy.

    actor_pcd_key controls the PCD the actor's PointNet embedding is computed from:
      "seg_pc"  -> synthetic (normal mode) or rendered (DISTILL_MODE)
      "mesh_pc" -> always synthetic (only present in DISTILL_MODE)
    """
    pcd_emb = _compute_pcd_embedding(policy_obs, diffusion, actor_pcd_key, downsample_points, device)
    return {
        "actor_pcd_emb":        pcd_emb[0].cpu().numpy(),
        "actor_ee_pose":        policy_obs["ee_pose"][0].cpu().numpy(),
        "actor_hand_joint_pos": policy_obs["hand_joint_pos"][0].cpu().numpy(),
    }


def _format_actor_obs_flat(policy_obs: dict) -> np.ndarray:
    """Flat state obs for standard (non-asymmetric) MultiInputPolicy."""
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


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _get_camera_frame(isaac_env) -> np.ndarray | None:
    """Read RGB from train_camera for env 0. Returns (H, W, 3) uint8 or None."""
    try:
        rgba = isaac_env.scene["train_camera"].data.output["rgb"][0].cpu().numpy()
        return rgba[:, :, :3]
    except Exception:
        return None


def _write_video(frames: list, path: str, fps: int = 10):
    if not frames:
        return
    import imageio
    imageio.mimsave(path, frames, fps=fps)
    print(f"[eval_pcd_gap] video -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mode_cfg       = MODE_CFG[args_cli.mode]
    use_distill    = mode_cfg["distill"]
    base_pcd_key   = mode_cfg["base_pcd_key"]   # PCD fed to diffusion base
    actor_pcd_key  = mode_cfg["actor_pcd_key"]  # PCD embedded for actor; None = no noise policy
    no_noise       = actor_pcd_key is None

    print(f"\n[eval_pcd_gap] mode={args_cli.mode}")
    print(f"  diffusion base reads: {base_pcd_key}")
    if no_noise:
        print(f"  actor: disabled (zero noise)")
    elif args_cli.asymmetric_ac:
        print(f"  actor embedding reads: {actor_pcd_key}")
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    diffusion, _ = _load_cfm_checkpoint(args_cli.diffusion_checkpoint, device)
    # downsample_points comes from the shape_meta baked into the loaded policy
    downsample_points = diffusion.obs_encoder.shape_meta["obs"]["seg_pc"]["shape"][-1]
    diffusion_horizon = diffusion.horizon
    action_dim        = diffusion.action_dim

    if not no_noise:
        if args_cli.ppo_checkpoint is None:
            raise ValueError(f"--ppo_checkpoint is required for mode '{args_cli.mode}'")
        custom_objects = {"rollout_buffer_class": _RolloutBuffer}
        if args_cli.asymmetric_ac:
            custom_objects["policy_class"] = AsymmetricActorCriticPolicy
        agent = PPO.load(args_cli.ppo_checkpoint, device=device, custom_objects=custom_objects)
    else:
        agent = None

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

    run_mode = DISTILL_MODE if use_distill else EVAL_MODE
    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        run_mode,
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
    isaac_env   = env.unwrapped
    episode_steps = int(isaac_env.max_episode_length)

    _base_dir = os.path.dirname(args_cli.ppo_checkpoint) if args_cli.ppo_checkpoint else args_cli.diffusion_checkpoint
    output_dir = args_cli.output_dir or os.path.join(_base_dir, "eval_pcd_gap", args_cli.mode)
    os.makedirs(output_dir, exist_ok=True)
    if use_distill:
        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)

    print(f"[eval_pcd_gap] output -> {output_dir}")

    finger_filter = LPFilter(alpha=args_cli.finger_smooth_alpha).to(device) \
        if args_cli.finger_smooth_alpha < 1.0 else None

    n_success = 0
    results = []

    with torch.inference_mode():
        pbar = tqdm(range(args_cli.num_episodes), desc=f"eval [{args_cli.mode}]")
        for ep_idx in pbar:
            obs_raw, _ = env.reset()
            if finger_filter is not None:
                finger_filter.reset()
                # seed filter with current finger positions so first action isn't jarring
                current_fingers = isaac_env.scene["robot"].data.joint_pos[:, 7:]
                finger_filter(current_fingers)
            warmup_act = isaac_env.cfg.warmup_action(isaac_env)
            for _ in range(args_cli.num_warmup_steps):
                obs_raw, _, _, _, _ = env.step(warmup_act)

            frames = []

            for _ in range(episode_steps):
                policy_obs = obs_raw["policy"]

                # --- actor obs: which PCD the noise policy sees ---
                if no_noise:
                    residual_raw = torch.zeros(1, args_cli.n_residual, device=device)
                    noise = torch.randn(1, diffusion_horizon, action_dim, device=device)
                elif args_cli.asymmetric_ac:
                    actor_obs = _format_actor_obs_asymmetric(
                        policy_obs, diffusion, actor_pcd_key, downsample_points, device
                    )
                    ppo_action_np, _ = agent.predict(actor_obs)
                    ppo_action = torch.as_tensor(ppo_action_np, device=device).unsqueeze(0)
                    residual_raw = ppo_action[:, :args_cli.n_residual]
                    noise = ppo_action[:, args_cli.n_residual:].reshape(-1, diffusion_horizon, action_dim)
                else:
                    ppo_action_np, _ = agent.predict(_format_actor_obs_flat(policy_obs)[0])
                    ppo_action = torch.as_tensor(ppo_action_np, device=device).unsqueeze(0)
                    residual_raw = ppo_action[:, :args_cli.n_residual]
                    noise = ppo_action[:, args_cli.n_residual:].reshape(-1, diffusion_horizon, action_dim)

                # --- diffusion base: which PCD the base policy sees ---
                diff_obs = _format_diffusion_obs(policy_obs, base_pcd_key, downsample_points, device)
                base = diffusion.predict_action(diff_obs, noise)["action_pred"][:, 0]

                env_action = base.clone()
                if finger_filter is not None:
                    env_action[:, 6:] = finger_filter(env_action[:, 6:])
                if args_cli.n_residual > 0:
                    residual = (residual_raw + 1.0) / 2.0 * (
                        residual_upper - residual_lower
                    ) + residual_lower
                    env_action[:, :6] += residual[:, :6]
                    env_action[:, 6:] += residual[:, 6:]
                env_action[:, :3] = env_action[:, :3].clamp(-0.03, 0.03)
                env_action[:, 3:6] = env_action[:, 3:6].clamp(-0.05, 0.05)

                success_before = isaac_env.cfg.is_success(isaac_env)
                obs_raw, _, terminated, truncated, _ = env.step(env_action)

                if use_distill:
                    frame = _get_camera_frame(isaac_env)
                    if frame is not None:
                        frames.append(frame)

                if terminated.any() or truncated.any():
                    break

            success = bool(success_before.cpu()[0])
            if success:
                n_success += 1
            results.append({"episode": ep_idx, "success": success})
            pbar.set_postfix(success=n_success, rate=f"{100*n_success/(ep_idx+1):.1f}%")

            if use_distill and frames:
                tag = "success" if success else "fail"
                video_path = os.path.join(output_dir, "videos", f"ep{ep_idx:03d}_{tag}.mp4")
                _write_video(frames, video_path, fps=args_cli.video_fps)

    success_rate = n_success / args_cli.num_episodes
    summary = {
        "mode":         args_cli.mode,
        "base_pcd_key": base_pcd_key,
        "actor_pcd_key": "n/a (no noise policy)" if no_noise else (actor_pcd_key if args_cli.asymmetric_ac else "n/a (flat state actor)"),
        "n_success":    n_success,
        "n_episodes":   args_cli.num_episodes,
        "success_rate": success_rate,
        "episodes":     results,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[eval_pcd_gap] {args_cli.mode}: {n_success}/{args_cli.num_episodes} "
          f"({100*success_rate:.1f}%) -> {results_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
