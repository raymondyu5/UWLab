"""
Collect CFM training data by rolling out a trained RL (SB3 PPO) policy.

The RL policy was trained on privileged state observations (object pose, contact, etc.)
via train_stage2_ppo.py.  This script runs the policy on the non-PPO task variant in
RL_MODE so that mesh-based seg_pc is available, concatenates the same obs_keys the
policy was trained on to form its flat input, and saves point-cloud + proprioception
(no privileged info) for downstream CFM training.

Usage:
    ./uwlab.sh -p scripts/reinforcement_learning/rialto/collect_rl_rollouts.py \\
        --task UW-FrankaLeap-GraspCube-JointAbs-PPO-Collect-v0 \\
        --checkpoint logs/rialto/stage2/cube_grasp_vf01_cw30/stage1_0516_1537/model_000250.zip \\
        --obs_keys arm_joint_pos hand_joint_pos manipulated_object_pose target_object_pose contact_obs object_in_tip \\
        --output_dir data_storage/rialto_rl/cube_grasp \\
        --target_episodes 1000 --num_envs 1024 --headless

With video + scatter plots (use small num_envs):
    ./uwlab.sh -p scripts/reinforcement_learning/rialto/collect_rl_rollouts.py \\
        --task ... --checkpoint ... --num_envs 8 --target_episodes 10 \\
        --record_video --enable_cameras --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Collect sim rollouts from an RL policy for CFM training."
)
parser.add_argument("--task", type=str, required=True,
                    help="Non-PPO IsaacLab task name (mesh-based seg_pc + dict obs).")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to SB3 PPO .zip checkpoint.")
parser.add_argument("--obs_keys", nargs="+",
                    default=["arm_joint_pos", "hand_joint_pos", "manipulated_object_pose"],
                    help="Obs keys to concatenate into flat policy input (must match training).")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory to write zarr episodes into.")
parser.add_argument("--target_episodes", type=int, default=1000,
                    help="Number of successful episodes to collect.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--success_key", type=str, default="is_lifted",
                    help="Metric key used as success criterion (e.g. is_lifted, is_success).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--record_video", action="store_true", default=False,
                    help="Record a preview MP4 video (requires --enable_cameras; use small --num_envs).")
parser.add_argument("--video_steps", type=int, default=2000,
                    help="Max steps to record for the preview video.")
parser.add_argument("--eval_dir", type=str, default=None,
                    help="Directory for eval outputs (scatter plots + video). "
                         "Default: <output_dir>/eval/")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.record_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── all Isaac-dependent imports after AppLauncher ──────────────────────────

import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
import gymnasium as gym
from tqdm import tqdm
from stable_baselines3 import PPO

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap import (
    RL_MODE,
    EVAL_MODE,
    parse_franka_leap_env_cfg,
)


# ── Scatter plot ───────────────────────────────────────────────────────────

def _save_scatter_plot(output_dir: str, xs, ys, successes) -> None:
    xs = np.array(xs)
    ys = np.array(ys)
    successes = np.array(successes, dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 6))
    for mask, color, label in [
        (~successes, "red", "fail"),
        (successes, "green", "success"),
    ]:
        if mask.any():
            ax.scatter(xs[mask], ys[mask], c=color, marker="x", s=40,
                       linewidths=1.2, label=label, alpha=0.7)

    n_pos = int(successes.sum())
    n_tot = len(successes)
    rate = 100 * n_pos / n_tot if n_tot else 0.0
    ax.set_title(f"Spawn outcomes  {n_pos}/{n_tot} success ({rate:.1f}%)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "scatter_success.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[collect_rl] scatter plot -> {out_path}")


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

    eval_dir = args_cli.eval_dir or os.path.join(args_cli.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # ── 1. Load RL policy ─────────────────────────────────────────────────
    print(f"[collect_rl] Loading checkpoint: {args_cli.checkpoint}")
    model = PPO.load(args_cli.checkpoint, device=device)
    print(f"[collect_rl] obs_keys={args_cli.obs_keys}")

    # ── 2. Create Isaac Sim env ───────────────────────────────────────────
    # EVAL_MODE enables the fixed_camera so env.render() works for video.
    # Mesh-based seg_pc works in both RL_MODE and EVAL_MODE.
    run_mode = EVAL_MODE if args_cli.record_video else RL_MODE
    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        run_mode,
        device=str(device),
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed

    if run_mode == EVAL_MODE:
        # train_camera is for rendered seg_pc (DISTILL_MODE); not needed here.
        env_cfg.scene.train_camera = None
        if hasattr(env_cfg.events, "reset_camera"):
            env_cfg.events.reset_camera = None

    render_mode = "rgb_array" if args_cli.record_video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    isaac_env = env.unwrapped

    # ── 3. Collection loop ────────────────────────────────────────────────
    num_envs = args_cli.num_envs
    n_success = 0
    n_attempts = 0
    # Per-round counters (one round = all num_envs episodes done together).
    round_success = 0
    round_attempts = 0
    success_key = args_cli.success_key
    ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    env_data = [defaultdict(list) for _ in range(num_envs)]

    obs_raw, _ = env.reset()
    policy_obs = obs_raw["policy"]

    # ── Scatter tracking: initial object XY per env vs success ───────────
    scatter_xs: list[float] = []
    scatter_ys: list[float] = []
    scatter_successes: list[bool] = []
    init_pos_local: np.ndarray | None = None
    env_origins = None
    try:
        env_origins = isaac_env.scene.env_origins
        init_pos_w = isaac_env.scene["grasp_object"].data.root_pos_w.clone()
        init_pos_local = (init_pos_w - env_origins).cpu().numpy().copy()
    except (KeyError, AttributeError):
        print("[collect_rl] Warning: could not find 'grasp_object' in scene; scatter plot will be skipped.")

    # ── Video recording ───────────────────────────────────────────────────
    video_frames: list[np.ndarray] = []
    video_step_count = 0

    if not isinstance(policy_obs, dict):
        raise RuntimeError(
            f"policy_obs is a {type(policy_obs).__name__}, not a dict. "
            "The task must use concatenate_terms=False (use a -PPO-Collect-v0 task, "
            "not -PPO-v0 or a plain task that happens to be flat)."
        )
    if "seg_pc" not in policy_obs:
        raise RuntimeError(
            f"'seg_pc' not in policy_obs keys: {list(policy_obs.keys())}. "
            "The task config must include a seg_pc obs term "
            "(use a -PPO-Collect-v0 task, not the plain -PPO-v0 variant)."
        )
    # Determine the key order for building the flat policy input.
    #
    # For -PPO-Collect-v0 tasks (StateCollectCfg): the dict contains exactly the
    # obs_keys + seg_pc, in the insertion order from the base StateCfg __post_init__.
    # That insertion order IS the order the policy was trained on (StateCfg
    # concatenate_terms=True iterates the same dict in the same order). We must
    # use the dict's natural order, not the obs_keys argument order — they can differ
    # (e.g. target_object_pose is inserted before manipulated_object_pose in all grasp
    # tasks, but GRASP_OBS lists them the other way around).
    #
    # For non-PPO tasks (soccer, cup_pour): the dict also contains ee_pose, joint_pos,
    # and other unwanted keys. The policy was trained via ProprioVecWrapper which
    # selected and ordered by obs_keys. Fall back to obs_keys order for those.
    policy_obs_keys = [k for k in policy_obs.keys() if k != "seg_pc"]
    if set(policy_obs_keys) == set(args_cli.obs_keys):
        # PPO-Collect task: dict order matches StateCfg training order exactly.
        flat_keys = policy_obs_keys
    else:
        # Non-PPO task: extra keys present; use obs_keys order (ProprioVecWrapper).
        flat_keys = args_cli.obs_keys
    obs_dim = sum(int(policy_obs[k].shape[-1]) for k in flat_keys)
    print(f"[collect_rl] policy_obs keys: {list(policy_obs.keys())}")
    print(f"[collect_rl] flat policy input order: {flat_keys}  (obs_dim={obs_dim})")

    pbar = tqdm(total=args_cli.target_episodes, desc="Successful episodes")

    with torch.inference_mode():
        while n_success < args_cli.target_episodes:
            # ── Record pre-step obs for each env ─────────────────────────
            for i in range(num_envs):
                for key in ("seg_pc", "arm_joint_pos", "hand_joint_pos"):
                    env_data[i][key].append(policy_obs[key][i].cpu().numpy())

            # ── Build flat policy input in training obs order ─────────────
            parts = [policy_obs[k].float().reshape(num_envs, -1) for k in flat_keys]
            flat_obs = torch.cat(parts, dim=-1).cpu().numpy()  # (num_envs, obs_dim)

            # ── RL policy inference ───────────────────────────────────────
            action, _ = model.predict(flat_obs, deterministic=True)
            action_t = torch.tensor(action, dtype=torch.float32, device=device)

            # Record action before step
            for i in range(num_envs):
                env_data[i]["actions"].append(action[i])

            # ── Read metrics BEFORE env.step (auto-reset clears them) ─────
            metrics = isaac_env.metrics.get_metrics()
            _zero = torch.zeros(num_envs, dtype=torch.bool, device=device)
            ever_success |= torch.as_tensor(
                metrics.get(success_key, _zero), device=device
            ).bool()

            # ── Video frame ───────────────────────────────────────────────
            if args_cli.record_video and video_step_count < args_cli.video_steps:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)
                video_step_count += 1

            # ── Step ──────────────────────────────────────────────────────
            obs_raw, reward, terminated, truncated, _ = env.step(action_t)
            new_policy_obs = obs_raw["policy"]

            # Record post-step data
            for i in range(num_envs):
                r = reward[i]
                env_data[i]["rewards"].append(float(r.item() if hasattr(r, "item") else r))
                env_data[i]["dones"].append(
                    bool(terminated[i].cpu() or truncated[i].cpu())
                )

            # ── Handle done envs ──────────────────────────────────────────
            done_mask = terminated.cpu().bool() | truncated.cpu().bool()

            if done_mask.any():
                for i in done_mask.nonzero(as_tuple=True)[0].tolist():
                    n_attempts += 1
                    is_success = bool(ever_success[i])

                    # Scatter: record this episode's initial position + outcome.
                    if init_pos_local is not None:
                        scatter_xs.append(float(init_pos_local[i, 0]))
                        scatter_ys.append(float(init_pos_local[i, 1]))
                        scatter_successes.append(is_success)
                        # Capture init_pos for the just-auto-reset env.
                        new_pos_w = isaac_env.scene["grasp_object"].data.root_pos_w[i]
                        init_pos_local[i] = (new_pos_w - env_origins[i]).cpu().numpy()

                    round_success += int(is_success)
                    round_attempts += 1

                    if is_success and n_success < args_cli.target_episodes:
                        episode_data = {k: np.array(v) for k, v in env_data[i].items()}
                        _save_episode_zarr(args_cli.output_dir, n_success, episode_data)
                        n_success += 1
                        pbar.update(1)

                    # Print per-round rate once a full batch of envs has been processed.
                    if round_attempts == num_envs:
                        pbar.set_postfix(
                            attempts=n_attempts,
                            round_rate=f"{round_success/round_attempts:.1%}",
                            total_rate=f"{n_success/n_attempts:.1%}",
                        )
                        round_success = 0
                        round_attempts = 0
                    elif round_attempts < num_envs:
                        pbar.set_postfix(attempts=n_attempts, total_rate=f"{n_success/n_attempts:.1%}")

                    env_data[i] = defaultdict(list)
                    ever_success[i] = False

                    # Stop processing the batch once we've hit the collection target.
                    if n_success >= args_cli.target_episodes:
                        break

            policy_obs = new_policy_obs

    pbar.close()
    env.close()

    print(f"\n[collect_rl] Done. {n_success} successful episodes out of {n_attempts} attempts"
          f"  ({n_success/max(n_attempts, 1):.1%} success rate)")
    print(f"[collect_rl] Data saved to: {args_cli.output_dir}")

    # ── Save eval outputs ─────────────────────────────────────────────────
    if scatter_xs:
        _save_scatter_plot(eval_dir, scatter_xs, scatter_ys, scatter_successes)

    if video_frames:
        import imageio
        cfg = isaac_env.cfg
        video_fps = 1.0 / (cfg.sim.dt * cfg.decimation)
        video_path = os.path.join(eval_dir, "collection_preview.mp4")
        imageio.mimsave(video_path, video_frames, fps=video_fps)
        print(f"[collect_rl] video -> {video_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
