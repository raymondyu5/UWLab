"""
FPO evaluation: periodic deterministic rollouts with success/video/scatter logging.

Mirrors the logic of RFSEvalCallback but as a standalone function (no SB3).

Key correctness points:
  - Metrics are read BEFORE each env.step() call to capture pre-auto-reset state.
  - policy.model is set to eval() for the duration and restored to train() after.
  - fpo_env.reset() is called after eval to restore training state; the returned
    obs is passed back to the training loop as the new current_obs.
"""

import glob
import os
import numpy as np
import torch
import wandb

from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import SpawnCfg


def run_fpo_eval(
    fpo_env,
    spawn_cfg: SpawnCfg,
    log_dir: str,
    t_so_far: int,
    record_video: bool = True,
) -> dict:
    """
    Run one round of FPO evaluation.

    Args:
        fpo_env:      FPOWrapper instance.
        spawn_cfg:    SpawnCfg — fixed poses or random trials.
        log_dir:      Root log dir; output goes to log_dir/eval/step_XXXXXXXXXX/.
        t_so_far:     Current training timestep (for logging and dir naming).
        record_video: Whether to capture and log video.

    Returns:
        new_obs: dict — ppo_obs from fpo_env.reset() after eval, to be used as
                 current_obs in the training loop.
    """
    step_tag   = f"step_{t_so_far:010d}"
    output_dir = os.path.join(log_dir, "eval", step_tag)

    isaac_env = fpo_env.unwrapped
    cfg       = isaac_env.cfg
    video_fps = 1.0 / (cfg.sim.dt * cfg.decimation)

    logger = EvalLogger(
        output_dir,
        record_video=record_video,
        record_plots=True,
        video_fps=video_fps,
    )

    # Switch to eval mode for the duration.
    fpo_env.policy.model.eval()
    fpo_env.policy.obs_encoder.eval()

    try:
        if spawn_cfg.poses:
            _run_fixed_pose_eval(fpo_env, spawn_cfg, logger, isaac_env, record_video)
        else:
            _run_random_eval(fpo_env, spawn_cfg, logger, isaac_env, record_video)
    finally:
        fpo_env.policy.model.train()
        # obs_encoder stays eval (it's frozen).

    results = logger.finalize()

    print(
        f"[FPOEval] t={t_so_far:,}  "
        f"success={results['n_success']}/{results['n_total']} "
        f"({100*results['success_rate']:.1f}%)  "
        f"ever={results['n_success_ever']}/{results['n_total']} "
        f"({100*results['success_rate_ever']:.1f}%)",
        flush=True,
    )

    _log_to_wandb(results, output_dir, t_so_far)

    # Reset env to restore training state; return obs for the training loop.
    new_obs, _ = fpo_env.reset()
    return new_obs


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

def _episode_steps(fpo_env) -> int:
    """Steps remaining in episode after warmup has been consumed."""
    ep_buf = int(fpo_env.unwrapped.episode_length_buf[0].item())
    max_ep = int(fpo_env.unwrapped.max_episode_length)
    return max(1, max_ep - ep_buf)


def _run_fixed_pose_eval(fpo_env, spawn_cfg, logger, isaac_env, record_video):
    num_envs = fpo_env.num_envs
    device   = fpo_env.device

    for pose_idx, pose in enumerate(spawn_cfg.poses):
        record_this = record_video and pose_idx == 0
        fpo_env._collect_substep_frames = record_this

        obs, _ = fpo_env.reset_to_spawn(pose)
        episode_steps = _episode_steps(fpo_env)

        logger.begin_episode(pose.name, {"x": pose.x, "y": pose.y, "yaw": pose.yaw})

        last_success  = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_success  = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_grasped  = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_extra: dict[str, torch.Tensor] = {}
        per_env_success      = [False] * num_envs
        per_env_success_ever = [False] * num_envs
        per_env_grasped      = [False] * num_envs
        per_env_extra: dict[str, list] = {}
        recorded = [False] * num_envs

        for step_idx in range(episode_steps):
            # Read metrics BEFORE step (pre-auto-reset state).
            active = torch.tensor(
                [not recorded[i] for i in range(num_envs)],
                dtype=torch.bool, device=device,
            )
            if active.any():
                metrics_np = _safe_get_metrics(isaac_env)
                if metrics_np is not None:
                    m_s = torch.from_numpy(metrics_np.get(
                        "is_success", np.zeros(num_envs, dtype=bool)
                    )).to(device).bool()
                    m_g = torch.from_numpy(metrics_np.get(
                        "is_grasped", np.zeros(num_envs, dtype=bool)
                    )).to(device).bool()
                    last_success[active] = m_s[active]
                    ever_success[active] |= m_s[active]
                    ever_grasped[active] |= m_g[active]
                    for key, arr in metrics_np.items():
                        if key in ("is_success", "is_grasped"):
                            continue
                        if key not in ever_extra:
                            ever_extra[key]    = torch.zeros(num_envs, dtype=torch.bool, device=device)
                            per_env_extra[key] = [False] * num_envs
                        m = torch.from_numpy(arr).to(device).bool()
                        ever_extra[key][active] |= m[active]

            _, reward, done, _ = fpo_env.step()

            # Capture per-env success at the moment of termination (pre-reset metrics).
            for i in range(num_envs):
                if done[i] and not recorded[i]:
                    per_env_success[i]      = bool(last_success[i])
                    per_env_success_ever[i] = bool(ever_success[i])
                    per_env_grasped[i]      = bool(ever_grasped[i])
                    for key in ever_extra:
                        per_env_extra[key][i] = bool(ever_extra[key][i])
                    recorded[i] = True

            # Record step for logger (env 0 only for EE/obj pose).
            ee_pose  = _get_ee_pose(fpo_env)
            obj_pose = isaac_env.scene["grasp_object"].data.root_pos_w[0].cpu().numpy()
            action   = fpo_env.last_action[0].cpu().numpy() if fpo_env.last_action is not None else np.zeros(fpo_env.cfm_action_dim)
            frame    = fpo_env.last_substep_frames[0] if (fpo_env.last_substep_frames and len(fpo_env.last_substep_frames) > 0) else None
            logger.record_step(ee_pose=ee_pose, object_pose=obj_pose, action=action, frame=frame)

            if all(recorded):
                break

        n_success      = sum(per_env_success)
        n_success_ever = sum(per_env_success_ever)
        n_grasped      = sum(per_env_grasped)
        extra_counts   = {k: int(sum(vals)) for k, vals in per_env_extra.items()}
        extra_counts["n_grasped"]      = n_grasped
        extra_counts["n_success_ever"] = n_success_ever

        logger.end_episode(
            n_success / num_envs,
            n_success=n_success,
            n_total=num_envs,
            n_success_ever=n_success_ever,
            extra_metrics=extra_counts,
        )

    fpo_env._collect_substep_frames = False


def _run_random_eval(fpo_env, spawn_cfg, logger, isaac_env, record_video):
    num_envs    = fpo_env.num_envs
    device      = fpo_env.device
    num_trials  = max(1, spawn_cfg.num_trials)

    for trial_idx in range(num_trials):
        record_this = record_video and trial_idx == 0
        fpo_env._collect_substep_frames = record_this

        obs, _ = fpo_env.reset()
        episode_steps = _episode_steps(fpo_env)

        # Capture initial object positions for scatter plot.
        init_pos_w     = isaac_env.scene["grasp_object"].data.root_pos_w.clone()
        init_pos_local = (init_pos_w - isaac_env.scene.env_origins).cpu().numpy()

        logger.begin_episode(f"random_trial_{trial_idx}", None)

        last_success  = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_success  = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_grasped  = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_extra: dict[str, torch.Tensor] = {}
        per_env_extra: dict[str, list] = {}
        per_env_success      = [False] * num_envs
        per_env_success_ever = [False] * num_envs
        per_env_grasped      = [False] * num_envs
        recorded   = [False] * num_envs
        done_steps: list[int | None] = [None] * num_envs

        for step_idx in range(episode_steps):
            # Read metrics BEFORE step (pre-auto-reset state).
            active = torch.tensor(
                [not recorded[i] for i in range(num_envs)],
                dtype=torch.bool, device=device,
            )
            if active.any():
                metrics_np = _safe_get_metrics(isaac_env)
                if metrics_np is not None:
                    m_s = torch.from_numpy(metrics_np.get(
                        "is_success", np.zeros(num_envs, dtype=bool)
                    )).to(device).bool()
                    m_g = torch.from_numpy(metrics_np.get(
                        "is_grasped", np.zeros(num_envs, dtype=bool)
                    )).to(device).bool()
                    last_success[active] = m_s[active]
                    ever_success[active] |= m_s[active]
                    ever_grasped[active] |= m_g[active]
                    for key, arr in metrics_np.items():
                        if key in ("is_success", "is_grasped"):
                            continue
                        if key not in ever_extra:
                            ever_extra[key]    = torch.zeros(num_envs, dtype=torch.bool, device=device)
                            per_env_extra[key] = [False] * num_envs
                        m = torch.from_numpy(arr).to(device).bool()
                        ever_extra[key][active] |= m[active]

            _, reward, done, _ = fpo_env.step()

            for i in range(num_envs):
                if done[i] and not recorded[i]:
                    per_env_success[i]      = bool(last_success[i])
                    per_env_success_ever[i] = bool(ever_success[i])
                    per_env_grasped[i]      = bool(ever_grasped[i])
                    for key in ever_extra:
                        per_env_extra[key][i] = bool(ever_extra[key][i])
                    recorded[i]  = True
                    done_steps[i] = step_idx

            # Record step for logger (env 0 only for EE/obj pose).
            ee_pose  = _get_ee_pose(fpo_env)
            obj_pose = isaac_env.scene["grasp_object"].data.root_pos_w[0].cpu().numpy()
            action   = fpo_env.last_action[0].cpu().numpy() if fpo_env.last_action is not None else np.zeros(fpo_env.cfm_action_dim)
            frame    = fpo_env.last_substep_frames[0] if (fpo_env.last_substep_frames and len(fpo_env.last_substep_frames) > 0) else None
            logger.record_step(ee_pose=ee_pose, object_pose=obj_pose, action=action, frame=frame)

            if all(recorded):
                break

        n_success      = sum(per_env_success)
        n_success_ever = sum(per_env_success_ever)
        n_grasped      = sum(per_env_grasped)
        extra_counts   = {k: int(sum(vals)) for k, vals in per_env_extra.items()}
        extra_counts["n_grasped"]      = n_grasped
        extra_counts["n_success_ever"] = n_success_ever

        logger.end_episode(
            n_success / num_envs,
            n_success=n_success,
            n_total=num_envs,
            n_success_ever=n_success_ever,
            extra_metrics=extra_counts,
        )

        secondary_key = next(iter(per_env_extra), None)
        logger.record_scatter_points(
            xs=init_pos_local[:, 0],
            ys=init_pos_local[:, 1],
            successes=per_env_success,
            secondary=per_env_extra.get(secondary_key) if secondary_key else None,
            secondary_name=secondary_key or "secondary",
        )

    fpo_env._collect_substep_frames = False


def _safe_get_metrics(isaac_env) -> dict | None:
    try:
        return isaac_env.metrics.get_metrics()
    except Exception:
        return None


def _get_ee_pose(fpo_env) -> np.ndarray:
    policy_obs = fpo_env.last_obs.get("policy", {}) if fpo_env.last_obs else {}
    ee = policy_obs.get("ee_pose", policy_obs.get("right_ee_pose"))
    if ee is not None:
        return ee[0, :3].detach().cpu().numpy()
    return np.zeros(3)


def _log_to_wandb(results: dict, output_dir: str, t_so_far: int):
    if wandb.run is None:
        return

    log_dict = {
        "eval/success_rate_end":  results["success_rate"],
        "eval/success_rate_ever": results["success_rate_ever"],
        "eval/n_success":         results["n_success"],
        "eval/n_total":           results["n_total"],
    }
    for key, rate in results.get("extra_metric_rates", {}).items():
        log_dict[f"eval/{key}_rate"] = rate

    scatter_success = os.path.join(output_dir, "scatter_success.png")
    if os.path.isfile(scatter_success):
        log_dict["eval/scatter_success"] = wandb.Image(scatter_success)

    for extra_png in glob.glob(os.path.join(output_dir, "scatter_*.png")):
        if extra_png == scatter_success:
            continue
        key = os.path.splitext(os.path.basename(extra_png))[0]  # e.g. scatter_is_grasped
        log_dict[f"eval/{key}"] = wandb.Image(extra_png)

    video_dir = os.path.join(output_dir, "videos")
    if os.path.isdir(video_dir):
        videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        if videos:
            log_dict["eval/video"] = wandb.Video(videos[0])

    wandb.log(log_dict, step=t_so_far)
