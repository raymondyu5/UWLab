"""
Evaluate a BC (CFM) policy inside the Isaac Sim environment.

Usage (inside container):
    ./isaaclab.sh -p scripts/play_bc.py \\
        --eval_cfg configs/eval/pink_cup_bc_joint_abs.yaml \\
        checkpoint=/path/to/outputs/2026-03-04/12-00-00 \\
        record_video=true

./uwlab.sh -p scripts/eval/play_bc.py --eval_cfg configs/eval/bottle_pour_bc.yaml --enable_cameras --record_video --checkpoint logs/bc_cfm_pcd_bourbon_0312 --headless

The checkpoint dir must contain:
    .hydra/config.yaml   (saved by Hydra at train time)
    checkpoints/best.ckpt (or latest.ckpt)

obs_keys can be overridden in eval config; if not set, uses the checkpoint's training config.
image_keys, downsample_points, and policy architecture are loaded from the training config.
"""

import argparse
import os
import sys

from uwlab.utils.paths import setup_third_party_paths
setup_third_party_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a BC policy in Isaac Sim.")
parser.add_argument("--eval_cfg", type=str, required=True,
                    help="Path to eval config YAML (configs/eval/*.yaml)")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Override num_envs from eval config")
parser.add_argument("--sim_type", type=str, choices=["eval", "distill"], default="eval",
                    help="eval: use eval mode. distill: use distill mode.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default=None,
                    help="Override output directory for results")
parser.add_argument("overrides", nargs="*",
                    help="Key=value overrides for eval config (e.g. checkpoint=/path record_video=true)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Enable cameras when record_video is requested (needed for env.render())
import yaml as _yaml
_record_video = False
if args_cli.eval_cfg and os.path.isfile(args_cli.eval_cfg):
    with open(args_cli.eval_cfg) as _f:
        _cfg = _yaml.safe_load(_f)
    _record_video = bool(_cfg.get("record_video", False))
    for kv in args_cli.overrides:
        key, _, val = kv.partition("=")
        if key == "record_video":
            _record_video = str(val).lower() == "true"
            break
if _record_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import yaml
import isaaclab.utils.math as math_utils

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy
from uwlab.eval.bc_obs_formatter import BCObsFormatter
from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import load_spawn_cfg, SpawnCfg

from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    EVAL_MODE,
    DISTILL_MODE,
    parse_franka_leap_env_cfg,
)
import uwlab_tasks  # noqa: F401  registers gym envs


def _load_eval_cfg(path: str, overrides: list) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        key, _, val = kv.partition("=")
        # simple type coercion
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.lstrip("-").replace(".", "").isdigit():
            val = float(val) if "." in val else int(val)
        cfg[key] = val
    return cfg


def _load_train_cfg(checkpoint_dir: str) -> dict:
    """Load the Hydra-saved training config from the checkpoint directory."""
    candidates = [
        os.path.join(checkpoint_dir, ".hydra", "config.yaml"),
        os.path.join(checkpoint_dir, "config.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f"Training config not found in {checkpoint_dir}. "
        "Expected .hydra/config.yaml or config.yaml"
    )


def _build_policy(train_cfg: dict, checkpoint_dir: str, device: torch.device,
                 eval_overrides: dict | None = None, ckpt: dict | None = None) -> CFMPCDPolicy:
    ds_cfg = train_cfg["dataset"]
    pol_cfg = dict(train_cfg["policy"])
    num_inference_steps = None
    if eval_overrides and "num_inference_steps" in eval_overrides:
        num_inference_steps = int(eval_overrides["num_inference_steps"])

    if ckpt is None:
        ckpt_path = _find_checkpoint(checkpoint_dir, eval_overrides and eval_overrides.get("checkpoint_name"))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    policy = load_cfm_policy(ckpt=ckpt, train_cfg=train_cfg, device=device,
                             num_inference_steps=num_inference_steps)
    return policy, ds_cfg


def load_cfm_policy(
    ckpt_path: str | None = None,
    *,
    ckpt: dict | None = None,
    train_cfg: dict | None = None,
    device: str | torch.device = "cpu",
    num_inference_steps: int | None = None,
) -> CFMPCDPolicy:
    """Load a CFMPCDPolicy from a checkpoint file or pre-loaded ckpt dict.

    Can be called with just a path:
        policy = load_cfm_policy("path/to/best.ckpt")
    """
    if ckpt is None:
        assert ckpt_path is not None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Config: prefer embedded cfg (new checkpoints), fall back to caller-supplied
    if train_cfg is None:
        train_cfg = ckpt.get("cfg")
    if train_cfg is None:
        raise ValueError("No cfg in checkpoint and no train_cfg supplied. "
                         "Pass a checkpoint dir to _build_policy or retrain with the updated workspace.")

    ds_cfg  = train_cfg["dataset"]
    pol_cfg = dict(train_cfg["policy"])
    pn_cfg  = train_cfg["pointnet"]
    if num_inference_steps is not None:
        pol_cfg["num_inference_steps"] = int(num_inference_steps)

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
    return policy


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
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}/checkpoints/")


def _get_object_asset(env):
    """Get the manipulated object (grasp_object for grasp tasks, object otherwise)."""
    scene = env.unwrapped.scene
    try:
        return scene["grasp_object"]
    except KeyError:
        return scene["object"]


def _set_object_pose(env, x_offset: float, y_offset: float, yaw_offset: float,
                     default_pos: tuple, default_rot_quat: tuple, reset_height: float):
    """Directly set object pose after env.reset() to control spawn position."""
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    obj = _get_object_asset(env)

    pos = torch.tensor(
        [[default_pos[0] + x_offset, default_pos[1] + y_offset, reset_height]],
        device=device, dtype=torch.float32,
    ).repeat(num_envs, 1)

    base_quat = torch.tensor([list(default_rot_quat)], device=device, dtype=torch.float32).repeat(num_envs, 1)
    delta_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(num_envs, device=device),
        torch.zeros(num_envs, device=device),
        torch.full((num_envs,), yaw_offset, device=device),
    )
    quat = math_utils.quat_mul(delta_quat, base_quat)

    pos_world = pos + env.unwrapped.scene.env_origins
    env_ids = torch.arange(num_envs, device=device)
    obj.write_root_pose_to_sim(torch.cat([pos_world, quat], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device), env_ids=env_ids)


def _check_success(env) -> np.ndarray:
    return env.unwrapped.cfg.is_success(env.unwrapped).cpu().numpy()


def _check_partial_success(env) -> np.ndarray:
    if hasattr(env.unwrapped.cfg, "is_partial_success"):
        return env.unwrapped.cfg.is_partial_success(env.unwrapped).cpu().numpy()
    num_envs = env.unwrapped.num_envs
    return np.zeros(num_envs, dtype=bool)


def main():
    eval_cfg = _load_eval_cfg(args_cli.eval_cfg, args_cli.overrides)

    checkpoint_dir = eval_cfg.get("checkpoint")
    if checkpoint_dir is None:
        raise ValueError("eval_cfg must specify 'checkpoint'")
    checkpoint_dir = os.path.expanduser(checkpoint_dir)

    device = torch.device(eval_cfg.get("device", "cuda:0"))
    num_envs = args_cli.num_envs or eval_cfg.get("num_envs", 1)
    action_horizon = int(eval_cfg.get("action_horizon", 1))
    record_video = bool(eval_cfg.get("record_video", False))
    record_plots = bool(eval_cfg.get("record_plots", True))

    # Load training config and build policy.
    # Prefer cfg embedded in checkpoint; fall back to .hydra/config.yaml for old checkpoints.
    ckpt_path = _find_checkpoint(checkpoint_dir, eval_cfg.get("checkpoint_name"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg = ckpt.get("cfg") or _load_train_cfg(checkpoint_dir)
    policy, ds_cfg = _build_policy(train_cfg, checkpoint_dir, device, eval_overrides=eval_cfg, ckpt=ckpt)

    obs_keys = list(eval_cfg.get("obs_keys") or ds_cfg["obs_keys"])
    image_keys = list(ds_cfg["image_keys"])
    downsample_points = int(ds_cfg["downsample_points"])
    n_obs_steps = int(train_cfg["n_obs_steps"])
    formatter = BCObsFormatter(obs_keys, image_keys, downsample_points, device,
                               n_obs_steps=n_obs_steps,
                               action_dim=policy.action_dim if policy.use_action_history else 0)

    # Create env
    task_id = eval_cfg["task_id"]
    env_cfg = parse_franka_leap_env_cfg(
        task_id,
        run_mode=EVAL_MODE if args_cli.sim_type == "eval" else DISTILL_MODE,
        device=str(device),
        num_envs=num_envs,
    )
    env_cfg.seed = args_cli.seed
    # In distill mode, train_camera is needed for RenderedSegPC — keep it alive.
    # In eval mode (synthetic PCD), cameras are not needed so disable them to save overhead.
    if args_cli.sim_type == "eval":
        env_cfg.scene.train_camera = None
        if hasattr(env_cfg.events, "reset_camera"):
            env_cfg.events.reset_camera = None
    env_cfg.scene.fixed_camera = None
    if hasattr(env_cfg.events, "reset_fixed_camera"):
        env_cfg.events.reset_fixed_camera = None
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if record_video else None)

    # Load spawn config
    spawn_name = eval_cfg.get("spawn", None)
    spawn_dir = os.path.join(os.path.dirname(args_cli.eval_cfg), "spawns")
    if spawn_name is not None and os.path.isdir(spawn_dir):
        spawn_cfg = load_spawn_cfg(spawn_name, spawn_dir)
    else:
        # No spawn config: run num_episodes with random init
        n_episodes = int(eval_cfg.get("num_episodes", 10))
        spawn_cfg = SpawnCfg(poses=[], num_trials=n_episodes)

    # Output dir: logs/eval/<eval_config_stem>/<checkpoint_basename>/
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        eval_base = os.path.join(os.path.dirname(checkpoint_dir.rstrip("/")), "eval")
        eval_config_stem = os.path.splitext(os.path.basename(args_cli.eval_cfg))[0]
        checkpoint_basename = os.path.basename(checkpoint_dir.rstrip("/"))
        output_dir = os.path.join(eval_base, eval_config_stem, checkpoint_basename)
    os.makedirs(output_dir, exist_ok=True)
    cfg = env.unwrapped.cfg
    video_fps = 1.0 / (cfg.sim.dt * cfg.decimation)
    logger = EvalLogger(
        output_dir,
        record_video=record_video,
        record_plots=record_plots,
        video_fps=video_fps,
    )

    _spawn_defaults = env.unwrapped.cfg.object_spawn_defaults
    default_pos = tuple(eval_cfg.get("object_default_pos", _spawn_defaults["default_pos"]))
    default_rot = tuple(eval_cfg.get("object_default_rot", _spawn_defaults["default_rot"]))
    reset_height = float(eval_cfg.get("object_reset_height", _spawn_defaults["reset_height"]))

    # Build episode list: 1 trial per pose (num_envs runs in parallel per pose)
    if spawn_cfg.poses:
        episodes = [(pose.name, pose) for pose in spawn_cfg.poses]
    else:
        episodes = [(None, None)] * spawn_cfg.num_trials

    isaac_env = env.unwrapped
    internal_warmup = int(getattr(isaac_env.cfg, "num_warmup_steps", 0))
    num_warmup_steps = int(eval_cfg.get("num_warmup_steps", 5))
    episode_steps = int(eval_cfg.get("episode_steps", isaac_env.max_episode_length)) - internal_warmup - num_warmup_steps

    with torch.inference_mode():
        for ep_idx, (spawn_name_ep, spawn_pose) in enumerate(episodes):
            obs_raw, _ = env.reset()
            policy_obs = obs_raw["policy"]
            formatter.reset()  # clear obs history at episode boundary

            # Override object pose if spawn pose specified
            if spawn_pose is not None:
                _set_object_pose(env, spawn_pose.x, spawn_pose.y, spawn_pose.yaw,
                                 default_pos, default_rot, reset_height)

            # Warm-up: hold position using task-defined safe action
            warmup_act = isaac_env.cfg.warmup_action(isaac_env)
            assert warmup_act.shape == (num_envs, isaac_env.action_manager.total_action_dim), \
                f"warmup_action shape {warmup_act.shape} doesn't match action dim {isaac_env.action_manager.total_action_dim}"
            for _ in range(num_warmup_steps):
                obs_raw, _, _, _, _ = env.step(warmup_act)
            policy_obs = obs_raw["policy"]

            spawn_pose_dict = (
                {"x": spawn_pose.x, "y": spawn_pose.y, "yaw": spawn_pose.yaw}
                if spawn_pose is not None else None
            )
            logger.begin_episode(spawn_name_ep, spawn_pose_dict)

            action_seq = None
            per_env_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
            per_env_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            per_env_partial = torch.zeros(num_envs, dtype=torch.bool, device=device)

            for step in range(episode_steps):
                if step % action_horizon == 0 and not per_env_done.all():
                    obs_dict = formatter.format(policy_obs)
                    result = policy.predict_action(obs_dict)
                    action_seq = result["action"]  # (B, n_action_steps, A)

                action_idx = step % action_horizon
                if action_idx >= action_seq.shape[1]:
                    action_idx = action_seq.shape[1] - 1
                action_step = action_seq[:, action_idx].clone()  # (B, A)

                # Hold position for already-done envs so they don't receive
                # stale policy actions after their auto-reset.
                if per_env_done.any():
                    action_step[per_env_done] = warmup_act[per_env_done]

                # Check success on current state BEFORE stepping to avoid
                # reading the auto-reset state when the episode truncates.
                pre_step_success = _check_success(env)
                active = ~per_env_done
                per_env_success |= torch.from_numpy(pre_step_success).to(device) & active
                per_env_partial |= torch.from_numpy(_check_partial_success(env)).to(device) & active

                obs_raw, reward, terminated, truncated, info = env.step(action_step)
                policy_obs = obs_raw["policy"]

                reset_mask = terminated | truncated
                formatter.update_action(action_step)
                if reset_mask.any():
                    formatter.reset_envs(reset_mask, policy_obs)

                per_env_done |= reset_mask

                ee_pose = isaac_env.cfg.observations.policy.ee_pose.func(
                    isaac_env, **isaac_env.cfg.observations.policy.ee_pose.params)
                obj = _get_object_asset(env)
                obj_pose = obj.data.root_pos_w - isaac_env.scene.env_origins  # (B, 3)

                frame = env.render() if record_video else None
                logger.record_step(
                    ee_pose.cpu().numpy()[0],
                    obj_pose.cpu().numpy()[0],
                    action_step.cpu().numpy()[0],
                    frame=frame,
                )

                if per_env_done.all():
                    break

            n_success = int(per_env_success.sum())
            n_partial = int(per_env_partial.sum())
            logger.end_episode(n_success / num_envs, n_success=n_success, n_total=num_envs,
                               partial_success=bool(n_partial > 0))
            status = "SUCCESS" if n_success > 0 else ("partial" if n_partial > 0 else "fail")
            print(f"[play_bc] Episode {ep_idx+1}/{len(episodes)}: {status} "
                  f"({n_success}/{num_envs} success, {n_partial}/{num_envs} partial)"
                  + (f" (spawn={spawn_name_ep})" if spawn_name_ep else ""))

    results = logger.finalize()
    print(f"\n[play_bc] Done. Success rate: {results['success_rate']:.1%} ({results['n_success']}/{results['n_episodes']})")
    print(f"[play_bc] Partial success: {results['partial_success_rate']:.1%} ({results['n_partial_success']}/{results['n_episodes']})")
    print(f"[play_bc] Results saved to: {output_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()