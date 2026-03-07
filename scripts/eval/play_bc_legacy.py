"""
Deploy an IsaacLab-format BC checkpoint inside a UWLab env.

IsaacLab checkpoints have a different layout than UWLab's:
  ckpt["state_dicts"]["ema_model"]  (weights)
  ckpt["cfg"]                       (OmegaConf DictConfig, embedded at save time)

Usage (inside container):
    ./uwlab.sh -p scripts/eval/play_bc_legacy.py \\
        --checkpoint /workspace/uwlab/logs/cup_pick_h4 \\
        --task UW-FrankaLeap-GraspPinkCup-IkRel-v0 \\
        --obs_keys joint_pos \\
        --num_envs 1 \\
        --num_episodes 10 \\
        --action_horizon 1 \\
        --record_video \\
        --enable_cameras \\
        --headless
"""

import argparse
import os
import sys

# Ensure third-party packages are importable regardless of whether isaac-sim's
# python.sh resets PYTHONPATH. These paths are bind-mounted inside the container. TODO: kinda a hack for now 
_EXTRA_PATHS = [
    "/workspace/uwlab/third_party/diffusion_policy",
    "/workspace/uwlab/third_party/pip_packages",
]
for _p in _EXTRA_PATHS:
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to IsaacLab checkpoint dir (contains checkpoints/latest.ckpt)")
parser.add_argument("--task", type=str, default="UW-FrankaLeap-GraspPinkCup-IkRel-v0")
parser.add_argument("--obs_keys", type=str, nargs="+", default=["joint_pos"],
                    help="Env obs keys to concatenate into agent_pos")
parser.add_argument("--image_keys", type=str, nargs="+", default=["seg_pc"])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--action_horizon", type=int, default=1,
                    help="Steps to execute before re-querying policy")
parser.add_argument("--num_warmup_steps", type=int, default=5)
parser.add_argument("--ckpt_name", type=str, default="latest.ckpt")
parser.add_argument("--record_video", action="store_true",
                    help="Save per-episode videos from fixed_camera (requires --enable_cameras)")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Where to save results. Defaults to <checkpoint>/eval_legacy/")
parser.add_argument("--spawn", type=str, default=None,
                    help="Spawn config name in configs/eval/spawns/ (e.g. 'cardinal_3x3'). "
                         "If set, num_episodes is ignored and episodes are drawn from the spawn grid.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy
from uwlab.eval.bc_obs_formatter import BCObsFormatter
from uwlab.eval.eval_logger import EvalLogger
from uwlab.eval.spawn import load_spawn_cfg, SpawnPose

import uwlab_tasks  # noqa: F401


def _load_legacy_checkpoint(checkpoint_dir: str, ckpt_name: str):
    """Load an IsaacLab-format checkpoint. Returns (state_dict, cfg)."""
    path = os.path.join(checkpoint_dir, "checkpoints", ckpt_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dicts"]["ema_model"]
    cfg = ckpt["cfg"]
    return state_dict, cfg


def _build_policy(state_dict, cfg, device: torch.device) -> CFMPCDPolicy:
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
    return policy


def _get_frame(env) -> np.ndarray:
    """Grab the latest RGB frame via env.render(). Returns (H, W, 3) uint8."""
    frame = env.render()  # (H, W, 3) or None
    return frame


def main():
    device = torch.device("cuda:0")

    state_dict, cfg = _load_legacy_checkpoint(args_cli.checkpoint, args_cli.ckpt_name)
    policy = _build_policy(state_dict, cfg, device)

    downsample_points = int(cfg.downsample_points)
    formatter = BCObsFormatter(args_cli.obs_keys, args_cli.image_keys, downsample_points, device)

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab.managers import SceneEntityCfg
    from uwlab_assets.robots.franka_leap import FRANKA_LEAP_EE_BODY, FRANKA_LEAP_EE_OFFSET
    from uwlab_tasks.manager_based.manipulation.grasp.mdp import ee_pose_w

    env_cfg = parse_env_cfg(args_cli.task, device=str(device), num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.record_video else None)

    isaac_env = env.unwrapped
    episode_steps = int(isaac_env.max_episode_length)

    output_dir = args_cli.output_dir or os.path.join(args_cli.checkpoint, "eval_legacy")
    logger = EvalLogger(output_dir, record_video=args_cli.record_video, record_plots=True)

    ee_cfg = SceneEntityCfg("robot")

    # Build episode list: either spawn grid or plain resets.
    if args_cli.spawn:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _uwlab_dir = os.path.abspath(os.path.join(_script_dir, "../../"))
        spawn_cfg = load_spawn_cfg(args_cli.spawn, os.path.join(_uwlab_dir, "configs/eval/spawns"))
        poses = spawn_cfg.poses if spawn_cfg.poses else [SpawnPose()]
        episode_list = [(pose, trial)
                        for pose in poses
                        for trial in range(spawn_cfg.num_trials)]
    else:
        episode_list = [(None, ep) for ep in range(args_cli.num_episodes)]

    def _reset_env(pose):
        obs_raw, _ = env.reset()
        if pose is not None:
            obj = isaac_env.scene["grasp_object"]
            defaults = isaac_env.cfg.object_spawn_defaults
            default_pos = torch.tensor(defaults["default_pos"], dtype=torch.float32, device=device)
            default_rot = torch.tensor(defaults["default_rot"], dtype=torch.float32, device=device)
            new_pos = default_pos.clone()
            new_pos[0] += pose.x
            new_pos[1] += pose.y
            root_state = obj.data.default_root_state.clone()
            root_state[:, :3] = new_pos.unsqueeze(0) + isaac_env.scene.env_origins
            root_state[:, 3:7] = default_rot.unsqueeze(0)
            root_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(root_state)
            isaac_env.sim.step()
            obs_raw = {"policy": isaac_env.observation_manager.compute()["policy"]}
        warmup_act = isaac_env.cfg.warmup_action(isaac_env)
        for _ in range(args_cli.num_warmup_steps):
            obs_raw, _, _, _, _ = env.step(warmup_act)
        return obs_raw

    with torch.inference_mode():
        for ep_idx, (pose, _) in enumerate(episode_list):
            obs_raw = _reset_env(pose)
            policy_obs = obs_raw["policy"] if isinstance(obs_raw, dict) else obs_raw

            spawn_name = pose.name if pose is not None else None
            spawn_pose_dict = {"x": pose.x, "y": pose.y, "yaw": pose.yaw} if pose is not None else None
            logger.begin_episode(spawn_name, spawn_pose_dict)
            action_seq = None
            success = False

            for step in range(episode_steps):
                if step % args_cli.action_horizon == 0:
                    obs_dict = formatter.format(policy_obs)
                    result = policy.predict_action(obs_dict)
                    action_seq = result["action"]  # (B, n_action_steps, A)

                action_idx = min(step % args_cli.action_horizon, action_seq.shape[1] - 1)
                action_step = action_seq[:, action_idx]  # (B, A)

                obs_raw, _, terminated, truncated, _ = env.step(action_step)
                policy_obs = obs_raw["policy"] if isinstance(obs_raw, dict) else obs_raw

                frame = _get_frame(env) if args_cli.record_video else None
                obj = isaac_env.scene["grasp_object"]
                obj_pose = (obj.data.root_pos_w - isaac_env.scene.env_origins).cpu().numpy()[0]
                ee_pose_tensor = ee_pose_w(isaac_env, asset_cfg=ee_cfg,
                                           ee_body_name=FRANKA_LEAP_EE_BODY,
                                           ee_offset=FRANKA_LEAP_EE_OFFSET)
                logger.record_step(ee_pose_tensor.cpu().numpy()[0], obj_pose,
                                   action_step.cpu().numpy()[0], frame)

                success = bool(isaac_env.cfg.is_success(isaac_env).cpu()[0])
                if success or terminated.any() or truncated.any():
                    break

            logger.end_episode(success)
            print(f"[play_bc_legacy] Episode {ep_idx+1}/{len(episode_list)}: "
                  f"{'SUCCESS' if success else 'fail'}")

    results = logger.finalize()
    print(f"\n[play_bc_legacy] Success rate: {results['success_rate']:.1%} "
          f"({results['n_success']}/{results['n_episodes']})")
    print(f"[play_bc_legacy] Results -> {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
