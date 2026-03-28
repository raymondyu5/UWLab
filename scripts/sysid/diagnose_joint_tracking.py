"""
Diagnostic: measure arm joint tracking error under the current fixed decimation,
and show how error would decay if we ran extra physics steps (blocking simulation).

For each control step:
  1. Run env.step(action) normally (decimation=6 physics steps, IK re-runs each sub-step).
  2. Record L-inf error: max_j |q[j] - q_target[j]| for the 7 arm joints.
  3. Then run up to EXTRA_STEPS additional physics steps with the SAME joint targets
     (no IK re-run), recording how the error decays. This simulates blocking.

Output (printed + numpy save):
  - Per-step tracking error array (shape: [total_steps])
  - Per-step convergence curve under extra steps (shape: [total_steps, EXTRA_STEPS+1])
  - Summary statistics

Usage (inside container):
    ./uwlab.sh -p scripts/eval/diagnose_joint_tracking.py \\
        --checkpoint logs/bc_cfm_pcd_bourbon_0312 \\
        --task UW-FrankaLeap-PourBottle-IkRel-v0 \\
        --num_envs 1 \\
        --num_episodes 3 \\
        --enable_cameras \\
        --headless
"""

import argparse
import os
import sys

from uwlab.utils.paths import setup_third_party_paths
setup_third_party_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="UW-FrankaLeap-PourBottle-IkRel-v0")
parser.add_argument("--obs_keys", type=str, nargs="+", default=["joint_pos"])
parser.add_argument("--image_keys", type=str, nargs="+", default=["seg_pc"])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--action_horizon", type=int, default=1)
parser.add_argument("--ckpt_name", type=str, default="latest.ckpt")
parser.add_argument("--extra_steps", type=int, default=30,
                    help="Extra physics steps to simulate after each control step (for convergence curve).")
parser.add_argument("--record_video", action="store_true", default=False)
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
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy
from uwlab.eval.bc_obs_formatter import BCObsFormatter

import uwlab_tasks  # noqa: F401

ARM_DOF = 7  # first 7 joints are the Franka arm


def _load_policy(checkpoint_dir: str, ckpt_name: str, device):
    """Load CFM policy from either IsaacLab legacy or UWLab checkpoint format."""
    import dill, yaml
    path = os.path.join(checkpoint_dir, "checkpoints", ckpt_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=dill)

    if "state_dicts" in ckpt and "ema_model" in ckpt["state_dicts"]:
        # IsaacLab legacy format: cfg is embedded OmegaConf object
        state_dict = ckpt["state_dicts"]["ema_model"]
        cfg = ckpt["cfg"]
        shape_meta = {
            "action": {"shape": list(cfg.shape_meta.action.shape)},
            "obs": {
                "agent_pos": {"shape": list(cfg.shape_meta.obs.agent_pos.shape), "type": "low_dim"},
                "seg_pc": {"shape": list(cfg.shape_meta.obs.seg_pc.shape), "type": "pcd"},
            },
        }
        sigma = float(cfg.policy.noise_scheduler.sigma)
        horizon = int(cfg.horizon)
        n_action_steps = int(cfg.n_action_steps) + int(getattr(cfg, "n_latency_steps", 0))
        n_obs_steps = int(cfg.n_obs_steps)
        down_dims = tuple(cfg.policy.down_dims)
        obs_keys = list(cfg.dataset.obs_keys)
        downsample_points = int(cfg.downsample_points)
    else:
        # UWLab format: ema_model at top level, config.yaml alongside
        state_dict = ckpt["ema_model"]
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.yaml not found at {config_path}")
        with open(config_path) as f:
            train_cfg = yaml.safe_load(f)
        agent_pos_dim = int(state_dict["normalizer.params_dict.agent_pos.scale"].shape[0])
        action_dim = int(state_dict["normalizer.params_dict.action.scale"].shape[0])
        downsample_points = int(train_cfg["dataset"]["downsample_points"])
        shape_meta = {
            "action": {"shape": [action_dim]},
            "obs": {
                "agent_pos": {"shape": [agent_pos_dim], "type": "low_dim"},
                "seg_pc": {"shape": [3, downsample_points], "type": "pcd"},
            },
        }
        pol = train_cfg["policy"]
        ds = train_cfg["dataset"]
        sigma = float(pol.get("sigma", 0.0))
        horizon = int(train_cfg.get("horizon", 4))
        n_action_steps = int(train_cfg.get("n_action_steps", 8))
        n_obs_steps = int(train_cfg.get("n_obs_steps", 1))
        down_dims = tuple(pol["down_dims"])
        obs_keys = ds.get("obs_keys", ds.get("obs_key", []))

    pcd_model = PointNet(
        in_channels=3,
        local_channels=(64, 64, 64, 128, 1024),
        global_channels=(512, 256),
        use_bn=False,
    )
    obs_encoder = MultiPCDObsEncoder(shape_meta=shape_meta, pcd_model=pcd_model)
    noise_scheduler = ConditionalFlowMatcher(sigma=sigma)
    policy = CFMPCDPolicy(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=5,
        diffusion_step_embed_dim=256,
        down_dims=down_dims,
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()
    print(f"[diag] Policy loaded: horizon={horizon}, n_action_steps={n_action_steps}, obs_keys={obs_keys}")
    return policy, obs_keys, downsample_points


def _arm_tracking_error(robot) -> float:
    """L-inf error between actual and target arm joint positions, env 0 only."""
    q = robot.data.joint_pos[0, :ARM_DOF]
    q_target = robot.data.joint_pos_target[0, :ARM_DOF]
    return (q - q_target).abs().max().item()


def _run_extra_steps(isaac_env, robot, n_steps: int) -> list[float]:
    """
    Run n_steps physics steps holding current joint targets fixed (no IK re-run).
    Returns list of L-inf errors at each step.
    """
    errors = []
    for _ in range(n_steps):
        # Write current targets back (already set; this ensures they persist)
        isaac_env.scene.write_data_to_sim()
        isaac_env.sim.step(render=False)
        isaac_env.scene.update(dt=isaac_env.physics_dt)
        errors.append(_arm_tracking_error(robot))
    return errors


def main():
    device = torch.device("cuda:0")

    policy, obs_keys, downsample_points = _load_policy(args_cli.checkpoint, args_cli.ckpt_name, device)
    formatter = BCObsFormatter(obs_keys, args_cli.image_keys, downsample_points, device)

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task, device=str(device), num_envs=args_cli.num_envs)
    render_mode = "rgb_array" if args_cli.record_video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    isaac_env = env.unwrapped
    robot = isaac_env.scene["robot"]
    episode_steps = int(isaac_env.max_episode_length)

    output_dir = args_cli.output_dir or os.path.join(args_cli.checkpoint, "diagnose_tracking")
    os.makedirs(output_dir, exist_ok=True)

    import imageio
    video_writers = {}  # ep -> imageio writer

    # Storage across all episodes
    # after_decimation_errors[i] = L-inf error after 6 normal physics sub-steps at step i
    # convergence_curves[i] = [error_after_0_extra, ..., error_after_EXTRA_STEPS extra] at step i
    all_after_errors = []
    all_convergence = []  # shape: [N_steps, extra_steps+1]

    control_hz = 1.0 / (isaac_env.physics_dt * isaac_env.cfg.decimation)

    with torch.inference_mode():
        for ep in range(args_cli.num_episodes):
            obs_raw, _ = env.reset()
            policy_obs = obs_raw["policy"] if isinstance(obs_raw, dict) else obs_raw
            action_seq = None

            if args_cli.record_video:
                video_path = os.path.join(output_dir, f"episode_{ep:02d}.mp4")
                writer = imageio.get_writer(video_path, fps=int(control_hz))
            else:
                writer = None

            for step in range(episode_steps):
                if step % args_cli.action_horizon == 0:
                    obs_dict = formatter.format(policy_obs)
                    action_seq = policy.predict_action(obs_dict)["action"]

                action_idx = min(step % args_cli.action_horizon, action_seq.shape[1] - 1)
                action_step = action_seq[:, action_idx]

                obs_raw, _, terminated, truncated, _ = env.step(action_step)
                policy_obs = obs_raw["policy"] if isinstance(obs_raw, dict) else obs_raw

                if writer is not None:
                    frame = env.render()
                    if frame is not None:
                        writer.append_data(frame)

                # Error after normal 6-step decimation
                err_after = _arm_tracking_error(robot)
                all_after_errors.append(err_after)

                # Convergence curve: run extra steps holding same targets
                curve = [err_after] + _run_extra_steps(isaac_env, robot, args_cli.extra_steps)
                all_convergence.append(curve)

                if terminated.any() or truncated.any():
                    break

            if writer is not None:
                writer.close()
                print(f"[diag] Video saved: {video_path}")

            print(f"[diag] Episode {ep+1}/{args_cli.num_episodes} done "
                  f"({len(all_after_errors)} control steps total so far)")

    after_arr = np.array(all_after_errors)        # [N]
    conv_arr = np.array(all_convergence)           # [N, extra_steps+1]

    np.save(os.path.join(output_dir, "after_decimation_errors.npy"), after_arr)
    np.save(os.path.join(output_dir, "convergence_curves.npy"), conv_arr)

    print("\n=== Arm joint tracking error after normal decimation (6 physics steps) ===")
    print(f"  mean : {after_arr.mean():.4f} rad")
    print(f"  p50  : {np.percentile(after_arr, 50):.4f} rad")
    print(f"  p90  : {np.percentile(after_arr, 90):.4f} rad")
    print(f"  p99  : {np.percentile(after_arr, 99):.4f} rad")
    print(f"  max  : {after_arr.max():.4f} rad")

    print(f"\n=== Convergence under {args_cli.extra_steps} extra blocking steps ===")
    mean_curve = conv_arr.mean(axis=0)  # [extra_steps+1]
    for i, err in enumerate(mean_curve):
        label = f"+{i:2d} steps"
        bar = "#" * int(err / mean_curve[0] * 40)
        print(f"  {label}: {err:.4f} rad  {bar}")

    # Find where mean error drops below thresholds
    thresholds = [0.01, 0.005, 0.002]
    print("\n=== Steps to reach threshold (mean curve) ===")
    for thr in thresholds:
        idx = next((i for i, e in enumerate(mean_curve) if e < thr), None)
        print(f"  < {thr:.3f} rad: {'never in budget' if idx is None else f'+{idx} extra steps'}")

    print(f"\n[diag] Arrays saved to {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
