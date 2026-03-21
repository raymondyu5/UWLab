# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import imageio
from isaaclab.app import AppLauncher



# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--output_dir", type=str, default="logs/sysid/open_loop_trajectory", help="Path to output directory.")

parser.add_argument("--trajectory_file", type=str, default=None, help="Path to the trajectory file.")
parser.add_argument("--action_type", type=str, choices=['delta_ee', 'abs_ee', 'joint'], default='joint', help="Type of action to use.")
parser.add_argument("--blocking", action="store_true", default=False, help="Enable blocking control: loop env.step() until convergence.")
parser.add_argument("--blocking_max_steps", type=int, default=50, help="Max extra env.step() calls per trajectory step when blocking.")
parser.add_argument("--blocking_tol", type=float, default=0.005, help="L-inf joint error (rad) to consider converged.")

parser.add_argument(
    "--debug_cameras",
    action="store_true",
    help="Capture one frame from each candidate camera and save a comparison grid to output_dir, then exit.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import SceneEntityCfg
import numpy as np
import uwlab_assets.robots.franka_leap as franka_leap
from uwlab_tasks.utils.trajectory_utils import load_real_episode
import matplotlib

import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm


def load_episode(episode_file: str, action_type: str):
    """Load an episode and construct per-step actions for the requested action_type.

    For all modes, the gripper command always comes from episode['actions'] (last 16 dims).
    The arm command comes from:
      - delta_ee: episode['actions'] directly (dataset already in delta EE space)
      - abs_ee:   obs[i]['desired_pose']
      - joint:    obs[i]['ik_joint_pos_desired']
    """
    episode = load_real_episode(episode_file)
    obs_list = episode["obs"]
    raw_actions = episode["actions"]
    

    num_steps = min(len(obs_list), len(raw_actions))
    if num_steps == 0:
        raise ValueError("Episode has no steps: obs or actions list is empty.")

    obs_list = obs_list[:num_steps]
    actions = []

    for i in range(num_steps):
        real_action = np.asarray(raw_actions[i], dtype=np.float32).reshape(-1)

        if action_type == "delta_ee":
            base_action = real_action
        else:
            if real_action.shape[0] < 16:
                raise ValueError(
                    f"Expected at least 16 dims in episode['actions'] for gripper, "
                    f"got {real_action.shape[0]} at step {i}"
                )
            hand_cmd = real_action[-16:]

            obs_i = obs_list[i]
            if action_type == "abs_ee":
                raise ValueError("Absolute EE action type is not yet supported for open loop trajectory.")
            elif action_type == "joint":
                if "ik_joint_pos_desired" not in obs_i:
                    raise KeyError(
                        "Missing key 'ik_joint_pos_desired' in obs for joint "
                        f"action_type at step {i}"
                    )
                arm_cmd = np.asarray(
                    obs_i["ik_joint_pos_desired"], dtype=np.float32
                ).reshape(-1)
            else:
                raise ValueError(f"Unsupported action_type: {action_type}")

            base_action = np.concatenate([arm_cmd, hand_cmd], axis=-1)

        actions.append(base_action)

    return obs_list, actions


def _extract_sim_observation(env, obs) -> dict:
    """Extract EE pose, arm joints, and hand joints for env[0] in dataset format."""
    obs = obs['policy']

    # EE pose (x, y, z, qw, qx, qy, qz) in world frame with calibration offset
    ee_pose = obs['ee_pose'][0]
    ee_pose_np = ee_pose.detach().cpu().numpy().reshape(-1)

    # Joint positions: first 7 are arm, remaining 16 are hand
    joint_pos = obs['joint_pos'][0].detach().cpu().numpy().reshape(-1)
    arm_joints = joint_pos[:7]
    hand_joints = joint_pos[7:7 + 16]

    return {
        "cartesian_position": ee_pose_np,
        "joint_positions": arm_joints,
        "gripper_position": hand_joints,
    }


# For --debug_cameras: list of camera names to compare (fixed_camera is set from eye+look_at on reset).
CAMERA_CANDIDATES = ["fixed_camera"]

def save_camera_candidates_comparison(env, output_dir: str, candidate_names: list):
    """Capture one frame from each candidate camera and save a labeled grid for visual comparison."""
    scene = env.unwrapped.scene
    n = len(candidate_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)
    for idx, name in enumerate(candidate_names):
        ax = axes.flat[idx]
        cam = scene[name]
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        ax.imshow(rgb)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    for idx in range(n, axes.size):
        axes.flat[idx].axis("off")
    plt.suptitle("Camera extrinsics candidates (pick the view that matches your real setup)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "camera_candidates.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO]: Saved camera comparison to {path}")


def compare_trajectories(real_obs_list: list, sim_results: dict, output_dir: str):
    """Compare real vs sim trajectories and generate plots."""

    real_ee_poses = np.asarray(
        [np.asarray(obs["cartesian_position"])[:7] for obs in real_obs_list],
        dtype=np.float32,
    )
    real_arm_joints = np.asarray(
        [np.asarray(obs["joint_positions"])[:7] for obs in real_obs_list],
        dtype=np.float32,
    )
    real_hand_joints = np.asarray(
        [np.asarray(obs["gripper_position"]) for obs in real_obs_list],
        dtype=np.float32,
    )

    sim_ee_poses = np.asarray(
        [np.asarray(obs["cartesian_position"])[:7] for obs in sim_results],
        dtype=np.float32,
    )
    sim_arm_joints = np.asarray(
        [np.asarray(obs["joint_positions"])[:7] for obs in sim_results],
        dtype=np.float32,
    )
    sim_hand_joints = np.asarray(
        [np.asarray(obs["gripper_position"]) for obs in sim_results],
        dtype=np.float32,
    )

    # Ensure hand joints are at least (T, D) even if scalar in the dataset.
    if real_hand_joints.ndim == 1:
        real_hand_joints = real_hand_joints[:, None]
    if sim_hand_joints.ndim == 1:
        sim_hand_joints = sim_hand_joints[:, None]

    min_len = min(len(real_ee_poses), len(sim_ee_poses))
    real_ee_poses = real_ee_poses[:min_len]
    real_arm_joints = real_arm_joints[:min_len]
    real_hand_joints = real_hand_joints[:min_len]
    sim_ee_poses = sim_ee_poses[:min_len]
    sim_arm_joints = sim_arm_joints[:min_len]
    sim_hand_joints = sim_hand_joints[:min_len]

    timesteps = np.arange(min_len)

    # Plot 1: EE position comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    labels = ['x', 'y', 'z', 'qw', 'qx', 'qy']
    for i in range(6):
        ax = axes[i]
        ax.plot(timesteps, real_ee_poses[:, i], 'b-', label='Real', linewidth=2)
        ax.plot(timesteps, sim_ee_poses[:, i], 'r--', label='Sim', linewidth=1.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel(labels[i])
        ax.set_title(f'EE {labels[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Real vs Sim EE Pose (OPEN-LOOP REPLAY)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir + '/ee_pose_comparison.png', dpi=150)
    plt.close()

    # Plot 2: Joint position comparison (arm)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(7):
        ax = axes[i]
        ax.plot(timesteps, real_arm_joints[:, i], 'b-', label='Real', linewidth=2)
        ax.plot(timesteps, sim_arm_joints[:, i], 'r--', label='Sim', linewidth=1.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'Joint {i+1} (rad)')
        ax.set_title(f'Arm Joint {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[7].axis('off')

    plt.suptitle('Real vs Sim Arm Joint Positions (OPEN-LOOP REPLAY)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'arm_joint_comparison.png'), dpi=150)
    plt.close()

    # Plot 3: Hand joint position comparison (up to 16 joints, 4x4 grid)
    num_hand_joints = min(real_hand_joints.shape[1], 16)
    n_cols = 4
    n_rows = int(np.ceil(num_hand_joints / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    for i in range(num_hand_joints):
        ax = axes[i]
        ax.plot(timesteps, real_hand_joints[:, i], 'b-', label='Target', linewidth=2)
        ax.plot(timesteps, sim_hand_joints[:, i], 'r--', label='Achieved', linewidth=1.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'j{i} (rad)')
        mae_i = np.mean(np.abs(real_hand_joints[:, i] - sim_hand_joints[:, i]))
        ax.set_title(f'j{i} (MAE: {np.degrees(mae_i):.1f}°)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots (e.g., when num_hand_joints < 16).
    for j in range(num_hand_joints, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Hand Joints: Target vs Achieved', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'hand_joint_comparison.png'), dpi=150)
    plt.close()

    # Plot 4: Hand joint error over time
    fig, ax = plt.subplots(figsize=(12, 5))
    hand_joint_error_per_step = np.mean(np.abs(real_hand_joints - sim_hand_joints), axis=1)
    ax.plot(timesteps, np.degrees(hand_joint_error_per_step), 'g-', linewidth=2)
    ax.axhline(y=np.degrees(np.mean(hand_joint_error_per_step)), color='r', linestyle='--',
               label=f'Mean: {np.degrees(np.mean(hand_joint_error_per_step)):.2f}°')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Hand Joint Error (degrees)')
    ax.set_title('Hand Joint Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hand_joint_error.png'), dpi=150)
    plt.close()

    # Compute metrics
    position_error = np.linalg.norm(real_ee_poses[:, :3] - sim_ee_poses[:, :3], axis=1)
    mean_position_error = np.mean(position_error)
    max_position_error = np.max(position_error)
    final_position_error = position_error[-1]

    arm_joint_error = np.abs(real_arm_joints - sim_arm_joints)
    mean_arm_joint_error = np.mean(arm_joint_error)
    max_arm_joint_error = np.max(arm_joint_error)

    hand_joint_error = np.abs(real_hand_joints - sim_hand_joints)
    mean_hand_joint_error = np.mean(hand_joint_error)
    max_hand_joint_error = np.max(hand_joint_error)

    # Plot 3: Position error over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, position_error * 1000, 'b-', linewidth=2)
    ax.axhline(y=mean_position_error * 1000, color='r', linestyle='--', label=f'Mean: {mean_position_error*1000:.1f}mm')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('EE Position Error Over Time (OPEN-LOOP REPLAY)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_error.png'), dpi=150)
    plt.close()

    metrics = {
        'mode': 'open_loop_replay',
        'mean_ee_position_error_m': float(mean_position_error),
        'max_ee_position_error_m': float(max_position_error),
        'final_ee_position_error_m': float(final_position_error),
        'mean_ee_position_error_mm': float(mean_position_error * 1000),
        'max_ee_position_error_mm': float(max_position_error * 1000),
        'final_ee_position_error_mm': float(final_position_error * 1000),
        'mean_arm_joint_error_rad': float(mean_arm_joint_error),
        'max_arm_joint_error_rad': float(max_arm_joint_error),
        'mean_hand_joint_error_rad': float(mean_hand_joint_error),
        'max_hand_joint_error_rad': float(max_hand_joint_error),
        'num_steps': min_len,
    }

    print("\n" + "=" * 60)
    print("TRAJECTORY COMPARISON METRICS (OPEN-LOOP REPLAY)")
    print("=" * 60)
    print(f"  Mean EE position error: {mean_position_error*1000:.2f} mm")
    print(f"  Max EE position error: {max_position_error*1000:.2f} mm")
    print(f"  Final EE position error: {final_position_error*1000:.2f} mm")
    print(f"  Mean arm joint error: {mean_arm_joint_error:.4f} rad")
    print(f"  Mean hand joint error: {mean_hand_joint_error:.4f} rad")
    print(f"  Trajectory length: {min_len} steps")

    return metrics


def replay_open_loop_trajectory(env, traj_obs: list, traj_actions: list, output_dir: str, action_type: str, camera_name: str = "fixed_camera",
                                blocking: bool = False, blocking_max_steps: int = 50, blocking_tol: float = 0.005):
    """Replay an open loop trajectory."""

    # Ensure the episode horizon in seconds is at least as long as the trajectory.
    # Use the config's decimation and sim.dt, since those live on the cfg, not the env.


    print("[INFO]: Resetting environment...")
    obs, _ = env.reset()
    print("[INFO]: Environment reset.")

    obs_list = [_extract_sim_observation(env, obs)]
    frames = []

    camera = env.unwrapped.scene[camera_name]
    action_space_shape = env.action_space.shape
    expected_action_dim = action_space_shape[-1]
    num_steps = len(traj_actions)

    robot = env.unwrapped.scene["robot"]
    ARM_DOF = 7
    blocking_steps_used = []

    with torch.inference_mode():
        for i in tqdm(range(num_steps)):
            base_action = torch.as_tensor(traj_actions[i], dtype=torch.float32).reshape(-1)

            if base_action.shape[-1] != expected_action_dim:
                raise ValueError(
                    f"Action dim mismatch at step {i}: got {base_action.shape[-1]}, "
                    f"expected {expected_action_dim}"
                )

            # Broadcast to all envs if needed
            if len(action_space_shape) == 1:
                action = base_action
            elif len(action_space_shape) == 2:
                action = torch.broadcast_to(base_action, action_space_shape)
            else:
                raise ValueError(f"Unsupported action space shape: {action_space_shape}")

            obs, _, _, _, _ = env.step(action)

            # Blocking: keep stepping with same action until arm converges
            extra = 0
            if blocking:
                for extra in range(1, blocking_max_steps + 1):
                    q = robot.data.joint_pos[0, :ARM_DOF]
                    if action_type == "joint":
                        q_target = base_action[:ARM_DOF].to(q.device)
                    else:
                        q_target = robot.data.joint_pos_target[0, :ARM_DOF]
                    err = (q - q_target).abs().max().item()
                    if err < blocking_tol:
                        break
                    obs, _, _, _, _ = env.step(action)
            blocking_steps_used.append(extra)

            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
            frames.append(rgb)
            obs_list.append(_extract_sim_observation(env, obs))
    
    ## save video
    output_video_path = os.path.join(output_dir, 'open_loop_trajectory.mp4')
    imageio.mimwrite(output_video_path, frames, fps=30)
    print(f"[INFO]: Video saved to {output_video_path}")

    if blocking and blocking_steps_used:
        arr = np.array(blocking_steps_used)
        print(f"\n=== Blocking stats (tol={blocking_tol} rad, max={blocking_max_steps}) ===")
        print(f"  mean extra steps: {arr.mean():.1f}")
        print(f"  p50:  {np.percentile(arr, 50):.0f}")
        print(f"  p90:  {np.percentile(arr, 90):.0f}")
        print(f"  max:  {arr.max():.0f}")
        print(f"  converged ({arr < blocking_max_steps} / {len(arr)}): "
              f"{(arr < blocking_max_steps).sum()}/{len(arr)} steps")

    return obs_list

def main():
    """Zero actions agent with Isaac Lab environment."""
    if args_cli.action_type == "delta_ee":
        task = "UW-FrankaLeap-IkRel-v0"
    elif args_cli.action_type == "abs_ee":
        task = "UW-FrankaLeap-IkAbs-v0"
    elif args_cli.action_type == "joint":
        task = "UW-FrankaLeap-JointAbs-v0"
    else:
        raise ValueError(f"Invalid action type: {args_cli.action_type}")

    output_dir = args_cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg = parse_env_cfg(
        task, device=args_cli.device, num_envs=num_envs, use_fabric=not args_cli.disable_fabric
    )
    traj_obs, traj_actions = load_episode(args_cli.trajectory_file, args_cli.action_type)
    env_cfg.episode_length_s = len(traj_actions) * env_cfg.decimation * env_cfg.sim.dt

    if args_cli.debug_cameras:
        env = gym.make(task, cfg=env_cfg)
        print("[INFO]: Resetting environment for camera debug...")
        env.reset()
        save_camera_candidates_comparison(env, output_dir, CAMERA_CANDIDATES)
        print("[INFO]: Use --camera <name> for videos (e.g. fixed_camera_no_inv). Exiting.")
        return

    if args_cli.trajectory_file is None:
        raise ValueError("--trajectory_file must be provided.")


    env = gym.make(task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Episode length: {env_cfg.episode_length_s}")
    print(f"[INFO]: Max episode steps: {env.unwrapped.max_episode_length}")
    

    sim_obs = replay_open_loop_trajectory(
        env, traj_obs, traj_actions, output_dir, args_cli.action_type,
        blocking=args_cli.blocking,
        blocking_max_steps=args_cli.blocking_max_steps,
        blocking_tol=args_cli.blocking_tol,
    )
    metrics = compare_trajectories(traj_obs, sim_obs, output_dir)

    metrics_path = os.path.join(output_dir, "open_loop_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO]: Metrics saved to {metrics_path}")



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
