# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop rollout a trajectory and save per-step viewport frames with sim point cloud
overlaid on real point cloud. Combines test_open_loop_trajectory and visualize_synthetic_pc."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Open-loop rollout trajectory and save per-step sim/real PC overlay frames."
)
parser.add_argument(
    "--trajectory_file",
    type=str,
    required=True,
    help="Path to trajectory: directory (e.g. episode_30) or episode .npy file.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/sysid/open_loop_pc_overlay",
    help="Output directory for frames and video.",
)
parser.add_argument(
    "--action_type",
    type=str,
    choices=["delta_ee", "abs_ee", "joint"],
    default="joint",
    help="Type of action to use.",
)
parser.add_argument(
    "--task",
    type=str,
    default="UW-FrankaLeap-GraspBottle-JointAbs-v0",
    help="Task with seg_pc (e.g. PourBottle, GraspPinkCup). Must have seg_pc observation.",
)
parser.add_argument(
    "--trajectory_downsample",
    type=int,
    default=2048,
    help="Downsample real point cloud to this many points.",
)
parser.add_argument(
    "--point_width_m",
    type=float,
    default=0.018,
    help="Point diameter in metres for USD Points.",
)
parser.add_argument(
    "--point_size",
    type=int,
    default=6,
    help="Point size in pixels for debug-draw.",
)
parser.add_argument(
    "--camera",
    type=str,
    default="fixed_camera",
    help="Camera name for viewport capture.",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["open_loop", "direct_joints"],
    default="open_loop",
    help="open_loop: rollout actions through controller. direct_joints: teleport robot to real joint positions each step (no controller).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print debug info when reset occurs (episode_length_buf, max_episode_length, etc.).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import numpy as np
import torch
import imageio
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg

import uwlab_assets.robots.franka_leap as franka_leap
from uwlab_tasks.utils.trajectory_utils import load_real_episode
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    ARM_RESET,
    HAND_RESET,
    EVAL_MODE,
    parse_franka_leap_env_cfg,
)
from uwlab_tasks.manager_based.manipulation.grasp.mdp.events import reset_robot_joints
from uwlab.utils.math import fps_points
from tqdm import tqdm


def load_episode(episode_file: str, action_type: str):
    """Load an episode and construct per-step actions for the requested action_type."""
    episode = load_real_episode(episode_file)
    obs_list = episode["obs"]
    raw_actions = episode["actions"]

    num_steps = min(len(obs_list), len(raw_actions))
    if num_steps == 0:
        raise ValueError("Episode has no steps: obs or actions list is empty.")

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
                if "commanded_ee_position" in obs_i:
                    arm_cmd = np.asarray(obs_i["commanded_ee_position"], dtype=np.float32).reshape(-1)
                else:
                    raise KeyError(
                        "Missing key 'commanded_ee_position' in obs for absolute EE "
                        f"action_type at step {i}"
                    )
            elif action_type == "joint":
                if "commanded_joint_positions" in obs_i:
                    arm_cmd = np.asarray(obs_i["commanded_joint_positions"], dtype=np.float32).reshape(-1)
                elif "ik_joint_pos_desired" in obs_i:
                    arm_cmd = np.asarray(obs_i["ik_joint_pos_desired"], dtype=np.float32).reshape(-1)
                else:
                    raise KeyError(
                        "Missing key 'ik_joint_pos_desired' or 'commanded_joint_positions' in obs for joint "
                        f"action_type at step {i}"
                    )
            else:
                raise ValueError(f"Unsupported action_type: {action_type}")

            base_action = np.concatenate([arm_cmd, hand_cmd], axis=-1)

        actions.append(base_action)

    return obs_list, actions, episode.get("pointclouds")


def _to_Nx3(pc: np.ndarray) -> np.ndarray | None:
    """Ensure point cloud is (N, 3)."""
    pc = np.asarray(pc, dtype=np.float64)
    if pc.ndim == 1:
        return None
    if pc.ndim == 2 and pc.shape[0] == 3 and pc.shape[1] != 3:
        pc = pc.T
    return pc


def downsample_pc(pc: np.ndarray, num_points: int, device: torch.device) -> np.ndarray:
    """Downsample point cloud to num_points via FPS. Returns (num_points, 3)."""
    pc = np.asarray(pc, dtype=np.float64)
    if pc.ndim == 2 and pc.shape[0] == 3 and pc.shape[1] != 3:
        pc = pc.T
    n = pc.shape[0]
    if n <= num_points:
        return pc
    pc_t = torch.from_numpy(pc).to(device).unsqueeze(0)
    pc_t = fps_points(pc_t, num_points)
    return pc_t[0].cpu().numpy()


def get_real_pc_for_frame(
    obs_list: list,
    pointclouds: list | None,
    frame_idx: int,
    env_device: torch.device,
    num_downsample: int,
) -> np.ndarray | None:
    """Get real point cloud for frame frame_idx."""
    if frame_idx >= len(obs_list):
        return None
    pc = None
    if pointclouds is not None and frame_idx < len(pointclouds):
        pc = _to_Nx3(np.asarray(pointclouds[frame_idx]))
    if pc is None and "seg_pc" in obs_list[frame_idx]:
        pc = _to_Nx3(np.asarray(obs_list[frame_idx]["seg_pc"]))
    if pc is not None and pc.shape[0] > 0:
        return downsample_pc(pc, num_downsample, env_device)
    return None


def _extract_sim_observation(obs: dict) -> dict | None:
    """Extract EE pose, arm joints, and hand joints from sim obs (policy group)."""
    policy = obs.get("policy")
    if not isinstance(policy, dict):
        return None
    ee_pose = policy.get("ee_pose")
    joint_pos = policy.get("joint_pos")
    if ee_pose is None or joint_pos is None:
        return None
    ee_pose_np = ee_pose[0].detach().cpu().numpy().reshape(-1)
    joint_pos_np = joint_pos[0].detach().cpu().numpy().reshape(-1)
    arm_joints = joint_pos_np[:7]
    hand_joints = joint_pos_np[7:7 + 16]
    return {
        "cartesian_position": ee_pose_np,
        "joint_positions": arm_joints,
        "gripper_position": hand_joints,
    }


def _extract_real_observation(obs: dict) -> dict | None:
    """Extract EE pose and joints from real obs (handles zarr/npy formats)."""
    cart = obs.get("cartesian_position")
    if cart is None:
        policy = obs.get("policy")
        if isinstance(policy, dict):
            cart = policy.get("ee_pose") or policy.get("cartesian_position")
    if cart is None:
        return None
    cart = np.asarray(cart).reshape(-1)[:7]

    arm = obs.get("joint_positions")
    hand = obs.get("gripper_position")
    if arm is None or hand is None:
        j = obs.get("joint_pos") or obs.get("joint_positions")
        policy = obs.get("policy")
        if isinstance(policy, dict) and j is None:
            j = policy.get("joint_pos")
        if j is not None:
            j = np.asarray(j).reshape(-1)
            if j.size >= 23:
                arm = j[:7]
                hand = j[7:23]
    if arm is None or hand is None:
        return None

    arm = np.asarray(arm).reshape(-1)[:7]
    hand = np.asarray(hand).reshape(-1)[:16]
    return {
        "cartesian_position": cart,
        "joint_positions": arm,
        "gripper_position": hand,
    }


def _extract_real_joints(obs: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract real arm (7) and hand (16) joint positions from obs."""
    if obs is None:
        return None, None

    def to_flat(a):
        a = np.asarray(a).reshape(-1)
        return a if a.size >= 23 else None

    policy = obs.get("policy")
    if isinstance(policy, dict):
        j = policy.get("joint_pos")
        if j is not None:
            j = to_flat(j)
            if j is not None:
                return j[:7], j[7:23]

    j = obs.get("joint_pos") or obs.get("joint_positions")
    if j is not None:
        j = to_flat(j)
        if j is not None:
            return j[:7], j[7:23]

    arm = obs.get("joint_positions")
    hand = obs.get("gripper_position")
    if arm is not None and hand is not None:
        arm = np.asarray(arm).reshape(-1)
        hand = np.asarray(hand).reshape(-1)
        if arm.size >= 7 and hand.size >= 16:
            return arm[:7], hand[:16]

    return None, None


def get_hold_action(env, obs: dict, action_type: str) -> torch.Tensor:
    """Build hold action in the format expected by the env for the given action_type."""
    unwrapped = env.unwrapped
    dev = unwrapped.device
    policy = obs.get("policy", {})
    joint_pos = policy.get("joint_pos")
    if joint_pos is None:
        raise KeyError("obs['policy']['joint_pos'] required for hold action")

    joint_pos_0 = joint_pos[0]
    if action_type == "joint":
        return joint_pos_0.unsqueeze(0)
    if action_type == "delta_ee":
        arm_part = torch.zeros(6, device=dev, dtype=torch.float32)
        hand_part = joint_pos_0[7:7 + 16]
        return torch.cat([arm_part, hand_part], dim=-1).unsqueeze(0)
    if action_type == "abs_ee":
        return unwrapped.cfg.warmup_action(unwrapped)
    raise ValueError(f"Unsupported action_type for hold: {action_type}")


def _build_hold_from_real_obs(obs: dict, action_type: str, device: torch.device, num_envs: int) -> torch.Tensor:
    """Build hold action from real (dataset) obs for use when teleporting to that obs."""
    arm, hand = _extract_real_joints(obs)
    if arm is None or hand is None:
        raise ValueError("Cannot extract joints from obs for hold")
    hand = np.asarray(hand).reshape(-1)[:16]

    if action_type == "joint":
        arr = np.concatenate([arm, hand], axis=-1)
    elif action_type == "delta_ee":
        arr = np.concatenate([np.zeros(6, dtype=np.float32), hand], axis=-1)
    elif action_type == "abs_ee":
        ee = obs.get("cartesian_position")
        if ee is None:
            raise KeyError("obs needs cartesian_position for abs_ee hold")
        ee = np.asarray(ee).reshape(-1)[:7]
        arr = np.concatenate([ee, hand], axis=-1)
    else:
        raise ValueError(f"Unsupported action_type: {action_type}")

    return torch.tensor(arr, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)


def reset_to_real_joints(env, first_real_obs, num_warmup_steps: int = 10, hold_from_obs: bool = False, action_type: str = "joint"):
    """Reset sim to match first-frame real joint positions."""
    arm, hand = _extract_real_joints(first_real_obs)
    if arm is None or hand is None:
        return None

    unwrapped = env.unwrapped
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
    reset_robot_joints(
        unwrapped,
        env_ids,
        SceneEntityCfg("robot"),
        arm.tolist(),
        hand.tolist(),
        arm_joint_limits=franka_leap.FRANKA_LEAP_ARM_JOINT_LIMITS,
    )
    if hold_from_obs:
        hold = _build_hold_from_real_obs(first_real_obs, action_type, unwrapped.device, unwrapped.num_envs)
    else:
        hold = torch.tensor(
            ARM_RESET + HAND_RESET,
            device=unwrapped.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(unwrapped.num_envs, 1)

    obs_after_reset = None
    for _ in range(num_warmup_steps):
        obs_after_reset, _, _, _, _ = env.step(hold)
    return obs_after_reset


def get_synthetic_seg_pc(obs) -> np.ndarray:
    """Get seg_pc from env observation."""
    seg_pc = obs["policy"]["seg_pc"][0]
    pc = seg_pc.detach().cpu().numpy()
    if pc.shape[0] == 3:
        pc = pc.T
    return np.asarray(pc, dtype=np.float64)


def draw_pointclouds_usd_points(
    real_pc: np.ndarray,
    sim_pc: np.ndarray,
    env_origin: np.ndarray,
    point_width_m: float = 0.018,
    prim_path_prefix: str = "/World/PointCloudViz",
) -> bool:
    """Spawn point clouds as UsdGeom.Points. Red = real, blue = sim."""
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
    except Exception:
        return False

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return False

    world_real = (real_pc + env_origin).astype(np.float32)
    world_sim = (sim_pc + env_origin).astype(np.float32)

    def add_points_prim(path: str, points_np: np.ndarray, rgb: tuple[float, float, float]) -> None:
        prim = stage.DefinePrim(path, "Points")
        pts = UsdGeom.Points(prim)
        points_list = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points_np]
        pts.CreatePointsAttr().Set(Vt.Vec3fArray(points_list))
        n = points_np.shape[0]
        pts.CreateWidthsAttr().Set(Vt.FloatArray([float(point_width_m)] * n))
        color_primvar = pts.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
        color_primvar.Set([Gf.Vec3f(rgb[0], rgb[1], rgb[2])])

    add_points_prim(f"{prim_path_prefix}/real_pc", world_real, (1.0, 0.0, 0.0))
    add_points_prim(f"{prim_path_prefix}/sim_pc", world_sim, (0.0, 0.4, 1.0))
    return True


def draw_pointclouds_debug_draw(
    real_pc: np.ndarray,
    sim_pc: np.ndarray,
    env_origin: np.ndarray,
    point_size: int = 6,
) -> bool:
    """Draw both point clouds using Isaac Sim debug draw."""
    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    except Exception:
        return False

    draw = omni_debug_draw.acquire_debug_draw_interface()
    if draw is None:
        return False

    if hasattr(draw, "clear_points"):
        draw.clear_points()
    if hasattr(draw, "clear_lines"):
        draw.clear_lines()

    world_real = real_pc + env_origin
    world_sim = sim_pc + env_origin

    def to_list_of_tuples(arr: np.ndarray) -> list:
        return [tuple(float(x) for x in row) for row in arr]

    points = to_list_of_tuples(world_real) + to_list_of_tuples(world_sim)
    n_real = len(world_real)
    n_sim = len(world_sim)
    colors = [(1.0, 0.0, 0.0, 0.8)] * n_real + [(0.0, 0.4, 1.0, 0.8)] * n_sim
    sizes = [point_size] * (n_real + n_sim)

    draw.draw_points(points, colors, sizes)
    return True


def capture_viewport_image(env, output_path: str, camera_name: str = "fixed_camera") -> bool:
    """Read the current camera RGB and save to output_path."""
    scene = env.unwrapped.scene
    try:
        cam = scene[camera_name]
    except KeyError:
        return False
    if "rgb" not in cam.data.output:
        return False
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(output_path, rgb)
    return True


def compare_trajectories(real_obs_list: list, sim_results: list, output_dir: str, mode: str = "open_loop") -> dict:
    """Compare real vs sim trajectories and generate plots (same as test_open_loop_trajectory)."""
    real_ee_poses = np.asarray(
        [np.asarray(o["cartesian_position"])[:7] for o in real_obs_list],
        dtype=np.float32,
    )
    real_arm_joints = np.asarray(
        [np.asarray(o["joint_positions"])[:7] for o in real_obs_list],
        dtype=np.float32,
    )
    real_hand_joints = np.asarray(
        [np.asarray(o["gripper_position"]) for o in real_obs_list],
        dtype=np.float32,
    )

    sim_ee_poses = np.asarray(
        [np.asarray(o["cartesian_position"])[:7] for o in sim_results],
        dtype=np.float32,
    )
    sim_arm_joints = np.asarray(
        [np.asarray(o["joint_positions"])[:7] for o in sim_results],
        dtype=np.float32,
    )
    sim_hand_joints = np.asarray(
        [np.asarray(o["gripper_position"]) for o in sim_results],
        dtype=np.float32,
    )

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
    replay_label = "DIRECT JOINTS" if mode == "direct_joints" else "OPEN-LOOP REPLAY"

    # Plot 1: EE position comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    labels = ["x", "y", "z", "qw", "qx", "qy"]
    for i in range(6):
        ax = axes[i]
        ax.plot(timesteps, real_ee_poses[:, i], "b-", label="Real", linewidth=2)
        ax.plot(timesteps, sim_ee_poses[:, i], "r--", label="Sim", linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(labels[i])
        ax.set_title(f"EE {labels[i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Real vs Sim EE Pose (OPEN-LOOP REPLAY)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "ee_pose_comparison.png"), dpi=150)
    plt.close()

    # Plot 2: Arm joint position comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i in range(7):
        ax = axes[i]
        ax.plot(timesteps, real_arm_joints[:, i], "b-", label="Real", linewidth=2)
        ax.plot(timesteps, sim_arm_joints[:, i], "r--", label="Sim", linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(f"Joint {i+1} (rad)")
        ax.set_title(f"Arm Joint {i+1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[7].axis("off")
    plt.suptitle(f"Real vs Sim Arm Joint Positions ({replay_label})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "arm_joint_comparison.png"), dpi=150)
    plt.close()

    # Plot 3: Hand joint position comparison
    num_hand_joints = min(real_hand_joints.shape[1], 16)
    n_cols = 4
    n_rows = int(np.ceil(num_hand_joints / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()
    for i in range(num_hand_joints):
        ax = axes[i]
        ax.plot(timesteps, real_hand_joints[:, i], "b-", label="Real", linewidth=2)
        ax.plot(timesteps, sim_hand_joints[:, i], "r--", label="Sim", linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(f"j{i} (rad)")
        mae_i = np.mean(np.abs(real_hand_joints[:, i] - sim_hand_joints[:, i]))
        ax.set_title(f"j{i} (MAE: {np.degrees(mae_i):.1f}°)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(num_hand_joints, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Hand Joints: Real vs Sim", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "hand_joint_comparison.png"), dpi=150)
    plt.close()

    # Plot 4: EE position error over time
    position_error = np.linalg.norm(real_ee_poses[:, :3] - sim_ee_poses[:, :3], axis=1)
    mean_position_error = np.mean(position_error)
    max_position_error = np.max(position_error)
    final_position_error = position_error[-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, position_error * 1000, "b-", linewidth=2)
    ax.axhline(y=mean_position_error * 1000, color="r", linestyle="--", label=f"Mean: {mean_position_error*1000:.1f}mm")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position Error (mm)")
    ax.set_title(f"EE Position Error Over Time ({replay_label})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "position_error.png"), dpi=150)
    plt.close()

    # Metrics
    mean_arm_joint_error = np.mean(np.abs(real_arm_joints - sim_arm_joints))
    max_arm_joint_error = np.max(np.abs(real_arm_joints - sim_arm_joints))
    mean_hand_joint_error = np.mean(np.abs(real_hand_joints - sim_hand_joints))
    max_hand_joint_error = np.max(np.abs(real_hand_joints - sim_hand_joints))

    mode_str = "direct_joints_pc_overlay" if mode == "direct_joints" else "open_loop_pc_overlay"
    metrics = {
        "mode": mode_str,
        "mean_ee_position_error_m": float(mean_position_error),
        "max_ee_position_error_m": float(max_position_error),
        "final_ee_position_error_m": float(final_position_error),
        "mean_ee_position_error_mm": float(mean_position_error * 1000),
        "max_ee_position_error_mm": float(max_position_error * 1000),
        "final_ee_position_error_mm": float(final_position_error * 1000),
        "mean_arm_joint_error_rad": float(mean_arm_joint_error),
        "max_arm_joint_error_rad": float(max_arm_joint_error),
        "mean_hand_joint_error_rad": float(mean_hand_joint_error),
        "max_hand_joint_error_rad": float(max_hand_joint_error),
        "num_steps": min_len,
    }

    print("\n" + "=" * 60)
    print(f"TRAJECTORY COMPARISON METRICS ({mode_str.upper()})")
    print("=" * 60)
    print(f"  Mean EE position error: {mean_position_error*1000:.2f} mm")
    print(f"  Max EE position error: {max_position_error*1000:.2f} mm")
    print(f"  Final EE position error: {final_position_error*1000:.2f} mm")
    print(f"  Mean arm joint error: {mean_arm_joint_error:.4f} rad")
    print(f"  Mean hand joint error: {mean_hand_joint_error:.4f} rad")
    print(f"  Trajectory length: {min_len} steps")

    return metrics


def main():
    task = args_cli.task
    output_dir = args_cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    traj_obs, traj_actions, pointclouds = load_episode(args_cli.trajectory_file, args_cli.action_type)
    num_steps = len(traj_actions)

    env_cfg = parse_franka_leap_env_cfg(
        task,
        EVAL_MODE,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    # Account for: (1) warmup in env.reset(), (2) warmup in reset_to_real_joints,
    # (3) 1 hold step for frame 0 overlay render, (4) num_steps trajectory steps.
    num_warmup_steps = getattr(env_cfg, "num_warmup_steps", 10)
    max_steps = 2 * num_warmup_steps + 1 +  2 * num_steps + 100 # 100 extra for safety
    episode_length_s = max_steps * env_cfg.decimation * env_cfg.sim.dt
    env_cfg.episode_length_s = episode_length_s
    if hasattr(env_cfg.terminations, "bottle_dropped"):
        env_cfg.terminations.bottle_dropped = None
    if hasattr(env_cfg.terminations, "bottle_too_far"):
        env_cfg.terminations.bottle_too_far = None
    if hasattr(env_cfg.terminations, "cup_toppled"):
        env_cfg.terminations.cup_toppled = None

    env = gym.make(task, cfg=env_cfg)
    if args_cli.debug:
        max_ep = int(env.unwrapped.max_episode_length)
        print(f"[DEBUG] num_steps={num_steps}, max_episode_length={max_ep}")

    obs, _ = env.reset()
    obs = reset_to_real_joints(env, traj_obs[0], num_warmup_steps=num_warmup_steps, hold_from_obs=True, action_type=args_cli.action_type)

    env_origin = env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
    action_space_shape = env.action_space.shape
    expected_action_dim = action_space_shape[-1]
    num_downsample = args_cli.trajectory_downsample

    frames = []
    sim_obs_list = []
    sim_obs_0 = _extract_sim_observation(obs)
    if sim_obs_0 is not None:
        sim_obs_list.append(sim_obs_0)
    
    REAL_PCD_LATENCY = 4 # 4  # 10 steps for real pcd to be ready
    with torch.inference_mode():
        # Frame 0: sim at real_obs[0] after reset, render against real_pcds[0]
        sim_pc_0 = get_synthetic_seg_pc(obs)
        real_pc_0 = get_real_pc_for_frame(
            traj_obs, pointclouds, REAL_PCD_LATENCY, env.unwrapped.device, num_downsample
        )
        if real_pc_0 is not None and real_pc_0.shape[0] > 0:
            draw_pointclouds_usd_points(
                real_pc_0, sim_pc_0, env_origin,
                point_width_m=args_cli.point_width_m,
            )
            # Step once so the viewport renders the overlay (hold keeps robot at real_obs[0])
            hold_action = get_hold_action(env, obs, args_cli.action_type)
            obs, _, _, _, _ = env.step(hold_action)
        rgb = env.unwrapped.scene[args_cli.camera].data.output["rgb"][0, ..., :3].cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        frames.append(rgb)
        imageio.imwrite(os.path.join(frames_dir, "frame_00000.png"), rgb)

        mode = args_cli.mode
        rollout_desc = "Direct joints" if mode == "direct_joints" else "Open-loop rollout"
        for i in tqdm(range(num_steps), desc=rollout_desc):
            if mode == "direct_joints":
                next_obs = traj_obs[i + 1] if i + 1 < len(traj_obs) else traj_obs[i]
                obs = reset_to_real_joints(env, next_obs, num_warmup_steps=1, hold_from_obs=True, action_type=args_cli.action_type)
                if obs is None:
                    raise ValueError(f"Cannot extract joints from traj_obs[{i+1}] for direct_joints mode")
                dev = env.unwrapped.device
                reward, terminated, truncated, info = None, torch.tensor([False], device=dev), torch.tensor([False], device=dev), {}
            else:
                base_action = torch.as_tensor(traj_actions[i], dtype=torch.float32).reshape(-1)
                if base_action.shape[-1] != expected_action_dim:
                    raise ValueError(
                        f"Action dim mismatch at step {i}: got {base_action.shape[-1]}, "
                        f"expected {expected_action_dim}"
                    )
                if len(action_space_shape) == 1:
                    action = base_action
                elif len(action_space_shape) == 2:
                    action = torch.broadcast_to(base_action, action_space_shape)
                else:
                    raise ValueError(f"Unsupported action space shape: {action_space_shape}")
                obs, reward, terminated, truncated, info = env.step(action)

            sim_obs_i = _extract_sim_observation(obs)
            if sim_obs_i is not None:
                sim_obs_list.append(sim_obs_i)

            if terminated.any() or truncated.any():
                print(f"[RESET] Frame {i}: terminated={terminated}, truncated={truncated}")
                if args_cli.debug:
                    unw = env.unwrapped
                    ep_buf = unw.episode_length_buf[0].item()
                    max_ep = int(unw.max_episode_length)
                    print(f"  episode_length_buf={ep_buf}, max_episode_length={max_ep}")
                    if hasattr(unw, "termination_manager"):
                        tm = unw.termination_manager
                        if hasattr(tm, "_terms"):
                            for name, term in tm._terms.items():
                                if term is not None and hasattr(term, "data"):
                                    val = term.data[0].item() if term.data.numel() > 0 else None
                                    print(f"  term[{name}]={val}")
            sim_pc = get_synthetic_seg_pc(obs)

            
            real_pc = get_real_pc_for_frame(
                traj_obs, pointclouds, i + 1 + REAL_PCD_LATENCY, env.unwrapped.device, num_downsample
            )

            if real_pc is not None and real_pc.shape[0] > 0:
                draw_pointclouds_usd_points(
                    real_pc, sim_pc, env_origin,
                    point_width_m=args_cli.point_width_m,
                )
            

            hold_action = get_hold_action(env, obs, args_cli.action_type)
            obs, reward, terminated, truncated, info = env.step(hold_action)
            if terminated.any() or truncated.any():
                print(f"[RESET] Frame {i} (after hold step): terminated={terminated}, truncated={truncated}")
                if args_cli.debug:
                    unw = env.unwrapped
                    ep_buf = unw.episode_length_buf[0].item()
                    max_ep = int(unw.max_episode_length)
                    print(f"  episode_length_buf={ep_buf}, max_episode_length={max_ep}")

            rgb = env.unwrapped.scene[args_cli.camera].data.output["rgb"][0, ..., :3].cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            frames.append(rgb)

            frame_path = os.path.join(frames_dir, f"frame_{i+1:05d}.png")
            imageio.imwrite(frame_path, rgb)

    video_name = "direct_joints_pc_overlay.mp4" if args_cli.mode == "direct_joints" else "open_loop_pc_overlay.mp4"
    video_path = os.path.join(output_dir, video_name)
    imageio.mimwrite(video_path, frames, fps=30)
    print(f"[INFO] Saved {len(frames)} frames to {frames_dir}")
    print(f"[INFO] Video saved to {video_path}")

    # Build real obs list and compare trajectories (joint/EE plots)
    real_obs_list = []
    for i in range(len(traj_obs)):
        ro = _extract_real_observation(traj_obs[i])
        if ro is not None:
            real_obs_list.append(ro)
    if real_obs_list and sim_obs_list:
        metrics = compare_trajectories(real_obs_list, sim_obs_list, output_dir, mode=args_cli.mode)
        metrics_name = "direct_joints_metrics.json" if args_cli.mode == "direct_joints" else "open_loop_metrics.json"
        metrics_path = os.path.join(output_dir, metrics_name)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Metrics saved to {metrics_path}")
    else:
        missing = "real" if not real_obs_list else "sim"
        print(f"[WARN] Skipping trajectory comparison: missing {missing} observations (ee_pose/joint_pos)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
