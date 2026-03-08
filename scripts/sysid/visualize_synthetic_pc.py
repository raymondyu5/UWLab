# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthesize the seg_pc point cloud for UW-FrankaLeap-GraspPinkCup-JointAbs and visualize
it against the first point cloud from a given trajectory.

Runs headless by default (no GUI). Use --viewport_pc to draw red/blue point clouds and save
a rendered camera image (e.g. ..._viewport.png); works headless. Without --headless, the
viewport also stays open for a few seconds. Launch Isaac Sim first (same as other sysid scripts).
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Synthesize seg_pc for GraspPinkCup and compare to trajectory first frame."
)
parser.add_argument(
    "--trajectory_path",
    type=str,
    required=True,
    help="Path to trajectory: directory (e.g. episode_30) or .npy file (e.g. CL8384200N1_000000.npy). "
    "For a directory, first point cloud is from sorted CL*.npy or obs[0]['seg_pc']. "
    "For a .npy file, the file is loaded as the first point cloud (or as episode dict with obs[0]['seg_pc']).",
)
parser.add_argument(
    "--output",
    type=str,
    default="logs/sysid/synthetic_pc_compare.png",
    help="Output path for the comparison figure(s). Multiple views saved with _view0, _view1, ... suffix.",
)
parser.add_argument(
    "--trajectory_downsample",
    type=int,
    default=2048,
    help="Downsample loaded trajectory point cloud to this many points (default 2048, matches seg_pc).",
)
parser.add_argument(
    "--reset_to_first_frame",
    action="store_true",
    help="Reset robot to first-frame joint positions from trajectory so synthetic PC matches real pose.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--viewport_pc",
    action="store_true",
    default=False,
    help="Draw red/blue point clouds and save a rendered image (works headless). "
    "Uses debug-draw extension if available, otherwise spawns UsdGeom.Points (no extension). "
    "Without --headless, also keeps the viewport open for a few seconds.",
)
parser.add_argument(
    "--point_width_m",
    type=float,
    default=0.018,
    help="Point diameter in metres for USD Points (default 0.018 = 18 mm). Increase if points are hidden by meshes.",
)
parser.add_argument(
    "--point_size",
    type=int,
    default=6,
    help="Point size in pixels for debug-draw (default 6). Increase if points are hidden by meshes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from glob import glob

from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    ARM_RESET,
    HAND_RESET,
    EVAL_MODE,
    parse_franka_leap_env_cfg,
)
from uwlab_tasks.manager_based.manipulation.grasp.mdp.events import reset_robot_joints
from isaaclab.managers import SceneEntityCfg
import uwlab_assets.robots.franka_leap as franka_leap


def load_first_point_cloud(trajectory_path: str):
    """Load the first point cloud from a trajectory path (directory or .npy file).

    Returns:
        np.ndarray: (N, 3) point cloud, or None if not found.
    """
    if trajectory_path.endswith(".npy"):
        data = np.load(trajectory_path, allow_pickle=True)
        try:
            data = data.item()
        except ValueError:
            data = np.asarray(data)
        if isinstance(data, dict):
            obs_list = data.get("obs", [])
            if obs_list and "seg_pc" in obs_list[0]:
                pc = np.asarray(obs_list[0]["seg_pc"])
            else:
                return None
        else:
            pc = np.asarray(data)
        return _to_Nx3(pc)

    # Directory: look for episode and/or CL*.npy
    episode_name = os.path.basename(trajectory_path.rstrip("/"))
    episode_id = episode_name.split("_")[-1] if "_" in episode_name else None
    episode_file = (
        os.path.join(trajectory_path, f"episode_{episode_id}.npy")
        if episode_id is not None
        else None
    )
    pcd_files = sorted(glob(os.path.join(trajectory_path, "CL*.npy")))

    if pcd_files:
        pc = np.load(pcd_files[0])
        return _to_Nx3(np.asarray(pc))

    if episode_file and os.path.isfile(episode_file):
        data = np.load(episode_file, allow_pickle=True).item()
        obs_list = data.get("obs", [])
        if obs_list and "seg_pc" in obs_list[0]:
            return _to_Nx3(np.asarray(obs_list[0]["seg_pc"]))

    return None


def _to_Nx3(pc: np.ndarray) -> np.ndarray:
    """Ensure point cloud is (N, 3)."""
    pc = np.asarray(pc, dtype=np.float64)
    if pc.ndim == 1:
        return None
    if pc.ndim == 2 and pc.shape[0] == 3 and pc.shape[1] != 3:
        pc = pc.T
    return pc


def _extract_real_joints(first_real_episode_obs: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract real arm (7) and hand (16) joint positions from first-frame obs.
    Tries policy joint_pos, flat joint_pos/joint_positions, and joint_positions+gripper_position.
    """
    if first_real_episode_obs is None:
        return None, None

    def to_flat(a):
        a = np.asarray(a).reshape(-1)
        return a if a.size >= 23 else None

    # Nested (gym-style): obs["policy"]["joint_pos"] -> (1, 23) or (23,)
    policy = first_real_episode_obs.get("policy")
    if isinstance(policy, dict):
        j = policy.get("joint_pos")
        if j is not None:
            j = to_flat(j)
            if j is not None:
                return j[:7], j[7:23]

    # Flat top-level
    j = first_real_episode_obs.get("joint_pos") or first_real_episode_obs.get("joint_positions")
    if j is not None:
        j = to_flat(j)
        if j is not None:
            return j[:7], j[7:23]

    # Dataset format: joint_positions (7) + gripper_position (16)
    arm = first_real_episode_obs.get("joint_positions")
    hand = first_real_episode_obs.get("gripper_position")
    if arm is not None and hand is not None:
        arm = np.asarray(arm).reshape(-1)
        hand = np.asarray(hand).reshape(-1)
        if arm.size >= 7 and hand.size >= 16:
            return arm[:7], hand[:16]

    return None, None


def draw_pointclouds_usd_points(
    trajectory_pc: np.ndarray,
    synthetic_pc: np.ndarray,
    env_origin: np.ndarray,
    point_width_m: float = 0.01,
    prim_path_prefix: str = "/World/PointCloudViz",
) -> bool:
    """Spawn point clouds as UsdGeom.Points prims (no extension required). Works headless.
    Red = trajectory, blue = synthetic. Returns True on success.
    """
    if not getattr(args_cli, "viewport_pc", False):
        return False

    try:
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
    except Exception:
        return False

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return False

    world_traj = (trajectory_pc + env_origin).astype(np.float32)
    world_syn = (synthetic_pc + env_origin).astype(np.float32)

    def add_points_prim(path: str, points_np: np.ndarray, rgb: tuple[float, float, float]) -> None:
        prim = stage.DefinePrim(path, "Points")
        pts = UsdGeom.Points(prim)
        points_list = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points_np]
        pts.CreatePointsAttr().Set(Vt.Vec3fArray(points_list))
        n = points_np.shape[0]
        pts.CreateWidthsAttr().Set(Vt.FloatArray([float(point_width_m)] * n))
        color_primvar = pts.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
        color_primvar.Set([Gf.Vec3f(rgb[0], rgb[1], rgb[2])])

    add_points_prim(f"{prim_path_prefix}/trajectory_pc", world_traj, (1.0, 0.0, 0.0))
    add_points_prim(f"{prim_path_prefix}/synthetic_pc", world_syn, (0.0, 0.4, 1.0))
    return True


def draw_pointclouds_debug_draw(
    trajectory_pc: np.ndarray,
    synthetic_pc: np.ndarray,
    env_origin: np.ndarray,
    point_size: int = 2,
) -> bool:
    """Draw both point clouds using Isaac Sim debug draw (viewport and offscreen render).
    Returns True if drawing was performed, False if draw unavailable (e.g. extension not loaded).
    Safe to call headless; points can appear in camera captures when rendering.
    """
    if not getattr(args_cli, "viewport_pc", False):
        return False

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

    world_traj = trajectory_pc + env_origin
    world_syn = synthetic_pc + env_origin

    def to_list_of_tuples(arr: np.ndarray) -> list:
        return [tuple(float(x) for x in row) for row in arr]

    points = to_list_of_tuples(world_traj) + to_list_of_tuples(world_syn)
    n_traj = len(world_traj)
    n_syn = len(world_syn)
    colors = [(1.0, 0.0, 0.0, 0.8)] * n_traj + [(0.0, 0.4, 1.0, 0.8)] * n_syn
    sizes = [point_size] * (n_traj + n_syn)

    draw.draw_points(points, colors, sizes)
    return True


def capture_viewport_image(env, output_path: str, camera_name: str = "fixed_camera") -> bool:
    """Read the current camera RGB and save to output_path. Caller must step the env first so the camera has fresh data."""
    scene = env.unwrapped.scene
    camera_name = str(camera_name)
    
    try:
        cam = scene[camera_name]
    except KeyError:
        return False
    if "rgb" not in cam.data.output:
        return False
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    import imageio
    imageio.imwrite(output_path, rgb)
    return True


def viewport_hold(simulation_app, hold_s: float = 5.0) -> None:
    """Keep the app running so the viewport can update (no-op when headless)."""
    if getattr(args_cli, "headless", True) or hold_s <= 0 or simulation_app is None:
        return
    import time
    end_time = time.time() + hold_s
    while time.time() < end_time:
        if getattr(simulation_app, "is_running", lambda: False)():
            if getattr(simulation_app, "update", None) is not None:
                simulation_app.update()
            else:
                time.sleep(0.016)
        else:
            break


def downsample_pc(pc: np.ndarray, num_points: int, seed: int = 0) -> np.ndarray:
    """Downsample point cloud to num_points by random choice. Returns (num_points, 3)."""
    n = pc.shape[0]
    if n <= num_points:
        return pc
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=num_points, replace=False)
    return pc[idx]


def get_synthetic_seg_pc(
    obs,
) -> np.ndarray:
    """Get seg_pc from env"""
    seg_pc = obs["policy"]["seg_pc"][0]
    return seg_pc.detach().cpu().numpy()


def reset_to_first_frame(env, first_real_episode_obs):
    num_reset_steps = 10
    arm, hand = _extract_real_joints(first_real_episode_obs)
    
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
    hold = torch.tensor(
        ARM_RESET + HAND_RESET,
        device=unwrapped.device,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(unwrapped.num_envs, 1)
    
    for _ in range(num_reset_steps):
        obs_after_reset, _, _, _, _ = env.step(hold)

    return obs_after_reset

def main():
    trajectory_path = args_cli.trajectory_path
    first_pc = load_first_point_cloud(trajectory_path)
    if first_pc is None:
        raise SystemExit(
            f"Could not load first point cloud from trajectory path: {trajectory_path}"
        )

    task = "UW-FrankaLeap-PourBottle-JointAbs-v0"
    env_cfg = parse_franka_leap_env_cfg(
        task,
        EVAL_MODE,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(task, cfg=env_cfg)
    obs, _ = env.reset()

    # Load first-frame obs from trajectory (for joint plot and optional reset_to_first_frame)
    episode_data = None
    if trajectory_path.endswith(".npy"):
        data = np.load(trajectory_path, allow_pickle=True).item()
        episode_data = data
    else:
        episode_name = os.path.basename(trajectory_path.rstrip("/"))
        episode_id = episode_name.split("_")[-1] if "_" in episode_name else None
        if episode_id is not None:
            ep_file = os.path.join(trajectory_path, f"episode_{episode_id}.npy")
            if os.path.isfile(ep_file):
                episode_data = np.load(ep_file, allow_pickle=True).item()
    first_real_episode_obs = episode_data["obs"][0] if episode_data and episode_data.get("obs") else None


    if args_cli.reset_to_first_frame:
        obs = reset_to_first_frame(env, first_real_episode_obs)

    synthetic_pc =  get_synthetic_seg_pc(
        obs,
    )

    sim_joint_pos = obs["policy"]["joint_pos"][0].detach().cpu().numpy()

    # seg_pc from env is (3, N); convert to (N, 3)
    if synthetic_pc.shape[0] == 3:
        synthetic_pc = synthetic_pc.T
    synthetic_pc = np.asarray(synthetic_pc, dtype=np.float64)

    # Downsample trajectory PC to match
    num_downsample = args_cli.trajectory_downsample
    first_pc = downsample_pc(first_pc, num_downsample)

    # Viewport / headless render: draw points (red=trajectory, blue=synthetic) and save camera image
    env_origin = env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
    if getattr(args_cli, "viewport_pc", False):
        drawn = draw_pointclouds_debug_draw(
            first_pc, synthetic_pc, env_origin, point_size=getattr(args_cli, "point_size", 6)
        )
        if not drawn:
            drawn = draw_pointclouds_usd_points(
                first_pc, synthetic_pc, env_origin,
                point_width_m=getattr(args_cli, "point_width_m", 0.018),
            )
            if drawn:
                print("[INFO] Drew point clouds using USD Points (no debug-draw extension).")
        if drawn:
            # Step once so the scene and camera render (including the drawn points)
            hold_action = torch.tensor(
                sim_joint_pos,
                device=env.unwrapped.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            env.step(hold_action)
            _vp_base = args_cli.output
            if _vp_base.endswith(".png") or _vp_base.endswith(".jpg"):
                _vp_base = _vp_base[:-4]
            viewport_image_path = f"{_vp_base}_viewport.png"
            if capture_viewport_image(env, viewport_image_path):
                print(f"[INFO] Saved viewport render with point clouds to {viewport_image_path}")
            else:
                print("[INFO] Drew point clouds but could not capture camera image.")
            viewport_hold(simulation_app, hold_s=5.0)
        else:
            print("[INFO] --viewport_pc set but could not draw (debug draw and USD stage unavailable).")

    env.close()

    # Multiple 3D views: (elev_deg, azim_deg, label)
    views = [
        (25, 45, "front_right"),
        (25, 135, "back_right"),
        (25, 225, "back_left"),
        (25, 315, "front_left"),
        (90, 0, "top"),
    ]

    import matplotlib.pyplot as plt

    base_path = args_cli.output
    if base_path.endswith(".png"):
        base_path = base_path[:-4]
    elif base_path.endswith(".jpg"):
        base_path = base_path[:-4]
    out_dir = os.path.dirname(base_path)
    os.makedirs(out_dir or ".", exist_ok=True)

    # Save raw points from each point cloud
    points_npy_path = f"{base_path}_points.npy"
    # Real (trajectory) joint positions for first frame
    real_arm, real_hand = _extract_real_joints(first_real_episode_obs)
    if first_real_episode_obs is not None and real_arm is None:
        print(f"[INFO] First-frame obs keys: {list(first_real_episode_obs.keys())}; could not find joint data.")
        if isinstance(first_real_episode_obs.get("policy"), dict):
            print(f"       policy keys: {list(first_real_episode_obs['policy'].keys())}")

    points_dict = {
        "trajectory_pc": first_pc,
        "synthetic_pc": synthetic_pc,
    }
    if real_arm is not None:
        points_dict["real_arm_joint_pos"] = real_arm
        points_dict["real_hand_joint_pos"] = real_hand
    points_dict["sim_arm_joint_pos"] = sim_joint_pos[:7]
    points_dict["sim_hand_joint_pos"] = sim_joint_pos[7:23]

    np.save(points_npy_path, points_dict, allow_pickle=True)
    print(f"[INFO] Saved raw points to {points_npy_path} (trajectory_pc: {first_pc.shape}, synthetic_pc: {synthetic_pc.shape})")

    # Joint position comparison (arm + gripper)
    fig_joints, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    sim_arm = sim_joint_pos[:7]
    sim_hand = sim_joint_pos[7:23]

    ax_arm = axes[0]
    x_arm = np.arange(7)
    ax_arm.bar(x_arm - 0.2, sim_arm, width=0.35, label="Sim", color="blue", alpha=0.7)
    if real_arm is not None:
        ax_arm.bar(x_arm + 0.2, real_arm, width=0.35, label="Real", color="red", alpha=0.7)
    ax_arm.set_ylabel("Joint position (rad)")
    ax_arm.set_title("Arm joints (panda_joint1–7)")
    ax_arm.set_xticks(x_arm)
    ax_arm.set_xticklabels([f"J{i+1}" for i in range(7)])
    ax_arm.legend()
    ax_arm.grid(True, alpha=0.3)

    ax_hand = axes[1]
    x_hand = np.arange(16)
    ax_hand.bar(x_hand - 0.2, sim_hand, width=0.35, label="Sim", color="blue", alpha=0.7)
    if real_hand is not None:
        ax_hand.bar(x_hand + 0.2, real_hand, width=0.35, label="Real", color="red", alpha=0.7)
    ax_hand.set_ylabel("Joint position (rad)")
    ax_hand.set_xlabel("Finger joint index")
    ax_hand.set_title("Gripper / hand joints (j0–j15)")
    ax_hand.set_xticks(x_hand)
    ax_hand.set_xticklabels([f"j{i}" for i in range(16)])
    ax_hand.legend()
    ax_hand.grid(True, alpha=0.3)

    fig_joints.suptitle("Joint positions: Sim vs Real (first frame)", fontsize=12)
    plt.tight_layout()
    joints_path = f"{base_path}_joints.png"
    plt.savefig(joints_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved joint comparison to {joints_path}")

    for elev, azim, label in views:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            first_pc[:, 0],
            first_pc[:, 1],
            first_pc[:, 2],
            c="red",
            s=1,
            alpha=0.5,
            label="Trajectory (first frame)",
        )
        ax.scatter(
            synthetic_pc[:, 0],
            synthetic_pc[:, 1],
            synthetic_pc[:, 2],
            c="blue",
            s=1,
            alpha=0.5,
            label="Synthetic (seg_pc)",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.legend()
        ax.set_title(f"Synthetic seg_pc vs trajectory first point cloud ({label})")
        ax.view_init(elev=elev, azim=azim)
        out_path = f"{base_path}_view_{label}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {label} view to {out_path}")

    print(f"  Trajectory PC: {first_pc.shape[0]} points (downsampled to {num_downsample})")
    print(f"  Synthetic PC:  {synthetic_pc.shape[0]} points")


if __name__ == "__main__":
    main()
    simulation_app.close()
