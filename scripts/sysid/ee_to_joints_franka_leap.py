# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find robot joint positions for a given end-effector position in the UW-FrankaLeap env.

Uses the UW-FrankaLeap-IkAbs-v0 environment: steps the sim with a constant desired EE pose
until the differential IK converges, then returns the resulting arm (7) and hand (16) joint positions.

Launch Isaac Sim first (same as other sysid scripts).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Find joint positions for a given EE position in UW-FrankaLeap env."
)
parser.add_argument(
    "--ee_pos",
    type=float,
    nargs=3,
    required=True,
    metavar=("X", "Y", "Z"),
    help="Desired end-effector position (m), in env world frame (same as ee_pose obs).",
)
parser.add_argument(
    "--ee_quat",
    type=float,
    nargs=4,
    default=None,
    metavar=("QW", "QX", "QY", "QZ"),
    help="Desired end-effector orientation (quat wxyz). Default: keep current orientation after reset.",
)
parser.add_argument(
    "--max_iters",
    type=int,
    default=500,
    help="Max simulation steps for IK convergence.",
)
parser.add_argument(
    "--position_tol",
    type=float,
    default=1e-4,
    help="Position error tolerance (m) to consider converged.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    HAND_RESET,
)


def ee_to_joints(
    ee_pos: tuple[float, float, float],
    ee_quat: tuple[float, float, float, float] | None = None,
    max_iters: int = 500,
    position_tol: float = 1e-4,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Find joint positions that achieve the given EE pose in UW-FrankaLeap.

    Args:
        ee_pos: Desired (x, y, z) in env world frame (same as observation ee_pose).
        ee_quat: Desired (qw, qx, qy, qz). If None, uses orientation after reset.
        max_iters: Max sim steps for IK convergence.
        position_tol: Position error (m) below which we consider converged.
        device: Device for the env.

    Returns:
        arm_joints: (7,) arm joint positions (rad).
        hand_joints: (16,) hand joint positions (rad).
        info: dict with "converged", "final_position_error", "num_steps", "achieved_ee_pose".
    """
    task = "UW-FrankaLeap-IkAbs-v0"
    env_cfg = parse_env_cfg(
        task,
        device=device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = (max_iters + 100) * env_cfg.decimation * env_cfg.sim.dt

    env = gym.make(task, cfg=env_cfg)
    obs, _ = env.reset()

    policy_obs = obs["policy"]
    ee_pose = policy_obs["ee_pose"][0]
    joint_pos = policy_obs["joint_pos"][0]

    desired_pos = torch.tensor(ee_pos, dtype=torch.float32, device=env.unwrapped.device)
    if ee_quat is not None:
        desired_quat = torch.tensor(
            ee_quat, dtype=torch.float32, device=env.unwrapped.device
        )
    else:
        desired_quat = ee_pose[3:7].clone()

    hand_joints = torch.tensor(
        HAND_RESET, dtype=torch.float32, device=env.unwrapped.device
    )
    action = torch.cat([desired_pos, desired_quat, hand_joints]).unsqueeze(0)

    converged = False
    step = 0
    err = float("inf")
    for step in range(max_iters):
        obs, _, _, _, _ = env.step(action)
        policy_obs = obs["policy"]
        ee_pose = policy_obs["ee_pose"][0]
        joint_pos = policy_obs["joint_pos"][0]

        err = (ee_pose[:3] - desired_pos).norm().item()
        if err < position_tol:
            converged = True
            break

    arm_joints = joint_pos[:7].clone()
    hand_joints_out = joint_pos[7 : 7 + 16].clone()
    info = {
        "converged": converged,
        "final_position_error_m": float(err),
        "num_steps": step + 1,
        "achieved_ee_pose": ee_pose.detach().cpu().numpy().tolist(),
    }
    env.close()
    return arm_joints, hand_joints_out, info


def main():
    arm_joints, hand_joints, info = ee_to_joints(
        ee_pos=tuple(args_cli.ee_pos),
        ee_quat=tuple(args_cli.ee_quat) if args_cli.ee_quat is not None else None,
        max_iters=args_cli.max_iters,
        position_tol=args_cli.position_tol,
        device=args_cli.device,
    )

    print("Target EE position (m):", list(args_cli.ee_pos))
    if args_cli.ee_quat is not None:
        print("Target EE quat (wxyz):", list(args_cli.ee_quat))
    print("Converged:", info["converged"])
    print("Final position error (m):", info["final_position_error_m"])
    print("Steps:", info["num_steps"])
    print("Achieved EE pose (pos + quat):", info["achieved_ee_pose"])
    print("\nArm joint positions (rad) [panda_joint1..panda_joint7]:")
    print(arm_joints.cpu().numpy().tolist())
    print("\nHand joint positions (rad) [j0..j15]:")
    print(hand_joints.cpu().numpy().tolist())
    print("\nFull joint position (23) for JointAbs / replay:")
    full = torch.cat([arm_joints, hand_joints]).cpu().numpy().tolist()
    print(full)


if __name__ == "__main__":
    main()
    simulation_app.close()
