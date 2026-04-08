# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils

# XY center-to-center distance thresholds (metres)
CUBE_POPTART_NEAR_DIST = 0.15      # outer "approaching" threshold
CUBE_POPTART_CONTACT_DIST = 0.10   # inner "touching" threshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cube_poptart_xy_dist(env) -> torch.Tensor:
    cube_pos = env.scene["grasp_object"].data.root_pos_w - env.scene.env_origins
    poptart_pos = env.scene["box"].data.root_pos_w - env.scene.env_origins
    return torch.norm(cube_pos[:, :2] - poptart_pos[:, :2], dim=1)


def _poptart_lin_speed(env) -> torch.Tensor:
    return torch.norm(env.scene["box"].data.root_lin_vel_w, dim=1)


def _poptart_toppled(env, angle_thresh_rad: float = 0.524) -> torch.Tensor:
    """True when poptart has rotated more than angle_thresh_rad from its upright spawn."""
    poptart = env.scene["box"]
    current_quat = poptart.data.root_quat_w
    # POPTART_PUSH_ROT is 90deg around X — corrects mesh from Y-up to Z-up
    spawn_quat = torch.tensor(
        [[0.707, 0.707, 0.0, 0.0]], device=env.device, dtype=current_quat.dtype
    ).expand(env.num_envs, -1)
    dot = (spawn_quat * current_quat).sum(dim=1)
    current_quat_corrected = torch.where(dot.unsqueeze(1) < 0, -current_quat, current_quat)
    q_rel = math_utils.quat_mul(math_utils.quat_conjugate(spawn_quat), current_quat_corrected)
    rotation_angle = 2.0 * torch.atan2(
        torch.norm(q_rel[:, 1:4], dim=1),
        torch.abs(q_rel[:, 0]),
    )
    return rotation_angle > angle_thresh_rad


def _joint_vel_l2(env, asset_name: str) -> torch.Tensor:
    robot = env.scene[asset_name]
    return torch.sum(robot.data.joint_vel ** 2, dim=1)


def _action_rate_l2(env) -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


def nan_to_num(val) -> torch.Tensor:
    return torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Metric predicates (used in metrics_spec)
# ---------------------------------------------------------------------------

def is_cube_near_poptart(env) -> torch.Tensor:
    """Cube XY center within CUBE_POPTART_NEAR_DIST of poptart center."""
    return _cube_poptart_xy_dist(env) < CUBE_POPTART_NEAR_DIST


def is_cube_touching_poptart(env) -> torch.Tensor:
    """Cube XY center within CUBE_POPTART_CONTACT_DIST of poptart center."""
    return _cube_poptart_xy_dist(env) < CUBE_POPTART_CONTACT_DIST


def is_success(env) -> torch.Tensor:
    """Cube XY center within contact distance of poptart (poptart may be moving)."""
    return is_cube_touching_poptart(env)


# ---------------------------------------------------------------------------
# Reward terms (wired as RewTerm in task __post_init__)
# ---------------------------------------------------------------------------

def push_approaching(env) -> torch.Tensor:
    """Sparse reward: 1.0 when cube is within CUBE_POPTART_NEAR_DIST of poptart."""
    return nan_to_num(is_cube_near_poptart(env).float())


def push_near(env) -> torch.Tensor:
    """Sparse reward: 1.0 when cube is within CUBE_POPTART_NEAR_DIST of poptart."""
    return nan_to_num(is_cube_near_poptart(env).float())


def push_success(env) -> torch.Tensor:
    """Sparse reward: 1.0 on success (cube contact distance to poptart)."""
    return nan_to_num(is_success(env).float())


def push_poptart_velocity(env) -> torch.Tensor:
    """Continuous penalty: poptart linear speed (m/s). Penalises sliding the poptart."""
    return nan_to_num(_poptart_lin_speed(env))


def push_poptart_topple(env) -> torch.Tensor:
    """Sparse penalty: 1.0 when poptart has tipped more than ~30 degrees."""
    return nan_to_num(_poptart_toppled(env).float())


def push_joint_vel_l2(env, asset_name: str = "robot") -> torch.Tensor:
    return nan_to_num(_joint_vel_l2(env, asset_name))


def push_action_rate_l2(env) -> torch.Tensor:
    return nan_to_num(_action_rate_l2(env))
