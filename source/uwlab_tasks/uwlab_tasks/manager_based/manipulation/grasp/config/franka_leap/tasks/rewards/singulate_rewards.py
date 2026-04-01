# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

GRASPED_Z_THRESHOLD = 0.15    # bottle centre z (env-local) above table to count as lifted

# Target: workspace centre at z=0.30. Must match SINGULATE_TARGET_POS in bottle_singulate.py.
SINGULATE_TARGET_POS = (0.55, 0.0, 0.30)
SUCCESS_DIST_THRESHOLD = 0.1  # 3-D distance to target to count as success

# Box approach corridor: box is "clear" when its XY distance from the bottle
# exceeds this threshold. The nominal spawn separation is ~0.10 m
# (BOTTLE_WIDTH/2 + gap + BOX_WIDTH/2 = 0.04 + 0.015 + 0.045),
# so this triggers as soon as the box has been pushed meaningfully away.
BOX_CLEAR_DIST_THRESHOLD = 0.12


# ---------------------------------------------------------------------------
# Internal helpers (defined first so predicates can call them)
# ---------------------------------------------------------------------------

def _box_bottle_xy_separation(env) -> torch.Tensor:
    """XY distance between box centre and bottle centre (env-local)."""
    bottle_pos = env.scene["grasp_object"].data.root_pos_w - env.scene.env_origins
    box_pos = env.scene["box"].data.root_pos_w - env.scene.env_origins
    return torch.norm(bottle_pos[:, :2] - box_pos[:, :2], dim=1)


def _joint_vel_l2(env, asset_name: str) -> torch.Tensor:
    robot = env.scene[asset_name]
    return torch.sum(robot.data.joint_vel ** 2, dim=1)


def _joint_pos_limits(env, asset_name: str, soft_ratio: float = 0.9) -> torch.Tensor:
    robot = env.scene[asset_name]
    joint_pos = robot.data.joint_pos
    lower = robot.data.soft_joint_pos_limits[:, :, 0]
    upper = robot.data.soft_joint_pos_limits[:, :, 1]
    lower_violation = torch.clamp(lower * soft_ratio - joint_pos, min=0.0)
    upper_violation = torch.clamp(joint_pos - upper * soft_ratio, min=0.0)
    return torch.sum(lower_violation + upper_violation, dim=1)


def _action_rate_l2(env) -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


def nan_to_num(val) -> torch.Tensor:
    return torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Metric predicates (return bool tensor, used by metrics_spec)
# ---------------------------------------------------------------------------

def is_grasped(env) -> torch.Tensor:
    bottle = env.scene["grasp_object"]
    pos = bottle.data.root_pos_w - env.scene.env_origins
    return pos[:, 2] > GRASPED_Z_THRESHOLD


def is_success(env) -> torch.Tensor:
    """Bottle centre within SUCCESS_DIST_THRESHOLD of the fixed target position."""
    bottle = env.scene["grasp_object"]
    pos = bottle.data.root_pos_w - env.scene.env_origins
    target = torch.tensor(list(SINGULATE_TARGET_POS), device=pos.device, dtype=pos.dtype)
    return torch.norm(pos - target.unsqueeze(0), dim=1) < SUCCESS_DIST_THRESHOLD


def is_box_clear(env) -> torch.Tensor:
    """Box has been pushed outside the approach corridor (XY sep from bottle > threshold)."""
    return _box_bottle_xy_separation(env) > BOX_CLEAR_DIST_THRESHOLD


# ---------------------------------------------------------------------------
# Reward terms (wire as RewTerm)
# ---------------------------------------------------------------------------

def singulate_grasped(env) -> torch.Tensor:
    return nan_to_num(is_grasped(env).float())


def singulate_success(env) -> torch.Tensor:
    return nan_to_num(is_success(env).float())


def singulate_box_separation(env) -> torch.Tensor:
    """Sparse reward: 1 when box has been pushed outside the approach corridor (XY sep > 0.10 m)."""
    return nan_to_num(is_box_clear(env).float())


def singulate_joint_vel_l2(env, asset_name: str = "robot") -> torch.Tensor:
    return nan_to_num(_joint_vel_l2(env, asset_name))


def singulate_joint_pos_limits(env, asset_name: str = "robot") -> torch.Tensor:
    return nan_to_num(_joint_pos_limits(env, asset_name))


def singulate_action_rate_l2(env) -> torch.Tensor:
    return nan_to_num(_action_rate_l2(env))
