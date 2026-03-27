# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Termination conditions for pour task.
# State and thresholds are task-wired through explicit params.

from __future__ import annotations

import torch


DEFAULT_RESET_HEIGHT_KEY = "_pour_bottle_reset_height"


def capture_bottle_reset_height(
    env,
    env_ids,
    object_name: str = "grasp_object",
    reset_height_key: str = DEFAULT_RESET_HEIGHT_KEY,
):
    """Capture object reset Z (env-local) for later termination checks."""
    if not hasattr(env, reset_height_key):
        setattr(env, reset_height_key, torch.zeros(env.num_envs, device=env.device, dtype=torch.float32))
    reset_heights = getattr(env, reset_height_key)
    object_state = env.scene[object_name]._data.root_state_w[env_ids, :7].clone()
    object_z_local = object_state[:, 2] - env.scene.env_origins[env_ids, 2]
    reset_heights[env_ids] = object_z_local


def bottle_dropped(
    env,
    object_name: str = "grasp_object",
    reset_height_key: str = DEFAULT_RESET_HEIGHT_KEY,
    z_margin: float = 0.05,
) -> torch.Tensor:
    """Terminate if bottle falls below its reset height."""
    if not hasattr(env, reset_height_key):
        raise RuntimeError(
            f"Missing reset height state '{reset_height_key}'. "
            "Wire capture_bottle_reset_height as a reset EventTerm before bottle_dropped."
        )
    reset_heights = getattr(env, reset_height_key)
    object_pose = env.scene[object_name]._data.root_state_w[:, :7].clone()
    object_pose[:, :3] -= env.scene.env_origins
    return object_pose[:, 2] < (reset_heights - z_margin)


def bottle_too_far(
    env,
    object_name: str = "grasp_object",
    max_xy_dist: float = 1.0,
) -> torch.Tensor:
    """Terminate if bottle moves more than 1m (XY) from robot base."""
    object_pose = env.scene[object_name]._data.root_state_w[:, :7].clone()
    object_pose[:, :3] -= env.scene.env_origins
    distance = torch.linalg.norm(object_pose[:, :2], dim=1)
    return distance > max_xy_dist


def cup_toppled(
    env,
    cup_name: str = "pink_cup",
    spawn_quat: tuple[float, float, float, float] = (0.707, 0.707, 0.0, 0.0),
    angle_thresh_rad: float = 0.524,
) -> torch.Tensor:
    """Terminate if cup rotates more than 30 degrees from spawn. Uses fixed PINK_CUP_POUR_ROT."""
    from isaaclab.utils.math import quat_conjugate, quat_mul

    cup = env.scene[cup_name]
    current_quat = cup.data.root_quat_w  # (num_envs, 4) wxyz
    spawn_quat_t = torch.tensor([list(spawn_quat)], device=env.device).expand(env.num_envs, -1)

    dot = (spawn_quat_t * current_quat).sum(dim=1)
    current_quat_corrected = torch.where(dot.unsqueeze(1) < 0, -current_quat, current_quat)
    q_rel = quat_mul(quat_conjugate(spawn_quat_t), current_quat_corrected)
    rotation_angle = 2.0 * torch.atan2(
        torch.norm(q_rel[:, 1:4], dim=1),
        torch.abs(q_rel[:, 0]),
    )
    return rotation_angle > angle_thresh_rad
