# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Termination conditions for pour task.
# Ported from SingleHandPourRew in IsaacLab's bimanual_franka_pour_rew.py.
# Each function takes a PourReward instance so it can read reset references
# without env.__dict__ hacks.

from __future__ import annotations

import torch


def bottle_dropped(env, pour_rew) -> torch.Tensor:
    """Terminate if bottle falls below its reset height."""
    if pour_rew.reset_init_height is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    object_pose = env.scene[pour_rew.object_name]._data.root_state_w[:, :7].clone()
    object_pose[:, :3] -= env.scene.env_origins
    return object_pose[:, 2] < (pour_rew.reset_init_height - 0.05)


def bottle_too_far(env, pour_rew) -> torch.Tensor:
    """Terminate if bottle moves more than 1m (XY) from robot base."""
    object_pose = env.scene[pour_rew.object_name]._data.root_state_w[:, :7].clone()
    object_pose[:, :3] -= env.scene.env_origins
    distance = torch.linalg.norm(object_pose[:, :2], dim=1)
    return distance > 1.0


def cup_toppled(env, pour_rew) -> torch.Tensor:
    """Terminate if cup rotates or moves significantly from its reset pose.

    Uses quaternion angle difference to detect toppling — frame-independent.
    Thresholds (from IsaacLab):
    - rotation > 30 deg
    - XY shift > 10 cm
    - Z change > 4 cm
    - 3D displacement > 12 cm
    """
    if pour_rew.cup_reset_pos_ref is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Per-env guard: skip envs whose reset references haven't been captured yet
    valid = torch.any(pour_rew.cup_reset_pos_ref != 0, dim=1)
    if not valid.any():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    cup_state = env.scene[pour_rew.cup_name]._data.root_state_w[:, :7].clone()
    current_pos = cup_state[:, :3] - env.scene.env_origins
    current_quat = cup_state[:, 3:7]

    reset_pos = pour_rew.cup_reset_pos_ref
    reset_quat = pour_rew.cup_reset_quat_ref

    # Position checks
    pos_shift_3d = torch.linalg.norm(current_pos - reset_pos, dim=1)
    xy_shift = torch.linalg.norm(current_pos[:, :2] - reset_pos[:, :2], dim=1)
    z_change = torch.abs(current_pos[:, 2] - reset_pos[:, 2])

    # Rotation check using quaternion double-cover correction
    dot_product = (reset_quat * current_quat).sum(dim=1)
    current_quat_corrected = torch.where(
        dot_product.unsqueeze(1) < 0,
        -current_quat,
        current_quat,
    )
    from isaaclab.utils.math import quat_conjugate, quat_mul
    q_rel = quat_mul(quat_conjugate(reset_quat), current_quat_corrected)
    rotation_angle = 2.0 * torch.atan2(
        torch.norm(q_rel[:, 1:4], dim=1),
        torch.abs(q_rel[:, 0]),
    )

    rotated = rotation_angle > 0.524      # 30 degrees
    knocked_away = xy_shift > 0.10        # 10 cm
    z_moved = z_change > 0.04            # 4 cm
    moved_far = pos_shift_3d > 0.12      # 12 cm

    return (rotated | knocked_away | z_moved | moved_far) & valid
