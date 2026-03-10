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
    """Terminate if cup rotates more than 30 degrees from its upright spawn orientation."""
    from isaaclab.utils.math import quat_conjugate, quat_mul
    cup_state = env.scene[pour_rew.cup_name]._data.root_state_w[:, :7].clone()
    current_quat = cup_state[:, 3:7]

    # Fixed upright orientation: PINK_CUP_POUR_ROT = (0.707, 0.707, 0.0, 0.0)
    spawn_quat = torch.tensor([[0.707, 0.707, 0.0, 0.0]], device=env.device).expand(env.num_envs, -1)

    dot = (spawn_quat * current_quat).sum(dim=1)
    current_quat_corrected = torch.where(dot.unsqueeze(1) < 0, -current_quat, current_quat)
    q_rel = quat_mul(quat_conjugate(spawn_quat), current_quat_corrected)
    rotation_angle = 2.0 * torch.atan2(
        torch.norm(q_rel[:, 1:4], dim=1),
        torch.abs(q_rel[:, 0]),
    )
    return rotation_angle > 0.524  # 30 degrees
