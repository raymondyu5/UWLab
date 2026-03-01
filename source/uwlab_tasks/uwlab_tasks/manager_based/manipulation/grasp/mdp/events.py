# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# reset_object_pose is an exact port of MultiRootStateCfg.reset_multi_root_state_uniform
# from IsaacLab/.../utils/grasp/config_rigids.py (lines 116-169) for a single object.
#
# reset_robot_joints writes the custom init pose from the pink cup YAML:
#   right_reset_joint_pose (arm) + right_reset_hand_joint_pose (hand)

from __future__ import annotations

import copy
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg


def reset_robot_joints(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    arm_joint_pos: list,
    hand_joint_pos: list,
):
    robot = env.scene[asset_cfg.name]
    num_envs_reset = len(env_ids)

    arm_pos = torch.tensor(arm_joint_pos, device=env.device, dtype=torch.float32)
    hand_pos = torch.tensor(hand_joint_pos, device=env.device, dtype=torch.float32)
    joint_pos = torch.cat([arm_pos, hand_pos], dim=0).unsqueeze(0).repeat(num_envs_reset, 1)
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_object_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    default_pos: tuple,
    default_rot_quat: tuple,
    pose_range: dict,
    reset_height: float,
):
    asset = env.scene[asset_cfg.name]

    default_root_state = torch.zeros((env.num_envs, 13), device=env.device, dtype=torch.float32)
    default_root_state[:, :3] = torch.tensor(list(default_pos), device=env.device, dtype=torch.float32)
    default_root_state[:, 3:7] = torch.tensor(list(default_rot_quat), device=env.device, dtype=torch.float32)

    asset.data.default_root_state[..., :7] = default_root_state[:, :7].clone()

    root_states = asset.data.default_root_state[env_ids].clone()

    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(orientations_delta, root_states[:, 3:7])

    velocity_range = {}
    range_list_vel = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges_vel = torch.tensor(range_list_vel, device=asset.device)
    rand_samples_vel = math_utils.sample_uniform(
        ranges_vel[:, 0], ranges_vel[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples_vel

    target_state = torch.cat([positions, orientations, velocities], dim=-1)
    # Overwrite z with fixed reset_height (matches IsaacLab line 162)
    target_state[:, 2] = reset_height

    asset.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(target_state[:, 7:], env_ids=env_ids)
