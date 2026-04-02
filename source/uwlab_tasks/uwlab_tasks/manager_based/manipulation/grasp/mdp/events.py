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


def _sample_spherical_point(origin, radius, theta_range_rad, phi_range_rad, num_envs, device):
    theta = torch.empty(num_envs, device=device).uniform_(*theta_range_rad)
    phi = torch.empty(num_envs, device=device).uniform_(*phi_range_rad)
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)
    offset = torch.stack([x, y, z], dim=1)
    if origin.ndim == 1:
        origin = origin.unsqueeze(0).expand(num_envs, -1)
    return origin + offset


def reset_camera_pose(
    env,
    env_ids: torch.Tensor,
    camera_name: str,
    random_pose_range: tuple,
    theta_range_rad: tuple,
    phi_range_rad: tuple,
):
    num_reset = len(env_ids)
    random_pose_range = torch.as_tensor(random_pose_range, device=env.device)
    bbox = random_pose_range[:6].reshape(2, 3)
    radius = torch.rand(num_reset, device=env.device) * (
        random_pose_range[7] - random_pose_range[6]) + random_pose_range[6]
    look_at = torch.rand((num_reset, 3), device=env.device) * (bbox[1] - bbox[0]) + bbox[0]
    eye = _sample_spherical_point(look_at, radius, theta_range_rad, phi_range_rad, num_reset, env.device)
    eye += env.scene.env_origins[env_ids]
    look_at += env.scene.env_origins[env_ids]
    env.scene[camera_name].set_world_poses_from_view(eye, look_at, env_ids=env_ids)


def set_fixed_camera_view(
    env,
    env_ids: torch.Tensor,
    camera_name: str,
    eye_offset: tuple[float, float, float],
    look_at_offset: tuple[float, float, float],
):
    """Set camera pose from eye position and look-at target (offsets relative to env origin)."""
    device = env.scene.env_origins.device
    eye = env.scene.env_origins[env_ids] + torch.tensor(
        eye_offset, device=device, dtype=torch.float32
    ).unsqueeze(0).expand(len(env_ids), 3)
    look_at = env.scene.env_origins[env_ids] + torch.tensor(
        look_at_offset, device=device, dtype=torch.float32
    ).unsqueeze(0).expand(len(env_ids), 3)
    env.scene[camera_name].set_world_poses_from_view(eye, look_at, env_ids=env_ids)


def apply_sysid_params_on_reset(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    params: dict,
):
    """Apply sysid params to robot on reset. Use with DelayedPDActuatorCfg scene."""
    import uwlab_assets.robots.franka_leap as franka_leap

    robot = env.scene[asset_cfg.name]
    arm_joint_ids = robot.find_joints(franka_leap.ARM_JOINT_NAMES)[0]
    if isinstance(arm_joint_ids, torch.Tensor):
        pass
    else:
        arm_joint_ids = torch.tensor(arm_joint_ids, device=env.device)
    franka_leap.apply_sysid_params_to_robot(
        robot, params, arm_joint_ids, robot.num_joints, env.device
    )


def reset_robot_joints(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    arm_joint_pos: list,
    hand_joint_pos: list,
    arm_joint_limits: dict | None = None,
):
    robot = env.scene[asset_cfg.name]
    num_envs_reset = len(env_ids)

    arm_pos = torch.tensor(arm_joint_pos, device=env.device, dtype=torch.float32)
    if arm_joint_limits is not None:
        order = [f"panda_joint{i}" for i in range(1, 8)]
        low = torch.tensor(
            [arm_joint_limits[j][0] for j in order],
            device=env.device,
            dtype=torch.float32,
        )
        high = torch.tensor(
            [arm_joint_limits[j][1] for j in order],
            device=env.device,
            dtype=torch.float32,
        )
        arm_pos = arm_pos.clamp(low, high)
    hand_pos = torch.tensor(hand_joint_pos, device=env.device, dtype=torch.float32)
    joint_pos = torch.cat([arm_pos, hand_pos], dim=0).unsqueeze(0).repeat(num_envs_reset, 1)
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_table_block(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    z_range: tuple,
):
    block = env.scene[asset_cfg.name]
    z_offsets = torch.empty(len(env_ids), device=env.device).uniform_(*z_range)
    root_state = block.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    root_state[:, 2] += z_offsets
    block.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)


def reset_object_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    default_pos: tuple,
    default_rot_quat: tuple,
    pose_range: dict,
    reset_height: float,
    table_block_name: str | None = None,
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
    # Overwrite Z: reset_height is env-local, so add env origin Z to get world Z.
    # If a table_block is provided, add its Z offset (how much it was raised from default).
    if table_block_name is not None:
        block = env.scene[table_block_name]
        block_z_offset = (
            block.data.root_state_w[env_ids, 2]
            - env.scene.env_origins[env_ids, 2]
            - block.data.default_root_state[env_ids, 2]
        )
        target_state[:, 2] = env.scene.env_origins[env_ids, 2] + reset_height + block_z_offset
    else:
        target_state[:, 2] = env.scene.env_origins[env_ids, 2] + reset_height

    asset.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(target_state[:, 7:], env_ids=env_ids)


def reset_bottle_and_box(
    env,
    env_ids: torch.Tensor,
    bottle_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    bottle_x_range: tuple,
    bottle_y_range: tuple,
    bottle_rot_quat: tuple,
    bottle_reset_height: float,
    bottle_width: float,
    bottle_length: float,
    box_width: float,
    box_length: float,
    box_rot_quat: tuple,
    box_reset_height: float,
    table_block_name: str | None = None,
):
    """Reset bottle to a random position and place box relative to it.

    The bottle is sampled uniformly from bottle_x_range × bottle_y_range.
    The box is placed flush against the -x face of the bottle, with its y
    sampled uniformly along the bottle's length.
    """
    n = len(env_ids)
    device = env.device
    origins = env.scene.env_origins[env_ids]

    # Optional table block z offset (same for both objects)
    block_z_offset = torch.zeros(n, device=device)
    if table_block_name is not None:
        block = env.scene[table_block_name]
        block_z_offset = (
            block.data.root_state_w[env_ids, 2]
            - env.scene.env_origins[env_ids, 2]
            - block.data.default_root_state[env_ids, 2]
        )

    # --- Bottle ---
    bottle_x = torch.empty(n, device=device).uniform_(*bottle_x_range)
    bottle_y = torch.empty(n, device=device).uniform_(*bottle_y_range)
    bottle_z = origins[:, 2] + bottle_reset_height + block_z_offset

    bottle_pos = torch.stack([origins[:, 0] + bottle_x, origins[:, 1] + bottle_y, bottle_z], dim=1)
    bottle_quat = torch.tensor(list(bottle_rot_quat), device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)

    bottle = env.scene[bottle_cfg.name]
    bottle.write_root_pose_to_sim(torch.cat([bottle_pos, bottle_quat], dim=-1), env_ids=env_ids)
    bottle.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

    # --- Box: behind bottle, up to flush against -x face of the bottle ---
    box_x_min = - box_width / 2
    box_x_max = -.01
    box_x = bottle_x - bottle_width / 2 - box_width / 2 + torch.empty(n, device=device).uniform_(box_x_min, box_x_max)
    # y sampled uniformly along the bottle's length
    #y_half_range = (bottle_length - box_length) / 2

    box_y_min = - box_length / 2
    box_y_max = bottle_length - box_length / 2
    box_y = bottle_y - bottle_length / 2 + torch.empty(n, device=device).uniform_(box_y_min, box_y_max)
    box_z = origins[:, 2] + box_reset_height + block_z_offset

    box_pos = torch.stack([origins[:, 0] + box_x, origins[:, 1] + box_y, box_z], dim=1)
    box_quat = torch.tensor(list(box_rot_quat), device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)

    box = env.scene[box_cfg.name]
    box.write_root_pose_to_sim(torch.cat([box_pos, box_quat], dim=-1), env_ids=env_ids)
    box.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


_scales_logged: set = set()


def log_object_mass(env, env_ids, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()


def log_object_scales(env, env_ids, asset_cfg: SceneEntityCfg):
    global _scales_logged
    if asset_cfg.name in _scales_logged:
        return
    _scales_logged.add(asset_cfg.name)
    try:
        import isaaclab.sim as sim_utils
        from pxr import UsdGeom
        stage = sim_utils.get_current_stage()
        prim_path_template = env.scene[asset_cfg.name].cfg.prim_path
        paths = sim_utils.find_matching_prim_paths(prim_path_template)[:4]
        scales = []
        for p in paths:
            prim = stage.GetPrimAtPath(p)
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if "scale" in op.GetOpName():
                    scales.append(tuple(round(v, 3) for v in op.Get()))
                    break
    except:
        pass