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
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


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


class reset_robot_joints_ee_box(ManagerTermBase):
    """Reset arm joints to configurations within an end-effector bounding box.

    At each reset, samples an EE position uniformly from the specified x/y/z box and
    solves IK on-the-fly to find the corresponding arm joint positions. Orientation is
    held fixed at ``ee_default_quat_w`` (world frame); if None, the current EE
    orientation is preserved per env. Hand joints are set to ``hand_joint_pos`` once
    before IK and are not modified by IK.

    Uses the same DLS IK + exponential smoothing pattern as
    ``reset_states/mdp/events.py``: 20 iterations × 0.25 gain ≈ 1% residual error.

    Required params (set in EventTerm.params):
        asset_cfg: SceneEntityCfg for the robot.
        arm_joint_pos: list[float] — fallback arm pose (starting point, 7 values).
        hand_joint_pos: list[float] — hand reset pose (16 values).
        ee_pos_box: dict with keys "x", "y", "z", each a (min, max) tuple in
            env-local coordinates (env origin will be added automatically).
        ee_default_quat_w: list[float] | None — [qw, qx, qy, qz] in world frame.
            If None, current EE orientation for each env is preserved.
        arm_joint_limits: dict | None — clamping limits keyed by joint name.
        num_ik_iters: int — IK iterations (default 20).

    How to get ee_default_quat_w:
        Run the env once, reset, and read obs["policy"]["ee_pose"][:, 3:7][0].tolist()
        at the start of an episode (arm at ARM_RESET). Alternatively leave None to
        preserve whatever orientation the arm ends episodes in.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        import uwlab_assets.robots.franka_leap as franka_leap

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset_cfg.resolve(env.scene)
        self.robot: Articulation = env.scene[asset_cfg.name]

        # Arm joint IDs for partial write_joint_state_to_sim calls
        arm_joint_ids, _ = self.robot.find_joints(franka_leap.ARM_JOINT_NAMES)
        self.arm_joint_ids: list[int] = arm_joint_ids
        self.n_arm_joints: int = len(arm_joint_ids)

        # DLS IK solver (Isaac Lab, with LEAP EE offset)
        solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=asset_cfg.name,
            joint_names=franka_leap.ARM_JOINT_NAMES,
            body_name=franka_leap.FRANKA_LEAP_EE_BODY,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=franka_leap.FRANKA_LEAP_EE_OFFSET,
            ),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = solver_cfg.class_type(solver_cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        arm_joint_pos: list,
        hand_joint_pos: list,
        ee_pos_box: dict,
        ee_default_quat_w: list | None = None,
        arm_joint_limits: dict | None = None,
        num_ik_iters: int = 20,
    ) -> None:
        num_reset = len(env_ids)

        # 1. Write arm (ARM_RESET) + hand to env_ids so IK starts from a known pose.
        arm_pos = torch.tensor(arm_joint_pos, device=env.device, dtype=torch.float32)
        if arm_joint_limits is not None:
            order = [f"panda_joint{i}" for i in range(1, 8)]
            low = torch.tensor(
                [arm_joint_limits[j][0] for j in order], device=env.device, dtype=torch.float32
            )
            high = torch.tensor(
                [arm_joint_limits[j][1] for j in order], device=env.device, dtype=torch.float32
            )
            arm_pos = arm_pos.clamp(low, high)
        hand_pos = torch.tensor(hand_joint_pos, device=env.device, dtype=torch.float32)
        joint_pos_init = torch.cat([arm_pos, hand_pos]).unsqueeze(0).repeat(num_reset, 1)
        self.robot.write_joint_state_to_sim(
            joint_pos_init, torch.zeros_like(joint_pos_init), env_ids=env_ids
        )

        # 2. Get current EE pose for ALL envs (base frame).
        #    Non-resetting envs keep their current pose as IK target (no change).
        #    Resetting envs will have their target overridden below.
        pos_b_all, quat_b_all = self.solver._compute_frame_pose()
        pos_b_all = pos_b_all.clone()
        quat_b_all = quat_b_all.clone()

        # 3. Sample target EE positions from box (env-local → world frame).
        box = ee_pos_box
        target_pos_w = torch.stack(
            [
                torch.empty(num_reset, device=env.device).uniform_(*box["x"]),
                torch.empty(num_reset, device=env.device).uniform_(*box["y"]),
                torch.empty(num_reset, device=env.device).uniform_(*box["z"]),
            ],
            dim=1,
        )
        target_pos_w += env.scene.env_origins[env_ids]

        # Orientation: use provided quat or preserve current per-env quat.
        if ee_default_quat_w is not None:
            target_quat_w = torch.tensor(
                ee_default_quat_w, device=env.device, dtype=torch.float32
            ).unsqueeze(0).expand(num_reset, -1)
        else:
            # Convert base-frame current quat → world frame then reuse it.
            # Since we only need consistent orientation across envs, use the
            # current per-env base-frame orientation unchanged as the target.
            target_quat_w = None  # handled below in base-frame directly

        # Convert world → robot base frame (or use base-frame quat directly).
        if target_quat_w is not None:
            target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
                self.robot.data.root_link_pos_w[env_ids],
                self.robot.data.root_link_quat_w[env_ids],
                target_pos_w,
                target_quat_w,
            )
        else:
            # Keep current per-env orientation; only change position.
            identity_quat = torch.zeros(num_reset, 4, device=env.device)
            identity_quat[:, 0] = 1.0  # [1, 0, 0, 0]
            target_pos_b, _ = math_utils.subtract_frame_transforms(
                self.robot.data.root_link_pos_w[env_ids],
                self.robot.data.root_link_quat_w[env_ids],
                target_pos_w,
                identity_quat,
            )
            target_quat_b = quat_b_all[env_ids]  # preserve current orientation

        # Override targets only for resetting envs.
        pos_b_all[env_ids] = target_pos_b
        quat_b_all[env_ids] = target_quat_b

        self.solver.process_actions(torch.cat([pos_b_all, quat_b_all], dim=1))

        # 4. Iterate DLS IK with 0.25 exponential smoothing (same as reset_states pattern).
        #    Writes only arm joints for env_ids; hand joints are untouched.
        for _ in range(num_ik_iters):
            self.solver.apply_actions()
            delta = 0.25 * (
                self.robot.data.joint_pos_target[env_ids][:, self.arm_joint_ids]
                - self.robot.data.joint_pos[env_ids][:, self.arm_joint_ids]
            )
            new_arm_pos = self.robot.data.joint_pos[env_ids][:, self.arm_joint_ids] + delta
            self.robot.write_joint_state_to_sim(
                position=new_arm_pos,
                velocity=torch.zeros(num_reset, self.n_arm_joints, device=env.device),
                joint_ids=self.arm_joint_ids,
                env_ids=env_ids,
            )


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
