# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Exact port of FrankaGraspRew from:
# IsaacLab/source/isaaclab_tasks/.../utils/grasp/franka_grasp_rew.py
#
# Changes from IsaacLab version:
# - asset_name parameterized (was "right_hand" hardcoded)
# - object_name parameterized (was env_cfg["params"]["spawn_multi_assets_name"])
# - fingers_name_list parameterized (was env_cfg["params"]["contact_sensor"]["spawn_contact_list"])
# - target_pos parameterized (was env_cfg["params"]["right_target_manipulated_object_pose"][:3])
# - init_height parameterized (was computed from RigidObject config pos[2])
# - finger_reward_scale kept exactly as original (palm_lower weight 0.5, fingertip 1.0, etc.)
# - Contact sensor prim paths adapted: {ENV_REGEX_NS}/Robot/{link_name} instead of
#   {ENV_REGEX_NS}/right_hand/right_hand/{link_name}
# - Sensor names: {link_name}_contact instead of right_{link_name}_contact
# - Wrist sensor: {ENV_REGEX_NS}/Robot/panda_link6, name: panda_link6_contact

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg


class GraspReward:

    def __init__(
        self,
        asset_name: str,
        object_name: str,
        fingers_name_list: list,
        init_height: float,
        target_pos: tuple,
        num_envs: int | None = None,
    ):
        self.asset_name = asset_name
        self.object_name = object_name
        self.fingers_name_list = fingers_name_list
        self.init_height = init_height
        self.target_pos = target_pos
        self._num_envs = num_envs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Reward scale per finger: [palm_lower, fingertip, thumb_fingertip, fingertip_2, fingertip_3]
        self.finger_reward_scale = torch.as_tensor([0.5, 1.0, 2, 1.5, 1.0]).to(
            self.device).unsqueeze(0)

        # Shared state set by grasp_rewards each step; initialized to None.
        # Obs methods fall back to scene data if called before first grasp_rewards run.
        self.object_pose = None
        self.contact_or_not = None
        self.finger_object_dev = None
        self.reset_init_height = None  # (num_envs,) — captured after reset; None until first capture
        self._cached_step = -1  # env.common_step_counter value at last _ensure_computed call

    def capture_reset_height(self, env, env_ids):
        """Capture actual object Z after reset (accounts for table height randomization).
        Wire as EventTerm(mode='reset') AFTER reset_object in the task config.
        """
        if self.reset_init_height is None:
            self.reset_init_height = torch.zeros(env.num_envs, device=env.device)
        obj = env.scene[self.object_name]
        obj_z_local = obj.data.root_state_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]
        self.reset_init_height[env_ids] = obj_z_local

    def _get_target_pose_tensor(self, num_envs, device):
        return torch.tensor(
            [[self.target_pos[0], self.target_pos[1], self.target_pos[2],
              1.0, 0.0, 0.0, 0.0]],
            device=device, dtype=torch.float32).repeat(num_envs, 1)

    def _get_init_height_tensor(self, num_envs, device):
        if self.reset_init_height is not None:
            return self.reset_init_height.to(device)
        return torch.full((num_envs,), self.init_height, device=device, dtype=torch.float32)

    # Maps plain finger name -> sensor link name in the sensor USD
    # fingertip/fingertip_2/fingertip_3 have dedicated sensor prims; palm_lower and thumb_fingertip do not
    _FINGER_SENSOR_LINK = {
        "palm_lower": "palm_lower",
        "fingertip": "fingertip_sensor",
        "thumb_fingertip": "thumb_sensor",
        "fingertip_2": "fingertip_2_sensor",
        "fingertip_3": "fingertip_3_sensor",
    }

    def setup_additional(self, scene_cfg):
        wrist_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link6",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
            debug_vis=False)
        setattr(scene_cfg, "panda_link6_contact", wrist_sensor)

    def setup_finger_entities(self, scene_cfg):
        # Register each finger as a RigidObjectCfg(spawn=None) scene entity so
        # env.scene[name] works for pose queries in get_finger_info.
        # Prim paths mirror IsaacLab's spanwn_robot_hand: {ENV_REGEX_NS}/Robot/right_hand/{link_name}
        for link_name in self.fingers_name_list:
            rigid_cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + link_name,
                spawn=None,
            )
            setattr(scene_cfg, link_name, rigid_cfg)

    def setup_finger_sensors(self, scene_cfg, object_prim_name: str = "Object"):
        filter_expr = ["{ENV_REGEX_NS}/" + object_prim_name]
        for link_name in self.fingers_name_list:
            sensor_link = self._FINGER_SENSOR_LINK.get(link_name, link_name)
            sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + sensor_link,
                update_period=0.0,
                history_length=3,
                filter_prim_paths_expr=filter_expr,
                debug_vis=False)
            setattr(scene_cfg, f"{link_name}_contact", sensor)

    def _ensure_computed(self, env):
        """Compute shared per-step state at most once per env step."""
        if env.common_step_counter != self._cached_step:
            self.get_object_info(env)
            self.get_finger_info(env)
            self.get_contact_info(env)
            self._cached_step = env.common_step_counter

    # Individual reward terms — register each as a separate RewTerm for per-term logging.
    # Weights to match original combined grasp_rewards(weight=4.0):
    #   rew_lift / rew_contact / rew_finger2object / rew_wrist_penalty → weight=4.0
    #   rew_joint_vel  → weight=-4e-3   (= 4.0 * 1e-3)
    #   rew_action_rate → weight=-2e-2  (= 4.0 * 5e-3)

    def rew_lift(self, env):
        self._ensure_computed(env)
        return self.liftobject_rewards(env)

    def rew_contact(self, env):
        self._ensure_computed(env)
        return self.object2fingercontact_rewards(env)

    def rew_finger2object(self, env):
        self._ensure_computed(env)
        return self.finger2object_rewards(env)

    def rew_wrist_penalty(self, env):
        self._ensure_computed(env)
        return self.penalty_contact(env)

    def rew_joint_vel(self, env):
        return self._joint_vel_l2(env)

    def rew_action_rate(self, env):
        return self._action_rate_l2(env)

    def grasp_rewards(self, env):
        self.get_object_info(env)
        self.get_finger_info(env)
        self.get_contact_info(env)

        rewards_finger2object = self.finger2object_rewards(env)
        rewards03_contact = self.object2fingercontact_rewards(env)
        rewards04_lift = self.liftobject_rewards(env)
        rewards04_link6 = self.penalty_contact(env)
        joint_vel_penalty = self._joint_vel_l2(env)
        action_rate_penalty = self._action_rate_l2(env)

        return (rewards04_lift + rewards_finger2object + rewards03_contact + rewards04_link6
                - joint_vel_penalty * 1.0e-3
                - action_rate_penalty * 5e-3)

    def penalty_contact(self, env):
        sensor = env.scene["panda_link6_contact"]
        force_data = torch.linalg.norm(
            sensor._data.net_forces_w.reshape(env.num_envs, 3), dim=1).unsqueeze(1)
        penalty_contact = torch.clip((force_data > 4).int() * 2 - 1, 0.0, 1.0) * -15
        return penalty_contact.reshape(-1)

    def liftobject_rewards(self, env):
        init_height = self._get_init_height_tensor(env.num_envs, env.device)
        lift_reward = torch.clip(self.object_pose[:, 2] - init_height, -0.1, 0.5)
        lift_reward = torch.clip(torch.clip(lift_reward / 0.2, -0.1, 1.0) * 80, -2, 60)

        self.lift_reward_scale = (self.object_pose[:, 2] - init_height >= 0.02).int()

        target_dist = torch.linalg.norm(self.object_target_dev, dim=1)
        target_dist_reward = torch.clip(1 - target_dist / 0.3, -0.1, 1.0) * 30
        x_penalty = torch.clip(
            (abs(self.object_target_dev[:, 0])) / 0.20, 0, 1.5) * -5 * self.lift_reward_scale

        still_or_not = abs(self.target_in_object_angle) < 0.1

        scale_topple = self.lift_reward_scale * 20 + 1
        topple_reward = torch.clip(
            torch.clip(still_or_not.int() * 2 - 1, -1, 0.2) * scale_topple, -1, 3)
        reward = (lift_reward + self.lift_reward_scale + target_dist_reward * self.lift_reward_scale
                  + x_penalty + topple_reward * 1.0)
        return reward

    def object2fingercontact_rewards(self, env):
        self.finger_rewards_scale = (torch.sum(self.contact_or_not, dim=1) >= 2).int()
        return torch.sum(self.contact_or_not, dim=1) * 3 + self.finger_rewards_scale * 2.0

    def finger2object_rewards(self, env):
        finger_dist = torch.clip(
            torch.linalg.norm(self.finger_object_dev, dim=2), 0.02, 0.8)
        reward = torch.clip((1.0 / (0.1 + finger_dist)) - 2.0, 0.0, 4.5)
        reward = torch.sum(reward, dim=1) / len(self.fingers_name_list) * 3.0
        return reward

    def get_finger_info(self, env):
        self.finger_pose = []
        for name in self.fingers_name_list:
            finger = env.scene[name]
            finger_pose = finger._data.root_state_w[:, :7].clone()
            finger_pose[:, :3] -= env.scene.env_origins
            self.finger_pose.append(finger_pose.unsqueeze(1))
        self.finger_pose = torch.cat(self.finger_pose, dim=1)
        finger_object_pose = self.object_pose.clone().unsqueeze(1).repeat_interleave(
            len(self.fingers_name_list), dim=1)
        self.finger_object_dev = (finger_object_pose[..., :3] - self.finger_pose[..., :, :3])

    def get_object_info(self, env):
        target_pose = self._get_target_pose_tensor(env.num_envs, env.device)
        target_pose[:, 3:7] = env.scene[self.object_name]._data.default_root_state[:, 3:7].clone()

        self.object_pose = env.scene[self.object_name]._data.root_state_w[:, :7].clone()
        self.object_pose[:, :3] -= env.scene.env_origins
        self.object_target_dev = target_pose[:, :3].clone() - self.object_pose[:, :3].clone()

        delta_quat = math_utils.quat_mul(
            target_pose[:, 3:7].clone(),
            math_utils.quat_inv(self.object_pose[:, 3:7]))
        detla_euler = math_utils.euler_xyz_from_quat(delta_quat)[:2]
        delta_xy_rotation = torch.cat(
            [detla_euler[0].unsqueeze(1), detla_euler[1].unsqueeze(1)], dim=1)
        self.target_in_object_angle = torch.sum(delta_xy_rotation, dim=1)

    def get_contact_info(self, env):
        sensor_data = []
        for name in self.fingers_name_list:
            sensor = env.scene[f"{name}_contact"]
            force_data = torch.linalg.norm(
                sensor._data.force_matrix_w.reshape(env.num_envs, 3), dim=1).unsqueeze(1)
            sensor_data.append(force_data)
        self.contact_or_not = (torch.cat(sensor_data, dim=1) > 4.0).int()

    # --- Obs term callables ---
    # These read shared state set by grasp_rewards (which runs before obs terms).
    # Wire them as ObsTerm(func=...) in the task __post_init__ after grasp_rewards.

    def obs_target_object_pose(self, env):
        # 7D: target xyz (from config) + current object rotation
        target_pose = self._get_target_pose_tensor(env.num_envs, env.device)
        target_pose[:, 3:7] = env.scene[self.object_name]._data.root_state_w[:, 3:7].clone()
        return target_pose

    def obs_manipulated_object_pose(self, env):
        # 7D: current object pose in env-relative world frame
        if self.object_pose is None:
            self.get_object_info(env)
        return self.object_pose

    def obs_contact(self, env):
        # (N, num_fingers): binary contact per finger
        if self.contact_or_not is None:
            self.get_contact_info(env)
        return self.contact_or_not

    def obs_object_in_tip(self, env):
        # (N, num_fingers*3): finger-to-object displacement vectors, flattened
        if self.finger_object_dev is None:
            self.get_finger_info(env)
        return self.finger_object_dev.reshape(env.num_envs, -1)


    def _joint_vel_l2(self, env):
        robot = env.scene[self.asset_name]
        return torch.sum(robot.data.joint_vel ** 2, dim=1)

    def _action_rate_l2(self, env):
        return torch.sum(
            (env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


class SimpleGraspReward:
    """Sparse reward set for grasp tasks.

    Terms (register each as a separate RewTerm):
        rew_grasped       — object z >= init_z + 0.12
        rew_lifted        — object z in [init_z + 0.20, init_z + 0.50]
        rew_success       — 3D dist(object, target_pos) <= 0.15
        rew_wrist_penalty — panda_link6 contacts table with >4 N
        rew_joint_vel     — L2 joint velocity penalty
        rew_action_rate   — L2 action-rate penalty

    Obs terms (wire as ObsTerm):
        obs_manipulated_object_pose — current object pose (7D, env-local)
        obs_target_object_pose      — fixed target pose (7D)
        obs_contact                 — binary finger contact per finger (N, num_fingers)
        obs_object_in_tip           — finger-to-object displacements (N, num_fingers*3)

    Setup (call in task __post_init__ before registering terms):
        setup_wrist_sensor(scene_cfg)
        setup_finger_entities(scene_cfg)
        setup_finger_sensors(scene_cfg, object_prim_name)
        capture_reset_height — register as EventTerm(mode="reset") after reset_object
    """

    GRASPED_Z = 0.12
    LIFTED_Z_LOW = 0.20
    LIFTED_Z_HIGH = 0.50
    SUCCESS_DIST = 0.15

    _FINGER_SENSOR_LINK = {
        "palm_lower": "palm_lower",
        "fingertip": "fingertip_sensor",
        "thumb_fingertip": "thumb_sensor",
        "fingertip_2": "fingertip_2_sensor",
        "fingertip_3": "fingertip_3_sensor",
    }

    def __init__(
        self,
        asset_name: str,
        object_name: str,
        fingers_name_list: list,
        init_height: float,
        target_pos: tuple,
    ):
        self.asset_name = asset_name
        self.object_name = object_name
        self.fingers_name_list = fingers_name_list
        self.init_height = init_height
        self.target_pos = target_pos

        self.reset_init_height = None  # (num_envs,) — captured after reset
        self._cached_step = -1
        self._object_pose = None    # (num_envs, 7) env-local
        self._contact_or_not = None # (num_envs, num_fingers)
        self._finger_object_dev = None  # (num_envs, num_fingers, 3)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def setup_wrist_sensor(self, scene_cfg):
        sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link6",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
            debug_vis=False,
        )
        setattr(scene_cfg, "panda_link6_contact", sensor)

    def setup_finger_entities(self, scene_cfg):
        for link_name in self.fingers_name_list:
            setattr(scene_cfg, link_name, RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + link_name,
                spawn=None,
            ))

    def setup_finger_sensors(self, scene_cfg, object_prim_name: str = "GraspObject"):
        filter_expr = ["{ENV_REGEX_NS}/" + object_prim_name]
        for link_name in self.fingers_name_list:
            sensor_link = self._FINGER_SENSOR_LINK.get(link_name, link_name)
            setattr(scene_cfg, f"{link_name}_contact", ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + sensor_link,
                update_period=0.0,
                history_length=3,
                filter_prim_paths_expr=filter_expr,
                debug_vis=False,
            ))

    # ------------------------------------------------------------------
    # Reset event — wire as EventTerm(mode="reset") after reset_object
    # ------------------------------------------------------------------

    def capture_reset_height(self, env, env_ids):
        if self.reset_init_height is None:
            self.reset_init_height = torch.zeros(env.num_envs, device=env.device)
        obj = env.scene[self.object_name]
        self.reset_init_height[env_ids] = (
            obj.data.root_state_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_computed(self, env):
        if env.common_step_counter != self._cached_step:
            obj_state = env.scene[self.object_name]._data.root_state_w[:, :7].clone()
            obj_state[:, :3] -= env.scene.env_origins
            self._object_pose = obj_state

            finger_poses = []
            for name in self.fingers_name_list:
                fp = env.scene[name]._data.root_state_w[:, :7].clone()
                fp[:, :3] -= env.scene.env_origins
                finger_poses.append(fp.unsqueeze(1))
            finger_poses = torch.cat(finger_poses, dim=1)
            obj_expanded = self._object_pose[:, :3].unsqueeze(1).expand_as(finger_poses[..., :3])
            self._finger_object_dev = obj_expanded - finger_poses[..., :3]

            sensor_data = []
            for name in self.fingers_name_list:
                sensor = env.scene[f"{name}_contact"]
                force = torch.linalg.norm(
                    sensor._data.force_matrix_w.reshape(env.num_envs, 3), dim=1
                ).unsqueeze(1)
                sensor_data.append(force)
            self._contact_or_not = (torch.cat(sensor_data, dim=1) > 4.0).int()

            self._cached_step = env.common_step_counter

    def _init_height_tensor(self, env):
        if self.reset_init_height is not None:
            return self.reset_init_height.to(env.device)
        return torch.full((env.num_envs,), self.init_height, device=env.device)

    def _target_pos_tensor(self, env):
        return torch.tensor(
            [list(self.target_pos)], device=env.device, dtype=torch.float32
        ).expand(env.num_envs, -1)

    # ------------------------------------------------------------------
    # Reward terms
    # ------------------------------------------------------------------

    def rew_grasped(self, env) -> torch.Tensor:
        """Sparse +1 when object z >= init_z + 0.12 m."""
        self._ensure_computed(env)
        z_above = self._object_pose[:, 2] - self._init_height_tensor(env)
        return (z_above >= self.GRASPED_Z).float()

    def rew_lifted(self, env) -> torch.Tensor:
        """Sparse +1 when object z in [init_z + 0.20, init_z + 0.50] m."""
        self._ensure_computed(env)
        z_above = self._object_pose[:, 2] - self._init_height_tensor(env)
        return ((z_above >= self.LIFTED_Z_LOW) & (z_above <= self.LIFTED_Z_HIGH)).float()

    def rew_success(self, env) -> torch.Tensor:
        """Sparse +1 when 3D distance from object to target_pos <= 0.15 m."""
        self._ensure_computed(env)
        dist = torch.linalg.norm(
            self._object_pose[:, :3] - self._target_pos_tensor(env), dim=1
        )
        return (dist <= self.SUCCESS_DIST).float()

    def rew_wrist_penalty(self, env) -> torch.Tensor:
        """Sparse -1 when panda_link6 contacts the table with > 4 N."""
        sensor = env.scene["panda_link6_contact"]
        force = torch.linalg.norm(
            sensor._data.net_forces_w.reshape(env.num_envs, 3), dim=1
        )
        return -(force > 4.0).float()

    def rew_joint_vel(self, env) -> torch.Tensor:
        robot = env.scene[self.asset_name]
        return torch.sum(robot.data.joint_vel ** 2, dim=1)

    def rew_action_rate(self, env) -> torch.Tensor:
        return torch.sum(
            (env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1
        )

    # ------------------------------------------------------------------
    # Obs terms
    # ------------------------------------------------------------------

    def obs_manipulated_object_pose(self, env) -> torch.Tensor:
        """Current object pose (7D) in env-local frame."""
        self._ensure_computed(env)
        return self._object_pose

    def obs_target_object_pose(self, env) -> torch.Tensor:
        """Fixed target pose (7D): target_pos xyz + object's default rotation."""
        target = self._target_pos_tensor(env)
        default_quat = env.scene[self.object_name]._data.default_root_state[:, 3:7].clone()
        return torch.cat([target, default_quat], dim=1)

    def obs_contact(self, env) -> torch.Tensor:
        """Binary contact per finger (N, num_fingers)."""
        self._ensure_computed(env)
        return self._contact_or_not

    def obs_object_in_tip(self, env) -> torch.Tensor:
        """Object-center to finger displacement vectors, flattened (N, num_fingers*3)."""
        self._ensure_computed(env)
        return self._finger_object_dev.reshape(env.num_envs, -1)
