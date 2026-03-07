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

    def _get_target_pose_tensor(self, num_envs, device):
        return torch.tensor(
            [[self.target_pos[0], self.target_pos[1], self.target_pos[2],
              1.0, 0.0, 0.0, 0.0]],
            device=device, dtype=torch.float32).repeat(num_envs, 1)

    def _get_init_height_tensor(self, num_envs, device):
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
                - joint_vel_penalty * 1.0e-2
                - action_rate_penalty * 5e-2)

    def penalty_contact(self, env):
        sensor = env.scene["panda_link6_contact"]
        force_data = torch.linalg.norm(
            sensor._data.net_forces_w.reshape(env.num_envs, 3), dim=1).unsqueeze(1)
        penalty_contact = torch.clip((force_data > 4).int() * 2 - 1, 0.0, 1.0) * -15
        return penalty_contact.reshape(-1)

    def liftobject_rewards(self, env):
        init_height = self._get_init_height_tensor(env.num_envs, env.device)
        lift_reward = torch.clip(self.object_pose[:, 2] - init_height, -0.1, 0.5)
        lift_reward = torch.clip(torch.clip(lift_reward / 0.4, -0.1, 1.0) * 80, -2, 60)

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


class PourReward:
    """Reward class for bourbon pouring task.

    Standalone port of SingleHandPourRew from IsaacLab's bimanual_franka_pour_rew.py.

    Changes from IsaacLab version:
    - No bimanual wrapper — single hand only, no hand_side prefix
    - No init_robot_pose offset (always zeros in UWLab)
    - Reset references stored as instance attrs (not env.__dict__)
    - reset_init_height computed once at reset, not every step
    - Single get_target_object_pose with cup_top_z + 0.03 (reward-side offset)
    - No _shared_tensors class-var hack
    - Contact/finger sensor prim paths use UWLab conventions (Robot/right_hand/*)
    """

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
        cup_name: str,
        fingers_name_list: list,
        init_height: float,
        bottle_cap_offset: tuple,
    ):
        self.asset_name = asset_name
        self.object_name = object_name
        self.cup_name = cup_name
        self.fingers_name_list = fingers_name_list
        self.init_height = init_height
        self.bottle_cap_offset = torch.tensor(bottle_cap_offset, dtype=torch.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Reward scale per finger: [palm_lower, thumb_fingertip, fingertip, fingertip_2, fingertip_3]
        self.finger_reward_scale = torch.as_tensor([0.5, 1.0, 2, 1.5, 1.0]).to(
            self.device).unsqueeze(0)
        # Grasp target: 6cm toward body end from center (cap is at -X, body at +X)
        self.grasp_target_offset = torch.tensor([0.06, 0.0, 0.0], dtype=torch.float32)

        # Reset references — populated by capture_reset_references event, None until first reset
        self.cup_reset_pos_ref = None    # (num_envs, 3)
        self.cup_reset_quat_ref = None   # (num_envs, 4)
        self.default_bottle_quat = None  # (num_envs, 4)
        self.reset_init_height = None    # (num_envs,) — actual env-local bottle Z at reset

        # Shared state updated by pour_rewards each step
        self.object_pose = None
        self.cup_pose = None
        self.cup_center_xy = None
        self.cup_top_z = None
        self.contact_or_not = None
        self.finger_object_dev = None
        self.finger_pose = None
        self.tip_pos = None

    # ------------------------------------------------------------------
    # Scene setup — called in task __post_init__
    # ------------------------------------------------------------------

    def setup_wrist_sensor(self, scene_cfg):
        from isaaclab.sensors.contact_sensor import ContactSensorCfg
        wrist_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link6",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
            debug_vis=False,
        )
        setattr(scene_cfg, "panda_link6_contact", wrist_sensor)

    def setup_finger_entities(self, scene_cfg):
        from isaaclab.assets import RigidObjectCfg
        for link_name in self.fingers_name_list:
            rigid_cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + link_name,
                spawn=None,
            )
            setattr(scene_cfg, link_name, rigid_cfg)

    def setup_finger_sensors(self, scene_cfg, object_prim_name: str = "GraspObject"):
        from isaaclab.sensors.contact_sensor import ContactSensorCfg
        filter_expr = ["{ENV_REGEX_NS}/" + object_prim_name]
        for link_name in self.fingers_name_list:
            sensor_link = self._FINGER_SENSOR_LINK.get(link_name, link_name)
            sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + sensor_link,
                update_period=0.0,
                history_length=3,
                filter_prim_paths_expr=filter_expr,
                debug_vis=False,
            )
            setattr(scene_cfg, f"{link_name}_contact", sensor)

    # ------------------------------------------------------------------
    # Reset reference capture — register as EventTerm(mode="reset")
    # Must run AFTER reset_object and reset_cup_object events.
    # ------------------------------------------------------------------

    def capture_reset_references(self, env, env_ids):
        """Capture cup and bottle orientations immediately after reset.

        Stored as instance attributes so termination functions can read them
        without touching env.__dict__.
        """
        num_envs = env.num_envs
        device = env.device

        if self.cup_reset_pos_ref is None:
            self.cup_reset_pos_ref = torch.zeros(num_envs, 3, device=device)
            self.cup_reset_quat_ref = torch.zeros(num_envs, 4, device=device)
            self.default_bottle_quat = torch.zeros(num_envs, 4, device=device)
            self.reset_init_height = torch.zeros(num_envs, device=device)

        cup_state = env.scene[self.cup_name]._data.root_state_w[env_ids, :7].clone()
        cup_state[:, :3] -= env.scene.env_origins[env_ids]
        self.cup_reset_pos_ref[env_ids] = cup_state[:, :3]
        self.cup_reset_quat_ref[env_ids] = cup_state[:, 3:7]

        bottle_state = env.scene[self.object_name]._data.root_state_w[env_ids, :7].clone()
        self.default_bottle_quat[env_ids] = bottle_state[:, 3:7]

        # Actual env-local z of bottle at reset (world z minus env origin z)
        bottle_z_local = bottle_state[:, 2] - env.scene.env_origins[env_ids, 2]
        self.reset_init_height[env_ids] = bottle_z_local

    # ------------------------------------------------------------------
    # Main reward
    # ------------------------------------------------------------------

    def pour_rewards(self, env):
        self.get_object_info(env)
        self.get_cup_info(env)
        self.get_finger_info(env)
        self.get_contact_info(env)

        r_approach = self.finger2object_rewards(env)
        r_contact = self.object2fingercontact_rewards(env)
        r_cap = self.cap_to_target_rewards(env)
        r_orientation = self.pour_orientation_rewards(env)
        r_link6 = self.penalty_contact(env)
        r_cup_topple = self.cup_topple_penalty(env)

        joint_vel_penalty = self._joint_vel_l2(env)
        joint_limit_penalty = self._joint_pos_limits(env)
        action_rate_penalty = self._action_rate_l2(env)

        final = (r_approach * 0.30
                 + r_contact * 0.3
                 + r_cap * 1.0
                 + r_orientation * 0.5
                 + r_link6
                 + r_cup_topple
                 - joint_vel_penalty * 1.0e-3
                 - joint_limit_penalty * 6.0e-1
                 - action_rate_penalty * 5e-3)

        return torch.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------

    def finger2object_rewards(self, env):
        finger_dist = torch.clip(
            torch.linalg.norm(self.finger_object_dev, dim=2), 0.02, 0.8)
        reward = torch.clip((1.0 / (0.1 + finger_dist)) - 2.0, 0.0, 4.5)
        reward[:, :-2] *= 2.4 / 3
        reward[:, -2:] *= 1.3
        reward = torch.sum(reward, dim=1) / reward.shape[1]
        return reward

    def object2fingercontact_rewards(self, env):
        finger_rewards_scale = (torch.sum(self.contact_or_not, dim=1) >= 2).int()
        return torch.sum(self.contact_or_not.to(torch.float32), dim=1) + finger_rewards_scale * 1.0

    def cap_to_target_rewards(self, env):
        """Reward moving bottle cap to 3cm above cup center. Gated by 5cm lift."""
        self._compute_tip_pos()
        reset_init_height = self._get_reset_init_height(env)

        target = torch.zeros_like(self.tip_pos)
        target[:, :2] = self.cup_center_xy
        target[:, 2] = self.cup_top_z + 0.11
        dist_3d = torch.linalg.norm(self.tip_pos - target, dim=1)
        dist_3d = torch.nan_to_num(dist_3d, nan=10.0, posinf=10.0, neginf=10.0)
        lifted = (self.object_pose[:, 2] - reset_init_height >= 0.05).float()
        return torch.clip(1 - dist_3d / 0.5, 0.0, 1.0) * 20 * lifted

    def pour_orientation_rewards(self, env):
        """Reward keeping bottle at spawn orientation. Gated by 5cm lift."""
        if self.default_bottle_quat is None or torch.all(self.default_bottle_quat == 0):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

        reset_init_height = self._get_reset_init_height(env)
        delta_quat = math_utils.quat_mul(
            self.object_pose[:, 3:7],
            math_utils.quat_inv(self.default_bottle_quat))
        axis_angle = math_utils.axis_angle_from_quat(delta_quat)
        rotation_magnitude = torch.linalg.norm(axis_angle, dim=1)
        rotation_magnitude = torch.nan_to_num(rotation_magnitude, nan=1.0, posinf=1.0, neginf=0.0)
        lifted = (self.object_pose[:, 2] - reset_init_height >= 0.05).float()
        return torch.clip(1 - rotation_magnitude / 0.5, 0.0, 1.0) * 5 * lifted

    def cup_topple_penalty(self, env):
        """Penalize knocking over the cup relative to its spawn orientation."""
        if self.cup_reset_quat_ref is None:
            return torch.zeros(env.num_envs, device=env.device)
        valid = torch.any(self.cup_reset_quat_ref != 0, dim=1)
        if not valid.any():
            return torch.zeros(env.num_envs, device=env.device)

        current_quat = self.cup_pose[:, 3:7]
        ref_quat = self.cup_reset_quat_ref
        rel_euler = math_utils.euler_xyz_from_quat(
            math_utils.quat_mul(current_quat, math_utils.quat_conjugate(ref_quat)))
        cup_tilt = torch.abs(rel_euler[0]) + torch.abs(rel_euler[1])
        return (cup_tilt > 0.3).float() * -10.0

    def penalty_contact(self, env):
        """Penalty for wrist (link6) contact with table."""
        sensor = env.scene["panda_link6_contact"]
        force_data = torch.linalg.norm(
            sensor._data.net_forces_w.reshape(env.num_envs, 3), dim=1).unsqueeze(1)
        return torch.clip((force_data > 4).int() * 2 - 1, 0.0, 1.0).reshape(-1) * -0.5

    # ------------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------------

    def get_object_info(self, env):
        self.object_pose = env.scene[self.object_name]._data.root_state_w[:, :7].clone()
        self.object_pose[:, :3] -= env.scene.env_origins

    def get_cup_info(self, env):
        self.cup_pose = env.scene[self.cup_name]._data.root_state_w[:, :7].clone()
        self.cup_pose[:, :3] -= env.scene.env_origins
        self.cup_center_xy = self.cup_pose[:, :2]
        self.cup_top_z = self.cup_pose[:, 2] + 0.15  # cup is ~15cm tall, origin at bottom

    def get_finger_info(self, env):
        self.finger_pose = []
        for name in self.fingers_name_list:
            finger = env.scene[name]
            finger_pose = finger._data.root_state_w[:, :7].clone()
            finger_pose[:, :3] -= env.scene.env_origins
            self.finger_pose.append(finger_pose.unsqueeze(1))
        self.finger_pose = torch.cat(self.finger_pose, dim=1)

        # Approach target: bottle body (10cm toward grip end from center)
        local_offset = self.grasp_target_offset.to(env.device).unsqueeze(0).expand(
            self.object_pose.shape[0], -1)
        world_offset = math_utils.quat_apply(self.object_pose[:, 3:7], local_offset)
        grasp_target_pos = self.object_pose[:, :3] + world_offset

        grasp_target_expanded = grasp_target_pos.unsqueeze(1).repeat_interleave(
            len(self.fingers_name_list), dim=1)
        self.finger_object_dev = grasp_target_expanded[..., :3] - self.finger_pose[..., :3]

    def get_contact_info(self, env):
        sensor_data = []
        for name in self.fingers_name_list:
            sensor = env.scene[f"{name}_contact"]
            force_data = torch.linalg.norm(
                sensor._data.force_matrix_w.reshape(env.num_envs, 3), dim=1).unsqueeze(1)
            sensor_data.append(force_data)
        self.contact_or_not = (torch.cat(sensor_data, dim=1) > 2.0).int()

    def get_target_object_pose(self, env):
        """Compute target bottle pose: cap 3cm above cup rim, centered over cup."""
        cup_top_z = self.cup_pose[:, 2] + 0.15
        cup_center_xy = self.cup_pose[:, :2]

        target_quat = self.default_bottle_quat.clone().to(env.device) if self.default_bottle_quat is not None \
            else torch.zeros(env.num_envs, 4, device=env.device)

        cap_offset = self.bottle_cap_offset.to(env.device).unsqueeze(0).expand(env.num_envs, -1)
        cap_offset_world = math_utils.quat_apply(target_quat, cap_offset)

        desired_cap_pos = torch.zeros(env.num_envs, 3, device=env.device)
        desired_cap_pos[:, :2] = cup_center_xy
        desired_cap_pos[:, 2] = cup_top_z + 0.11

        target_pos = desired_cap_pos - cap_offset_world

        target_pose = torch.zeros(env.num_envs, 7, device=env.device)
        target_pose[:, :3] = target_pos
        target_pose[:, 3:7] = target_quat
        return target_pose

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_tip_pos(self):
        local_offset = self.bottle_cap_offset.to(self.object_pose.device).unsqueeze(0).expand(
            self.object_pose.shape[0], -1)
        world_offset = math_utils.quat_apply(self.object_pose[:, 3:7], local_offset)
        self.tip_pos = self.object_pose[:, :3] + world_offset

    def _get_reset_init_height(self, env):
        if self.reset_init_height is not None:
            return self.reset_init_height
        return torch.full((env.num_envs,), self.init_height, device=env.device)

    def _joint_vel_l2(self, env):
        robot = env.scene[self.asset_name]
        return torch.sum(robot.data.joint_vel ** 2, dim=1)

    def _joint_pos_limits(self, env, soft_ratio: float = 0.9):
        robot = env.scene[self.asset_name]
        joint_pos = robot.data.joint_pos
        lower = robot.data.soft_joint_pos_limits[:, :, 0]
        upper = robot.data.soft_joint_pos_limits[:, :, 1]
        lower_violation = torch.clamp(lower * soft_ratio - joint_pos, min=0.0)
        upper_violation = torch.clamp(joint_pos - upper * soft_ratio, min=0.0)
        return torch.sum(lower_violation + upper_violation, dim=1)

    def _action_rate_l2(self, env):
        return torch.sum(
            (env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)

    # ------------------------------------------------------------------
    # Obs term callables — wire as ObsTerm(func=...) in task __post_init__
    # ------------------------------------------------------------------

    def obs_cup_pose(self, env):
        """Cup position (3D) in env-relative frame."""
        if self.cup_pose is None:
            self.get_cup_info(env)
        return self.cup_pose[:, :3]

    def obs_manipulated_object_pose(self, env):
        """Bottle pose (7D) in env-relative frame."""
        if self.object_pose is None:
            self.get_object_info(env)
        return self.object_pose

    def obs_target_object_pose(self, env):
        """Computed target bottle pose (7D): cap 3cm above cup."""
        if self.cup_pose is None:
            self.get_cup_info(env)
        return self.get_target_object_pose(env)

    def obs_contact(self, env):
        """(N, num_fingers) binary contact per finger."""
        if self.contact_or_not is None:
            self.get_contact_info(env)
        return self.contact_or_not

    def obs_object_in_tip(self, env):
        """(N, num_fingers*3) finger-to-object displacement vectors, flattened."""
        if self.finger_object_dev is None:
            self.get_finger_info(env)
        return self.finger_object_dev.reshape(env.num_envs, -1)
