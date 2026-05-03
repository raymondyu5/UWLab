# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg

BULB_BODY_NAME = "body"  # name of the bulb rigid body in the screw_lamp articulation
BULB_Z_OFFSET = 0.18  # height of bulb above lamp base (updated for new real-scale lightbulb asset)


class ScrewReward:
    """Reward terms for the screw lightbulb task.

    Rewards:
        rew_rotation      — dense per-step rotation reward from AngleCounterCommand
        rew_contact       — finger contact with lamp bulb
        rew_proximity     — finger proximity to lamp bulb
        rew_wrist_penalty — panda_link6 contact with table
        rew_joint_vel     — L2 joint velocity penalty
        rew_action_rate   — L2 action-rate penalty
        rew_is_success    — sparse: cumulative rotation > 2pi

    Obs terms:
        obs_object_pose   — lamp root pose in env-local frame (z overridden to bulb height)
        obs_rotate_angle  — AngleCounterCommand output: [joint_pos, delta, sum_angle]
        obs_contact       — binary finger contact (N, num_fingers)
        obs_object_in_tip — finger-to-bulb displacement vectors, flattened
    """

    _FINGER_SENSOR_LINK = {
        "palm_lower": "palm_lower",
        "fingertip": "fingertip_sensor",
        "thumb_fingertip": "thumb_sensor",
        "fingertip_2": "fingertip_2_sensor",
        "fingertip_3": "fingertip_3_sensor",
    }

    def __init__(self, asset_name: str, object_name: str, fingers_name_list: list):
        self.asset_name = asset_name
        self.object_name = object_name
        self.fingers_name_list = fingers_name_list

        self._cached_step = -1
        self._object_pose = None
        self._contact_or_not = None
        self._finger_object_dev = None
        self._base_spawn_pos = None

    def setup_wrist_sensor(self, scene_cfg):
        setattr(scene_cfg, "panda_link6_contact", ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link6",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
            debug_vis=False,
        ))

    def setup_finger_entities(self, scene_cfg):
        for link_name in self.fingers_name_list:
            setattr(scene_cfg, link_name, RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + link_name,
                spawn=None,
            ))

    def setup_finger_sensors(self, scene_cfg, object_prim_name: str):
        filter_expr = ["{ENV_REGEX_NS}/" + object_prim_name + "/" + BULB_BODY_NAME]
        for link_name in self.fingers_name_list:
            sensor_link = self._FINGER_SENSOR_LINK.get(link_name, link_name)
            setattr(scene_cfg, f"{link_name}_contact", ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + sensor_link,
                update_period=0.0,
                history_length=3,
                filter_prim_paths_expr=filter_expr,
                debug_vis=False,
            ))

    def _ensure_computed(self, env):
        if env.common_step_counter == self._cached_step:
            return

        lamp = env.scene[self.object_name]

        obj_state = lamp._data.root_state_w[:, :7].clone()
        obj_state[:, :3] -= env.scene.env_origins
        obj_state[:, 2] += BULB_Z_OFFSET
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
                sensor._data.net_forces_w.reshape(env.num_envs, 3), dim=1
            ).unsqueeze(1)
            sensor_data.append(force)
        self._contact_or_not = (torch.cat(sensor_data, dim=1) > 1.0).int()

        self._cached_step = env.common_step_counter

    def rew_rotation(self, env) -> torch.Tensor:
        angle_counter = env.command_manager.get_command("angle_counter")
        delta = torch.clip(angle_counter[..., 1].reshape(-1), -0.5, 0.5)
        reward = torch.where(delta > 0, delta * 100.0, delta * 30.0)
        not_yet_done = (angle_counter[..., -1].reshape(-1) < 2 * torch.pi).float()
        return reward * not_yet_done

    def rew_contact(self, env) -> torch.Tensor:
        self._ensure_computed(env)
        n_contact = torch.sum(self._contact_or_not, dim=1).float()
        grasped = (n_contact >= 2).float()
        return n_contact * 1.0 + grasped * 1.0

    def rew_index_thumb_contact(self, env) -> torch.Tensor:
        index_force = torch.linalg.norm(
            env.scene["fingertip_contact"]._data.net_forces_w.reshape(env.num_envs, 3), dim=1
        )
        thumb_force = torch.linalg.norm(
            env.scene["thumb_fingertip_contact"]._data.net_forces_w.reshape(env.num_envs, 3), dim=1
        )
        index_contact = (index_force > 1.0).float()
        thumb_contact = (thumb_force > 1.0).float()
        return index_contact + thumb_contact + index_contact * thumb_contact

    def rew_proximity(self, env) -> torch.Tensor:
        self._ensure_computed(env)
        dist = torch.clip(torch.linalg.norm(self._finger_object_dev, dim=2), 0.02, 0.8)
        reward = torch.clip((1.0 / (0.1 + dist)) - 2.0, 0.0, 4.5)
        return torch.sum(reward, dim=1) / len(self.fingers_name_list)

    def rew_wrist_penalty(self, env) -> torch.Tensor:
        sensor = env.scene["panda_link6_contact"]
        force = torch.linalg.norm(sensor._data.net_forces_w.reshape(env.num_envs, 3), dim=1)
        return -(force > 4.0).float()

    def rew_joint_vel(self, env) -> torch.Tensor:
        robot = env.scene[self.asset_name]
        return torch.sum(robot.data.joint_vel ** 2, dim=1)

    def rew_action_rate(self, env) -> torch.Tensor:
        return torch.sum(
            (env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1
        )

    def rew_arm_action_rate(self, env, num_arm_joints: int = 7) -> torch.Tensor:
        delta = env.action_manager.action - env.action_manager.prev_action
        return torch.sum(delta[:, :num_arm_joints] ** 2, dim=1)

    def rew_arm_action_rate_curriculum(self, env, num_arm_joints: int = 7) -> torch.Tensor:
        delta = env.action_manager.action - env.action_manager.prev_action
        raw = torch.sum(delta[:, :num_arm_joints] ** 2, dim=1)
        total_agent_steps = env.common_step_counter * env.num_envs
        scale = min(5.0 + 195.0 * (total_agent_steps / 200_000_000), 200.0)
        return raw * scale

    def rew_joint_vel_curriculum(self, env) -> torch.Tensor:
        robot = env.scene[self.asset_name]
        raw = torch.sum(robot.data.joint_vel ** 2, dim=1)
        total_agent_steps = env.common_step_counter * env.num_envs
        scale = min(0.001 + 0.024 * (total_agent_steps / 200_000_000), 0.025)
        return raw * scale

    def rew_action_l2(self, env) -> torch.Tensor:
        return torch.sum(env.action_manager.action ** 2, dim=1)

    def apply_gravity_compensation(self, env) -> torch.Tensor:
        lamp = env.scene[self.object_name]
        force = torch.zeros(env.num_envs, lamp.num_bodies, 3, device=env.device)
        force[:, 1, 2] = 0.5 * 9.81  # upward force on bulb body (index 1), matches set_bulb_mass=0.5kg
        lamp.set_external_force_and_torque(force, torch.zeros_like(force))
        return torch.zeros(env.num_envs, device=env.device)

    def rew_base_stability(self, env) -> torch.Tensor:
        lamp = env.scene[self.object_name]
        base_pos = lamp.data.root_pos_w[:, :3]
        buf = env.unwrapped.episode_length_buf

        if self._base_spawn_pos is None:
            self._base_spawn_pos = base_pos.clone()

        reset_mask = buf <= 1
        self._base_spawn_pos[reset_mask] = base_pos[reset_mask].clone()

        displacement = torch.linalg.norm(base_pos - self._base_spawn_pos, dim=1)
        return displacement

    def rew_joint_pos_limits(self, env, soft_ratio: float = 0.9) -> torch.Tensor:
        robot = env.scene[self.asset_name]
        joint_pos = robot.data.joint_pos
        lower = robot.data.soft_joint_pos_limits[:, :, 0]
        upper = robot.data.soft_joint_pos_limits[:, :, 1]
        lower_violation = torch.clamp(lower * soft_ratio - joint_pos, min=0.0)
        upper_violation = torch.clamp(joint_pos - upper * soft_ratio, min=0.0)
        return torch.nan_to_num(torch.sum(lower_violation + upper_violation, dim=1), nan=0.0, posinf=0.0, neginf=0.0)

    def rew_is_success(self, env) -> torch.Tensor:
        angle_counter = env.command_manager.get_command("angle_counter")
        return (angle_counter[..., -1].reshape(-1) > 2 * torch.pi).float()

    def metric_success_2pi(self, env) -> torch.Tensor:
        angle_counter = env.command_manager.get_command("angle_counter")
        return (angle_counter[..., -1].reshape(-1) > 2 * torch.pi).float()

    def metric_cumulative_rotation(self, env) -> torch.Tensor:
        angle_counter = env.command_manager.get_command("angle_counter")
        return angle_counter[..., -1].reshape(-1)

    def obs_object_pose(self, env) -> torch.Tensor:
        self._ensure_computed(env)
        return self._object_pose

    def obs_rotate_angle(self, env) -> torch.Tensor:
        return env.command_manager.get_command("angle_counter")

    def obs_contact(self, env) -> torch.Tensor:
        self._ensure_computed(env)
        return self._contact_or_not

    def obs_object_in_tip(self, env) -> torch.Tensor:
        self._ensure_computed(env)
        return self._finger_object_dev.reshape(env.num_envs, -1)
