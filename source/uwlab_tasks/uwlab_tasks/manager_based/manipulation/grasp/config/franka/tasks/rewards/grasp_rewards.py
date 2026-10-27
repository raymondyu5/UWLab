# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ported from ../../franka_leap/tasks/rewards/grasp_rewards.py::SimpleGraspReward for the
# standard 2-finger Franka gripper (panda_leftfinger / panda_rightfinger), instead of the
# 5-fingertip LEAP hand. Differences from the LEAP version:
# - fingers_name_list is fixed to the two Franka gripper finger links (real contact-sensor-
#   capable links, no "_sensor" alias links or intermediate "right_hand" body needed).
# - Contact/finger prim paths are "{ENV_REGEX_NS}/Robot/<link>" (LEAP version nests under
#   .../Robot/right_hand/<link>).

from __future__ import annotations

import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
import isaaclab.sim as sim_utils

FRANKA_GRIPPER_FINGERS = ["panda_leftfinger", "panda_rightfinger"]


class SimpleGraspReward:
    """Sparse reward set for grasp tasks with a standard 2-finger Franka gripper.

    Terms (register each as a separate RewTerm):
        rew_grasped       — object z >= init_z + 0.12
        rew_lifted        — object z in [init_z + 0.20, init_z + 0.50]
        rew_success       — 3D dist(object, target_pos) <= 0.15
        rew_wrist_penalty — panda_link6 contacts table with >4 N
        rew_proximity     — dense: fingers close to object (optional, not wired by default)
        rew_joint_vel     — L2 joint velocity penalty
        rew_action_rate   — L2 action-rate penalty

    Obs terms (wire as ObsTerm):
        obs_manipulated_object_pose — current object pose (7D, env-local)
        obs_target_object_pose      — fixed target pose (7D)
        obs_contact                 — binary finger contact per finger (N, 2)
        obs_object_in_tip           — finger-to-object displacements (N, 2*3)

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

    def __init__(
        self,
        asset_name: str,
        object_name: str,
        init_height: float,
        target_pos: tuple,
        fingers_name_list: list | None = None,
    ):
        self.asset_name = asset_name
        self.object_name = object_name
        self.fingers_name_list = fingers_name_list or list(FRANKA_GRIPPER_FINGERS)
        self.init_height = init_height
        self.target_pos = target_pos

        self.reset_init_height = None  # (num_envs,) — captured after reset
        self._cached_step = -1
        self._object_pose = None    # (num_envs, 7) env-local
        self._contact_or_not = None  # (num_envs, num_fingers)
        self._finger_object_dev = None  # (num_envs, num_fingers, 3)
        self._target_marker: VisualizationMarkers | None = None

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def setup_wrist_sensor(self, scene_cfg, filter_prim_paths=None):
        if filter_prim_paths is None:
            filter_prim_paths = ["{ENV_REGEX_NS}/Table"]
        sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link6",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=filter_prim_paths,
            debug_vis=False,
        )
        setattr(scene_cfg, "panda_link6_contact", sensor)

    def setup_finger_entities(self, scene_cfg):
        for link_name in self.fingers_name_list:
            setattr(scene_cfg, link_name, RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                spawn=None,
            ))

    def setup_finger_sensors(self, scene_cfg, object_prim_name: str = "GraspObject"):
        filter_expr = ["{ENV_REGEX_NS}/" + object_prim_name]
        for link_name in self.fingers_name_list:
            setattr(scene_cfg, f"{link_name}_contact", ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
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

    def _update_target_marker(self, env) -> None:
        if self._target_marker is None:
            cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/grasp_target",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            self._target_marker = VisualizationMarkers(cfg)
        target_env_local = self._target_pos_tensor(env)
        world_positions = target_env_local + env.scene.env_origins
        self._target_marker.visualize(translations=world_positions)

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
            self._contact_or_not = (torch.cat(sensor_data, dim=1) > 0.5).int()

            self._update_target_marker(env)
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

    def rew_proximity(self, env) -> torch.Tensor:
        """Dense reward for both fingers being close to the object."""
        self._ensure_computed(env)
        dist = torch.clip(torch.linalg.norm(self._finger_object_dev, dim=2), 0.02, 0.8)
        reward = torch.clip((1.0 / (0.1 + dist)) - 2.0, 0.0, 4.5)
        return torch.sum(reward, dim=1) / len(self.fingers_name_list)

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
        """Binary contact per finger (N, 2)."""
        self._ensure_computed(env)
        return self._contact_or_not

    def obs_object_in_tip(self, env) -> torch.Tensor:
        """Object-center to finger displacement vectors, flattened (N, 2*3)."""
        self._ensure_computed(env)
        return self._finger_object_dev.reshape(env.num_envs, -1)
