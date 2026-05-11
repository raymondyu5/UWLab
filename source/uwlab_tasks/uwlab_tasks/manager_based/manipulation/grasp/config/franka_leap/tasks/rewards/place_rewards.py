# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .grasp_rewards import SimpleGraspReward


class PlaceReward(SimpleGraspReward):
    """Extends SimpleGraspReward for pick-and-place tasks.

    Inherits all setup, grasp, lift, and penalty terms unchanged.

    Additional terms (register each as a RewTerm):
        rew_proximity   — continuous [0,1]: plate approaching slot, gated by lift
        rew_orientation — continuous [0,1]: plate becoming vertical, gated by lift
        rew_success     — overrides parent: pos within threshold AND plate vertical

    target_pos should be the slot center position (env-local coords).
    If rack_name and slot_offset are provided, target_pos is ignored and the goal
    is computed dynamically as rack_root_pos (env-local) + slot_offset.

    Thresholds:
        SLOT_POS_THRESHOLD = 0.04   # 1 cm
        ORIENT_THRESHOLD   = 0.85   # |plate_z · world_x| >= 0.85  (within ~32° of target axis)
    """

    SLOT_POS_THRESHOLD = 0.04
    ORIENT_THRESHOLD = 0.85

    def __init__(
        self,
        rack_name: str | None = None,
        slot_offset: tuple | None = None,
        slot_rot_local: tuple | None = None,
        success_box_x_lo_offset: float = 0.00,
        success_box_x_hi_offset: float = 0.09,
        success_box_y_front: float = 0.07,
        success_box_y_rear: float = 0.02,
        success_box_y_offset: float = 0.03,
        success_box_z_min: float = 0.05,
        success_box_z_max: float = 0.14,
        home_joints: list | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rack_name = rack_name
        self.slot_offset = slot_offset
        self.slot_rot_local = slot_rot_local
        self.success_box_x_lo_offset = success_box_x_lo_offset
        self.success_box_x_hi_offset = success_box_x_hi_offset
        self.success_box_y_front = success_box_y_front
        self.success_box_y_rear = success_box_y_rear
        self.success_box_y_offset = success_box_y_offset
        self.success_box_z_min = success_box_z_min
        self.success_box_z_max = success_box_z_max
        self._slot_marker: VisualizationMarkers | None = None
        self._placed = None
        self._home_joints = torch.tensor(home_joints, dtype=torch.float32) if home_joints is not None else None

    def _target_pos_tensor(self, env):
        if self.rack_name is not None and self.slot_offset is not None:
            rack_pos = env.scene[self.rack_name].data.root_pos_w - env.scene.env_origins
            offset = torch.tensor([list(self.slot_offset)], device=env.device, dtype=torch.float32)
            return rack_pos + offset
        return super()._target_pos_tensor(env)

    def _target_quat_tensor(self, env) -> torch.Tensor:
        """Target bowl quaternion in world frame: q_rack * q_slot_local (WXYZ)."""
        if self.rack_name is not None and self.slot_rot_local is not None:
            rack_quat = env.scene[self.rack_name].data.root_state_w[:, 3:7]
            q_local = torch.tensor([list(self.slot_rot_local)], device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
            return math_utils.quat_mul(rack_quat, q_local)
        return env.scene[self.object_name]._data.default_root_state[:, 3:7].clone()

    def _quat_similarity(self, env) -> torch.Tensor:
        """Absolute dot product between current bowl quat and target quat. Near 1 = well-aligned."""
        target_quat = self._target_quat_tensor(env)
        bowl_quat = self._object_pose[:, 3:7]
        return torch.abs(torch.sum(bowl_quat * target_quat, dim=1))

    def _ensure_computed(self, env):
        needs_update = (env.common_step_counter != self._cached_step)
        super()._ensure_computed(env)
        if needs_update:
            self._update_slot_marker(env)
            self._update_placed_flag(env)

    def _update_placed_flag(self, env) -> None:
        if self._placed is None:
            self._placed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._placed |= self.rew_success(env).bool()

    def reset_placed_flag(self, env, env_ids):
        if self._placed is not None:
            self._placed[env_ids] = False

    def _update_slot_marker(self, env) -> None:
        """Lazily create and update a green sphere marker at the target slot position (world frame)."""
        if self._slot_marker is None:
            cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/slot_target",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            self._slot_marker = VisualizationMarkers(cfg)
        target_env_local = self._target_pos_tensor(env)
        world_positions = target_env_local + env.scene.env_origins
        self._slot_marker.visualize(translations=world_positions)

    def rew_contact(self, env) -> torch.Tensor:
        """Continuous: reward for finger-bowl contact. Max ~17."""
        self._ensure_computed(env)
        finger_scale = (torch.sum(self._contact_or_not, dim=1) >= 2).int()
        return torch.sum(self._contact_or_not, dim=1) * 3 + finger_scale * 2.0

    def rew_finger2object(self, env) -> torch.Tensor:
        """Continuous [0, ~15]: reward for finger proximity to bowl."""
        self._ensure_computed(env)
        finger_dist = torch.clip(torch.linalg.norm(self._finger_object_dev, dim=2), 0.02, 0.8)
        reward = torch.clip((1.0 / (0.1 + finger_dist)) - 2.0, 0.0, 4.5)
        return torch.sum(reward, dim=1) / len(self.fingers_name_list) * 3.0

    PLACE_LIFT_THRESHOLD = 0.03  # 3 cm above spawn — just off the table, much lower than GRASPED_Z (12 cm)
                                  # slot Z is only ~8 cm above bowl spawn, so rewards must stay active at slot level

    def _is_lifted(self, env) -> torch.Tensor:
        init_z = self._init_height_tensor(env)
        return (self._object_pose[:, 2] - init_z >= self.PLACE_LIFT_THRESHOLD).float()

    def rew_proximity(self, env) -> torch.Tensor:
        """Continuous [0, 1]: plate approaching slot, only active when lifted."""
        self._ensure_computed(env)
        dist = torch.linalg.norm(
            self._object_pose[:, :3] - self._target_pos_tensor(env), dim=1
        )
        return torch.clip(1.0 - dist / 0.5, 0.0, 1.0) * self._is_lifted(env)

    def rew_orientation(self, env) -> torch.Tensor:
        """Continuous [0, 1]: bowl quaternion alignment with slot target, only active when lifted."""
        self._ensure_computed(env)
        return self._quat_similarity(env) * self._is_lifted(env)

    def rew_success(self, env) -> torch.Tensor:
        """Sparse +1: bowl inside success box AND correctly oriented AND no finger contact."""
        self._ensure_computed(env)
        rack_pos = env.scene[self.rack_name].data.root_pos_w - env.scene.env_origins
        bowl = self._object_pose[:, :3]
        cy = rack_pos[:, 1] + self.success_box_y_offset
        x_ok = (bowl[:, 0] >= rack_pos[:, 0] + self.success_box_x_lo_offset) & \
               (bowl[:, 0] <= rack_pos[:, 0] + self.success_box_x_hi_offset)
        y_ok = (bowl[:, 1] >= cy - self.success_box_y_front) & \
               (bowl[:, 1] <= cy + self.success_box_y_rear)
        z_ok = (bowl[:, 2] >= self.success_box_z_min) & (bowl[:, 2] <= self.success_box_z_max)
        orient_ok = self._quat_similarity(env) >= self.ORIENT_THRESHOLD
        released = torch.sum(self._contact_or_not, dim=1) == 0
        return (x_ok & y_ok & z_ok & orient_ok & released).float()

    def metric_near_slot(self, env) -> torch.Tensor:
        """Sparse +1: plate within 15cm of slot."""
        self._ensure_computed(env)
        dist = torch.linalg.norm(
            self._object_pose[:, :3] - self._target_pos_tensor(env), dim=1
        )
        return (dist <= 0.15).float()

    def metric_placed_pos(self, env) -> torch.Tensor:
        """Sparse +1: bowl inside success box (position only, no orientation/release check)."""
        self._ensure_computed(env)
        rack_pos = env.scene[self.rack_name].data.root_pos_w - env.scene.env_origins
        bowl = self._object_pose[:, :3]
        cy = rack_pos[:, 1] + self.success_box_y_offset
        x_ok = (bowl[:, 0] >= rack_pos[:, 0] + self.success_box_x_lo_offset) & \
               (bowl[:, 0] <= rack_pos[:, 0] + self.success_box_x_hi_offset)
        y_ok = (bowl[:, 1] >= cy - self.success_box_y_front) & \
               (bowl[:, 1] <= cy + self.success_box_y_rear)
        z_ok = (bowl[:, 2] >= self.success_box_z_min) & (bowl[:, 2] <= self.success_box_z_max)
        return (x_ok & y_ok & z_ok).float()

    def metric_placed_orient(self, env) -> torch.Tensor:
        """Sparse +1: bowl quaternion aligns with slot target orientation."""
        self._ensure_computed(env)
        return (self._quat_similarity(env) >= self.ORIENT_THRESHOLD).float()


    HOME_DIST_THRESHOLD = 0.2  # joint-space L2 distance to consider "at home"

    def rew_return_home(self, env) -> torch.Tensor:
        """Sparse +1: at home joints after placing, gated by placed flag."""
        self._ensure_computed(env)
        if self._placed is None or self._home_joints is None:
            return torch.zeros(env.num_envs, device=env.device)
        home = self._home_joints.to(env.device)
        joint_pos = env.scene[self.asset_name].data.joint_pos
        dist = torch.linalg.norm(joint_pos - home.unsqueeze(0), dim=1)
        return self._placed.float() * (dist < self.HOME_DIST_THRESHOLD).float()

    def is_done(self, env) -> torch.Tensor:
        """True when placed AND joints within HOME_DIST_THRESHOLD of home."""
        if self._placed is None or self._home_joints is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        home = self._home_joints.to(env.device)
        joint_pos = env.scene[self.asset_name].data.joint_pos
        dist = torch.linalg.norm(joint_pos - home.unsqueeze(0), dim=1)
        return self._placed & (dist < self.HOME_DIST_THRESHOLD)

    def obs_target_object_pose(self, env) -> torch.Tensor:
        """Target pose (7D): slot position + target quaternion in world frame."""
        target_pos = self._target_pos_tensor(env)
        target_quat = self._target_quat_tensor(env)
        return torch.cat([target_pos, target_quat], dim=1)
