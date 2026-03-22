# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic joint position action that clamps the first N joints (arm) to given limits."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions import JointPositionAction, RelativeJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, RelativeJointPositionActionCfg
from isaaclab.utils import configclass


class BoundedJointPositionAction(JointPositionAction):
    """Joint position action that clamps the first len(arm_joint_order) joints to arm_joint_limits."""

    def __init__(self, cfg: BoundedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        order = cfg.arm_joint_order
        limits = cfg.arm_joint_limits
        self._num_arm_joints = len(order)
        self._arm_low = torch.tensor(
            [limits[j][0] for j in order],
            device=self.device,
            dtype=torch.float32,
        )
        self._arm_high = torch.tensor(
            [limits[j][1] for j in order],
            device=self.device,
            dtype=torch.float32,
        )

    def process_actions(self, actions: torch.Tensor):
        # Parent fills _processed_actions; we clamp the arm prefix before apply_actions uses it.
        super().process_actions(actions)
        self._processed_actions[:, : self._num_arm_joints] = self._processed_actions[
            :, : self._num_arm_joints
        ].clamp(self._arm_low, self._arm_high)


class BoundedRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint position action that clamps the resulting position of the first N joints to arm_joint_limits."""

    def __init__(self, cfg: BoundedRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        order = cfg.arm_joint_order
        limits = cfg.arm_joint_limits
        self._num_arm_joints = len(order)
        self._arm_low = torch.tensor(
            [limits[j][0] for j in order],
            device=self.device,
            dtype=torch.float32,
        )
        self._arm_high = torch.tensor(
            [limits[j][1] for j in order],
            device=self.device,
            dtype=torch.float32,
        )

    def apply_actions(self):
        resulting = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        resulting[:, : self._num_arm_joints] = resulting[:, : self._num_arm_joints].clamp(
            self._arm_low, self._arm_high
        )
        self._asset.set_joint_position_target(resulting, joint_ids=self._joint_ids)


@configclass
class BoundedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Config for BoundedRelativeJointPositionAction. Clamps resulting joint positions (current + delta) to arm limits."""

    class_type: type[RelativeJointPositionAction] = BoundedRelativeJointPositionAction

    arm_joint_limits: dict[str, tuple[float, float]] = {}
    arm_joint_order: list[str] = []


@configclass
class BoundedJointPositionActionCfg(JointPositionActionCfg):
    """Config for BoundedJointPositionAction. Same arm_joint_limits interface as BoundedDifferentialInverseKinematicsActionCfg."""

    class_type: type[JointPositionAction] = BoundedJointPositionAction

    arm_joint_limits: dict[str, tuple[float, float]] = {}
    """Joint name -> (low, high) in rad. Same type as BoundedDifferentialInverseKinematicsActionCfg."""
    arm_joint_order: list[str] = []
    """Order of arm joints: the first len(arm_joint_order) joints in this action are clamped; limits are looked up by name in this order."""
