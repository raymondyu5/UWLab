# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic differential IK action that clamps computed joint targets to given limits."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


class BoundedDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Differential IK action that clamps joint targets to arm_joint_limits (keyed by joint name)."""

    def __init__(self, cfg: BoundedDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        order = list(self._joint_names)
        limits = cfg.arm_joint_limits
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
        # Parent computes joint_pos_des and sets it in apply_actions; we clamp after by reading back and re-setting.
        # Execution is synchronous: apply_actions() runs to completion before the env calls write_data_to_context()
        # and scene.update(), so the sim only ever sees the clamped targets.
        super().apply_actions()
        targets = self._asset.data.joint_pos_target[:, self._joint_ids].clone()
        targets = targets.clamp(self._arm_low, self._arm_high)
        self._asset.set_joint_position_target(targets, self._joint_ids)


@configclass
class BoundedDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Config for BoundedDifferentialInverseKinematicsAction. Same arm_joint_limits interface as BoundedJointPositionActionCfg."""

    class_type: type[DifferentialInverseKinematicsAction] = BoundedDifferentialInverseKinematicsAction

    arm_joint_limits: dict[str, tuple[float, float]] = {}
    """Joint name -> (low, high) in rad. Keys must match the joints controlled by this IK action (joint_names)."""
