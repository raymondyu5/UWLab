# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from . import franka_leap as fl
from .bounded_joint_position_action import BoundedJointPositionActionCfg, BoundedRelativeJointPositionActionCfg
from .bounded_differential_ik_action import BoundedDifferentialInverseKinematicsActionCfg

##
# Individual action terms (all enforce FRANKA_LEAP_ARM_JOINT_LIMITS for the arm)
##


# Unbounded 23-DOF joint position (for rare use when limits must be bypassed)
FRANKA_LEAP_JOINT_POSITION_UNBOUNDED = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*", "j[0-9]+"],
    scale=1.0,
    use_default_offset=False,
)

# 23-DOF joint position with arm clamped to conservative limits (default for Franka Leap)
_ARM_JOINT_ORDER = [f"panda_joint{i}" for i in range(1, 8)]
FRANKA_LEAP_JOINT_POSITION = BoundedJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*", "j[0-9]+"],
    scale=1.0,
    use_default_offset=False,
    arm_joint_limits=dict(fl.FRANKA_LEAP_ARM_JOINT_LIMITS),
    arm_joint_order=_ARM_JOINT_ORDER,
)

# IK-relative arm (6D delta EE pose), arm targets clamped to limits; used alongside hand joint position
FRANKA_LEAP_IK_REL_ARM = BoundedDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_link7",
    controller=DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    ),
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=fl.FRANKA_LEAP_EE_OFFSET),
    arm_joint_limits=dict(fl.FRANKA_LEAP_ARM_JOINT_LIMITS),
)

# IK-relative arm (6D delta EE pose), arm targets clamped to limits; used alongside hand joint position
FRANKA_LEAP_IK_ABS_ARM = BoundedDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_link7",
    controller=DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    ),
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=fl.FRANKA_LEAP_EE_OFFSET),
    arm_joint_limits=dict(fl.FRANKA_LEAP_ARM_JOINT_LIMITS),
)


# Absolute joint position for hand only (16-DOF)
FRANKA_LEAP_HAND_JOINT_POSITION = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["j[0-9]+"],
    scale=1.0,
    use_default_offset=False,
)

##
# Action space configs (used as env actions field)
##

@configclass
class FrankaLeapJointPositionAction:
    """Alias for FrankaLeapJointPositionAction; arm joints always clamped to limits."""
    joint_pos = FRANKA_LEAP_JOINT_POSITION


@configclass
class FrankaLeapIkRelArmHandJointAction:
    """IK-relative arm (6D) + absolute hand joint position (16D)."""
    arm_action = FRANKA_LEAP_IK_REL_ARM
    hand_action = FRANKA_LEAP_HAND_JOINT_POSITION



@configclass
class FrankaLeapIkAbsArmHandJointAction:
    """IK-absolute arm (6D) + absolute hand joint position (16D)."""
    arm_action = FRANKA_LEAP_IK_ABS_ARM
    hand_action = FRANKA_LEAP_HAND_JOINT_POSITION


# Delta arm joint position (7D) + absolute hand joint position (16D)
FRANKA_LEAP_ARM_JOINT_RELATIVE = BoundedRelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    scale=1.0,
    use_zero_offset=True,
    arm_joint_limits=dict(fl.FRANKA_LEAP_ARM_JOINT_LIMITS),
    arm_joint_order=_ARM_JOINT_ORDER,
)


@configclass
class FrankaLeapJointRelArmHandJointAction:
    """Delta arm joint position (7D) + absolute hand joint position (16D)."""
    arm_action = FRANKA_LEAP_ARM_JOINT_RELATIVE
    hand_action = FRANKA_LEAP_HAND_JOINT_POSITION
