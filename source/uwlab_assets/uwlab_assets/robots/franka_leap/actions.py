# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass

##
# Individual action terms
##

# Combined 23-DOF absolute joint position (7 arm + 16 hand)
FRANKA_LEAP_JOINT_POSITION = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*", "j[0-9]+"],
    scale=1.0,
    use_default_offset=False,
)

# IK-relative arm (6D delta EE pose), used alongside hand joint position
FRANKA_LEAP_IK_REL_ARM = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_link7",
    controller=DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    ),
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
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
    """23-DOF absolute joint position: 7 arm + 16 hand."""
    joint_pos = FRANKA_LEAP_JOINT_POSITION


@configclass
class FrankaLeapIkRelArmHandJointAction:
    """IK-relative arm (6D) + absolute hand joint position (16D)."""
    arm_action = FRANKA_LEAP_IK_REL_ARM
    hand_action = FRANKA_LEAP_HAND_JOINT_POSITION
