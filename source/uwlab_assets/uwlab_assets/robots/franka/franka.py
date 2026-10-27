# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Standard Franka Panda arm + parallel (2-finger) gripper. No LEAP hand.
# Uses the Isaac Lab Nucleus asset (same one used by the factory_extension tasks).

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

FRANKA_USD = f"{ISAACLAB_NUCLEUS_DIR}/Factory/franka_mimic.usd"

##
# EE calibration
##

FRANKA_EE_BODY = "panda_hand"
FRANKA_EE_OFFSET = (0.0, 0.0, 0.1034)  # matches body_offset used by the IK actions in action.py

##
# Actuator configs
##

FRANKA_ARM_ACTUATOR_CFG = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=87.0,
        velocity_limit_sim=2.175,
        stiffness=400.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=12.0,
        velocity_limit_sim=2.61,
        stiffness=400.0,
        damping=80.0,
    ),
}

FRANKA_GRIPPER_ACTUATOR_CFG = {
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint[1-2]"],
        effort_limit_sim=200.0,
        velocity_limit_sim=0.2,
        stiffness=2e3,
        damping=1e2,
    ),
}

##
# Default joint positions ("ready" pose) and limits
##

FRANKA_DEFAULT_ARM_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
}

FRANKA_DEFAULT_GRIPPER_JOINT_POS = {
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}

FRANKA_DEFAULT_JOINT_POS = FRANKA_DEFAULT_ARM_JOINT_POS | FRANKA_DEFAULT_GRIPPER_JOINT_POS

FRANKA_ARM_JOINT_LIMITS = {
    "panda_joint1": (-2.8973, 2.8973),
    "panda_joint2": (-1.7628, 1.7628),
    "panda_joint3": (-2.8973, 2.8973),
    "panda_joint4": (-3.0718, -0.0698),
    "panda_joint5": (-2.8973, 2.8973),
    "panda_joint6": (-0.0175, 3.7525),
    "panda_joint7": (-2.8973, 2.8973),
}

##
# ArticulationCfg
##

FRANKA_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRANKA_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=1,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=FRANKA_DEFAULT_JOINT_POS,
    ),
    soft_joint_pos_limit_factor=1.0,
)

IMPLICIT_FRANKA = FRANKA_ARTICULATION.copy()
IMPLICIT_FRANKA.actuators = FRANKA_ARM_ACTUATOR_CFG | FRANKA_GRIPPER_ACTUATOR_CFG
