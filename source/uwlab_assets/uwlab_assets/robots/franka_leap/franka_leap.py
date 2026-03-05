# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# USD lives in UWLab/assets/robot/franka_leap/ (not committed to git).
# Inside the container it is mounted at /workspace/uwlab/assets/ by run_singularity.sh.
# TODO: upload to UWLAB_CLOUD_ASSETS_DIR when bucket credentials are available.
#
# Real-robot reference: ai_logs/franka_urdf_feb17 (limits, effort, velocity, dynamics).
# Conservative limits (prior to safety filter): ai_logs/FRANKA_LIMITS_FOR_SIM.md.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

FRANKA_LEAP_ASSETS_DIR = "/workspace/uwlab/assets/robot/franka_leap"

##
# EE calibration offset
#
# Measured offset from panda_link7 to the physical EE tip, in panda_link7 frame.
# This matches the frame reported by the real robot's Cartesian controller.
# Used in the ee_pose_w obs term — single source of truth, never recomputed elsewhere.
##

FRANKA_LEAP_EE_BODY = "panda_link7"
FRANKA_LEAP_EE_OFFSET = (0.0, 0.0, 0.107)  # (x, y, z) metres

##
# Actuator configs — conservative real limits (prior to safety filter)
# See ai_logs/FRANKA_LIMITS_FOR_SIM.md.
#
# vs franka_urdf_feb17:
#   - Effort: matches (86 J1–4, 11.5 J5–7). Velocity/position: conservative (see FRANKA_LIMITS_FOR_SIM).
#   - Damping/friction: per-joint from URDF, set via ImplicitActuatorCfg below.
#   - Inertia: real robot in URDF; sim from USD — match if USD was built from same URDF.
##

# Per-joint dynamics from ai_logs/franka_urdf_feb17 (<dynamics damping="..." friction="..."/>).
# Damping: ImplicitActuatorCfg expects Nm·s/rad (joint drive Kd); URDF dynamics damping is typically
# the same units, so values should match when the URDF follows that convention.
# Friction: URDF joint friction is not well standardized; verify scale/units in Isaac Sim or tune empirically.
FRANKA_LEAP_ARM_JOINT_DAMPING_URDF = {
    "panda_joint1": 10.0,
    "panda_joint2": 5.0,
    "panda_joint3": 5.0,
    "panda_joint4": 1.0,
    "panda_joint5": 2.0,
    "panda_joint6": 1.0,
    "panda_joint7": 1.0,
}
FRANKA_LEAP_ARM_JOINT_FRICTION_URDF = {
    "panda_joint1": 5.0,
    "panda_joint2": 2.0,
    "panda_joint3": 2.0,
    "panda_joint4": 0.5,
    "panda_joint5": 1.0,
    "panda_joint6": 0.5,
    "panda_joint7": 0.5,
}

# Franka arm: high-PD config used for IK / joint position control
FRANKA_LEAP_ARM_ACTUATOR_CFG = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit=86.0,
        velocity_limit=1.575,
        stiffness=400.0,
        damping=FRANKA_LEAP_ARM_JOINT_DAMPING_URDF,
        friction=FRANKA_LEAP_ARM_JOINT_FRICTION_URDF,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit=11.5,
        velocity_limit=2.01,
        stiffness=400.0,
        damping=FRANKA_LEAP_ARM_JOINT_DAMPING_URDF,
        friction=FRANKA_LEAP_ARM_JOINT_FRICTION_URDF,
    ),
}

# LEAP hand
FRANKA_LEAP_HAND_ACTUATOR_CFG = {
    "hand": ImplicitActuatorCfg(
        joint_names_expr=["j[0-9]+"],
        stiffness=20.0,
        damping=1.0,
        armature=0.001,
        friction=0.2,
        velocity_limit=8.48,
        effort_limit=0.95,
    ),
}

##
# Default joint positions
##

FRANKA_LEAP_DEFAULT_ARM_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": 0.0,
    "panda_joint3": 0.0,
    "panda_joint4": -0.20,
    "panda_joint5": 0.0,
    "panda_joint6": 2.0,
    "panda_joint7": 0.0,
}

FRANKA_LEAP_DEFAULT_HAND_JOINT_POS = {
    "j0": 0.0, "j1": 0.0, "j2": 0.0, "j3": 0.0,
    "j4": 0.0, "j5": 0.0, "j6": 0.0, "j7": 0.0,
    "j8": 0.0, "j9": 0.0, "j10": 0.0, "j11": 0.0,
    "j12": 0.0, "j13": 0.0, "j14": 0.0, "j15": 0.0,
}

FRANKA_LEAP_DEFAULT_JOINT_POS = FRANKA_LEAP_DEFAULT_ARM_JOINT_POS | FRANKA_LEAP_DEFAULT_HAND_JOINT_POS

##
# Joint limits (Franka arm only — conservative, prior to safety filter)
# Used to clamp joint targets. This must be implemented at the controller level. See ai_logs/FRANKA_LIMITS_FOR_SIM.md.
##

FRANKA_LEAP_ARM_JOINT_LIMITS = {
    "panda_joint1": (-2.60, 2.60),
    "panda_joint2": (-1.46, 1.46),
    "panda_joint3": (-2.60, 2.60),
    "panda_joint4": (-2.77, -0.37),
    "panda_joint5": (-2.60, 2.60),
    "panda_joint6": (0.28, 3.45),
    "panda_joint7": (-2.60, 2.60),
}

##
# ArticulationCfg
##

FRANKA_LEAP_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FRANKA_LEAP_ASSETS_DIR}/franka_right_leap_long_finger_sensor.usd",
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
        joint_pos=FRANKA_LEAP_DEFAULT_JOINT_POS,
    ),
    soft_joint_pos_limit_factor=1.0,
)

IMPLICIT_FRANKA_LEAP = FRANKA_LEAP_ARTICULATION.copy()
IMPLICIT_FRANKA_LEAP.actuators = FRANKA_LEAP_ARM_ACTUATOR_CFG | FRANKA_LEAP_HAND_ACTUATOR_CFG
