# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# USD lives in UWLab/assets/robot/franka_leap/ (not committed to git).
# Inside the container it is mounted at /workspace/uwlab/assets/ by run_singularity.sh.
# TODO: upload to UWLAB_CLOUD_ASSETS_DIR when bucket credentials are available.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
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
# Actuator configs — matched to IsaacLab training values
##

# Franka arm: high-PD config used for IK / joint position control
FRANKA_LEAP_ARM_ACTUATOR_CFG = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=86.0,
        velocity_limit_sim=1.575,
        stiffness=400.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=11.5,
        velocity_limit_sim=2.01,
        stiffness=400.0,
        damping=80.0,
    ),
}

# p and d gains from the real robot
FRANKA_LEAP_REAL_GAINS_ARM_ACTUATOR_CFG = {
    "panda_link1": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint1"],
        stiffness=60.0,
        damping=4.0,
        effort_limit_sim=86.0,
        velocity_limit_sim=1.575,
    ),  
    "panda_link2": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[2]"],
        stiffness=55.0,
        damping=6.0,
        effort_limit_sim=86.0,
        velocity_limit_sim=1.575,
    ),
    "panda_link3": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[3]"],
        stiffness=80.0,
        damping=5.0,
        effort_limit_sim=86.0,
        velocity_limit_sim=1.575,
    ),
    "panda_link4": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[4]"],
        stiffness=40.0,
        damping=5.0,
        effort_limit_sim=86.0,
        velocity_limit_sim=1.575,
    ),
    "panda_link5": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5]"],
        stiffness=55.0,
        damping=3.0,
        effort_limit_sim=11.5,
        velocity_limit_sim=2.01,
    ),
    "panda_link6": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[6]"],
        stiffness=40.0,
        damping=2.0,
        effort_limit_sim=11.5,
        velocity_limit_sim=2.01,
    ),
    "panda_link7": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[7]"],
        stiffness=16.0,
        damping=1.0,
        effort_limit_sim=11.5,
        velocity_limit_sim=2.01,
    ),
}


FRANKA_LEAP_REAL_GAINS_ARM_ACTUATOR_DELAYED_CFG= {
    "panda_link1": DelayedPDActuatorCfg(
        joint_names_expr=["panda_joint1"],
        stiffness=60.0,
        damping=4.0,
        effort_limit_sim=86.0,
        velocity_limit_sim=1.575,
        min_delay=0,
        max_delay=5,
    ),
    "panda_link2": DelayedPDActuatorCfg(
        joint_names_expr=["panda_joint[2]"],
        stiffness=55.0,
        damping=6.0,
        effort_limit_sim=86.0,
        velocity_limit_sim=1.575,
        min_delay=0,
        max_delay=5,
    ),
    "panda_link3": DelayedPDActuatorCfg(
        joint_names_expr=["panda_joint[3]"],
        stiffness=80.0,
        damping=5.0,
        effort_limit_sim=86.0,
        velocity_limit_sim=1.575,
        min_delay=0,
        max_delay=5,
    ),
    "panda_link4": DelayedPDActuatorCfg(
        joint_names_expr=["panda_joint[4]"],
        stiffness=40.0,
        damping=5.0,
        effort_limit_sim=86.0,
        velocity_limit_sim=1.575,
        min_delay=0,
        max_delay=5,
    ),
    "panda_link5": DelayedPDActuatorCfg(
        joint_names_expr=["panda_joint[5]"],
        stiffness=55.0,
        damping=3.0,
        effort_limit_sim=11.5,
        velocity_limit_sim=2.01,
        min_delay=0,
        max_delay=5,
    ),
    "panda_link6": DelayedPDActuatorCfg(
        joint_names_expr=["panda_joint[6]"],
        stiffness=40.0,
        damping=2.0,
        effort_limit_sim=11.5,
        velocity_limit_sim=2.01,
        min_delay=0,
        max_delay=5,
    ),
    "panda_link7": DelayedPDActuatorCfg(
        joint_names_expr=["panda_joint[7]"],
        stiffness=16.0,
        damping=1.0,
        effort_limit_sim=11.5,
        velocity_limit_sim=2.01,
        min_delay=0,
        max_delay=5,
    ),
}

##
# Sysid params — apply to env via apply_sysid_params_to_robot() after env creation.
# Replace with your sysid output. Encoder bias: add to initial arm joint pos when resetting.
##
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
ARM_ACTUATOR_NAMES = [f"panda_link{i}" for i in range(1, 8)]

# Example from sysid output (replace with your final_results.pt / printed values)
FRANKA_LEAP_SYSID_PARAMS = {
    "armature": [0.4131, 0.5835, 0.8336, 0.6845, 0.3521, 0.2363, 0.1509],
    "static_friction": [1.2008, 1.5891, 1.1150, 1.2567, 1.0867, 0.9516, 0.5117],
    "dynamic_ratio": [0.6789, 0.8146, 0.4764, 0.8223, 0.5187, 0.4928, 0.7358],
    "viscous_friction": [3.3143, 4.1785, 6.4565, 1.1896, 3.3427, 3.0174, 1.1646],
    "encoder_bias_rad": [-0.0015, -1.0798, -0.2323, -0.0183, 0.2623, 0.1980, -0.1067],
    "delay_steps": 3,
}


def apply_sysid_params_to_robot(robot, params: dict, arm_joint_ids, num_joints, device):
    """Apply sysid params (armature, friction, delay) to robot. Call after env creation/reset.

    params: dict with armature, static_friction, dynamic_ratio, viscous_friction (lists of 7),
            delay_steps (int). encoder_bias_rad is for initial pose offset, not applied here.
    """
    import torch

    arm = torch.tensor(params["armature"], dtype=torch.float32, device=device)
    sf = torch.tensor(params["static_friction"], dtype=torch.float32, device=device)
    dr = torch.tensor(params["dynamic_ratio"], dtype=torch.float32, device=device)
    vf = torch.tensor(params["viscous_friction"], dtype=torch.float32, device=device)
    delay = int(params["delay_steps"])

    N = robot.num_instances
    env_ids = torch.arange(N, device=device)

    armature_full = torch.zeros(N, num_joints, device=device)
    static_full = torch.zeros(N, num_joints, device=device)
    dynamic_full = torch.zeros(N, num_joints, device=device)
    viscous_full = torch.zeros(N, num_joints, device=device)

    armature_full[:, arm_joint_ids] = arm.unsqueeze(0).expand(N, -1)
    static_full[:, arm_joint_ids] = sf.unsqueeze(0).expand(N, -1)
    dynamic_full[:, arm_joint_ids] = (dr * sf).unsqueeze(0).expand(N, -1)
    viscous_full[:, arm_joint_ids] = vf.unsqueeze(0).expand(N, -1)

    robot.write_joint_armature_to_sim(armature_full, env_ids=env_ids)
    robot.write_joint_friction_coefficient_to_sim(
        static_full,
        joint_dynamic_friction_coeff=dynamic_full,
        joint_viscous_friction_coeff=viscous_full,
        env_ids=env_ids,
    )

    delay_int = torch.full((N,), delay, dtype=torch.int, device=device)
    for name in ARM_ACTUATOR_NAMES:
        act = robot.actuators[name]
        act.positions_delay_buffer.set_time_lag(delay_int)
        act.velocities_delay_buffer.set_time_lag(delay_int)
        act.efforts_delay_buffer.set_time_lag(delay_int)


# LEAP hand
FRANKA_LEAP_HAND_ACTUATOR_CFG = {
    "hand": ImplicitActuatorCfg(
        joint_names_expr=["j[0-9]+"],
        stiffness=20.0,
        damping=1.0,
        armature=0.001,
        friction=0.2,
        velocity_limit_sim=8.48,
        effort_limit_sim=0.95,
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
# Joint limits (Franka arm only — used to clamp reset poses)
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

