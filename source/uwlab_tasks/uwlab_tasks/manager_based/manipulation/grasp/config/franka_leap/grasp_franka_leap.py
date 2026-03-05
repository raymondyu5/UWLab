# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Base robot-specific scene config for Franka-LEAP grasp tasks.
# Subclassed by tasks/pink_cup.py, tasks/cube.py, etc.
# Each task subclass adds the object, wires reward + seg_pc, and sets reset params.

from dataclasses import MISSING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from ... import grasp_env
from ...mdp import ee_pose_w
from isaaclab.managers import ObservationTermCfg as ObsTerm


@configclass
class FrankaLeapGraspSceneCfg(grasp_env.GraspSceneCfg):
    robot = franka_leap.IMPLICIT_FRANKA_LEAP.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class FrankaLeapGraspEnv(grasp_env.GraspEnv):
    scene: FrankaLeapGraspSceneCfg = FrankaLeapGraspSceneCfg(
        num_envs=1, env_spacing=2.5)



    def __post_init__(self):
        super().__post_init__()
        # Wire EE pose obs with Franka-LEAP calibration values
        self.observations.policy.ee_pose.params["asset_cfg"] = SceneEntityCfg("robot")
        self.observations.policy.ee_pose.params["ee_body_name"] = franka_leap.FRANKA_LEAP_EE_BODY
        self.observations.policy.ee_pose.params["ee_offset"] = franka_leap.FRANKA_LEAP_EE_OFFSET

        self.physics_hz = 60.0 # physics update frequency (number of physics updates per second)
        self.decimation = 6 # control update period (number of physics updates per control update)
        
        self.dt = 1 / self.physics_hz # physics update period (seconds per physics update)
        self.control_hz = 1 / (self.dt * self.decimation) # control update frequency (number of control updates per second)
    
