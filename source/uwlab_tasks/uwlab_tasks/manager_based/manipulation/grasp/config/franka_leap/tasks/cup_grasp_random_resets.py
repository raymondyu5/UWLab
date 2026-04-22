# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Cup grasp task with robot arm reset poses sampled from a JSON file.
# Identical to GraspPinkCup in every way except reset_robot draws a random
# arm pose from /workspace/uwlab/assets/reset_poses_cup_grasp.json each episode.

import json
import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from ....mdp import reset_robot_joints_from_poses
from ..grasp_franka_leap import ARM_RESET, HAND_RESET
from .pink_cup import GraspPinkCupFrankaLeapCfg

RESET_POSES_PATH = "/workspace/uwlab/assets/reset_poses_cup_grasp.json"


@configclass
class GraspPinkCupRandomResetsFrankaLeapCfg(GraspPinkCupFrankaLeapCfg):

    def __post_init__(self):
        super().__post_init__()

        with open(RESET_POSES_PATH) as f:
            arm_joint_poses = json.load(f)["poses"]

        self.events.reset_robot = EventTerm(
            func=reset_robot_joints_from_poses,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "arm_joint_poses": arm_joint_poses,
                "hand_joint_pos": HAND_RESET,
                "arm_joint_limits": franka_leap.FRANKA_LEAP_ARM_JOINT_LIMITS,
                "canonical_arm_joint_pos": ARM_RESET,
                "canonical_reset_prob": 1.0,
            },
        )


@configclass
class GraspPinkCupRandomResetsFrankaLeapJointAbsCfg(GraspPinkCupRandomResetsFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        return env.scene["robot"].data.joint_pos.clone()


@configclass
class GraspPinkCupRandomResets7030FrankaLeapCfg(GraspPinkCupRandomResetsFrankaLeapCfg):

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_robot.params["canonical_reset_prob"] = 0.70


@configclass
class GraspPinkCupRandomResets7030FrankaLeapJointAbsCfg(GraspPinkCupRandomResets7030FrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        return env.scene["robot"].data.joint_pos.clone()
