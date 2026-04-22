# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Cube push task with robot arm reset poses sampled from a JSON file.
# Identical to PushCubeToPoptart in every way except reset_robot draws a random
# arm pose from /workspace/uwlab/assets/reset_poses_cube_push.json each episode.

import json
import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from isaaclab.envs import mdp as isaac_mdp

from ....mdp import reset_robot_joints_from_poses, apply_force_after_step, clear_external_force
from ..grasp_franka_leap import ARM_RESET, HAND_RESET
from .cube_push import PushCubeToPoptartFrankaLeapCfg, CUBE_PUSH_SPAWN_POS

RESET_POSES_PATH = "/workspace/uwlab/assets/reset_poses_cube_push.json"


@configclass
class PushCubeToPoptartRandomResetsFrankaLeapCfg(PushCubeToPoptartFrankaLeapCfg):
    table_z_range: tuple = (0.0, 0.05)

    def __post_init__(self):
        super().__post_init__()

        self.events.clear_force_on_cube = EventTerm(
            func=clear_external_force,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("grasp_object")},
        )

        self.events.apply_force_to_cube = EventTerm(
            func=apply_force_after_step,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "force_range": (-0.2, 0.2),
                "torque_range": (0.0, 0.0),
                "min_episode_step": 30,
            },
        )

        self.events.randomize_object_mass = EventTerm(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="reset",
            min_step_count_between_reset=800,
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "mass_distribution_params": (1.0, 2.5),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.events.randomize_object_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="reset",
            min_step_count_between_reset=800,
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "static_friction_range": (0.7, 1.2),
                "dynamic_friction_range": (0.7, 1.2),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )

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
class PushCubeToPoptartRandomResetsFrankaLeapJointAbsCfg(PushCubeToPoptartRandomResetsFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        return env.scene["robot"].data.joint_pos.clone()
