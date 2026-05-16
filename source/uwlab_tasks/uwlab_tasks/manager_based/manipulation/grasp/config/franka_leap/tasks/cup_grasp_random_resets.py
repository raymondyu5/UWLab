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
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from ....mdp import reset_robot_joints_from_poses
from ..grasp_franka_leap import ARM_RESET, HAND_RESET
from .pink_cup import GraspPinkCupFrankaLeapCfg, PINK_CUP_SPAWN_POS, PINK_CUP_TARGET_POS

# ---------------------------------------------------------------------------
# Module-level obs/rew functions for Hydra-compatible PPO config.
# ---------------------------------------------------------------------------

def _cup_obs_object_pose(env) -> torch.Tensor:
    state = env.scene["grasp_object"]._data.root_state_w[:, :7].clone()
    state[:, :3] -= env.scene.env_origins
    return state


def _cup_rew_grasped(env) -> torch.Tensor:
    pos = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    return (pos[:, 2] - PINK_CUP_SPAWN_POS[2] >= 0.12).float()


def _cup_rew_lifted(env) -> torch.Tensor:
    pos = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    z_above = pos[:, 2] - PINK_CUP_SPAWN_POS[2]
    return ((z_above >= 0.20) & (z_above <= 0.50)).float()


def _cup_rew_success(env) -> torch.Tensor:
    pos = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    target = torch.tensor(list(PINK_CUP_TARGET_POS), device=env.device, dtype=torch.float32)
    dist = torch.linalg.norm(pos - target.unsqueeze(0), dim=1)
    return (dist <= 0.15).float()


def _cup_rew_joint_vel(env) -> torch.Tensor:
    return torch.sum(env.scene["robot"].data.joint_vel ** 2, dim=1)


def _cup_rew_action_rate(env) -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


def _cup_obs_target_pose(env) -> torch.Tensor:
    """Fixed target pose (7D): PINK_CUP_TARGET_POS xyz + object's default rotation."""
    target = torch.tensor([list(PINK_CUP_TARGET_POS)], device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
    default_quat = env.scene["grasp_object"]._data.default_root_state[:, 3:7].clone()
    return torch.cat([target, default_quat], dim=1)


_FINGER_CONTACT_NAMES = [
    "palm_lower_contact", "fingertip_contact", "thumb_fingertip_contact",
    "fingertip_2_contact", "fingertip_3_contact",
]
_FINGER_BODY_NAMES = ["palm_lower", "fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]


def _cup_obs_contact(env) -> torch.Tensor:
    """Binary finger contact with object (5D)."""
    parts = []
    for name in _FINGER_CONTACT_NAMES:
        force = torch.linalg.norm(
            env.scene[name]._data.force_matrix_w.reshape(env.num_envs, 3), dim=1)
        parts.append((force > 4.0).int().unsqueeze(1))
    return torch.cat(parts, dim=1).float()


def _cup_obs_object_in_tip(env) -> torch.Tensor:
    """Object-to-fingertip displacement vectors, flattened (15D)."""
    obj_pos = env.scene["grasp_object"]._data.root_state_w[:, :3]
    parts = [obj_pos - env.scene[name].data.root_pos_w for name in _FINGER_BODY_NAMES]
    return torch.cat(parts, dim=1)


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
class GraspPinkCupRandomResetsFrankaLeapJointAbsStateCfg(GraspPinkCupRandomResetsFrankaLeapJointAbsCfg):
    """PPO-friendly variant: module-level obs/rew functions only (Hydra-safe), no seg_pc.

    Observation space: arm_joint_pos (7) + hand_joint_pos (16) + object_pose (7) = 30D flat.
    Matches the BC training obs keys so BC checkpoints transfer directly.
    """
    run_mode: str = "rl_mode"

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.manipulated_object_pose = ObsTerm(func=_cup_obs_object_pose)
        self.observations.policy.target_object_pose = ObsTerm(func=_cup_obs_target_pose)
        self.observations.policy.contact_obs = ObsTerm(func=_cup_obs_contact)
        self.observations.policy.object_in_tip = ObsTerm(func=_cup_obs_object_in_tip)
        self.observations.policy.joint_pos = None
        self.observations.policy.ee_pose = None
        self.observations.policy.seg_pc = None
        self.observations.policy.concatenate_terms = True
        self.rewards.grasped = RewTerm(func=_cup_rew_grasped, weight=1.0)
        self.rewards.lifted = RewTerm(func=_cup_rew_lifted, weight=5.0)
        self.rewards.success = RewTerm(func=_cup_rew_success, weight=10.0)
        self.rewards.wrist = None
        self.rewards.joint_vel = RewTerm(func=_cup_rew_joint_vel, weight=-1e-3)
        self.rewards.action_rate = RewTerm(func=_cup_rew_action_rate, weight=-0.5)
        self.events.capture_reset_height = None
        self.metrics_spec = {
            "is_success": _cup_rew_success,
            "is_lifted": _cup_rew_lifted,
            "is_grasped": _cup_rew_grasped,
        }


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
