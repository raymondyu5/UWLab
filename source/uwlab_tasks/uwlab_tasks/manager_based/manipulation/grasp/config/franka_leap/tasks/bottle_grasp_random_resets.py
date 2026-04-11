# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Bottle grasp task with robot arm reset poses sampled from a JSON file.
# Identical to GraspBottle in every way except reset_robot draws a random
# arm pose from /workspace/uwlab/assets/reset_poses.json each episode.

import json
import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from ....mdp import reset_robot_joints_from_poses
from ..grasp_franka_leap import ARM_RESET, HAND_RESET
from .bottle import BOTTLE_SPAWN_POS, BOTTLE_TARGET_POS, GraspBottleFrankaLeapCfg

# ---------------------------------------------------------------------------
# Standalone module-level obs/rew functions for Hydra-compatible PPO config.
# These replace SimpleGraspReward bound methods which cannot survive the
# Hydra serialize → deserialize round-trip (bound method __module__ points to
# grasp_rewards, but the method is not a module-level attribute there).
# ---------------------------------------------------------------------------

_GRASP_GRASPED_Z = 0.12
_GRASP_LIFTED_Z_LOW = 0.20
_GRASP_LIFTED_Z_HIGH = 0.50
_GRASP_SUCCESS_DIST = 0.15


def _grasp_obs_object_pose(env) -> torch.Tensor:
    """Bottle pose (7D) in env-relative frame."""
    state = env.scene["grasp_object"]._data.root_state_w[:, :7].clone()
    state[:, :3] -= env.scene.env_origins
    return state


def _grasp_obs_target_pose(env) -> torch.Tensor:
    """Fixed target pose (7D): BOTTLE_TARGET_POS xyz + object's default rotation."""
    target = torch.tensor([list(BOTTLE_TARGET_POS)], device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
    default_quat = env.scene["grasp_object"]._data.default_root_state[:, 3:7].clone()
    return torch.cat([target, default_quat], dim=1)


def _grasp_rew_grasped(env) -> torch.Tensor:
    """Sparse +1 when object z >= BOTTLE_SPAWN_POS[2] + 0.12 m."""
    pos = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    return (pos[:, 2] - BOTTLE_SPAWN_POS[2] >= _GRASP_GRASPED_Z).float()


def _grasp_rew_lifted(env) -> torch.Tensor:
    """Sparse +1 when object z in [BOTTLE_SPAWN_POS[2] + 0.20, + 0.50] m."""
    pos = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    z_above = pos[:, 2] - BOTTLE_SPAWN_POS[2]
    return ((z_above >= _GRASP_LIFTED_Z_LOW) & (z_above <= _GRASP_LIFTED_Z_HIGH)).float()


def _grasp_rew_success(env) -> torch.Tensor:
    """Sparse +1 when 3D distance to BOTTLE_TARGET_POS <= 0.15 m."""
    pos = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    target = torch.tensor(list(BOTTLE_TARGET_POS), device=env.device, dtype=torch.float32)
    dist = torch.linalg.norm(pos - target.unsqueeze(0), dim=1)
    return (dist <= _GRASP_SUCCESS_DIST).float()


def _grasp_rew_joint_vel(env) -> torch.Tensor:
    return torch.sum(env.scene["robot"].data.joint_vel ** 2, dim=1)


def _grasp_rew_action_rate(env) -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)

RESET_POSES_PATH = "/workspace/uwlab/assets/reset_poses.json"


@configclass
class GraspBottleRandomResetsFrankaLeapCfg(GraspBottleFrankaLeapCfg):

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
                "canonical_reset_prob": 0.70,
            },
        )


@configclass
class GraspBottleRandomResetsFrankaLeapJointAbsCfg(GraspBottleRandomResetsFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        # Hold the randomly sampled reset pose, not the default ARM_RESET.
        return env.scene["robot"].data.joint_pos.clone()


@configclass
class GraspBottleRandomResetsFrankaLeapJointAbsStateCfg(GraspBottleRandomResetsFrankaLeapJointAbsCfg):
    """PPO-friendly variant: module-level obs/rew functions only (Hydra-safe), no seg_pc.

    Replaces all SimpleGraspReward bound methods with standalone module-level functions
    so the config survives the Hydra serialize → deserialize round-trip.
    Observation space: joint_pos (23) + arm_joint_pos (7) + ee_pose (7) +
                       hand_joint_pos (16) + object_pose (7) + target_pose (7) = 67D flat.
    """
    run_mode: str = "rl_mode"

    def __post_init__(self):
        super().__post_init__()
        # Replace bound method obs terms with module-level equivalents
        self.observations.policy.target_object_pose = ObsTerm(func=_grasp_obs_target_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=_grasp_obs_object_pose)
        self.observations.policy.contact_obs = None
        self.observations.policy.object_in_tip = None
        self.observations.policy.seg_pc = None
        self.observations.policy.concatenate_terms = True
        # Replace bound method reward terms with module-level equivalents
        self.rewards.grasped = RewTerm(func=_grasp_rew_grasped, weight=1.0)
        self.rewards.lifted = RewTerm(func=_grasp_rew_lifted, weight=5.0)
        self.rewards.success = RewTerm(func=_grasp_rew_success, weight=10.0)
        self.rewards.wrist = None  # requires bound method sensor — removed
        self.rewards.joint_vel = RewTerm(func=_grasp_rew_joint_vel, weight=-1e-3)
        self.rewards.action_rate = RewTerm(func=_grasp_rew_action_rate, weight=-5e-3)
        # Remove bound method event; replace metrics_spec with module-level equivalents
        self.events.capture_reset_height = None
        self.metrics_spec = {
            "is_success": _grasp_rew_success,
            "is_lifted": _grasp_rew_lifted,
            "is_grasped": _grasp_rew_grasped,
        }


@configclass
class GraspBottleRandomResetsFrankaLeapIkRelCfg(GraspBottleRandomResetsFrankaLeapCfg):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class GraspBottleRandomResetsFrankaLeapIkAbsCfg(GraspBottleRandomResetsFrankaLeapCfg):
    actions = franka_leap.FrankaLeapIkAbsArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        ee_body_idx = robot.body_names.index(franka_leap.FRANKA_LEAP_EE_BODY)
        ee_state = robot._data.body_state_w[:, ee_body_idx, :7].clone()
        ee_state[:, :3] -= env.scene.env_origins
        offset_pos = torch.tensor([list(franka_leap.FRANKA_LEAP_EE_OFFSET)], device=env.device).repeat(env.num_envs, 1)
        offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
        pos, quat = math_utils.combine_frame_transforms(ee_state[:, :3], ee_state[:, 3:7], offset_pos, offset_rot)
        hand_joints = robot.data.joint_pos[:, len(ARM_RESET):]
        return torch.cat([pos, quat, hand_joints], dim=-1)
