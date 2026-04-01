# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Bottle singulate task for Franka-LEAP.
# Goal: pick up (singulate) a bourbon bottle placed next to a poptart box obstacle.

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

import uwlab_assets.robots.franka_leap as franka_leap

from .rewards.singulate_rewards import (
    SINGULATE_TARGET_POS,
    is_grasped,
    is_success,
    is_box_clear,
    singulate_grasped,
    singulate_success,
    singulate_box_separation,
    singulate_joint_vel_l2,
    singulate_joint_pos_limits,
    singulate_action_rate_l2,
)

from ....mdp import (
    CachedSamplePC,
    reset_object_pose,
    reset_bottle_and_box,
    reset_table_block,
    capture_bottle_reset_height,
    bottle_too_far,
    cup_toppled,
)
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET, ARM_NUM_POINTS, HAND_NUM_POINTS
from .bottle import (
    GraspBottleSceneCfg,
    BOTTLE_SPAWN_ROT,
    BOTTLE_OBJECT_NUM_POINTS,
)

# ---------------------------------------------------------------------------
# Object dimensions (metres)
# ---------------------------------------------------------------------------
BOTTLE_WIDTH = 0.08
BOTTLE_LENGTH = 0.25

BOX_WIDTH = 0.09
BOX_LENGTH = 0.135   # height when box is standing upright

BOX_USD = "/workspace/uwlab/assets/poptart/rigid_object.usd"
BOX_OBJECT_NUM_POINTS = 128

# ---------------------------------------------------------------------------
# Bottle randomisation bounds (centre of bottle)
# x: [.38 + bottle_width/2, .68 - bottle_width/2]
# y: [.11 - bottle_length/2, .16 - bottle_length/2]
# ---------------------------------------------------------------------------
BOTTLE_X_RANGE = (0.38 + BOTTLE_WIDTH / 2, 0.68 - BOTTLE_WIDTH / 2)   # (0.42, 0.64)
BOTTLE_Y_RANGE = (0.11 - BOTTLE_LENGTH / 2, 0.16 - BOTTLE_LENGTH / 2)  # (-0.015, 0.035)
BOTTLE_RESET_HEIGHT = 0.11

BOX_SPAWN_ROT = (1.0, 0.0, 0.0, 0.0)  # upright, axis-aligned
BOX_RESET_HEIGHT = BOX_LENGTH / 2      # 0.0675  (box stands upright on table)

# Nominal centres (midpoint of randomisation range) — used for scene init_state and object_spawn_defaults.
_BOTTLE_CENTER_X = (BOTTLE_X_RANGE[0] + BOTTLE_X_RANGE[1]) / 2   # 0.53
_BOTTLE_CENTER_Y = (BOTTLE_Y_RANGE[0] + BOTTLE_Y_RANGE[1]) / 2   # 0.01
SINGULATE_BOTTLE_SPAWN_POS = (_BOTTLE_CENTER_X, _BOTTLE_CENTER_Y, BOTTLE_RESET_HEIGHT)
BOX_SPAWN_POS = (_BOTTLE_CENTER_X - BOTTLE_WIDTH / 2 - BOX_WIDTH / 2, _BOTTLE_CENTER_Y, BOX_RESET_HEIGHT)

SINGULATE_HORIZON = 176


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def obs_box_pose(env) -> torch.Tensor:
    """Box position (3D) in env-relative frame."""
    box_state = env.scene["box"]._data.root_state_w[:, :7].clone()
    box_state[:, :3] -= env.scene.env_origins
    return box_state[:, :3]


def obs_manipulated_object_pose(env) -> torch.Tensor:
    """Bottle pose (7D) in env-relative frame."""
    bottle_pose = env.scene["grasp_object"]._data.root_state_w[:, :7].clone()
    bottle_pose[:, :3] -= env.scene.env_origins
    return bottle_pose


def obs_target_object_pose(env) -> torch.Tensor:
    """Target bottle pose: fixed workspace centre at z=0.30, spawn orientation."""
    target_quat = env.scene["grasp_object"].data.default_root_state[:, 3:7].clone()
    target_pose = torch.zeros(env.num_envs, 7, device=env.device, dtype=torch.float32)
    target_pose[:, 0] = SINGULATE_TARGET_POS[0]
    target_pose[:, 1] = SINGULATE_TARGET_POS[1]
    target_pose[:, 2] = SINGULATE_TARGET_POS[2]
    target_pose[:, 3:7] = target_quat
    return target_pose


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------

@configclass
class SingulateBottleSceneCfg(GraspBottleSceneCfg):
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BOX_SPAWN_POS,
            rot=BOX_SPAWN_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOX_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------

@configclass
class SingulateBottleFrankaLeapCfg(grasp_franka_leap.FrankaLeapGraspEnvCfg):
    scene: SingulateBottleSceneCfg = SingulateBottleSceneCfg(num_envs=1, env_spacing=2.5)
    table_z_range: tuple = (0.0, 0.0)
    distill_include_entity_names: tuple[str, ...] = ("robot", "grasp_object", "box")

    def __post_init__(self):
        super().__post_init__()

        self.object_spawn_defaults = {
            "default_pos": list(SINGULATE_BOTTLE_SPAWN_POS),
            "default_rot": list(BOTTLE_SPAWN_ROT),
            "reset_height": SINGULATE_BOTTLE_SPAWN_POS[2],
        }

        # Disable the base-class reset_object event; reset_bottle_and_box handles both objects.
        self.events.reset_object = None

        self.setup_horizon(horizon=SINGULATE_HORIZON)

        # Rewards
        self.rewards.grasped = RewTerm(func=singulate_grasped, weight=1.0)
        self.rewards.box_separation = RewTerm(func=singulate_box_separation, weight=2.0)
        self.rewards.success = RewTerm(func=singulate_success, weight=10.0)
        self.rewards.joint_vel = RewTerm(
            func=singulate_joint_vel_l2,
            weight=-1.0e-3,
            params={"asset_name": "robot"},
        )
        self.rewards.joint_limit = RewTerm(
            func=singulate_joint_pos_limits,
            weight=-6.0e-1,
            params={"asset_name": "robot"},
        )
        self.rewards.action_rate = RewTerm(func=singulate_action_rate_l2, weight=-5.0e-3)

        # Metrics
        self.metrics_spec = {
            "is_success": is_success,
            "is_grasped": is_grasped,
            "is_box_clear": is_box_clear,
        }

        # Observations
        self.observations.policy.box_pose = ObsTerm(func=obs_box_pose)
        self.observations.policy.target_object_pose = ObsTerm(func=obs_target_object_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=obs_manipulated_object_pose)

        # Point cloud
        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object", "box"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[BOTTLE_OBJECT_NUM_POINTS, BOX_OBJECT_NUM_POINTS],
            num_downsample_points=2048,
            pcd_crop_region=self.pcd_crop_region,
            pcd_noise=0.0,
        )
        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.get_seg_pc)

        # Events
        self.events.reset_table_block = EventTerm(
            func=reset_table_block,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("table_block"),
                "z_range": self.table_z_range,
            },
        )

        self.events.reset_bottle_and_box = EventTerm(
            func=reset_bottle_and_box,
            mode="reset",
            params={
                "bottle_cfg": SceneEntityCfg("grasp_object"),
                "box_cfg": SceneEntityCfg("box"),
                "bottle_x_range": BOTTLE_X_RANGE,
                "bottle_y_range": BOTTLE_Y_RANGE,
                "bottle_rot_quat": BOTTLE_SPAWN_ROT,
                "bottle_reset_height": BOTTLE_RESET_HEIGHT,
                "bottle_width": BOTTLE_WIDTH,
                "bottle_length": BOTTLE_LENGTH,
                "box_width": BOX_WIDTH,
                "box_length": BOX_LENGTH,
                "box_rot_quat": BOX_SPAWN_ROT,
                "box_reset_height": BOX_RESET_HEIGHT,
                "table_block_name": "table_block",
            },
        )

        self.events.capture_reset_references = EventTerm(
            func=capture_bottle_reset_height,
            mode="reset",
            params={"object_name": "grasp_object"},
        )

        # Terminations
        self.terminations.bottle_too_far = DoneTerm(
            func=bottle_too_far,
            params={
                "object_name": "grasp_object",
                "max_xy_dist": 1.0,
            },
            time_out=False,
        )
        self.terminations.box_toppled = DoneTerm(
            func=cup_toppled,
            params={
                "cup_name": "box",
                "spawn_quat": BOX_SPAWN_ROT,
                "angle_thresh_rad": 0.524,
            },
            time_out=False,
        )


# ---------------------------------------------------------------------------
# Action-space variants
# ---------------------------------------------------------------------------

@configclass
class SingulateBottleFrankaLeapJointAbsCfg(SingulateBottleFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class SingulateBottleFrankaLeapJointRelCfg(SingulateBottleFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class SingulateBottleFrankaLeapIkRelCfg(SingulateBottleFrankaLeapCfg):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class SingulateBottleFrankaLeapIkAbsCfg(SingulateBottleFrankaLeapCfg):
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
