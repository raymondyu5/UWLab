# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Bottle pour task for Franka-LEAP.
# Source values from:
#   IsaacLab/source/config/task/hand_env/leap_franka/grasp/rl_env_bourbon_pour_pink_cup_delta_joint.yaml

import torch
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from .rewards.pour_rewards import BOTTLE_CAP_OFFSET, is_grasped, is_healthy_z, is_near_miss, is_success
from .rewards.pour_rewards import (
    PourReward,
    pour_action_rate_l2,
    pour_cup_topple,
    pour_grasped,
    pour_joint_pos_limits,
    pour_joint_vel_l2,
    pour_success,
    pour_xy_healthy,
    pour_xy_near_miss,
)

from ....mdp import (
    CachedSamplePC,
    reset_object_pose,
    reset_table_block,
    capture_bottle_reset_height,
    bottle_dropped,
    bottle_too_far,
    cup_toppled,
)
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET, ARM_NUM_POINTS, HAND_NUM_POINTS
from .shared_params import ARM_MESH_DIR, HAND_MESH_DIR, FINGERS_NAME_LIST
from .bottle import (
    GraspBottleSceneCfg,
    BOTTLE_USD,
    BOTTLE_SPAWN_POS,
    BOTTLE_SPAWN_ROT,
    BOTTLE_OBJECT_NUM_POINTS,
)
from .pink_cup import PINK_CUP_USD
from .pink_cup import PINK_CUP_OBJECT_NUM_POINTS
# Spawn positions centered within real-world workspace bounds:
# Bottle: x∈[0.42,0.64], y∈[-0.20,-0.01]  →  center (0.53, -0.105), range ±(0.11, 0.095)
# Cup:    x∈[0.41,0.65], y∈[0.17,0.29]    →  center (0.53,  0.23),  range ±(0.12, 0.06)
BOTTLE_POUR_SPAWN_POS = (0.53, -0.105, 0.11)
PINK_CUP_POUR_POS = (0.53, 0.23, 0.07)
PINK_CUP_POUR_ROT = (0.707, 0.707, 0.0, 0.0)

POUR_HORIZON = 176


def obs_cup_pose(env) -> torch.Tensor:
    """Cup position (3D) in env-relative frame."""
    cup_pose = env.scene["pink_cup"]._data.root_state_w[:, :7].clone()
    cup_pose[:, :3] -= env.scene.env_origins
    return cup_pose[:, :3]


def obs_manipulated_object_pose(env) -> torch.Tensor:
    """Bottle pose (7D) in env-relative frame."""
    bottle_pose = env.scene["grasp_object"]._data.root_state_w[:, :7].clone()
    bottle_pose[:, :3] -= env.scene.env_origins
    return bottle_pose


def obs_target_object_pose(env) -> torch.Tensor:
    """Target bottle pose with cap centered over cup at a pouring-friendly height."""
    cup_pose = env.scene["pink_cup"]._data.root_state_w[:, :7].clone()
    cup_pose[:, :3] -= env.scene.env_origins
    cup_top_z = cup_pose[:, 2] + 0.15
    cup_center_xy = cup_pose[:, :2]

    target_quat = env.scene["grasp_object"].data.default_root_state[:, 3:7].clone()
    cap_offset = torch.tensor(list(BOTTLE_CAP_OFFSET), device=env.device, dtype=torch.float32).unsqueeze(0).expand(env.num_envs, -1)
    cap_offset_world = math_utils.quat_apply(target_quat, cap_offset)

    desired_cap_pos = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    desired_cap_pos[:, :2] = cup_center_xy
    desired_cap_pos[:, 2] = cup_top_z + 0.10

    target_pose = torch.zeros(env.num_envs, 7, device=env.device, dtype=torch.float32)
    target_pose[:, :3] = desired_cap_pos - cap_offset_world
    target_pose[:, 3:7] = target_quat
    return target_pose


@configclass
class PourBottleSceneCfg(GraspBottleSceneCfg):
    pink_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PinkCup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=PINK_CUP_POUR_POS,
            rot=PINK_CUP_POUR_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=PINK_CUP_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
    )


@configclass
class PourBottleFrankaLeapCfg(grasp_franka_leap.FrankaLeapGraspEnvCfg):
    scene: PourBottleSceneCfg = PourBottleSceneCfg(num_envs=1, env_spacing=2.5)
    table_z_range: tuple = (0.0, 0.0)  # set to (0.0, 0.05) to enable table height randomization
    distill_include_entity_names: tuple[str, ...] = ("robot", "grasp_object", "pink_cup")

    def __post_init__(self):
        super().__post_init__()

        self.object_spawn_defaults = {
            "default_pos": list(BOTTLE_POUR_SPAWN_POS),
            "default_rot": list(BOTTLE_SPAWN_ROT),
            "reset_height": BOTTLE_POUR_SPAWN_POS[2],
        }

        
        self.setup_horizon(horizon=POUR_HORIZON)
        self.rewards.grasped = RewTerm(func=pour_grasped, weight=1.0)
        self.rewards.xy_healthy = RewTerm(func=pour_xy_healthy, weight=2.0)
        self.rewards.xy_near_miss = RewTerm(func=pour_xy_near_miss, weight=5.0)
        self.rewards.success = RewTerm(func=pour_success, weight=10.0)
        self.rewards.cup_topple = RewTerm(func=pour_cup_topple, weight=-10.0)
        self.rewards.joint_vel = RewTerm(
            func=pour_joint_vel_l2,
            weight=-1.0e-3,
            params={"asset_name": "robot"},
        )
        self.rewards.joint_limit = RewTerm(
            func=pour_joint_pos_limits,
            weight=-6.0e-1,
            params={"asset_name": "robot"},
        )
        self.rewards.action_rate = RewTerm(func=pour_action_rate_l2, weight=-5.0e-3)

        # Task-defined boolean-ish metrics used by eval scripts.
        self.metrics_spec = {
            "is_success": is_success,
            "is_grasped": is_grasped,
            "is_healthy_z": is_healthy_z,
            "is_near_miss": is_near_miss,
        }

        self.observations.policy.cup_pose = ObsTerm(func=obs_cup_pose)
        self.observations.policy.target_object_pose = ObsTerm(func=obs_target_object_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=obs_manipulated_object_pose)

        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object", "pink_cup"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[BOTTLE_OBJECT_NUM_POINTS, PINK_CUP_OBJECT_NUM_POINTS],
            num_downsample_points=2048,
            pcd_crop_region=self.pcd_crop_region,
            pcd_noise=0.0,
        )

        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.get_seg_pc)

        self.events.reset_table_block = EventTerm(
            func=reset_table_block,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("table_block"),
                "z_range": self.table_z_range,
            },
        )

        # Reset bottle
        self.events.reset_object = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "default_pos": BOTTLE_POUR_SPAWN_POS,
                "default_rot_quat": BOTTLE_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.11, 0.11),
                    "y": (-0.095, 0.095),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": BOTTLE_POUR_SPAWN_POS[2],
                "table_block_name": "table_block",
            },
        )

        # Reset pink cup (placed to the right of robot, slight XY jitter)
        self.events.reset_cup_object = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("pink_cup"),
                "default_pos": PINK_CUP_POUR_POS,
                "default_rot_quat": PINK_CUP_POUR_ROT,
                "pose_range": {
                    "x": (-0.12, 0.12),
                    "y": (-0.06, 0.06),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": PINK_CUP_POUR_POS[2],
                "table_block_name": "table_block",
            },
        )

        # Capture reset references AFTER both objects have been reset
        self.events.capture_reset_references = EventTerm(
            func=capture_bottle_reset_height,
            mode="reset",
            params={"object_name": "grasp_object"},
        )

        # Terminations
        # self.terminations.bottle_dropped = DoneTerm(
        #     func=bottle_dropped,
        #     params={
        #         "object_name": "grasp_object",
        #         "z_margin": 0.05,
        #     },
        #     time_out=False,
        # )
        self.terminations.bottle_too_far = DoneTerm(
            func=bottle_too_far,
            params={
                "object_name": "grasp_object",
                "max_xy_dist": 1.0,
            },
            time_out=False,
        )
        self.terminations.cup_toppled = DoneTerm(
            func=cup_toppled,
            params={
                "cup_name": "pink_cup",
                "spawn_quat": PINK_CUP_POUR_ROT,
                "angle_thresh_rad": 0.524,
            },
            time_out=False,
        )


@configclass
class PourBottleFrankaLeapJointAbsCfg(PourBottleFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class PourBottleFrankaLeapJointRelCfg(PourBottleFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class PourBottleFrankaLeapIkRelCfg(PourBottleFrankaLeapCfg):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class PourBottleFrankaLeapIkAbsCfg(PourBottleFrankaLeapCfg):
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
