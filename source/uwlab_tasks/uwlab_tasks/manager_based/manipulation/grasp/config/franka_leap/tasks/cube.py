# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Cube grasp task for Franka-LEAP.

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from ....mdp import CachedSamplePC, reset_object_pose, reset_table_block, log_object_mass, log_object_scales
from .rewards.grasp_rewards import SimpleGraspReward
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET, ARM_NUM_POINTS, HAND_NUM_POINTS
from .shared_params import ARM_MESH_DIR, HAND_MESH_DIR, FINGERS_NAME_LIST

CUBE_USD = "/workspace/uwlab/assets/cube/rigid_object.usd"
CUBE_OBJECT_NUM_POINTS = 128

# TODO: update spawn pos/rot and pose_range once real-world workspace bounds are known
CUBE_SPAWN_POS = (0.50, 0.0, 0.07)
CUBE_SPAWN_ROT = (1.0, 0.0, 0.0, 0.0)
CUBE_TARGET_POS = (0.55, 0.0, 0.35)
CUBE_HORIZON = 128
CUBE_SUCCESS_HEIGHT = 0.20
CUBE_GRASPED_HEIGHT = 0.12

# ---------------------------------------------------------------------------
# Module-level obs/rew functions for Hydra-compatible PPO config.
# ---------------------------------------------------------------------------

def _cube_obs_object_pose(env) -> torch.Tensor:
    state = env.scene["grasp_object"]._data.root_state_w[:, :7].clone()
    state[:, :3] -= env.scene.env_origins
    return state


def _cube_rew_grasped(env) -> torch.Tensor:
    pos = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    return (pos[:, 2] - CUBE_SPAWN_POS[2] >= 0.12).float()


def _cube_rew_lifted(env) -> torch.Tensor:
    pos = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    z_above = pos[:, 2] - CUBE_SPAWN_POS[2]
    return ((z_above >= 0.20) & (z_above <= 0.50)).float()


def _cube_rew_success(env) -> torch.Tensor:
    pos = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    target = torch.tensor(list(CUBE_TARGET_POS), device=env.device, dtype=torch.float32)
    dist = torch.linalg.norm(pos - target.unsqueeze(0), dim=1)
    return (dist <= 0.15).float()


def _cube_rew_joint_vel(env) -> torch.Tensor:
    return torch.sum(env.scene["robot"].data.joint_vel ** 2, dim=1)


def _cube_rew_action_rate(env) -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


def _cube_obs_target_pose(env) -> torch.Tensor:
    """Fixed target pose (7D): CUBE_TARGET_POS xyz + object's default rotation."""
    target = torch.tensor([list(CUBE_TARGET_POS)], device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
    default_quat = env.scene["grasp_object"]._data.default_root_state[:, 3:7].clone()
    return torch.cat([target, default_quat], dim=1)


_FINGER_CONTACT_NAMES = [
    "palm_lower_contact", "fingertip_contact", "thumb_fingertip_contact",
    "fingertip_2_contact", "fingertip_3_contact",
]
_FINGER_BODY_NAMES = ["palm_lower", "fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]


def _cube_obs_contact(env) -> torch.Tensor:
    """Binary finger contact with object (5D)."""
    parts = []
    for name in _FINGER_CONTACT_NAMES:
        force = torch.linalg.norm(
            env.scene[name]._data.force_matrix_w.reshape(env.num_envs, 3), dim=1)
        parts.append((force > 4.0).int().unsqueeze(1))
    return torch.cat(parts, dim=1).float()


def _cube_obs_object_in_tip(env) -> torch.Tensor:
    """Object-to-fingertip displacement vectors, flattened (15D)."""
    obj_pos = env.scene["grasp_object"]._data.root_state_w[:, :3]
    parts = [obj_pos - env.scene[name].data.root_pos_w for name in _FINGER_BODY_NAMES]
    return torch.cat(parts, dim=1)


@configclass
class GraspCubeSceneCfg(grasp_franka_leap.FrankaLeapGraspSceneCfg):
    grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CUBE_SPAWN_POS,
            rot=CUBE_SPAWN_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUBE_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
    )

    table_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableBlock",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(CUBE_SPAWN_POS[0], CUBE_SPAWN_POS[1], 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/uwlab/assets/table/table_block.usd",
            scale=(1.2, 1.0, 0.10),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
            ),
        ),
    )


@configclass
class GraspCubeFrankaLeapCfg(grasp_franka_leap.FrankaLeapGraspEnvCfg):
    scene: GraspCubeSceneCfg = GraspCubeSceneCfg(num_envs=1, env_spacing=2.5)
    table_z_range: tuple = (0.0, 0.05)

    def _cube_z_above_table(self, env) -> torch.Tensor:
        obj = env.scene["grasp_object"]
        pos = obj.data.root_pos_w - env.scene.env_origins
        block = env.scene["table_block"]
        table_z_offset = (
            block.data.root_state_w[:, 2]
            - env.scene.env_origins[:, 2]
            - block.data.default_root_state[:, 2]
        )
        return pos[:, 2], table_z_offset

    def is_grasped(self, env) -> torch.Tensor:
        z, table_z_offset = self._cube_z_above_table(env)
        return z >= (CUBE_GRASPED_HEIGHT + table_z_offset)

    def is_success(self, env) -> torch.Tensor:
        z, table_z_offset = self._cube_z_above_table(env)
        return z >= (CUBE_SUCCESS_HEIGHT + table_z_offset)

    def __post_init__(self):
        super().__post_init__()

        self.object_spawn_defaults = {
            "default_pos": tuple(CUBE_SPAWN_POS),
            "default_rot": tuple(CUBE_SPAWN_ROT),
            "reset_height": float(CUBE_SPAWN_POS[2]),
        }

        self.setup_horizon(horizon=CUBE_HORIZON)

        simple_rew = SimpleGraspReward(
            asset_name="robot",
            object_name="grasp_object",
            fingers_name_list=FINGERS_NAME_LIST,
            init_height=CUBE_SPAWN_POS[2],
            target_pos=CUBE_TARGET_POS,
        )
        simple_rew.setup_wrist_sensor(self.scene)
        simple_rew.setup_finger_entities(self.scene)
        simple_rew.setup_finger_sensors(self.scene, object_prim_name="GraspObject")

        self.rewards.grasped     = RewTerm(func=simple_rew.rew_grasped,       weight=1.0)
        self.rewards.lifted      = RewTerm(func=simple_rew.rew_lifted,        weight=5.0)
        self.rewards.success     = RewTerm(func=simple_rew.rew_success,       weight=10.0)
        self.rewards.wrist       = RewTerm(func=simple_rew.rew_wrist_penalty, weight=-2.0)
        self.rewards.joint_vel   = RewTerm(func=simple_rew.rew_joint_vel,     weight=-1e-3)
        self.rewards.action_rate = RewTerm(func=simple_rew.rew_action_rate,   weight=-5e-3)
        self.metrics_spec = {"is_success": simple_rew.rew_success, "is_lifted": simple_rew.rew_lifted, "is_grasped": simple_rew.rew_grasped}

        self.observations.policy.target_object_pose = ObsTerm(func=simple_rew.obs_target_object_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=simple_rew.obs_manipulated_object_pose)
        self.observations.policy.contact_obs = ObsTerm(func=simple_rew.obs_contact)
        self.observations.policy.object_in_tip = ObsTerm(func=simple_rew.obs_object_in_tip)

        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[CUBE_OBJECT_NUM_POINTS],
            num_downsample_points=2048,
            pcd_crop_region=self.pcd_crop_region,
            pcd_noise=0.02,
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

        self.events.reset_object = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "default_pos": CUBE_SPAWN_POS,
                "default_rot_quat": CUBE_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.20, 0.20),
                    "y": (-0.20, 0.20),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": CUBE_SPAWN_POS[2],
                "table_block_name": "table_block",
            },
        )

        self.events.capture_reset_height = EventTerm(
            func=simple_rew.capture_reset_height,
            mode="reset",
            params={},
        )

        self.events.randomize_object_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="reset",
            min_step_count_between_reset=800,
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "static_friction_range": (0.3, 0.6),
                "dynamic_friction_range": (0.3, 0.6),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )

        self.events.randomize_object_mass = EventTerm(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="reset",
            min_step_count_between_reset=800,
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "mass_distribution_params": (1.0, 2.0),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.events.randomize_object_scale = EventTerm(
            func=isaac_mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "scale_range": (0.9, 1.2),
            },
        )

        self.events.log_object_mass = EventTerm(
            func=log_object_mass,
            mode="reset",
            min_step_count_between_reset=800,
            params={"asset_cfg": SceneEntityCfg("grasp_object")},
        )

        self.events.log_object_scales = EventTerm(
            func=log_object_scales,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("grasp_object")},
        )


@configclass
class GraspCubeFrankaLeapJointAbsCfg(GraspCubeFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class GraspCubeFrankaLeapJointAbsStateCfg(GraspCubeFrankaLeapJointAbsCfg):
    """PPO-friendly variant: module-level obs/rew functions only (Hydra-safe), no seg_pc.

    Observation space: arm_joint_pos (7) + hand_joint_pos (16) + object_pose (7) = 30D flat.
    Matches the BC training obs keys so BC checkpoints transfer directly.
    """
    run_mode: str = "rl_mode"

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.manipulated_object_pose = ObsTerm(func=_cube_obs_object_pose)
        self.observations.policy.target_object_pose = ObsTerm(func=_cube_obs_target_pose)
        self.observations.policy.contact_obs = ObsTerm(func=_cube_obs_contact)
        self.observations.policy.object_in_tip = ObsTerm(func=_cube_obs_object_in_tip)
        self.observations.policy.joint_pos = None
        self.observations.policy.ee_pose = None
        self.observations.policy.seg_pc = None
        self.observations.policy.concatenate_terms = True
        self.rewards.grasped = RewTerm(func=_cube_rew_grasped, weight=1.0)
        self.rewards.lifted = RewTerm(func=_cube_rew_lifted, weight=5.0)
        self.rewards.success = RewTerm(func=_cube_rew_success, weight=10.0)
        self.rewards.wrist = None
        self.rewards.joint_vel = RewTerm(func=_cube_rew_joint_vel, weight=-1e-3)
        self.rewards.action_rate = RewTerm(func=_cube_rew_action_rate, weight=-5e-3)
        self.events.capture_reset_height = None
        self.metrics_spec = {
            "is_success": _cube_rew_success,
            "is_lifted": _cube_rew_lifted,
            "is_grasped": _cube_rew_grasped,
        }


@configclass
class GraspCubeFrankaLeapJointAbsStateCollectCfg(GraspCubeFrankaLeapJointAbsStateCfg):
    """StateCfg with seg_pc re-enabled and dict obs for RL rollout collection.

    Inherits the correct module-level obs functions from StateCfg (same as training),
    then re-adds mesh-based seg_pc and switches to dict obs so the collection script
    can feed the flat privileged-state input to the policy and save seg_pc separately.
    """

    def __post_init__(self):
        super().__post_init__()
        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[CUBE_OBJECT_NUM_POINTS],
            num_downsample_points=2048,
            pcd_crop_region=self.pcd_crop_region,
            pcd_noise=0.02,
        )
        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.get_seg_pc)
        self.observations.policy.concatenate_terms = False


@configclass
class GraspCubeFrankaLeapIkRelCfg(GraspCubeFrankaLeapCfg):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class GraspCubeFrankaLeapIkAbsCfg(GraspCubeFrankaLeapCfg):
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
