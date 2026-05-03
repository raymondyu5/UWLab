# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Credit card grasp task for Franka-LEAP.
# Goal: grasp the credit card lying flat on top of the shelf cube.
# The card is spawned near the +Y side edge so fingers can approach from the side.

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

from ....mdp import CachedSamplePC
from .rewards.grasp_rewards import SimpleGraspReward
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET, ARM_NUM_POINTS, HAND_NUM_POINTS
from .shared_params import FINGERS_NAME_LIST

SHELF_USD = "/workspace/uwlab/assets/block_new/rigid_object.usda"
CARD_USD = "/workspace/uwlab/assets/credit_card/rigid_object.usda"

CARD_OBJECT_NUM_POINTS = 256

# Shelf cube spawned as a fixed kinematic platform.
SHELF_SPAWN_POS = (0.55, -0.20, 0.0)
SHELF_SPAWN_ROT = (0.707, 0.0, 0.0, 0.707)  # 90° around Z

CARD_SHELF_OFFSET = (0.000, 0.050, 0.112)  # offset from shelf spawn origin; z = block_top(0.100) + gap(0.012)
CARD_SPAWN_POS = tuple(s + o for s, o in zip(SHELF_SPAWN_POS, CARD_SHELF_OFFSET))
CARD_SPAWN_ROT = (0.707, 0.0, 0.0, 0.707)  # 90° around Z, same as shelf

CARD_TARGET_POS = (CARD_SPAWN_POS[0], 0.0, 0.35)


def _reset_shelf_and_card(env, env_ids, shelf_pose_range: dict):
    shelf = env.scene["shelf_cube"]
    card = env.scene["grasp_object"]
    n = len(env_ids)

    # Randomize shelf position: default local pos + env origin + random offset
    shelf_state = shelf.data.default_root_state[env_ids].clone()
    shelf_state[:, :3] += env.scene.env_origins[env_ids]
    for i, key in enumerate(["x", "y", "z"]):
        lo, hi = shelf_pose_range.get(key, (0.0, 0.0))
        if lo != hi:
            shelf_state[:, i] += torch.empty(n, device=env.device).uniform_(lo, hi)
    shelf.write_root_pose_to_sim(shelf_state[:, :7], env_ids=env_ids)

    # Place card at new shelf world pos + fixed offset + random x, zero velocity
    offset = torch.tensor(CARD_SHELF_OFFSET, device=env.device, dtype=torch.float32)
    card_rot = torch.tensor(CARD_SPAWN_ROT, device=env.device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    card_state = torch.zeros(n, 13, device=env.device, dtype=torch.float32)
    card_state[:, :3] = shelf_state[:, :3] + offset
    card_state[:, 0] += torch.empty(n, device=env.device).uniform_(-0.05, 0.0)
    card_state[:, 3:7] = card_rot
    card.write_root_state_to_sim(card_state, env_ids=env_ids)


def _rew_arm_action_rate(env) -> torch.Tensor:
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(delta[:, :7] ** 2, dim=1)


def _rew_index_thumb_contact(env) -> torch.Tensor:
    index_force = torch.linalg.norm(
        env.scene["fingertip_contact"]._data.force_matrix_w.reshape(env.num_envs, 3), dim=1
    )
    thumb_force = torch.linalg.norm(
        env.scene["thumb_fingertip_contact"]._data.force_matrix_w.reshape(env.num_envs, 3), dim=1
    )
    index_contact = (index_force > 1.0).float()
    thumb_contact = (thumb_force > 1.0).float()
    return index_contact + thumb_contact + index_contact * thumb_contact
CARD_HORIZON = 128
CARD_GRASPED_HEIGHT = 0.10   # card z must exceed reset_z + this to count as grasped
CARD_SUCCESS_HEIGHT = 0.18   # card z must exceed reset_z + this to count as success


@configclass
class GraspCreditCardSceneCfg(grasp_franka_leap.FrankaLeapGraspSceneCfg):
    grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CARD_SPAWN_POS,
            rot=CARD_SPAWN_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=CARD_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
    )

    shelf_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ShelfCube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SHELF_SPAWN_POS,
            rot=SHELF_SPAWN_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SHELF_USD,
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
            ),
        ),
    )


@configclass
class GraspCreditCardFrankaLeapCfg(grasp_franka_leap.FrankaLeapGraspEnvCfg):
    scene: GraspCreditCardSceneCfg = GraspCreditCardSceneCfg(num_envs=1, env_spacing=2.5)
    distill_include_entity_names: tuple[str, ...] = ("robot", "grasp_object", "shelf_cube")

    def is_grasped(self, env) -> torch.Tensor:
        obj = env.scene["grasp_object"]
        pos = obj.data.root_pos_w - env.scene.env_origins
        return pos[:, 2] >= (CARD_GRASPED_HEIGHT + CARD_SPAWN_POS[2])

    def is_success(self, env) -> torch.Tensor:
        obj = env.scene["grasp_object"]
        pos = obj.data.root_pos_w - env.scene.env_origins
        return pos[:, 2] >= (CARD_SUCCESS_HEIGHT + CARD_SPAWN_POS[2])

    def __post_init__(self):
        super().__post_init__()

        self.object_spawn_defaults = {
            "default_pos": tuple(CARD_SPAWN_POS),
            "default_rot": tuple(CARD_SPAWN_ROT),
            "reset_height": float(CARD_SPAWN_POS[2]),
        }

        self.setup_horizon(horizon=CARD_HORIZON)

        simple_rew = SimpleGraspReward(
            asset_name="robot",
            object_name="grasp_object",
            fingers_name_list=FINGERS_NAME_LIST,
            init_height=CARD_SPAWN_POS[2],
            target_pos=CARD_TARGET_POS,
        )
        simple_rew.GRASPED_Z = 0.06
        simple_rew.LIFTED_Z_LOW = 0.10
        simple_rew.setup_wrist_sensor(self.scene, filter_prim_paths=["{ENV_REGEX_NS}/Table", "{ENV_REGEX_NS}/ShelfCube"])
        simple_rew.setup_finger_entities(self.scene)
        simple_rew.setup_finger_sensors(self.scene, object_prim_name="GraspObject")

        self.rewards.grasped             = RewTerm(func=simple_rew.rew_grasped,             weight=1.0)
        self.rewards.lifted              = RewTerm(func=simple_rew.rew_lifted,              weight=5.0)
        self.rewards.success             = RewTerm(func=simple_rew.rew_success,             weight=10.0)

        self.rewards.wrist               = RewTerm(func=simple_rew.rew_wrist_penalty,       weight=-2.0)
        self.rewards.joint_vel           = RewTerm(func=simple_rew.rew_joint_vel,           weight=-1e-3)
        self.rewards.action_rate         = RewTerm(func=simple_rew.rew_action_rate,         weight=-0.1)
        self.rewards.arm_action_rate     = RewTerm(func=_rew_arm_action_rate,               weight=-5.0)
        self.metrics_spec = {
            "is_success": simple_rew.rew_success,
            "is_lifted": simple_rew.rew_lifted,
            "is_grasped": simple_rew.rew_grasped,
        }

        self.observations.policy.target_object_pose = ObsTerm(func=simple_rew.obs_target_object_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=simple_rew.obs_manipulated_object_pose)
        self.observations.policy.contact_obs = ObsTerm(func=simple_rew.obs_contact)
        self.observations.policy.object_in_tip = ObsTerm(func=simple_rew.obs_object_in_tip)

        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object", "shelf_cube"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[CARD_OBJECT_NUM_POINTS, 512],
            num_downsample_points=2048,
            pcd_crop_region=self.pcd_crop_region,
            pcd_noise=0.02,
        )
        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.get_seg_pc)

        self.events.reset_object = EventTerm(
            func=_reset_shelf_and_card,
            mode="reset",
            params={
                "shelf_pose_range": {
                    "x": (-0.05, 0.10),
                    "y": (-0.03, 0.03),
                    "z": (-0.01, 0.03),
                },
            },
        )

        self.events.capture_reset_height = EventTerm(
            func=simple_rew.capture_reset_height,
            mode="reset",
            params={},
        )

        self.events.randomize_card_mass = EventTerm(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="reset",
            min_step_count_between_reset=CARD_HORIZON * 4,
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "mass_distribution_params": (0.5, 1.0),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.events.randomize_card_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="reset",
            min_step_count_between_reset=CARD_HORIZON * 4,
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "static_friction_range": (0.3, 1.0),
                "dynamic_friction_range": (0.3, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )

        self.events.set_shelf_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="reset",
            min_step_count_between_reset=CARD_HORIZON * 4,
            params={
                "asset_cfg": SceneEntityCfg("shelf_cube"),
                "static_friction_range": (0.3, 1.0),
                "dynamic_friction_range": (0.3, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )


@configclass
class GraspCreditCardFrankaLeapJointAbsCfg(GraspCreditCardFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class GraspCreditCardFrankaLeapIkRelCfg(GraspCreditCardFrankaLeapCfg):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class GraspCreditCardFrankaLeapIkAbsCfg(GraspCreditCardFrankaLeapCfg):
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
