# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Cup pour task for Franka-LEAP.
# Robot holds pink cup with a ping pong ball inside and pours it into a big cup.

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from ....mdp import CachedSamplePC, reset_object_pose, reset_table_block, bottle_too_far, cup_toppled
from .rewards.cup_pour_rewards import ball_in_cup, ball_proximity_to_cup, ball_too_far, reset_ball_in_cup
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET, ARM_NUM_POINTS, HAND_NUM_POINTS
from .shared_params import FINGERS_NAME_LIST
SMALL_CUP_USD = "/workspace/uwlab/assets/pink_cup/small_cup.usda"

# ── Asset paths ──
PING_PONG_BALL_USD = "/workspace/uwlab/assets/ping_pong_ball/object.usda"
BIG_CUP_USD = "/workspace/uwlab/assets/big_cup/object.usda"

# ── Spawn positions (placeholder — tune with PCD overlay) ──
PINK_CUP_POUR_SPAWN_POS = (0.54, -0.115, 0.07)
PINK_CUP_POUR_SPAWN_ROT = (1.0, 0.0, 0.0, 0.0)

BIG_CUP_SPAWN_POS = (0.55, 0.12, 0.07)
BIG_CUP_SPAWN_ROT = (1.0, 0.0, 0.0, 0.0)

BALL_SPAWN_Z_OFFSET = 0.30

# ── Ball success region (placeholder — tune with zero_agent visualization) ──
BIG_CUP_INNER_RADIUS = 0.05
BIG_CUP_HEIGHT = 0.12

# ── Task params ──
CUP_POUR_HORIZON = 128

CUP_GRASPED_Z_THRESH = 0.04


def cup_grasped(env) -> torch.Tensor:
    pos = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    return (pos[:, 2] - PINK_CUP_POUR_SPAWN_POS[2] >= CUP_GRASPED_Z_THRESH).float()

PINK_CUP_NUM_POINTS = 128
BIG_CUP_NUM_POINTS = 128
BALL_NUM_POINTS = 32


def obs_big_cup_pose(env) -> torch.Tensor:
    cup_pose = env.scene["big_cup"]._data.root_state_w[:, :7].clone()
    cup_pose[:, :3] -= env.scene.env_origins
    return cup_pose[:, :3]


def obs_ball_pose(env) -> torch.Tensor:
    ball_pose = env.scene["ping_pong_ball"]._data.root_state_w[:, :7].clone()
    ball_pose[:, :3] -= env.scene.env_origins
    return ball_pose[:, :3]


def obs_manipulated_object_pose(env) -> torch.Tensor:
    obj_pose = env.scene["grasp_object"]._data.root_state_w[:, :7].clone()
    obj_pose[:, :3] -= env.scene.env_origins
    return obj_pose


def obs_ball_velocity(env) -> torch.Tensor:
    ball_vel = env.scene["ping_pong_ball"]._data.root_state_w[:, 7:13]
    return ball_vel


@configclass
class CupPourSceneCfg(grasp_franka_leap.FrankaLeapGraspSceneCfg):
    grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=PINK_CUP_POUR_SPAWN_POS,
            rot=PINK_CUP_POUR_SPAWN_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SMALL_CUP_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
    )

    big_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BigCup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BIG_CUP_SPAWN_POS,
            rot=BIG_CUP_SPAWN_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=BIG_CUP_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
    )

    ping_pong_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PingPongBall",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PINK_CUP_POUR_SPAWN_POS[0], PINK_CUP_POUR_SPAWN_POS[1],
                 PINK_CUP_POUR_SPAWN_POS[2] + BALL_SPAWN_Z_OFFSET),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=PING_PONG_BALL_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                linear_damping=0.5,
                angular_damping=2.0,
            ),
        ),
    )

    table_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableBlock",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PINK_CUP_POUR_SPAWN_POS[0], 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
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
class CupPourFrankaLeapCfg(grasp_franka_leap.FrankaLeapGraspEnvCfg):
    scene: CupPourSceneCfg = CupPourSceneCfg(num_envs=1, env_spacing=2.5)
    table_z_range: tuple = (0.0, 0.0)
    distill_include_entity_names: tuple[str, ...] = ("robot", "grasp_object", "big_cup", "ping_pong_ball")

    def is_success(self, env) -> torch.Tensor:
        return ball_in_cup(
            env,
            ball_name="ping_pong_ball",
            cup_name="big_cup",
            cup_inner_radius=BIG_CUP_INNER_RADIUS,
            cup_height=BIG_CUP_HEIGHT,
        )

    def __post_init__(self):
        super().__post_init__()

        self.object_spawn_defaults = {
            "default_pos": list(PINK_CUP_POUR_SPAWN_POS),
            "default_rot": list(PINK_CUP_POUR_SPAWN_ROT),
            "reset_height": PINK_CUP_POUR_SPAWN_POS[2],
        }

        self.setup_horizon(horizon=CUP_POUR_HORIZON)

        # ── Rewards ──
        self.rewards.ball_in_cup = RewTerm(
            func=ball_in_cup,
            weight=10.0,
            params={
                "ball_name": "ping_pong_ball",
                "cup_name": "big_cup",
                "cup_inner_radius": BIG_CUP_INNER_RADIUS,
                "cup_height": BIG_CUP_HEIGHT,
            },
        )
        self.rewards.grasped = RewTerm(func=cup_grasped, weight=1.0)
        self.rewards.joint_vel = RewTerm(
            func=lambda env: torch.sum(env.scene["robot"].data.joint_vel ** 2, dim=1),
            weight=-1e-3,
        )
        self.rewards.action_rate = RewTerm(
            func=lambda env: torch.sum(
                (env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1
            ),
            weight=-0.05,
        )

        self.metrics_spec = {
            "is_success": lambda env: ball_in_cup(
                env, "ping_pong_ball", "big_cup", BIG_CUP_INNER_RADIUS, BIG_CUP_HEIGHT,
            ),
            "is_grasped": cup_grasped,
        }

        # ── Observations ──
        self.observations.policy.big_cup_pose = ObsTerm(func=obs_big_cup_pose)
        self.observations.policy.ball_pose = ObsTerm(func=obs_ball_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=obs_manipulated_object_pose)
        self.observations.policy.ball_velocity = ObsTerm(func=obs_ball_velocity)

        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object", "big_cup", "ping_pong_ball"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[PINK_CUP_NUM_POINTS, BIG_CUP_NUM_POINTS, BALL_NUM_POINTS],
            num_downsample_points=2048,
            pcd_crop_region=self.pcd_crop_region,
            pcd_noise=0.0,
        )
        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.get_seg_pc)

        # ── Reset events ──
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
                "default_pos": PINK_CUP_POUR_SPAWN_POS,
                "default_rot_quat": PINK_CUP_POUR_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.08, 0.08),
                    "y": (-0.04, 0.04),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": PINK_CUP_POUR_SPAWN_POS[2],
                "table_block_name": "table_block",
            },
        )

        self.events.reset_big_cup = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("big_cup"),
                "default_pos": BIG_CUP_SPAWN_POS,
                "default_rot_quat": BIG_CUP_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.08, 0.08),
                    "y": (-0.08, 0.08),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": BIG_CUP_SPAWN_POS[2],
                "table_block_name": "table_block",
            },
        )

        # Ball reset AFTER pink cup so it lands inside
        self.events.reset_ball = EventTerm(
            func=reset_ball_in_cup,
            mode="reset",
            params={
                "ball_name": "ping_pong_ball",
                "cup_name": "grasp_object",
                "z_offset": BALL_SPAWN_Z_OFFSET,
            },
        )

        # Ball bounciness
        self.events.randomize_ball_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="reset",
            min_step_count_between_reset=800,
            params={
                "asset_cfg": SceneEntityCfg("ping_pong_ball"),
                "static_friction_range": (0.2, 0.4),
                "dynamic_friction_range": (0.2, 0.4),
                "restitution_range": (0.0, 0.1),
                "num_buckets": 64,
            },
        )

        self.events.randomize_object_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="reset",
            min_step_count_between_reset=800,
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "static_friction_range": (0.3, 0.7),
                "dynamic_friction_range": (0.3, 0.7),
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
                "mass_distribution_params": (0.8, 1.5),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.events.randomize_object_scale = EventTerm(
            func=isaac_mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "scale_range": (0.8, 1.1),
            },
        )

        # ── Terminations ──
        self.terminations.cup_too_far = DoneTerm(
            func=bottle_too_far,
            params={"object_name": "grasp_object", "max_xy_dist": 1.0},
            time_out=False,
        )
        self.terminations.ball_too_far = DoneTerm(
            func=ball_too_far,
            params={"ball_name": "ping_pong_ball", "max_xy_dist": 1.5},
            time_out=False,
        )
        self.terminations.big_cup_toppled = DoneTerm(
            func=cup_toppled,
            params={
                "cup_name": "big_cup",
                "spawn_quat": BIG_CUP_SPAWN_ROT,
                "angle_thresh_rad": 0.524,
            },
            time_out=False,
        )


@configclass
class CupPourFrankaLeapJointAbsCfg(CupPourFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class CupPourFrankaLeapIkRelCfg(CupPourFrankaLeapCfg):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
