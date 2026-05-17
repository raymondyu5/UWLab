# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Soccer ball push task for Franka-LEAP.
# Goal: push the soccer ball (stress ball) to a goal region (TBD).
# For now, no goal object — just the ball on the table with regularization rewards.

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
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

from ....mdp import (
    CachedSamplePC,
    reset_object_pose,
    reset_table_block,
    log_object_mass,
    log_object_scales,
)
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET, ARM_NUM_POINTS, HAND_NUM_POINTS

SOCCER_BALL_USD = "/workspace/uwlab/assets/soccer_ball/object.usda"
SOCCER_BALL_NUM_POINTS = 128

SOCCER_BALL_SPAWN_POS = (0.53, -0.125, 0.035)
SOCCER_BALL_SPAWN_ROT = (1.0, 0.0, 0.0, 0.0)

SOCCER_GOAL_USD = "/workspace/uwlab/assets/soccer_goal/object.usda"
SOCCER_GOAL_SPAWN_POS = (0.53, 0.15, 0.0)
SOCCER_GOAL_SPAWN_ROT = (1.0, 0.0, 0.0, 0.0)

SOCCER_PUSH_HORIZON = 160

GOAL_BOX_X = (-0.07, 0.07)
GOAL_BOX_Y = (-0.03, 0.03)
GOAL_BOX_Z = (0.01, 0.08)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def obs_manipulated_object_pose(env) -> torch.Tensor:
    ball_pose = env.scene["grasp_object"]._data.root_state_w[:, :7].clone()
    ball_pose[:, :3] -= env.scene.env_origins
    return ball_pose


def obs_goal_pose(env) -> torch.Tensor:
    goal_pose = env.scene["goal"]._data.root_state_w[:, :7].clone()
    goal_pose[:, :3] -= env.scene.env_origins
    return goal_pose


def obs_ball_velocity(env) -> torch.Tensor:
    vel = env.scene["grasp_object"].data.root_state_w[:, 7:13].clone()
    return vel


# ---------------------------------------------------------------------------
# Reward / termination helpers
# ---------------------------------------------------------------------------

def nan_to_num(val) -> torch.Tensor:
    return torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)


def ball_fell_off_table(env, object_name: str = "grasp_object", min_z: float = -0.05) -> torch.Tensor:
    obj = env.scene[object_name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return obj_z < min_z


def _ball_goal_pos(env):
    ball_pos = env.scene["grasp_object"].data.root_pos_w - env.scene.env_origins
    goal_pos = env.scene["goal"].data.root_pos_w - env.scene.env_origins
    return ball_pos, goal_pos


def _ball_in_goal_box(env) -> torch.Tensor:
    ball_pos, goal_pos = _ball_goal_pos(env)
    dx = ball_pos[:, 0] - goal_pos[:, 0]
    dy = ball_pos[:, 1] - goal_pos[:, 1]
    dz = ball_pos[:, 2] - goal_pos[:, 2]
    x_ok = (dx >= GOAL_BOX_X[0]) & (dx <= GOAL_BOX_X[1])
    y_ok = (dy >= GOAL_BOX_Y[0]) & (dy <= GOAL_BOX_Y[1])
    z_ok = (dz >= GOAL_BOX_Z[0]) & (dz <= GOAL_BOX_Z[1])
    return x_ok & y_ok & z_ok


def soccer_ball_in_goal(env) -> torch.Tensor:
    return nan_to_num(_ball_in_goal_box(env).float())


def soccer_is_success(env) -> torch.Tensor:
    return _ball_in_goal_box(env).float()


def soccer_is_grasped(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def soccer_joint_vel_l2(env, asset_name: str = "robot") -> torch.Tensor:
    robot = env.scene[asset_name]
    return nan_to_num(torch.sum(robot.data.joint_vel ** 2, dim=1))


def soccer_action_rate_l2(env) -> torch.Tensor:
    return nan_to_num(
        torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)
    )


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------

@configclass
class SoccerPushSceneCfg(grasp_franka_leap.FrankaLeapGraspSceneCfg):
    grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SOCCER_BALL_SPAWN_POS,
            rot=SOCCER_BALL_SPAWN_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCCER_BALL_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
    )

    goal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SOCCER_GOAL_SPAWN_POS,
            rot=SOCCER_GOAL_SPAWN_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCCER_GOAL_USD,
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
            ),
        ),
    )

    table_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableBlock",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(SOCCER_BALL_SPAWN_POS[0], SOCCER_BALL_SPAWN_POS[1], 0.0),
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


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------

@configclass
class SoccerPushFrankaLeapCfg(grasp_franka_leap.FrankaLeapGraspEnvCfg):
    scene: SoccerPushSceneCfg = SoccerPushSceneCfg(num_envs=1, env_spacing=2.5)
    table_z_range: tuple = (0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()

        self.object_spawn_defaults = {
            "default_pos": list(SOCCER_BALL_SPAWN_POS),
            "default_rot": list(SOCCER_BALL_SPAWN_ROT),
            "reset_height": SOCCER_BALL_SPAWN_POS[2],
        }

        self.setup_horizon(horizon=SOCCER_PUSH_HORIZON)

        # Rewards
        self.rewards.ball_in_goal = RewTerm(func=soccer_ball_in_goal, weight=10.0)
        self.rewards.joint_vel = RewTerm(
            func=soccer_joint_vel_l2,
            weight=-1.0e-3,
            params={"asset_name": "robot"},
        )
        self.rewards.action_rate = RewTerm(func=soccer_action_rate_l2, weight=-0.25)

        self.metrics_spec = {
            "is_success": soccer_is_success,
            "is_grasped": soccer_is_grasped,
        }

        # Observations
        self.observations.policy.manipulated_object_pose = ObsTerm(func=obs_manipulated_object_pose)
        self.observations.policy.goal_pose = ObsTerm(func=obs_goal_pose)
        self.observations.policy.ball_velocity = ObsTerm(func=obs_ball_velocity)

        self.distill_include_entity_names = ("robot", "grasp_object", "goal")

        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object", "goal"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[SOCCER_BALL_NUM_POINTS, 128],
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

        self.events.reset_object = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "default_pos": SOCCER_BALL_SPAWN_POS,
                "default_rot_quat": SOCCER_BALL_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.15, 0.15),
                    "y": (-0.15, 0.1),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": SOCCER_BALL_SPAWN_POS[2],
                "table_block_name": "table_block",
            },
        )

        self.events.randomize_object_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="reset",
            min_step_count_between_reset=800,
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "static_friction_range": (0.5, 1.0),
                "dynamic_friction_range": (0.5, 1.0),
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

        self.events.log_object_mass = EventTerm(
            func=log_object_mass,
            mode="reset",
            min_step_count_between_reset=800,
            params={"asset_cfg": SceneEntityCfg("grasp_object")},
        )

        self.events.randomize_object_scale = EventTerm(
            func=isaac_mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "scale_range": (0.8, 1.2),
            },
        )

        self.events.log_object_scales = EventTerm(
            func=log_object_scales,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("grasp_object")},
        )

        # Terminations
        self.terminations.ball_fell_off = DoneTerm(
            func=ball_fell_off_table,
            params={"object_name": "grasp_object", "min_z": -0.05},
            time_out=False,
        )


# ---------------------------------------------------------------------------
# Action-space variants
# ---------------------------------------------------------------------------

@configclass
class SoccerPushFrankaLeapJointAbsCfg(SoccerPushFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class SoccerPushFrankaLeapIkRelCfg(SoccerPushFrankaLeapCfg):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class SoccerPushFrankaLeapIkAbsCfg(SoccerPushFrankaLeapCfg):
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
