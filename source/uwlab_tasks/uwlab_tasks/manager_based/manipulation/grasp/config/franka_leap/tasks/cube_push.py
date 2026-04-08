# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Cube push task for Franka-LEAP.
# Goal: push the cube across the table until it contacts the poptart box.
# Poptart spawns like the pink cup in bottle_pour (target, right side).
# Cube spawns like the bottle in bottle_pour (manipulated object, front-left).

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

from .rewards.push_rewards import (
    is_success,
    is_cube_near_poptart,
    is_cube_touching_poptart,
    push_near,
    push_success,
    push_poptart_velocity,
    push_poptart_topple,
    push_joint_vel_l2,
    push_action_rate_l2,
)
from ....mdp import (
    CachedSamplePC,
    reset_object_pose,
    reset_table_block,
    bottle_too_far,
    cup_toppled,
    log_object_mass,
    log_object_scales,
)
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET, ARM_NUM_POINTS, HAND_NUM_POINTS
from .cube import GraspCubeSceneCfg, CUBE_SPAWN_ROT, CUBE_OBJECT_NUM_POINTS
from .bottle_singulate import BOX_USD, BOX_RESET_HEIGHT, BOX_OBJECT_NUM_POINTS

# Spawn positions:
# Cube: front-left (same XY as bottle in bottle_pour, cube table height)
# Poptart: right side (same XY as pink cup in bottle_pour, poptart standing height)
CUBE_PUSH_SPAWN_POS = (0.53, -0.105, 0.07)
POPTART_PUSH_POS = (0.53, 0.23, BOX_RESET_HEIGHT)  # BOX_RESET_HEIGHT = 0.0675
# 90deg around X: corrects the poptart mesh from Y-up to Z-up so it stands upright
POPTART_PUSH_ROT = (0.707, 0.707, 0.0, 0.0)

PUSH_HORIZON = 160


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def obs_poptart_pose(env) -> torch.Tensor:
    """Poptart position (3D) in env-relative frame."""
    poptart_pose = env.scene["box"]._data.root_state_w[:, :7].clone()
    poptart_pose[:, :3] -= env.scene.env_origins
    return poptart_pose[:, :3]


def obs_manipulated_object_pose(env) -> torch.Tensor:
    """Cube pose (7D) in env-relative frame."""
    cube_pose = env.scene["grasp_object"]._data.root_state_w[:, :7].clone()
    cube_pose[:, :3] -= env.scene.env_origins
    return cube_pose


def obs_target_object_pose(env) -> torch.Tensor:
    """Target cube pose: poptart XY at cube spawn height, cube spawn orientation."""
    poptart_pose = env.scene["box"]._data.root_state_w[:, :7].clone()
    poptart_pose[:, :3] -= env.scene.env_origins
    target_quat = env.scene["grasp_object"].data.default_root_state[:, 3:7].clone()
    target_pose = torch.zeros(env.num_envs, 7, device=env.device, dtype=torch.float32)
    target_pose[:, :2] = poptart_pose[:, :2]
    target_pose[:, 2] = CUBE_PUSH_SPAWN_POS[2]
    target_pose[:, 3:7] = target_quat
    return target_pose


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------

@configclass
class PushCubeToPoptartSceneCfg(GraspCubeSceneCfg):
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=POPTART_PUSH_POS,
            rot=POPTART_PUSH_ROT,
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
class PushCubeToPoptartFrankaLeapCfg(grasp_franka_leap.FrankaLeapGraspEnvCfg):
    scene: PushCubeToPoptartSceneCfg = PushCubeToPoptartSceneCfg(num_envs=1, env_spacing=2.5)
    table_z_range: tuple = (0.0, 0.0)
    distill_include_entity_names: tuple[str, ...] = ("robot", "grasp_object", "box")

    def __post_init__(self):
        super().__post_init__()

        self.object_spawn_defaults = {
            "default_pos": list(CUBE_PUSH_SPAWN_POS),
            "default_rot": list(CUBE_SPAWN_ROT),
            "reset_height": CUBE_PUSH_SPAWN_POS[2],
        }

        self.setup_horizon(horizon=PUSH_HORIZON)

        # Rewards
        self.rewards.push_near = RewTerm(func=push_near, weight=5.0)
        self.rewards.push_success = RewTerm(func=push_success, weight=10.0)
        self.rewards.poptart_velocity = RewTerm(func=push_poptart_velocity, weight=-2.0)
        self.rewards.poptart_topple = RewTerm(func=push_poptart_topple, weight=-10.0)
        self.rewards.joint_vel = RewTerm(
            func=push_joint_vel_l2,
            weight=-1.0e-3,
            params={"asset_name": "robot"},
        )
        self.rewards.action_rate = RewTerm(func=push_action_rate_l2, weight=-0.25)

        # Metrics
        self.metrics_spec = {
            "is_success": is_success,
            "is_cube_near_poptart": is_cube_near_poptart,
            "is_cube_touching_poptart": is_cube_touching_poptart,
        }

        # Observations
        self.observations.policy.poptart_pose = ObsTerm(func=obs_poptart_pose)
        self.observations.policy.target_object_pose = ObsTerm(func=obs_target_object_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=obs_manipulated_object_pose)

        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object", "box"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[CUBE_OBJECT_NUM_POINTS, BOX_OBJECT_NUM_POINTS],
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

        # Reset cube
        self.events.reset_object = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "default_pos": CUBE_PUSH_SPAWN_POS,
                "default_rot_quat": CUBE_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.11, 0.11),
                    "y": (-0.095, 0.095),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": CUBE_PUSH_SPAWN_POS[2],
                "table_block_name": "table_block",
            },
        )

        # Reset poptart
        self.events.reset_box_object = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "default_pos": POPTART_PUSH_POS,
                "default_rot_quat": POPTART_PUSH_ROT,
                "pose_range": {
                    "x": (-0.12, 0.12),
                    "y": (-0.06, 0.06),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": POPTART_PUSH_POS[2],
                "table_block_name": "table_block",
            },
        )

        self.events.log_object_scales = EventTerm(
            func=log_object_scales,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("grasp_object")},
        )

        self.events.log_box_scales = EventTerm(
            func=log_object_scales,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("box")},
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
                "scale_range": (1.0, 1.3),
            },
        )

        self.events.randomize_box_scale = EventTerm(
            func=isaac_mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "scale_range": (0.9, 1.1),
            },
        )

        # Terminations
        self.terminations.cube_too_far = DoneTerm(
            func=bottle_too_far,
            params={
                "object_name": "grasp_object",
                "max_xy_dist": 1.0,
            },
            time_out=False,
        )
        self.terminations.poptart_toppled = DoneTerm(
            func=cup_toppled,
            params={
                "cup_name": "box",
                "spawn_quat": POPTART_PUSH_ROT,
                "angle_thresh_rad": 0.524,
            },
            time_out=False,
        )


# ---------------------------------------------------------------------------
# Action-space variants
# ---------------------------------------------------------------------------

@configclass
class PushCubeToPoptartFrankaLeapJointAbsCfg(PushCubeToPoptartFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class PushCubeToPoptartFrankaLeapIkRelCfg(PushCubeToPoptartFrankaLeapCfg):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class PushCubeToPoptartFrankaLeapIkAbsCfg(PushCubeToPoptartFrankaLeapCfg):
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
