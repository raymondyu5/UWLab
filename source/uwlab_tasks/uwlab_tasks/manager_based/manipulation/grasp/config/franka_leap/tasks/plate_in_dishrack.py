# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.utils.math as math_utils
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

from ....mdp import CachedSamplePC, reset_object_pose, bottle_too_far
from .rewards.place_rewards import PlaceReward
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET, ARM_NUM_POINTS, HAND_NUM_POINTS
from .shared_params import FINGERS_NAME_LIST



PLATE_HORIZON = 160
PLATE_OBJECT_NUM_POINTS = 512
RACK_OBJECT_NUM_POINTS = 256

BOWL_USD = "/workspace/uwlab/assets/bowl_new/object.usda"
PLATE_RACK_USD = "/workspace/uwlab/assets/wooden_dishrack_sawed/object.usda"

# Real bowl and wooden dishrack assets — scale=(1,1,1), real-world dimensions.
# TODO: tune spawn positions after visual inspection with zero_agent.py
PLATE_SPAWN_POS = (0.58, -0.12, 0.05)
PLATE_SPAWN_ROT = (1.0, 0.0, 0.0, 0.0)

RACK_SPAWN_POS = (0.57, 0.14, 0.0)
RACK_SPAWN_ROT = (0.0, 0.0, 0.0, 1.0)  # 180 deg around Z

# Target slot pose derived from Blender (rack-local frame, Y-up->Z-up converted).
# slot_offset is auto-computed as SLOT_TARGET_POS - RACK_SPAWN_POS.
SLOT_TARGET_POS = (0.6047, 0.1259, 0.1283)
# Bowl rotation at slot in rack-local sim frame (WXYZ).
# Blender WXYZ=(0.555,0,0,-0.832) → convert Y-up→Z-up → q_bowl_world=(0.555,0,0.832,0).
# rack-local = conj(q_rack) * q_bowl_world = conj(0,0,0,1) * (0.555,0,0.832,0) = (0,0.832,0,-0.555)
# Verify: q_rack * q_local = (0,0,0,1)*(0,0.832,0,-0.555) = (0.555,0,0.832,0) ✓
SLOT_TARGET_ROT_LOCAL = (0.0, 0.832, 0.0, -0.555)

# Success bounding box — bowl must be inside this region (env-local, relative to rack position).
# Tune visually using zero_agent.py --success_box --plate_test.
SUCCESS_BOX_X_LO_OFFSET = 0.00  # x_min = rack_x + this
SUCCESS_BOX_X_HI_OFFSET = 0.06  # x_max = rack_x + this
SUCCESS_BOX_Y_FRONT = 0.07   # how far in front of rack center (toward bowl) the box starts
SUCCESS_BOX_Y_REAR  = 0.02   # how far past rack center (first half + small buffer)
SUCCESS_BOX_Y_OFFSET = 0.03  # shift box center in Y (tune if rack origin isn't centered)
SUCCESS_BOX_Z_MIN   = 0.05   # bowl must be above table
SUCCESS_BOX_Z_MAX   = 0.14   # bowl height ceiling


@configclass
class PlateInDishRackSceneCfg(grasp_franka_leap.FrankaLeapGraspSceneCfg):
    grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        init_state=RigidObjectCfg.InitialStateCfg(pos=PLATE_SPAWN_POS, rot=PLATE_SPAWN_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOWL_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=1.0,
                angular_damping=2.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
        ),
    )

    plate_rack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlateRack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=RACK_SPAWN_POS, rot=RACK_SPAWN_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=PLATE_RACK_USD,
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )


@configclass
class PlateInDishRackFrankaLeapCfg(grasp_franka_leap.FrankaLeapGraspEnvCfg):
    scene: PlateInDishRackSceneCfg = PlateInDishRackSceneCfg(num_envs=1, env_spacing=2.5)

    def is_success(self, env) -> torch.Tensor:
        return self._place_rew.is_done(env).float()

    def __post_init__(self):
        # Lower z_min from 0.035 to -0.01 so bowl on table (z~0.02) is not cropped out
        self.pcd_crop_region = [-0.10, -0.50, 0.005, 0.85, 0.50, 0.70]
        super().__post_init__()

        init = self.scene.grasp_object.init_state
        self.object_spawn_defaults = {
            "default_pos": tuple(init.pos),
            "default_rot": tuple(init.rot),
            "reset_height": float(init.pos[2]),
        }

        self.setup_horizon(horizon=PLATE_HORIZON)

        _slot_offset = (
            SLOT_TARGET_POS[0] - RACK_SPAWN_POS[0],
            SLOT_TARGET_POS[1] - RACK_SPAWN_POS[1],
            SLOT_TARGET_POS[2] - RACK_SPAWN_POS[2],
        )
        place_rew = PlaceReward(
            asset_name="robot",
            object_name="grasp_object",
            fingers_name_list=FINGERS_NAME_LIST,
            init_height=PLATE_SPAWN_POS[2],
            target_pos=SLOT_TARGET_POS,
            rack_name="plate_rack",
            slot_offset=_slot_offset,
            slot_rot_local=SLOT_TARGET_ROT_LOCAL,
            success_box_x_lo_offset=SUCCESS_BOX_X_LO_OFFSET,
            success_box_x_hi_offset=SUCCESS_BOX_X_HI_OFFSET,
            success_box_y_front=SUCCESS_BOX_Y_FRONT,
            success_box_y_rear=SUCCESS_BOX_Y_REAR,
            success_box_y_offset=SUCCESS_BOX_Y_OFFSET,
            success_box_z_min=SUCCESS_BOX_Z_MIN,
            success_box_z_max=SUCCESS_BOX_Z_MAX,
            home_joints=ARM_RESET + HAND_RESET,
        )
        place_rew.setup_wrist_sensor(self.scene)
        place_rew.setup_finger_entities(self.scene)
        place_rew.setup_finger_sensors(self.scene, object_prim_name="Bowl")
        self._place_rew = place_rew

        self.rewards.success      = RewTerm(func=place_rew.rew_success,      weight=50.0)
        self.rewards.return_home  = RewTerm(func=place_rew.rew_return_home,  weight=2.0)
        self.rewards.joint_vel    = RewTerm(func=place_rew.rew_joint_vel,    weight=-1e-3)
        self.rewards.action_rate  = RewTerm(func=place_rew.rew_action_rate,  weight=-5e-3)

        self.metrics_spec = {
            "is_success": place_rew.rew_success,
            "is_grasped": place_rew.rew_grasped,
            "is_near_slot": place_rew.metric_near_slot,
            "is_placed_pos": place_rew.metric_placed_pos,
            "is_placed_orient": place_rew.metric_placed_orient,
        }

        self.events.capture_reset_height = EventTerm(
            func=place_rew.capture_reset_height, mode="reset", params={}
        )
        self.events.reset_placed_flag = EventTerm(
            func=place_rew.reset_placed_flag, mode="reset", params={}
        )

        self.observations.policy.target_object_pose      = ObsTerm(func=place_rew.obs_target_object_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=place_rew.obs_manipulated_object_pose)
        self.observations.policy.contact_obs             = ObsTerm(func=place_rew.obs_contact)
        self.observations.policy.object_in_tip           = ObsTerm(func=place_rew.obs_object_in_tip)

        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object", "plate_rack"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[PLATE_OBJECT_NUM_POINTS, RACK_OBJECT_NUM_POINTS],
            num_downsample_points=2048,
            pcd_crop_region=self.pcd_crop_region,
            pcd_noise=0.02,
        )
        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.get_seg_pc)

        self.events.reset_object = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "default_pos": PLATE_SPAWN_POS,
                "default_rot_quat": PLATE_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.055, 0.055),
                    "y": (-0.048, 0.05),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": PLATE_SPAWN_POS[2],
                "table_block_name": None,
            },
        )

        self.events.randomize_plate_material = EventTerm(
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

        self.events.randomize_plate_mass = EventTerm(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="reset",
            min_step_count_between_reset=800,
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "mass_distribution_params": (0.2, 0.6),
                "operation": "abs",
                "distribution": "uniform",
            },
        )

        self.events.randomize_plate_scale = EventTerm(
            func=isaac_mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "scale_range": (0.9, 1.1),
            },
        )

        self.events.reset_rack = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("plate_rack"),
                "default_pos": RACK_SPAWN_POS,
                "default_rot_quat": RACK_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.02, 0.02),
                    "y": (-0.02, 0.02),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": RACK_SPAWN_POS[2],
                "table_block_name": None,
            },
        )

        self.terminations.plate_too_far = DoneTerm(
            func=bottle_too_far,
            params={"object_name": "grasp_object", "max_xy_dist": 1.0},
            time_out=False,
        )
        self.terminations.placed_and_home = DoneTerm(
            func=place_rew.is_done,
            time_out=False,
        )


# ---------------------------------------------------------------------------
# Module-level obs/rew functions for Hydra-compatible PPO config.
# ---------------------------------------------------------------------------

_PLATE_SLOT_OFFSET = (
    SLOT_TARGET_POS[0] - RACK_SPAWN_POS[0],
    SLOT_TARGET_POS[1] - RACK_SPAWN_POS[1],
    SLOT_TARGET_POS[2] - RACK_SPAWN_POS[2],
)

_FINGER_CONTACT_NAMES = [
    "palm_lower_contact", "fingertip_contact", "thumb_fingertip_contact",
    "fingertip_2_contact", "fingertip_3_contact",
]
_FINGER_BODY_NAMES = ["palm_lower", "fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]


def _plate_obs_object_pose(env) -> torch.Tensor:
    state = env.scene["grasp_object"]._data.root_state_w[:, :7].clone()
    state[:, :3] -= env.scene.env_origins
    return state


def _plate_obs_target_pose(env) -> torch.Tensor:
    """Dynamic rack slot target pose (7D): tracks rack position at reset."""
    rack_pos = env.scene["plate_rack"].data.root_pos_w - env.scene.env_origins
    offset = torch.tensor([list(_PLATE_SLOT_OFFSET)], device=env.device, dtype=torch.float32)
    target_pos = rack_pos + offset
    rack_quat = env.scene["plate_rack"].data.root_state_w[:, 3:7]
    slot_rot_local = torch.tensor(
        [list(SLOT_TARGET_ROT_LOCAL)], device=env.device, dtype=torch.float32
    ).expand(env.num_envs, -1)
    target_quat = math_utils.quat_mul(rack_quat, slot_rot_local)
    return torch.cat([target_pos, target_quat], dim=1)


def _plate_obs_contact(env) -> torch.Tensor:
    """Binary finger contact with object (5D)."""
    parts = []
    for name in _FINGER_CONTACT_NAMES:
        force = torch.linalg.norm(
            env.scene[name]._data.force_matrix_w.reshape(env.num_envs, 3), dim=1)
        parts.append((force > 4.0).int().unsqueeze(1))
    return torch.cat(parts, dim=1).float()


def _plate_obs_object_in_tip(env) -> torch.Tensor:
    """Object-to-fingertip displacement vectors, flattened (15D)."""
    obj_pos = env.scene["grasp_object"]._data.root_state_w[:, :3]
    parts = [obj_pos - env.scene[name].data.root_pos_w for name in _FINGER_BODY_NAMES]
    return torch.cat(parts, dim=1)


def _plate_rew_success(env) -> torch.Tensor:
    """Simplified success: bowl inside dishrack bounding box (position only)."""
    rack_pos = env.scene["plate_rack"].data.root_pos_w - env.scene.env_origins
    bowl = env.scene["grasp_object"]._data.root_state_w[:, :3] - env.scene.env_origins
    cy = rack_pos[:, 1] + SUCCESS_BOX_Y_OFFSET
    x_ok = (bowl[:, 0] >= rack_pos[:, 0] + SUCCESS_BOX_X_LO_OFFSET) & \
            (bowl[:, 0] <= rack_pos[:, 0] + SUCCESS_BOX_X_HI_OFFSET)
    y_ok = (bowl[:, 1] >= cy - SUCCESS_BOX_Y_FRONT) & \
            (bowl[:, 1] <= cy + SUCCESS_BOX_Y_REAR)
    z_ok = (bowl[:, 2] >= SUCCESS_BOX_Z_MIN) & (bowl[:, 2] <= SUCCESS_BOX_Z_MAX)
    return (x_ok & y_ok & z_ok).float()


def _plate_rew_joint_vel(env) -> torch.Tensor:
    return torch.sum(env.scene["robot"].data.joint_vel ** 2, dim=1)


def _plate_rew_action_rate(env) -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


@configclass
class PlateInDishRackFrankaLeapJointAbsCfg(PlateInDishRackFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class PlateInDishRackFrankaLeapJointAbsStateCfg(PlateInDishRackFrankaLeapJointAbsCfg):
    """PPO-friendly variant: module-level obs/rew functions only (Hydra-safe), no seg_pc.

    Observation space: arm_joint_pos (7) + hand_joint_pos (16) + object_pose (7) = 30D flat.
    Matches the BC training obs keys so BC checkpoints transfer directly.
    """
    run_mode: str = "rl_mode"

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.manipulated_object_pose = ObsTerm(func=_plate_obs_object_pose)
        self.observations.policy.target_object_pose = ObsTerm(func=_plate_obs_target_pose)
        self.observations.policy.contact_obs = ObsTerm(func=_plate_obs_contact)
        self.observations.policy.object_in_tip = ObsTerm(func=_plate_obs_object_in_tip)
        self.observations.policy.joint_pos = None
        self.observations.policy.ee_pose = None
        self.observations.policy.seg_pc = None
        self.observations.policy.concatenate_terms = True
        self.rewards.success = RewTerm(func=_plate_rew_success, weight=50.0)
        self.rewards.return_home = None
        self.rewards.joint_vel = RewTerm(func=_plate_rew_joint_vel, weight=-1e-3)
        self.rewards.action_rate = RewTerm(func=_plate_rew_action_rate, weight=-5e-3)
        self.events.capture_reset_height = None
        self.metrics_spec = {
            "is_success": _plate_rew_success,
        }


@configclass
class PlateInDishRackFrankaLeapJointAbsStateCollectCfg(PlateInDishRackFrankaLeapJointAbsStateCfg):
    """StateCfg with seg_pc re-enabled and dict obs for RL rollout collection."""

    def __post_init__(self):
        super().__post_init__()
        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object", "plate_rack"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[PLATE_OBJECT_NUM_POINTS, RACK_OBJECT_NUM_POINTS],
            num_downsample_points=2048,
            pcd_crop_region=self.pcd_crop_region,
            pcd_noise=0.02,
        )
        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.get_seg_pc)
        self.observations.policy.concatenate_terms = False
