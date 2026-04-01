# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Bottle grasp task for Franka-LEAP.
# Source values from:
#   IsaacLab/source/config/task/hand_env/leap_franka/grasp/rl_env_bourbon_pour_pink_cup_delta_joint.yaml
#   (RigidObject section, right_hand_object entry)

from source.uwlab_tasks.uwlab_tasks.manager_based.manipulation.grasp.mdp.observations import CachedSamplePC
import torch
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from .... import mdp
from ....mdp import reset_object_pose, reset_table_block
from .rewards.grasp_rewards import GraspReward
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET, ARM_NUM_POINTS, HAND_NUM_POINTS
from .shared_params import ARM_MESH_DIR, HAND_MESH_DIR, FINGERS_NAME_LIST

# Bottle spawn values from rl_env_bourbon_pour_pink_cup_delta_joint.yaml (RigidObject, right_hand_object):
# pos: [0.55, -0.10, 0.11], rot: identity (upright), discrete_yaw: -1.57 (cap faces right toward cup)
# pose_range: x±10cm, y±10cm
# target_pos: same as pink cup (0.60, 0.10, 0.40)

BOTTLE_USD = "/workspace/uwlab/assets/bourbon/rigid_object.usd"
BOTTLE_OBJECT_NUM_POINTS = 128

BOTTLE_SPAWN_POS = (0.55, -0.10, 0.11)
# -90deg around Z = (0.707, 0, 0, -0.707); local -X (cap) maps to world +Y at spawn
BOTTLE_SPAWN_ROT = (0.707, 0.0, 0.0, -0.707)
BOTTLE_TARGET_POS = (0.60, 0.10, 0.40)
BOTTLE_HORIZON = 180
BOTTLE_SUCCESS_HEIGHT = 0.25

@configclass
class GraspBottleSceneCfg(grasp_franka_leap.FrankaLeapGraspSceneCfg):
    grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BOTTLE_SPAWN_POS,
            rot=BOTTLE_SPAWN_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOTTLE_USD,
            scale=(1.1, 1.1, 1.1),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.10, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
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
class GraspBottleSysidSceneCfg(GraspBottleSceneCfg):
    """Bottle scene with delayed actuators for sysid-applied overlay. Keeps cameras for seg_pc."""

    robot = franka_leap.IMPLICIT_FRANKA_LEAP.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators=franka_leap.FRANKA_LEAP_REAL_GAINS_ARM_ACTUATOR_DELAYED_CFG
        | franka_leap.FRANKA_LEAP_HAND_ACTUATOR_CFG,
    )


@configclass
class GraspBottleFrankaLeapCfg(grasp_franka_leap.FrankaLeapGraspEnvCfg):
    scene: GraspBottleSceneCfg = GraspBottleSceneCfg(num_envs=1, env_spacing=2.5)
    table_z_range: tuple = (0.0, 0.05)  # set to (0.0, 0.0) to disable table height randomization

    def is_success(self, env) -> torch.Tensor:
        obj = env.scene["grasp_object"]
        pos = obj.data.root_pos_w - env.scene.env_origins
        block = env.scene["table_block"]
        table_z_offset = (
            block.data.root_state_w[:, 2]
            - env.scene.env_origins[:, 2]
            - block.data.default_root_state[:, 2]
        )
        return pos[:, 2] >= (BOTTLE_SUCCESS_HEIGHT + table_z_offset)

    def __post_init__(self):
        super().__post_init__()


        self.object_spawn_defaults = {
            "default_pos": list(BOTTLE_SPAWN_POS),
            "default_rot": list(BOTTLE_SPAWN_ROT),
            "reset_height": BOTTLE_SPAWN_POS[2],
        }

        self.setup_horizon(horizon=BOTTLE_HORIZON)

        grasp_rew = GraspReward(
            asset_name="robot",
            object_name="grasp_object",
            fingers_name_list=FINGERS_NAME_LIST,
            init_height=BOTTLE_SPAWN_POS[2],
            target_pos=BOTTLE_TARGET_POS,
        )
        grasp_rew.setup_additional(self.scene)
        grasp_rew.setup_finger_entities(self.scene)
        grasp_rew.setup_finger_sensors(self.scene, object_prim_name="GraspObject")

        self.rewards.lift = RewTerm(func=grasp_rew.rew_lift, weight=4.0)
        self.rewards.contact = RewTerm(func=grasp_rew.rew_contact, weight=4.0)
        self.rewards.finger2object = RewTerm(func=grasp_rew.rew_finger2object, weight=4.0)
        self.rewards.wrist_penalty = RewTerm(func=grasp_rew.rew_wrist_penalty, weight=4.0)
        self.rewards.joint_vel = RewTerm(func=grasp_rew.rew_joint_vel, weight=-4e-3)
        self.rewards.action_rate = RewTerm(func=grasp_rew.rew_action_rate, weight=-2e-2)
        self.observations.policy.target_object_pose = ObsTerm(func=grasp_rew.obs_target_object_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=grasp_rew.obs_manipulated_object_pose)
        self.observations.policy.contact_obs = ObsTerm(func=grasp_rew.obs_contact)
        self.observations.policy.object_in_tip = ObsTerm(func=grasp_rew.obs_object_in_tip)

        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["grasp_object"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[BOTTLE_OBJECT_NUM_POINTS],
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
                "default_pos": BOTTLE_SPAWN_POS,
                "default_rot_quat": BOTTLE_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.10, 0.10),
                    "y": (-0.10, 0.10),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": BOTTLE_SPAWN_POS[2],
                "table_block_name": "table_block",
            },
        )

        self.events.capture_reset_height = EventTerm(
            func=grasp_rew.capture_reset_height,
            mode="reset",
            params={},
        )

        self.events.randomize_object_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="reset",
            min_step_count_between_reset=800,
            params={
                "asset_cfg": SceneEntityCfg("grasp_object"),
                "static_friction_range": (0.3, 1.5),
                "dynamic_friction_range": (0.3, 1.2),
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


@configclass
class GraspBottleFrankaLeapJointAbsCfg(GraspBottleFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class GraspBottleFrankaLeapJointAbsSysidAppliedCfg(GraspBottleFrankaLeapJointAbsCfg):
    """Bottle JointAbs with sysid params applied on reset. Has seg_pc for overlay scripts."""

    scene: GraspBottleSysidSceneCfg = GraspBottleSysidSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.apply_sysid_params = EventTerm(
            func=mdp.apply_sysid_params_on_reset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "params": franka_leap.FRANKA_LEAP_SYSID_PARAMS,
            },
        )


@configclass
class GraspBottleFrankaLeapIkRelCfg(GraspBottleFrankaLeapCfg):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class GraspBottleFrankaLeapIkAbsCfg(GraspBottleFrankaLeapCfg):
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
