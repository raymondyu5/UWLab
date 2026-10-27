# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Pink cup grasp task for standard Franka (2-finger gripper, no LEAP hand).
# Adapted from ../../franka_leap/tasks/pink_cup.py: same object/spawn/table/randomization,
# reward and finger obs are re-derived for the 2-finger gripper (see rewards/grasp_rewards.py).
# State-based only (no seg_pc / camera pipeline — no gripper mesh PCD set up for this robot).

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

import uwlab_assets.robots.franka as franka

from ....mdp import reset_object_pose, reset_table_block, bottle_too_far
from .rewards.grasp_rewards import SimpleGraspReward
from .. import grasp_franka
from ..grasp_franka import ARM_RESET, GRIPPER_RESET

CUP_GRASP_HORIZON = 128
CUP_GRASP_TARGET_POS = (0.55, 0.10, 0.35)

CUP_USD = "/workspace/uwlab/assets/pink_cup/rigid_object.usd"
CUP_SPAWN_POS = (0.525, 0.1, 0.07)
CUP_SPAWN_ROT = (0.707, 0.707, 0.0, 0.0)


@configclass
class GraspCupFrankaSceneCfg(grasp_franka.FrankaGraspSceneCfg):
    grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CUP_SPAWN_POS,
            rot=CUP_SPAWN_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUP_USD,
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(CUP_SPAWN_POS[0], CUP_SPAWN_POS[1], 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/uwlab/assets/table/table_block.usd",
            scale=(1.2, 1.0, 0.10),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
            ),
        ),
    )


SUCCESS_HEIGHT = 0.20  # object z above table (local frame) to count as grasped
GRASPED_HEIGHT = 0.12  # 12cm above table


@configclass
class GraspCupFrankaCfg(grasp_franka.FrankaGraspEnvCfg):
    scene: GraspCupFrankaSceneCfg = GraspCupFrankaSceneCfg(num_envs=1, env_spacing=2.5)
    table_z_range: tuple = (0.0, 0.05)  # set to (0.0, 0.0) to disable table height randomization

    def _cup_z_above_table(self, env) -> torch.Tensor:
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
        z, table_z_offset = self._cup_z_above_table(env)
        return z >= (GRASPED_HEIGHT + table_z_offset)

    def is_success(self, env) -> torch.Tensor:
        z, table_z_offset = self._cup_z_above_table(env)
        return z >= (SUCCESS_HEIGHT + table_z_offset)

    def __post_init__(self):
        super().__post_init__()

        init = self.scene.grasp_object.init_state
        self.object_spawn_defaults = {
            "default_pos": tuple(init.pos),
            "default_rot": tuple(init.rot),
            "reset_height": float(init.pos[2]),
        }

        self.setup_horizon(horizon=CUP_GRASP_HORIZON)

        simple_rew = SimpleGraspReward(
            asset_name="robot",
            object_name="grasp_object",
            init_height=CUP_SPAWN_POS[2],
            target_pos=CUP_GRASP_TARGET_POS,
        )
        simple_rew.setup_wrist_sensor(self.scene)
        simple_rew.setup_finger_entities(self.scene)
        simple_rew.setup_finger_sensors(self.scene, object_prim_name="GraspObject")

        self.rewards.grasped     = RewTerm(func=simple_rew.rew_grasped,       weight=1.0)
        self.rewards.lifted      = RewTerm(func=simple_rew.rew_lifted,        weight=5.0)
        self.rewards.success     = RewTerm(func=simple_rew.rew_success,       weight=10.0)
        self.rewards.wrist       = RewTerm(func=simple_rew.rew_wrist_penalty, weight=-2.0)
        self.rewards.proximity   = RewTerm(func=simple_rew.rew_proximity,     weight=0.5)
        self.rewards.joint_vel   = RewTerm(func=simple_rew.rew_joint_vel,     weight=-1e-3)
        self.rewards.action_rate = RewTerm(func=simple_rew.rew_action_rate,   weight=-0.5)
        self.metrics_spec = {"is_success": simple_rew.rew_success, "is_lifted": simple_rew.rew_lifted, "is_grasped": simple_rew.rew_grasped}

        self.events.capture_reset_height = EventTerm(
            func=simple_rew.capture_reset_height, mode="reset", params={}
        )

        self.observations.policy.target_object_pose = ObsTerm(func=simple_rew.obs_target_object_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=simple_rew.obs_manipulated_object_pose)
        self.observations.policy.contact_obs = ObsTerm(func=simple_rew.obs_contact)
        self.observations.policy.object_in_tip = ObsTerm(func=simple_rew.obs_object_in_tip)

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
                "default_pos": CUP_SPAWN_POS,
                "default_rot_quat": CUP_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.075, 0.175),
                    "y": (-0.10, 0.10),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": CUP_SPAWN_POS[2],
                "table_block_name": "table_block",
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
                "scale_range": (0.8, 1.2),
            },
        )

        self.terminations.cup_too_far = DoneTerm(
            func=bottle_too_far,
            params={
                "object_name": "grasp_object",
                "max_xy_dist": 1.0,
            },
            time_out=False,
        )


@configclass
class GraspCupFrankaJointAbsCfg(GraspCupFrankaCfg):
    actions = franka.JointPositionAbsoluteAction()

    def warmup_action(self, env) -> torch.Tensor:
        """Hold at reset joint position — safe no-op for joint absolute control."""
        reset = torch.tensor(ARM_RESET + GRIPPER_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class GraspCupFrankaIkRelCfg(GraspCupFrankaCfg):
    actions = franka.IkDeltaAction()

    def warmup_action(self, env) -> torch.Tensor:
        """Zero delta EE + open gripper — safe no-op for IK-relative control."""
        arm = torch.zeros(env.num_envs, 6, device=env.device)
        gripper_open = torch.ones(env.num_envs, 1, device=env.device)
        return torch.cat([arm, gripper_open], dim=-1)


@configclass
class GraspCupFrankaIkAbsCfg(GraspCupFrankaCfg):
    actions = franka.IkAbsoluteAction()

    def warmup_action(self, env) -> torch.Tensor:
        """Hold current EE pose + open gripper — safe no-op for IK-absolute control."""
        robot = env.scene["robot"]
        ee_body_idx = robot.body_names.index(franka.FRANKA_EE_BODY)
        ee_state = robot._data.body_state_w[:, ee_body_idx, :7].clone()
        ee_state[:, :3] -= env.scene.env_origins
        offset_pos = torch.tensor([list(franka.FRANKA_EE_OFFSET)], device=env.device).repeat(env.num_envs, 1)
        offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
        pos, quat = math_utils.combine_frame_transforms(ee_state[:, :3], ee_state[:, 3:7], offset_pos, offset_rot)
        gripper_open = torch.ones(env.num_envs, 1, device=env.device)
        return torch.cat([pos, quat, gripper_open], dim=-1)
