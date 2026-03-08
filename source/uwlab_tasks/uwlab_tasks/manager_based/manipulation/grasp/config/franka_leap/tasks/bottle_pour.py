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

from ....mdp import PourReward, SynthesizePC, SamplePC, reset_object_pose, reset_table_block
from ....mdp import bottle_dropped, bottle_too_far, cup_toppled
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET
from .shared_params import ARM_MESH_DIR, HAND_MESH_DIR, FINGERS_NAME_LIST
from .bottle import (
    GraspBottleSceneCfg,
    BOTTLE_USD,
    BOTTLE_MESH,
    BOTTLE_SPAWN_POS,
    BOTTLE_SPAWN_ROT,
)
from .pink_cup import PINK_CUP_MESH, PINK_CUP_USD

# Pink cup pour-task spawn values from rl_env_bourbon_pour_pink_cup_synthetic_pc_force_pert.yaml (pour_config):
# cup_pos: [0.55, 0.10, 0.07], rot: X-axis 90deg = (0.707, 0.707, 0, 0) (w,x,y,z)
# cup_pose_range: x±5cm, y [0, +5cm]
PINK_CUP_POUR_POS = (0.55, 0.10, 0.07)
PINK_CUP_POUR_ROT = (0.707, 0.707, 0.0, 0.0)

# Bottle cap offset in local -X frame: 13.22cm (cap is at X=-0.132 in mesh frame)
BOTTLE_CAP_OFFSET = (-0.132179, 0.0, 0.0)

POUR_HORIZON = 180


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
    table_z_range: tuple = (0.0, 0.05)  # set to (0.0, 0.0) to disable table height randomization

    def is_success(self, env) -> torch.Tensor:
        # Cap tip XY within 5cm of cup center and above cup_z + 0.07
        # (matches IsaacLab eval_bc_policy.py pour success criterion)
        bottle = env.scene["grasp_object"]
        cup = env.scene["pink_cup"]
        bottle_pos = bottle.data.root_pos_w - env.scene.env_origins       # (N, 3)
        bottle_quat = bottle.data.root_quat_w                              # (N, 4) w,x,y,z
        cup_pos = cup.data.root_pos_w - env.scene.env_origins             # (N, 3)

        cap_offset = torch.tensor(list(BOTTLE_CAP_OFFSET), device=env.device)  # (3,)
        # Rotate cap offset by bottle quaternion: v' = v + 2w*(q×v) + 2*(q×(q×v))
        w = bottle_quat[:, 0:1]
        q = bottle_quat[:, 1:]   # xyz
        t = 2.0 * torch.linalg.cross(q, cap_offset.unsqueeze(0).expand_as(q))
        tip_pos = bottle_pos + cap_offset.unsqueeze(0) + w * t + torch.linalg.cross(q, t)

        xy_dist = torch.norm(tip_pos[:, :2] - cup_pos[:, :2], dim=1)
        above_cup = tip_pos[:, 2] > (cup_pos[:, 2] + 0.26)
        return (xy_dist < 0.05) & above_cup

    def __post_init__(self):
        super().__post_init__()



        self.object_spawn_defaults = {
            "default_pos": list(BOTTLE_SPAWN_POS),
            "default_rot": list(BOTTLE_SPAWN_ROT),
        }

        self.horizon = POUR_HORIZON
        self.episode_length_s = self.horizon * self.decimation * self.sim.dt

        pour_rew = PourReward(
            asset_name="robot",
            object_name="grasp_object",
            cup_name="pink_cup",
            fingers_name_list=FINGERS_NAME_LIST,
            init_height=BOTTLE_SPAWN_POS[2],
            bottle_cap_offset=BOTTLE_CAP_OFFSET,
        )
        pour_rew.setup_wrist_sensor(self.scene)
        pour_rew.setup_finger_entities(self.scene)
        pour_rew.setup_finger_sensors(self.scene, object_prim_name="GraspObject")

        self.rewards.pour_rewards = RewTerm(func=pour_rew.pour_rewards, weight=4.0)

        self.observations.policy.cup_pose = ObsTerm(func=pour_rew.obs_cup_pose)
        self.observations.policy.target_object_pose = ObsTerm(func=pour_rew.obs_target_object_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=pour_rew.obs_manipulated_object_pose)
        self.observations.policy.contact_obs = ObsTerm(func=pour_rew.obs_contact)
        self.observations.policy.object_in_tip = ObsTerm(func=pour_rew.obs_object_in_tip)

        synth_pc = SamplePC(
            asset_name="robot",
            object_names=["grasp_object", "pink_cup"],
            arm_mesh_dir=ARM_MESH_DIR,
            hand_mesh_dir=HAND_MESH_DIR,
            object_mesh_paths=[BOTTLE_MESH, PINK_CUP_MESH],
            num_arm_pcd=64,
            num_hand_pcd=64,
            num_object_pcd=512,
            num_downsample_points=2048,
        )
        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.synthesize_env)

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
                "default_pos": BOTTLE_SPAWN_POS,
                "default_rot_quat": BOTTLE_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": BOTTLE_SPAWN_POS[2],
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
                    "x": (-0.05, 0.05),
                    "y": (0.0, 0.05),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": PINK_CUP_POUR_POS[2],  # 0.07
                "table_block_name": "table_block",
            },
        )

        # Capture reset references AFTER both objects have been reset
        self.events.capture_reset_references = EventTerm(
            func=pour_rew.capture_reset_references,
            mode="reset",
            params={},
        )

        # Terminations
        self.terminations.bottle_dropped = DoneTerm(
            func=bottle_dropped,
            params={"pour_rew": pour_rew},
            time_out=False,
        )
        self.terminations.bottle_too_far = DoneTerm(
            func=bottle_too_far,
            params={"pour_rew": pour_rew},
            time_out=False,
        )
        self.terminations.cup_toppled = DoneTerm(
            func=cup_toppled,
            params={"pour_rew": pour_rew},
            time_out=False,
        )


@configclass
class PourBottleFrankaLeapJointAbsCfg(PourBottleFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


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
