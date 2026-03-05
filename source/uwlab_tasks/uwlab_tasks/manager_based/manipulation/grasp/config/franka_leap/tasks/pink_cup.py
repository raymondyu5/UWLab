# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Pink cup grasp task for Franka-LEAP.
# Source values from:
#   IsaacLab/source/config/task/hand_env/object/recenter_ycb.yaml (pink_cup entry)
#   IsaacLab/source/config/task/hand_env/leap_franka/grasp/rl_env_ycb_synthetic_pc_custom_init_pink_cup.yaml

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from ....mdp import GraspReward, SynthesizePC, reset_object_pose
from .. import grasp_franka_leap
from .shared_params import ARM_MESH_DIR, HAND_MESH_DIR, FINGERS_NAME_LIST

# Pink cup spawn values from rl_env_ycb_cam_custom_init_pink_cup.yaml (RigidObject section):
# pos: [0.55, 0.0, 0.11], rot: axis=[0], angles=[1.57] -> X-axis 90deg -> quat (0.707, 0.707, 0, 0)
# scale: [1.0, 1.0, 1.0] — the 0.35 scale in recenter_ycb.yaml is baked into rigid_object.usd already
# pose_range: x: [-0.05, 0.05], y: [-0.05, 0.05], z: [0,0], roll: [0,0], yaw: [0,0]

PINK_CUP_HORIZON = 200
PINK_CUP_TARGET_POS = (0.60, 0.10, 0.40) # from yaml


PINK_CUP_USD = "/workspace/uwlab/assets/pink_cup/rigid_object.usd"
PINK_CUP_MESH = "/workspace/uwlab/assets/pink_cup/textured_recentered.obj"

@configclass
class GraspPinkCupSceneCfg(grasp_franka_leap.FrankaLeapGraspSceneCfg):
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.11),
            rot=(0.707, 0.707, 0.0, 0.0),
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


SUCCESS_HEIGHT = 0.20  # object z above table (local frame) to count as grasped


@configclass
class GraspPinkCupFrankaLeap(grasp_franka_leap.FrankaLeapGraspEnv):
    scene: GraspPinkCupSceneCfg = GraspPinkCupSceneCfg(num_envs=1, env_spacing=2.5)

    @property
    def object_spawn_defaults(self) -> dict:
        """Default object spawn pose, read from scene config. Used by eval scripts."""
        init = self.scene.object.init_state
        return {
            "default_pos": tuple(init.pos),
            "default_rot": tuple(init.rot),
            "reset_height": float(init.pos[2]),
        }

    def is_success(self, env) -> torch.Tensor:
        """Returns bool tensor (num_envs,): True if object is lifted above SUCCESS_HEIGHT."""
        obj = env.scene["object"]
        pos = obj.data.root_pos_w - env.scene.env_origins
        return pos[:, 2] >= SUCCESS_HEIGHT

    def __post_init__(self):
        super().__post_init__()

        self.horizon = PINK_CUP_HORIZON
        self.episode_length_s = self.horizon * self.decimation * self.sim.dt

        # # --- Instantiate GraspReward and add finger contact sensors ---
        grasp_rew = GraspReward(
            asset_name="robot",
            object_name="object",
            fingers_name_list=FINGERS_NAME_LIST,
            init_height=0.11,
            target_pos=PINK_CUP_TARGET_POS,
        )
        grasp_rew.setup_additional(self.scene)
        grasp_rew.setup_finger_entities(self.scene)
        grasp_rew.setup_finger_sensors(self.scene, object_prim_name="Object")

        self.rewards.grasp_rewards = RewTerm(func=grasp_rew.grasp_rewards, weight=4.0)
        self.observations.policy.target_object_pose = ObsTerm(func=grasp_rew.obs_target_object_pose)
        self.observations.policy.manipulated_object_pose = ObsTerm(func=grasp_rew.obs_manipulated_object_pose)
        self.observations.policy.contact_obs = ObsTerm(func=grasp_rew.obs_contact)
        self.observations.policy.object_in_tip = ObsTerm(func=grasp_rew.obs_object_in_tip)

        # --- Instantiate SynthesizePC and wire as seg_pc obs term ---
        synth_pc = SynthesizePC(
            asset_name="robot",
            object_name="object",
            arm_mesh_dir=ARM_MESH_DIR,
            hand_mesh_dir=HAND_MESH_DIR,
            object_mesh_path=PINK_CUP_MESH,
            num_arm_pcd=64,
            num_hand_pcd=64,
            num_object_pcd=512,
            num_downsample_points=2048,
        )
        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.synthesize_env)

        self.events.reset_object = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "default_pos": (0.55, 0.0, 0.11),
                "default_rot_quat": (0.707, 0.707, 0.0, 0.0),
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": 0.11,
            },
        )

@configclass
class GraspPinkCupFrankaLeapJointAbs(GraspPinkCupFrankaLeap):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        """Hold at reset joint position — safe no-op for joint absolute control."""
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)


@configclass
class GraspPinkCupFrankaLeapIkRel(GraspPinkCupFrankaLeap):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        """Zero delta EE + zero hand — safe no-op for IK-relative control."""
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class GraspPinkCupFrankaLeapIkAbs(GraspPinkCupFrankaLeap):
    actions = franka_leap.FrankaLeapIkAbsArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        """Hold current EE pose + current hand joints — safe no-op for IK-absolute control."""
        robot = env.scene["robot"]
        ee_body_idx = robot.body_names.index(franka_leap.FRANKA_LEAP_EE_BODY)
        ee_state = robot._data.body_state_w[:, ee_body_idx, :7].clone()
        ee_state[:, :3] -= env.scene.env_origins
        offset_pos = torch.tensor([list(franka_leap.FRANKA_LEAP_EE_OFFSET)], device=env.device).repeat(env.num_envs, 1)
        offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
        pos, quat = math_utils.combine_frame_transforms(ee_state[:, :3], ee_state[:, 3:7], offset_pos, offset_rot)
        hand_joints = robot.data.joint_pos[:, len(ARM_RESET):]  # hand joints after arm
        return torch.cat([pos, quat, hand_joints], dim=-1)