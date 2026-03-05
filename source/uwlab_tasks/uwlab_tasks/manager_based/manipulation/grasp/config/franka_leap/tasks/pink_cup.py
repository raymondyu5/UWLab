# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Pink cup grasp task for Franka-LEAP.
# Source values from:
#   IsaacLab/source/config/task/hand_env/object/recenter_ycb.yaml (pink_cup entry)
#   IsaacLab/source/config/task/hand_env/leap_franka/grasp/rl_env_ycb_synthetic_pc_custom_init_pink_cup.yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from ....mdp import GraspReward, SynthesizePC, reset_robot_joints, reset_object_pose, reset_camera_pose
from .. import grasp_franka_leap
from .shared_params import ARM_MESH_DIR, HAND_MESH_DIR, FINGERS_NAME_LIST, ARM_RESET, HAND_RESET

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


@configclass
class GraspPinkCupFrankaLeap(grasp_franka_leap.FrankaLeapGraspEnv):
    scene: GraspPinkCupSceneCfg = GraspPinkCupSceneCfg(num_envs=1, env_spacing=2.5)
    

    def __post_init__(self):
        super().__post_init__()

        self.horizon = PINK_CUP_HORIZON
        self.episode_length_s = self.horizon * self.decimation * self.dt

        # --- Instantiate GraspReward and add finger contact sensors ---
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

        # --- Reset events ---
        self.events.reset_robot = EventTerm(
            func=reset_robot_joints,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "arm_joint_pos": ARM_RESET,
                "hand_joint_pos": HAND_RESET,
                "arm_joint_limits": franka_leap.FRANKA_LEAP_ARM_JOINT_LIMITS,
            },
        )
        # Camera pose randomization — exact params from rl_env_ycb_cam_custom_init_pink_cup.yaml
        # random_pose_range: [x_min, y_min, z_min, x_max, y_max, z_max, radius_min, radius_max]
        # phi_range_rad: elevation [1.0, 1.66] (~57-95°), theta_range_rad: azimuth [0.0, 0.5]
        self.events.reset_camera = EventTerm(
            func=reset_camera_pose,
            mode="reset",
            params={
                "camera_name": "camera",
                "random_pose_range": (0.4, -0.15, 0.10, 0.6, 0.15, 0.25, 0.8, 1.7),
                "theta_range_rad": (0.0, 0.5),
                "phi_range_rad": (1.0, 1.66),
            },
        )

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

@configclass
class GraspPinkCupFrankaLeapIkRel(GraspPinkCupFrankaLeap):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

@configclass
class GraspPinkCupFrankaLeapIkAbs(GraspPinkCupFrankaLeap):
    actions = franka_leap.FrankaLeapIkAbsArmHandJointAction()