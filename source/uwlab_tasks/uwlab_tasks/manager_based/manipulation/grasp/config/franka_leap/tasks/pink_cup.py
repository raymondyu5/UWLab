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

from ....mdp import GraspReward, SynthesizePC, reset_robot_joints, reset_object_pose
from .. import grasp_franka_leap

# Pink cup spawn values from recenter_ycb.yaml:
# pos: [0.5, 0.0, 0.07], rot: axis=[0], angles=[1.57] -> X-axis 90deg -> quat (0.707, 0.707, 0, 0)
# scale: [0.35, 0.35, 0.35]
# pose_range: x: [-0.05, 0.05], y: [-0.05, 0.05], z: [0,0], roll: [0,0], yaw: [0,0]

PINK_CUP_USD = "/workspace/uwlab/assets/pink_cup/rigid_object.usd"
PINK_CUP_MESH = "/workspace/uwlab/assets/pink_cup/textured_recentered.obj"
ARM_MESH_DIR = "/workspace/uwlab/assets/robot/franka_leap/raw_mesh"
HAND_MESH_DIR = "/workspace/uwlab/assets/robot/franka_leap/raw_mesh"

# From rl_env_ycb_synthetic_pc_custom_init_pink_cup.yaml:
#   right_reset_joint_pose (arm, 7D)
PINK_CUP_ARM_RESET = [
    3.1088299e-01, 4.0700440e-03, -3.1125304e-01, -2.0509737e+00,
    1.4107295e-03, 2.0548446e+00, 7.8060406e-01,
]
#   right_reset_hand_joint_pose (hand, 16D)
PINK_CUP_HAND_RESET = [
    0.35281801223754883, 0.6442744731903076, 0.29912877082824707, 0.34514832496643066,
    -0.03681302070617676, -0.06749272346496582, -0.09357023239135742, -0.14725971221923828,
    0.0659637451171875, 0.43411898612976074, 0.05982780456542969, 0.013808250427246094,
    0.03221607208251953, -0.009201288223266602, 0.029148101806640625, 0.0046045780181884766,
]

# Target pose for lift reward: right_target_manipulated_object_pose from YAML
PINK_CUP_TARGET_POS = (0.60, 0.10, 0.40)

FINGERS_NAME_LIST = ["palm_lower", "fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]


@configclass
class GraspPinkCupSceneCfg(grasp_franka_leap.FrankaLeapGraspSceneCfg):
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.07),
            rot=(0.707, 0.707, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=PINK_CUP_USD,
            scale=(0.35, 0.35, 0.35),
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
    actions = franka_leap.FrankaLeapJointPositionAction()

    def __post_init__(self):
        super().__post_init__()

        # horizon=200, decimation=3, dt=1/60 → episode_length_s = 200 * 3 / 60
        self.episode_length_s = 200 * 3 / 60.0

        # --- Instantiate GraspReward and add finger contact sensors ---
        grasp_rew = GraspReward(
            asset_name="robot",
            object_name="object",
            fingers_name_list=FINGERS_NAME_LIST,
            init_height=0.07,
            target_pos=PINK_CUP_TARGET_POS,
        )
        grasp_rew.setup_additional(self.scene)
        grasp_rew.setup_finger_sensors(self.scene, object_prim_name="Object")
        self.rewards.grasp_rewards = RewTerm(func=grasp_rew.grasp_rewards, weight=4.0)

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
                "arm_joint_pos": PINK_CUP_ARM_RESET,
                "hand_joint_pos": PINK_CUP_HAND_RESET,
            },
        )
        self.events.reset_object = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "default_pos": (0.5, 0.0, 0.07),
                "default_rot_quat": (0.707, 0.707, 0.0, 0.0),
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": 0.07,
            },
        )
