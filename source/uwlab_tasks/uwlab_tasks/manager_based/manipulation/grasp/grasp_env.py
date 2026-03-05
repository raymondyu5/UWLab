# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import time_out
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from . import mdp

DEFAULT_GRASP_HORIZON = 100
DEFAULT_GRASP_DECIMATION = 3
DEFAULT_GRASP_PHYSICS_HZ = 60.0

@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.3, 0.0, 0.0], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/uwlab/assets/table/table_instanceable.usd",
            scale=[1.0, 1.0, 1.0]),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.0]),
        spawn=sim_utils.UsdFileCfg(usd_path="/workspace/uwlab/assets/table/default_environment.usd"),
        collision_group=-1,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.03,
        height=480,
        width=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.0, 0.5, 0.8),
            rot=(0.4619, -0.1913, 0.4619, -0.7391),
            convention="ros",
        ),
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        joint_pos = ObsTerm(
            func=mdp.joint_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        ee_pose = ObsTerm(
            func=mdp.ee_pose_w,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "ee_body_name": MISSING,
                "ee_offset": MISSING,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    pass


@configclass
class RewardsCfg:
    pass


@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_robot_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "arm_joint_pos": MISSING,
            "hand_joint_pos": MISSING,
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_object_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "default_pos": MISSING,
            "default_rot_quat": MISSING,
            "pose_range": MISSING,
            "reset_height": MISSING,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class GraspEnv(ManagerBasedRLEnvCfg):

    scene: GraspSceneCfg = GraspSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = MISSING
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = DEFAULT_GRASP_DECIMATION
        self.physics_hz = DEFAULT_GRASP_PHYSICS_HZ  
        self.horizon = DEFAULT_GRASP_HORIZON
        
        self.sim = SimulationCfg(
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            physx=PhysxCfg(
                bounce_threshold_velocity=0.2,
                gpu_max_rigid_contact_count=2**20,
                gpu_max_rigid_patch_count=2**23,
            ),
        )

        self.sim.dt = 1.0 / self.physics_hz
        self.sim.render_interval = self.decimation
        self.control_hz = 1 / (self.sim.dt * self.decimation) # control update frequency (number of control updates per second)
        self.episode_length_s = self.horizon * self.decimation * self.sim.dt


        self.scene.replicate_physics = False
