# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sysid config based on GraspFrankaLeapJointAbsCfg with real robot gains.

Uses the same env as RL (GraspFrankaLeapJointAbsCfg) but overrides the robot to use
FRANKA_LEAP_REAL_GAINS_ARM_ACTUATOR_DELAYED_CFG (DelayedPDActuatorCfg) for sysid.
run_mode=RL_MODE disables cameras.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from ... import mdp
from .grasp_franka_leap import (
    FrankaLeapEmptySceneCfg,
    GraspFrankaLeapJointAbsCfg,
    GraspFrankaLeapJointRelCfg,
    RL_MODE,
    ARM_NUM_POINTS,
    HAND_NUM_POINTS,
)


@configclass
class FrankaLeapSysidSceneCfg(FrankaLeapEmptySceneCfg):
    """Same as FrankaLeapEmptySceneCfg but robot uses real gains for sysid. No cameras (rl_mode)."""

    #fixed_camera = None  # no scene cameras; viewport (ViewerCfg) used for sim.render() when recording
    robot = franka_leap.REAL_GAINS_FRANKA_LEAP.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class FrankaLeapSysidSceneWithCameraCfg(FrankaLeapSysidSceneCfg):
    """Sysid scene with fixed camera for video recording. Adds overhead; use only when --record_video_every > 0."""

    train_camera = None  # disable to reduce overhead; only fixed_camera needed for recording
    fixed_camera = None
    # fixed_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/FixedCamera",
    #     update_period=0.03,
    #     height=480,
    #     width=480,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,gains
    #         pos=(1.43, 0.24, 0.6),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #         convention="ros",
    #     ),
    # )


@configclass
class GraspFrankaLeapSysidJointAbsCfg(GraspFrankaLeapJointAbsCfg):
    """GraspFrankaLeapJointAbsCfg with real robot gains for sysid. Same env as RL."""

    scene: FrankaLeapSysidSceneCfg = FrankaLeapSysidSceneCfg(num_envs=512, env_spacing=2.0)
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.4327373524611016, 0.2400519659762369, 0.6),
        lookat=(0.0, -0.15, 0.0),
        origin_type="env",
        env_index=0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.run_mode = RL_MODE
        self.episode_length_s = 99999.0

        # synth_pc = mdp.CachedSamplePC(
        #     asset_name="robot",
        #     object_names=[],
        #     num_arm_pcd=ARM_NUM_POINTS,
        #     num_hand_pcd=HAND_NUM_POINTS,
        #     num_object_pcd=[],
        #     num_downsample_points=2048,
        # )
        # self.observations.policy.seg_pc = ObsTerm(func=synth_pc.get_seg_pc)


@configclass
class GraspFrankaLeapSysidJointRelCfg(GraspFrankaLeapJointRelCfg):
    """Sysid empty scene with real gains and mesh seg_pc, but joint-relative arm actions (7D delta + 16D hand).

    Same runtime settings as GraspFrankaLeapSysidJointAbsCfg; only the action term differs
    (FrankaLeapJointRelArmHandJointAction). Use for scripts that expect relative arm deltas.
    """

    scene: FrankaLeapSysidSceneCfg = FrankaLeapSysidSceneCfg(num_envs=512, env_spacing=2.0)
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.4327373524611016, 0.2400519659762369, 0.6),
        lookat=(0.0, -0.15, 0.0),
        origin_type="env",
        env_index=0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.run_mode = RL_MODE
        self.episode_length_s = 99999.0

        synth_pc = mdp.CachedSamplePC(
            asset_name="robot",
            object_names=[],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[],
            num_downsample_points=2048,
            pcd_noise=0.02,
        )
        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.get_seg_pc)


@configclass
class GraspFrankaLeapJointAbsSysidAppliedCfg(GraspFrankaLeapJointAbsCfg):
    """GraspFrankaLeapJointAbsCfg with sysid params applied on reset.

    Uses delayed actuator scene and applies FRANKA_LEAP_SYSID_PARAMS each reset.
    Update franka_leap.FRANKA_LEAP_SYSID_PARAMS with your sysid output.
    Encoder bias: add to arm_joint_pos in reset_robot when resetting to real data.
    """

    scene: FrankaLeapSysidSceneCfg = FrankaLeapSysidSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_fixed_camera = None  # Sysid scene has no fixed_camera
        self.events.reset_camera = None  # Sysid scene has no train_camera
        self.events.apply_sysid_params = EventTerm(
            func=mdp.apply_sysid_params_on_reset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "params": franka_leap.FRANKA_LEAP_SYSID_PARAMS,
            },
        )
