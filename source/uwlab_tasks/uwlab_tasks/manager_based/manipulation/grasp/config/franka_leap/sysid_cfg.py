# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene and manager-based env config for Franka LEAP arm system identification (CMA-ES).

Uses open-loop joint position replay: same FrankaLeapJointPositionAction as RL.
Arm-only sysid (7 joints). ImplicitActuatorCfg, no delay.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import time_out

import uwlab_assets.robots.franka_leap as franka_leap

from ...mdp import joint_pos_w

# Default simulation timestep for sysid (60 Hz, matches typical Franka control rate)
SYSID_SIM_DT = 1.0 / 60.0


@configclass
class FrankaLeapSysidSceneCfg(InteractiveSceneCfg):
    """Scene for system identification: robot + ground + light, no objects."""

    robot = franka_leap.IMPLICIT_FRANKA_LEAP.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=franka_leap.IMPLICIT_FRANKA_LEAP.init_state.joint_pos,
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class SysidObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=joint_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SysidRewardsCfg:
    pass


@configclass
class SysidTerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class FrankaLeapSysidEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based env for Franka LEAP arm sysid: joint position replay, decimation=1."""

    scene: FrankaLeapSysidSceneCfg = FrankaLeapSysidSceneCfg(num_envs=512, env_spacing=2.0)
    actions = franka_leap.FrankaLeapJointPositionAction()
    observations: SysidObservationsCfg = SysidObservationsCfg()
    rewards: SysidRewardsCfg = SysidRewardsCfg()
    terminations: SysidTerminationsCfg = SysidTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 1
        self.episode_length_s = 99999.0
        self.sim.dt = SYSID_SIM_DT
