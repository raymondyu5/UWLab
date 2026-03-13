# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .grasp_franka_leap import (
    EVAL_MODE,
    DISTILL_MODE,
    RL_MODE,
    RUN_MODES,
    parse_franka_leap_env_cfg,
)

########## Pink Cup #########
gym.register(
    id="UW-FrankaLeap-GraspPinkCup-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.pink_cup:GraspPinkCupFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="UW-FrankaLeap-GraspPinkCup-IkRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.pink_cup:GraspPinkCupFrankaLeapIkRelCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="UW-FrankaLeap-GraspPinkCup-IkAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.pink_cup:GraspPinkCupFrankaLeapIkAbsCfg",
    },
    disable_env_checker=True,
)


########## Bottle Grasp #########

gym.register(
    id="UW-FrankaLeap-GraspBottle-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle:GraspBottleFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspBottle-IkRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle:GraspBottleFrankaLeapIkRelCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspBottle-IkAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle:GraspBottleFrankaLeapIkAbsCfg",
    },
    disable_env_checker=True,
)


########## Bottle Pour #########

gym.register(
    id="UW-FrankaLeap-PourBottle-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-PourBottle-IkRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapIkRelCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-PourBottle-IkAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapIkAbsCfg",
    },
    disable_env_checker=True,
)


########## Empty #########

gym.register(
    id="UW-FrankaLeap-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_franka_leap:GraspFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-IkAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_franka_leap:GraspFrankaLeapIkAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-IkRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_franka_leap:GraspFrankaLeapIkRelCfg",
    },
    disable_env_checker=True,
)