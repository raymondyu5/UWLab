# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .grasp_franka import (
    EVAL_MODE,
    DISTILL_MODE,
    RL_MODE,
    RUN_MODES,
    parse_franka_env_cfg,
)

########## Cup Grasp #########

gym.register(
    id="UW-Franka-GraspCup-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka:FrankaGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cup_grasp:GraspCupFrankaJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-Franka-GraspCup-IkRel-v0",
    entry_point=f"{__name__}.grasp_franka:FrankaGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cup_grasp:GraspCupFrankaIkRelCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-Franka-GraspCup-IkAbs-v0",
    entry_point=f"{__name__}.grasp_franka:FrankaGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cup_grasp:GraspCupFrankaIkAbsCfg",
    },
    disable_env_checker=True,
)
