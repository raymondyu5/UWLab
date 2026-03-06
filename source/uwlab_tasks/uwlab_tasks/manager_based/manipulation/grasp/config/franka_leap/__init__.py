# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

########## Pink Cup #########
gym.register(
    id="UW-FrankaLeap-GraspPinkCup-JointAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.pink_cup:GraspPinkCupFrankaLeapJointAbs",
    },
    disable_env_checker=True,
)


gym.register(
    id="UW-FrankaLeap-GraspPinkCup-IkRel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.pink_cup:GraspPinkCupFrankaLeapIkRel",
    },
    disable_env_checker=True,
)


gym.register(
    id="UW-FrankaLeap-GraspPinkCup-IkAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.pink_cup:GraspPinkCupFrankaLeapIkAbs",
    },
    disable_env_checker=True,
)


########## Bottle Grasp #########

gym.register(
    id="UW-FrankaLeap-GraspBottle-JointAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle:GraspBottleFrankaLeapJointAbs",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspBottle-IkRel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle:GraspBottleFrankaLeapIkRel",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspBottle-IkAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle:GraspBottleFrankaLeapIkAbs",
    },
    disable_env_checker=True,
)


########## Bottle Pour #########

gym.register(
    id="UW-FrankaLeap-PourBottle-JointAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapJointAbs",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-PourBottle-IkRel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapIkRel",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-PourBottle-IkAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapIkAbs",
    },
    disable_env_checker=True,
)


########## Empty #########

gym.register(
    id="UW-FrankaLeap-JointAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_franka_leap:GraspFrankaLeapJointAbs",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-IkAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_franka_leap:GraspFrankaLeapIkAbs",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-IkRel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_franka_leap:GraspFrankaLeapIkRel",
    },
    disable_env_checker=True,
)