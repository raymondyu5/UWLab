# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
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
    id="UW-FrankaLeap-GraspPinkCup-JointAbs-SysidApplied-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.pink_cup:GraspPinkCupFrankaLeapJointAbsSysidAppliedCfg",
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


########## Pink Cup (random arm resets from JSON) #########

gym.register(
    id="UW-FrankaLeap-GraspPinkCupRandomResets-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cup_grasp_random_resets:GraspPinkCupRandomResetsFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspPinkCupRandomResets7030-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cup_grasp_random_resets:GraspPinkCupRandomResets7030FrankaLeapJointAbsCfg",
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
    id="UW-FrankaLeap-GraspBottle-JointAbs-SysidApplied-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle:GraspBottleFrankaLeapJointAbsSysidAppliedCfg",
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
    id="UW-FrankaLeap-PourBottle-EeBox-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapEeBoxJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-PourBottle-EeBoxHandRand-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapEeBoxHandRandJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-PourBottle-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-PourBottle-JointAbs-PPO-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapJointAbsStateCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PourBottleJointAbsPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
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

gym.register(
    id="UW-FrankaLeap-PourBottle-JointRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour:PourBottleFrankaLeapJointRelCfg",
    },
    disable_env_checker=True,
)


########## Bottle Pour (random arm resets from JSON) #########

gym.register(
    id="UW-FrankaLeap-PourBottleRandomResets-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour_random_resets:PourBottleRandomResetsFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-PourBottleRandomResets-JointAbs-PPO-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_pour_random_resets:PourBottleRandomResetsFrankaLeapJointAbsStateCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PourBottleJointAbsPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


########## Cube Grasp #########

gym.register(
    id="UW-FrankaLeap-GraspCube-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cube:GraspCubeFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspCube-IkRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cube:GraspCubeFrankaLeapIkRelCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspCube-IkAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cube:GraspCubeFrankaLeapIkAbsCfg",
    },
    disable_env_checker=True,
)


########## Bottle Grasp (random arm resets from JSON) #########

gym.register(
    id="UW-FrankaLeap-GraspBottleRandomResets-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_grasp_random_resets:GraspBottleRandomResetsFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspBottleRandomResets-JointAbs-PPO-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_grasp_random_resets:GraspBottleRandomResetsFrankaLeapJointAbsStateCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PourBottleJointAbsPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspBottleRandomResets-IkRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_grasp_random_resets:GraspBottleRandomResetsFrankaLeapIkRelCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspBottleRandomResets-IkAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_grasp_random_resets:GraspBottleRandomResetsFrankaLeapIkAbsCfg",
    },
    disable_env_checker=True,
)


########## Cube Grasp (random arm resets from JSON) #########

gym.register(
    id="UW-FrankaLeap-GraspCubeRandomResets-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cube_grasp_random_resets:GraspCubeRandomResetsFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)


########## Cube Push #########

gym.register(
    id="UW-FrankaLeap-PushCubeToPoptart-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cube_push:PushCubeToPoptartFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-PushCubeToPoptart-IkRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cube_push:PushCubeToPoptartFrankaLeapIkRelCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-PushCubeToPoptart-IkAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cube_push:PushCubeToPoptartFrankaLeapIkAbsCfg",
    },
    disable_env_checker=True,
)


########## Cube Push (random arm resets from JSON) #########

gym.register(
    id="UW-FrankaLeap-PushCubeToPoptartRandomResets-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.cube_push_random_resets:PushCubeToPoptartRandomResetsFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)


########## Bottle Singulate #########

gym.register(
    id="UW-FrankaLeap-SingulateBottle-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_singulate:SingulateBottleFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-SingulateBottle-JointRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_singulate:SingulateBottleFrankaLeapJointRelCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-SingulateBottle-IkRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_singulate:SingulateBottleFrankaLeapIkRelCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-SingulateBottle-IkAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.bottle_singulate:SingulateBottleFrankaLeapIkAbsCfg",
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


########## Plate in Dishrack #########

gym.register(
    id="UW-FrankaLeap-PlateInDishRack-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.plate_in_dishrack:PlateInDishRackFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)


########## Screw Lightbulb #########

gym.register(
    id="UW-FrankaLeap-ScrewLightbulb-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.screw_lightbulb:ScrewLightbulbFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-ScrewLightbulb-HighFriction-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.screw_lightbulb:ScrewLightbulbFrankaLeapHighFrictionJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-ScrewLightbulb-LightBulb-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.screw_lightbulb:ScrewLightbulbFrankaLeapLightBulbJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-ScrewLightbulb-Unfixed-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.screw_lightbulb:ScrewLightbulbFrankaLeapUnfixedJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-ScrewLightbulb-Curriculum-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.screw_lightbulb:ScrewLightbulbFrankaLeapCurriculumJointAbsCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="UW-FrankaLeap-ScrewLightbulb-TallBigBase-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.screw_lightbulb:ScrewLightbulbFrankaLeapTallBigBaseJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-ScrewLightbulb-RealLightbulb-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.screw_lightbulb:ScrewLightbulbFrankaLeapRealLightbulbJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-ScrewLightbulb-HighContact-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.screw_lightbulb:ScrewLightbulbFrankaLeapHighContactJointAbsCfg",
    },
    disable_env_checker=True,
)

########## Credit Card Grasp #########

gym.register(
    id="UW-FrankaLeap-GraspCreditCard-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.credit_card_grasp:GraspCreditCardFrankaLeapJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspCreditCard-IkRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.credit_card_grasp:GraspCreditCardFrankaLeapIkRelCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-GraspCreditCard-IkAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.credit_card_grasp:GraspCreditCardFrankaLeapIkAbsCfg",
    },
    disable_env_checker=True,
)


########## Sysid (arm-only CMA-ES) #########

gym.register(
    id="UW-FrankaLeap-Sysid-JointAbs-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sysid_cfg:GraspFrankaLeapSysidJointAbsCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-Sysid-JointRel-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sysid_cfg:GraspFrankaLeapSysidJointRelCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-FrankaLeap-JointAbs-SysidApplied-v0",
    entry_point=f"{__name__}.grasp_franka_leap:FrankaLeapGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sysid_cfg:GraspFrankaLeapJointAbsSysidAppliedCfg",
    },
    disable_env_checker=True,
)