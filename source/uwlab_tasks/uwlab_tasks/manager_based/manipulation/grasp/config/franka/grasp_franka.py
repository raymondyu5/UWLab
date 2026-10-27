# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Base robot-specific scene config for standard-Franka (2-finger gripper, no LEAP hand) grasp tasks.
# Mirrors ../grasp_franka_leap.py but for the plain Franka arm + parallel gripper.
# Subclassed by tasks/cup_grasp.py.

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka as franka

from ... import grasp_env
from ...mdp import (
    reset_robot_joints,
    reset_camera_pose,
    set_fixed_camera_view,
)
from ...mdp.metrics import EnvMetrics
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.utils import parse_env_cfg

############################################
################ task info #################
############################################

ARM_RESET = [
    0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741,
]

GRIPPER_RESET = [0.04, 0.04]  # open

# Valid values for env_cfg.run_mode.
EVAL_MODE = "eval_mode"
DISTILL_MODE = "distill_mode"
RL_MODE = "rl_mode"
RUN_MODES = (EVAL_MODE, DISTILL_MODE, RL_MODE)


@configclass
class FrankaGraspSceneCfg(grasp_env.GraspSceneCfg):
    robot = franka.IMPLICIT_FRANKA.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class FrankaGraspEnvCfg(grasp_env.GraspEnvCfg):
    scene: FrankaGraspSceneCfg = FrankaGraspSceneCfg(num_envs=1, env_spacing=2.5)
    num_warmup_steps: int = 15
    # Runtime mode: "rl_mode" | "distill_mode" | "eval_mode". "rl_mode" disables scene cameras.
    # Standard-Franka tasks default to rl_mode since there's no gripper mesh PCD pipeline set up.
    run_mode: str = "rl_mode"
    # Scene entity name of the primary task object for eval logging.
    eval_object_name: str | None = "grasp_object"

    def warmup_action(self, env) -> torch.Tensor:
        raise NotImplementedError("Error: warmup_action must be implemented in the subclass for FrankaGraspEnvCfg")

    def __post_init__(self):
        super().__post_init__()
        # Wire EE pose obs with Franka calibration values
        self.observations.policy.ee_pose.params["asset_cfg"] = SceneEntityCfg("robot")
        self.observations.policy.ee_pose.params["ee_body_name"] = franka.FRANKA_EE_BODY
        self.observations.policy.ee_pose.params["ee_offset"] = franka.FRANKA_EE_OFFSET

        # --- Reset events ---
        self.events.reset_robot = EventTerm(
            func=reset_robot_joints,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "arm_joint_pos": ARM_RESET,
                "hand_joint_pos": GRIPPER_RESET,
                "arm_joint_limits": franka.FRANKA_ARM_JOINT_LIMITS,
            },
        )
        # Camera reset events only matter when run_mode != rl_mode (cameras are stripped in rl_mode).
        self.events.reset_camera = EventTerm(
            func=reset_camera_pose,
            mode="reset",
            params={
                "camera_name": "train_camera",
                "random_pose_range": (0.4, -0.15, 0.10, 0.6, 0.15, 0.25, 0.8, 1.7),
                "theta_range_rad": (0.0, 0.5),
                "phi_range_rad": (1.0, 1.66),
            },
        )
        self.events.reset_fixed_camera = EventTerm(
            func=set_fixed_camera_view,
            mode="reset",
            params={
                "camera_name": "fixed_camera",
                "eye_offset": (1.4327373524611016, 0.2400519659762369, 0.6),
                "look_at_offset": (0.0, -0.15, 0.0),
            },
        )


def _apply_rl_mode(cfg: FrankaGraspEnvCfg) -> None:
    """Configure env for rl_mode: disable scene cameras and their reset events."""
    cfg.scene.train_camera = None
    cfg.scene.fixed_camera = None
    if hasattr(cfg.events, "reset_camera"):
        cfg.events.reset_camera = None
    if hasattr(cfg.events, "reset_fixed_camera"):
        cfg.events.reset_fixed_camera = None


def parse_franka_env_cfg(task: str, run_mode: str, **kwargs) -> FrankaGraspEnvCfg:
    """Parse env config for standard-Franka grasp tasks.

    Args:
        task: The name of the environment.
        run_mode: The runtime mode of the environment (RL_MODE, DISTILL_MODE, EVAL_MODE).
        **kwargs: Additional arguments to pass to parse_env_cfg.
    """
    assert run_mode in RUN_MODES, f"run_mode must be one of {RUN_MODES}, got {run_mode!r}"
    env_cfg = parse_env_cfg(task, **kwargs)
    env_cfg.run_mode = run_mode
    return env_cfg


class FrankaGraspEnv(ManagerBasedRLEnv):
    """Runtime RL environment for standard-Franka grasp tasks with built-in warmup on reset.

    Same reset-then-warmup behavior as FrankaLeapGraspEnv (see ../grasp_franka_leap.py).
    """

    def __init__(self, cfg: FrankaGraspEnvCfg, **kwargs):
        run_mode = cfg.run_mode
        self._apply_run_mode(cfg, run_mode)
        super().__init__(cfg, **kwargs)

        metrics_spec = getattr(cfg, "metrics_spec", None)
        self.metrics = EnvMetrics(metrics_spec or {}, env=self)

    def _apply_run_mode(self, cfg: FrankaGraspEnvCfg, run_mode: str) -> None:
        assert run_mode in RUN_MODES, f"run_mode must be one of {RUN_MODES}, got {run_mode!r}"
        if run_mode == RL_MODE:
            _apply_rl_mode(cfg)
        elif run_mode == DISTILL_MODE:
            raise NotImplementedError(
                "distill_mode is not set up for standard-Franka tasks (no gripper mesh PCD pipeline)."
            )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        cfg = self.cfg
        warmup_action = cfg.warmup_action(self)

        for _ in range(cfg.num_warmup_steps):
            obs, _, terminated, truncated, info = super().step(warmup_action)

        return obs, info
