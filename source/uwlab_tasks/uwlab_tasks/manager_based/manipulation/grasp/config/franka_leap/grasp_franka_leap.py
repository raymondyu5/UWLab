# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Base robot-specific scene config for Franka-LEAP grasp tasks.
# Subclassed by tasks/pink_cup.py, tasks/cube.py, etc.
# Each task subclass adds the object, wires reward + seg_pc, and sets reset params.

from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import torch
import isaaclab.utils.math as math_utils

import uwlab_assets.robots.franka_leap as franka_leap

from ... import grasp_env
from ...mdp import (
    ee_pose_w,
    reset_robot_joints,
    reset_camera_pose,
    set_fixed_camera_view,
    RenderedSegPC,
)
from isaaclab.sensors import CameraCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg

############################################
################ task info #################
############################################

#   right_reset_joint_pose (arm, 7D)
ARM_RESET = [
    3.1088299e-01, 4.0700440e-03, -3.1125304e-01, -2.0509737e+00,
    1.4107295e-03, 2.0548446e+00, 7.8060406e-01,
]

#   right_reset_hand_joint_pose (hand, 16D)
HAND_RESET = [
    0.35281801223754883, 0.6442744731903076, 0.29912877082824707, 0.34514832496643066,
    -0.03681302070617676, -0.06749272346496582, -0.09357023239135742, -0.14725971221923828,
    0.0659637451171875, 0.43411898612976074, 0.05982780456542969, 0.013808250427246094,
    0.03221607208251953, -0.009201288223266602, 0.029148101806640625, 0.0046045780181884766,
]

# PCD crop region from real world. Replaced max z with .67 instead of .7
# Because for this camera, the view angle doesn't allow for any points above .67
PCD_CROP_REGION_CL8384200N1 = [-0.10, -0.50, 0.035, 0.85, 0.50, 0.67]

# Number of points to sample from each mesh component for each part of the scene
ARM_NUM_POINTS = 256
HAND_NUM_POINTS = 64

# Valid values for env_cfg.run_mode.
EVAL_MODE = "eval_mode"
DISTILL_MODE = "distill_mode"
RL_MODE = "rl_mode"
RUN_MODES = (EVAL_MODE, DISTILL_MODE, RL_MODE)

@configclass
class FrankaLeapGraspSceneCfg(grasp_env.GraspSceneCfg):
    robot = franka_leap.IMPLICIT_FRANKA_LEAP.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class FrankaLeapGraspEnvCfg(grasp_env.GraspEnvCfg):
    scene: FrankaLeapGraspSceneCfg = FrankaLeapGraspSceneCfg(
        num_envs=1, env_spacing=2.5)
    num_warmup_steps: int = 10
    # Runtime mode: "rl_mode" | "distill_mode" | "eval_mode". "rl_mode" disables scene cameras and their reset events.
    run_mode: str = "eval_mode"
    pcd_crop_region: list[float] = PCD_CROP_REGION_CL8384200N1
    # When True in distill_mode, use fixed_camera for RenderedSegPC instead of train_camera.
    # Useful for overlay scripts where train_camera depth may fail (e.g. headless).
    distill_use_fixed_camera: bool = False
    # When set in distill_mode, use this exact camera for RenderedSegPC (e.g. "fixed_camera").
    # Overrides distill_use_fixed_camera. Use the same camera as capture for overlay scripts.
    distill_camera_name: str | None = None

    def warmup_action(self, env) -> torch.Tensor:
        """Hold at reset joint position — safe no-op for joint absolute control."""
        raise NotImplementedError("Error: warmup_action must be implemented in the subclass for FrankaLeapGraspEnvCfg")

    def __post_init__(self):
        super().__post_init__()
        # Wire EE pose obs with Franka-LEAP calibration values
        self.observations.policy.ee_pose.params["asset_cfg"] = SceneEntityCfg("robot")
        self.observations.policy.ee_pose.params["ee_body_name"] = franka_leap.FRANKA_LEAP_EE_BODY
        self.observations.policy.ee_pose.params["ee_offset"] = franka_leap.FRANKA_LEAP_EE_OFFSET

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
                "camera_name": "train_camera",
                "random_pose_range": (0.4, -0.15, 0.10, 0.6, 0.15, 0.25, 0.8, 1.7),
                "theta_range_rad": (0.0, 0.5),
                "phi_range_rad": (1.0, 1.66),
            },
        )
        # Fixed camera: set from eye + look_at on every reset (position relative to robot base).
        self.events.reset_fixed_camera = EventTerm(
            func=set_fixed_camera_view,
            mode="reset",
            params={
                "camera_name": "fixed_camera",
                "eye_offset": (1.4327373524611016, 0.2400519659762369, 0.6),
                "look_at_offset": (0.0, -0.15, 0.0),
            },
        )

@configclass
class FrankaLeapEmptySceneCfg(FrankaLeapGraspSceneCfg):
    # Must spawn a real prim: with spawn=None the asset only wraps existing prims, so the path would not exist.
    # Use a tiny cuboid so the scene has no visible object.

    # Empty env needs a spawned prim at GraspObject (spawn=None would cause "Could not find prim").
    # Use a tiny invisible cuboid so the scene has no visible object

    grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.11),
            rot=(0.707, 0.707, 0.0, 0.0),
        ),
        spawn=CuboidCfg(
            size=(0.001, 0.001, 0.001),  # 1mm cube, effectively invisible,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
    )

@configclass
class FrankaLeapEmptyGraspEnvCfg(FrankaLeapGraspEnvCfg):
    scene: FrankaLeapEmptySceneCfg = FrankaLeapEmptySceneCfg(
        num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object= None

@configclass
class GraspFrankaLeapJointAbsCfg(FrankaLeapEmptyGraspEnvCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        """Hold at reset joint position — safe no-op for joint absolute control."""
        reset = torch.tensor(ARM_RESET + HAND_RESET, device=env.device, dtype=torch.float32)
        return reset.unsqueeze(0).repeat(env.num_envs, 1)

@configclass
class GraspFrankaLeapIkRelCfg(FrankaLeapEmptyGraspEnvCfg):
    actions = franka_leap.FrankaLeapIkRelArmHandJointAction()

    def warmup_action(self, env) -> torch.Tensor:
        """Zero delta EE + zero hand — safe no-op for IK-relative control."""
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@configclass
class GraspFrankaLeapIkAbsCfg(FrankaLeapEmptyGraspEnvCfg):
    def warmup_action(self, env) -> torch.Tensor:
        """Hold current EE pose + current hand joints — safe no-op for IK-absolute control."""
        robot = env.scene["robot"]
        ee_body_idx = robot.body_names.index(franka_leap.FRANKA_LEAP_EE_BODY)
        ee_state = robot._data.body_state_w[:, ee_body_idx, :7].clone()
        ee_state[:, :3] -= env.scene.env_origins
        offset_pos = torch.tensor([list(franka_leap.FRANKA_LEAP_EE_OFFSET)], device=env.device).repeat(env.num_envs, 1)
        offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
        pos, quat = math_utils.combine_frame_transforms(ee_state[:, :3], ee_state[:, 3:7], offset_pos, offset_rot)
        hand_joints = robot.data.joint_pos[:, len(ARM_RESET):]  # hand joints after arm
        return torch.cat([pos, quat, hand_joints], dim=-1)


def _apply_distill_mode(cfg: FrankaLeapGraspEnvCfg) -> None:
    """Configure env for distill_mode: camera with depth, seg_pc from RenderedSegPC.
    Uses fixed_camera when distill_use_fixed_camera=True or distill_camera_name is set (e.g. overlay scripts)."""
    camera_name = getattr(cfg, "distill_camera_name", None)
    if camera_name is not None:
        if not hasattr(cfg.scene, camera_name) or getattr(cfg.scene, camera_name) is None:
            raise ValueError(f"distill_camera_name={camera_name!r} but scene has no such camera")
        cam_cfg = getattr(cfg.scene, camera_name)
    elif getattr(cfg, "distill_use_fixed_camera", False):
        cam_cfg = cfg.scene.fixed_camera
        if cam_cfg is None:
            raise ValueError("distill_use_fixed_camera requires fixed_camera; scene.fixed_camera is None")
        camera_name = "fixed_camera"
    else:
        cam_cfg = cfg.scene.train_camera
        if cam_cfg is None:
            raise ValueError("distill_mode requires train_camera; scene.train_camera is None")
        camera_name = "train_camera"

    focal_length = cam_cfg.spawn.focal_length
    horizontal_aperture = cam_cfg.spawn.horizontal_aperture
    rendered_pc = RenderedSegPC(
        camera_name=camera_name,
        depth_key="depth",
        pcd_crop_region=cfg.pcd_crop_region,
        num_downsample_points=2048,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        include_entity_names=("robot", "grasp_object"),
    )
    if not hasattr(cfg.observations.policy, "seg_pc") or cfg.observations.policy.seg_pc is None:
        raise ValueError(
            "distill_mode requires a task that defines seg_pc (e.g. GraspPinkCup, PourBottle)"
        )
    cfg.observations.policy.seg_pc = ObsTerm(func=rendered_pc.get_seg_pc)


def _apply_rl_mode(cfg: FrankaLeapGraspEnvCfg) -> None:
    """Configure env for rl_mode: disable scene cameras and their reset events."""
    cfg.scene.train_camera = None
    cfg.scene.fixed_camera = None
    if hasattr(cfg.events, "reset_camera"):
        cfg.events.reset_camera = None
    if hasattr(cfg.events, "reset_fixed_camera"):
        cfg.events.reset_fixed_camera = None


def parse_franka_leap_env_cfg(task: str, run_mode: str, **kwargs) -> FrankaLeapGraspEnvCfg:
    """Parse env config for Franka-LEAP grasp tasks. 
    
    Args:
        task: The name of the environment.
        run_mode: The runtime mode of the environment (RL_MODE, DISTILL_MODE, EVAL_MODE).
        **kwargs: Additional arguments to pass to parse_env_cfg.

    Returns:
        The parsed environment configuration.
    run_mode is required (RL_MODE, DISTILL_MODE, EVAL_MODE)."""

    assert run_mode in RUN_MODES, f"run_mode must be one of {RUN_MODES}, got {run_mode!r}"
    env_cfg = parse_env_cfg(task, **kwargs)
    env_cfg.run_mode = run_mode
    return env_cfg

class FrankaLeapGraspEnv(ManagerBasedRLEnv):
    """Runtime RL environment for Franka-LEAP grasp tasks with built-in warmup on reset.

    After the standard ManagerBasedRLEnv reset (which runs all reset EventTerms),
    this env optionally executes `num_warmup_steps` steps using the task's
    `warmup_action(env)` to allow sim/observations to settle before the first
    observation is returned to the caller.

    Runtime options: set env_cfg.run_mode to EVAL_MODE | DISTILL_MODE | RL_MODE.
    "rl_mode" disables scene cameras (train_camera, fixed_camera) and their reset events.
    """

    def __init__(self, cfg: FrankaLeapGraspEnvCfg, **kwargs):
        # apply runtime variables to the environment given the mode (eval, rl, or distill)
        run_mode = cfg.run_mode
        self._apply_run_mode(cfg, run_mode) 
        super().__init__(cfg, **kwargs)

    def _apply_run_mode(self, cfg: FrankaLeapGraspEnvCfg, run_mode: str) -> None:
        """Apply run_mode to env config. rl_mode disables scene cameras. distill_mode uses depth-rendered seg_pc."""
        assert run_mode in RUN_MODES, f"run_mode must be one of {RUN_MODES}, got {run_mode!r}"
        if run_mode == RL_MODE:
            _apply_rl_mode(cfg)
        elif run_mode == DISTILL_MODE:
            _apply_distill_mode(cfg)

    def reset(self, *args, **kwargs):
        # Run the standard reset behavior (events, managers, etc.).
        obs, info = super().reset(*args, **kwargs)

        # If the config doesn't define a warmup_action, just return immediately.
        cfg = self.cfg
        warmup_fn = cfg.warmup_action

        # Compute the warmup action once and reuse it for all warmup steps.
        warmup_action = warmup_fn(self)

        for _ in range(cfg.num_warmup_steps):
            # Use the parent step implementation so that all managers and counters
            # are updated exactly as in normal interaction.
            obs, _, terminated, truncated, info = super().step(warmup_action)


        return obs, info