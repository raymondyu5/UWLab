# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.franka_leap as franka_leap

from ....mdp import CachedSamplePC, reset_object_pose, reset_articulation_joint_state
from .rewards.screw_rewards import ScrewReward
from .. import grasp_franka_leap
from ..grasp_franka_leap import ARM_RESET, HAND_RESET, ARM_NUM_POINTS, HAND_NUM_POINTS
from .shared_params import FINGERS_NAME_LIST

SCREW_LAMP_HORIZON = 256
SCREW_LAMP_OBJECT_NUM_POINTS = 128

SCREW_LAMP_USD = "/workspace/uwlab/assets/screw_lamp_tall_base/screw_lamp.usd"
SCREW_LAMP_SPAWN_POS = (0.545, 0.13, 0.0)  # z ignored by reset_object_pose; use reset_height instead
SCREW_LAMP_SPAWN_ROT = (1.0, 0.0, 0.0, 0.0)
SCREW_LAMP_TABLE_BLOCK_Z = 0.02  # calibrated: lamp base settles at ~0.05 with this block z


class AngleCounterCommand(CommandTerm):
    """Tracks per-env joint position, per-step delta, and cumulative rotation of the screw joint."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.object_name = cfg.object_name
        self.last_angle = None
        self.delta = None
        self.sum_angle = None
        self._prev_step = None

    def _init_buffers(self):
        cur = self._env.scene[self.object_name]._data.joint_pos
        self.last_angle = cur.clone()
        self.delta = torch.zeros_like(cur)
        self.sum_angle = torch.zeros_like(cur)
        self._prev_step = self._env.unwrapped.episode_length_buf.clone()

    @property
    def command(self) -> torch.Tensor:
        cur = self._env.scene[self.object_name]._data.joint_pos  # (num_envs, n_joints)
        buf = self._env.unwrapped.episode_length_buf

        if self.last_angle is None:
            self._init_buffers()
            return torch.cat([cur, self.delta, self.sum_angle], dim=-1)

        new_step = buf > self._prev_step
        if new_step.any():
            delta = torch.atan2(
                torch.sin(self.last_angle - cur),
                torch.cos(self.last_angle - cur),
            )
            self.delta[new_step] = delta[new_step]
            self.sum_angle[new_step] += self.delta[new_step]
            self.last_angle[new_step] = cur[new_step]
            self._prev_step[new_step] = buf[new_step]

        return torch.cat([cur, self.delta, self.sum_angle], dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        if self.last_angle is None:
            return
        cur = self._env.scene[self.object_name]._data.joint_pos
        self.last_angle[env_ids] = cur[env_ids].clone()
        self.delta[env_ids] = 0.0
        self.sum_angle[env_ids] = 0.0
        self._prev_step[env_ids] = -1

    def _update_metrics(self): pass
    def _update_command(self): pass
    def _set_debug_vis_impl(self, debug_vis: bool): pass
    def _debug_vis_callback(self, event): pass


@configclass
class AngleCounterCfg(CommandTermCfg):
    class_type: type = AngleCounterCommand
    object_name: str = "screw_lamp"

    def __post_init__(self):
        self.resampling_time_range = (1e6, 1e6)


@configclass
class ScrewCommandsCfg:
    angle_counter: AngleCounterCfg = AngleCounterCfg(object_name="screw_lamp")


@configclass
class ScrewLightbulbSceneCfg(grasp_franka_leap.FrankaLeapGraspSceneCfg):
    grasp_object = None  # satisfies MISSING from base; not used in this task

    screw_lamp = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ScrewLamp",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=SCREW_LAMP_SPAWN_POS,
            rot=SCREW_LAMP_SPAWN_ROT,
            joint_pos={".*": 0.0},
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCREW_LAMP_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
        ),
        actuators={
            "screw_lamp": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=200.0,
                velocity_limit=300.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.1,
            ),
        },
    )


@configclass
class ScrewLightbulbFrankaLeapCfg(grasp_franka_leap.FrankaLeapGraspEnvCfg):
    scene: ScrewLightbulbSceneCfg = ScrewLightbulbSceneCfg(num_envs=1, env_spacing=2.5)
    commands: ScrewCommandsCfg = ScrewCommandsCfg()

    def is_success(self, env) -> torch.Tensor:
        angle_counter = env.command_manager.get_command("angle_counter")
        return (angle_counter[..., -1].reshape(-1) > 2 * math.pi).float()

    def __post_init__(self):
        super().__post_init__()

        self.eval_object_name = "screw_lamp"
        self.events.reset_object = None

        self.setup_horizon(horizon=SCREW_LAMP_HORIZON)

        # angle_counter is already declared as a class field in ScrewCommandsCfg

        screw_rew = ScrewReward(
            asset_name="robot",
            object_name="screw_lamp",
            fingers_name_list=FINGERS_NAME_LIST,
        )
        screw_rew.setup_wrist_sensor(self.scene)
        screw_rew.setup_finger_entities(self.scene)
        screw_rew.setup_finger_sensors(self.scene, object_prim_name="ScrewLamp")

        self.rewards.rotation    = RewTerm(func=screw_rew.rew_rotation,           weight=1.0)
        self.rewards.contact     = RewTerm(func=screw_rew.rew_contact,           weight=0.0)

        self.rewards.proximity   = RewTerm(func=screw_rew.rew_proximity,         weight=0.1)
        self.rewards.wrist       = RewTerm(func=screw_rew.rew_wrist_penalty,  weight=0.0)
        self.rewards.joint_vel        = RewTerm(func=screw_rew.rew_joint_vel,         weight=-1e-3)
        self.rewards.action_rate      = RewTerm(func=screw_rew.rew_action_rate,       weight=-0.1)
        self.rewards.arm_action_rate  = RewTerm(func=screw_rew.rew_arm_action_rate,   weight=-5.0)

        self.rewards.joint_pos_limits        = RewTerm(func=screw_rew.rew_joint_pos_limits,      weight=-100.0)
        self.rewards.base_stability         = RewTerm(func=screw_rew.rew_base_stability,        weight=0.0)
        self._screw_rew = screw_rew
        self.metrics_spec = {
            "is_success": screw_rew.rew_is_success,
            "success_2pi": screw_rew.metric_success_2pi,
            "cumulative_rotation": screw_rew.metric_cumulative_rotation,
        }

        self.observations.policy.object_pose   = ObsTerm(func=screw_rew.obs_object_pose)
        self.observations.policy.rotate_angle  = ObsTerm(func=screw_rew.obs_rotate_angle)
        self.observations.policy.contact_obs   = ObsTerm(func=screw_rew.obs_contact)
        self.observations.policy.object_in_tip = ObsTerm(func=screw_rew.obs_object_in_tip)

        synth_pc = CachedSamplePC(
            asset_name="robot",
            object_names=["screw_lamp"],
            num_arm_pcd=ARM_NUM_POINTS,
            num_hand_pcd=HAND_NUM_POINTS,
            num_object_pcd=[SCREW_LAMP_OBJECT_NUM_POINTS],
            num_downsample_points=2048,
            pcd_crop_region=self.pcd_crop_region,
            pcd_noise=0.02,
        )
        self.observations.policy.seg_pc = ObsTerm(func=synth_pc.get_seg_pc)

        self.events.reset_lamp_joints = EventTerm(
            func=reset_articulation_joint_state,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("screw_lamp"),
                "joint_pos": 0.0,
            },
        )

        self.events.reset_lamp_pose = EventTerm(
            func=reset_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("screw_lamp"),
                "default_pos": SCREW_LAMP_SPAWN_POS,
                "default_rot_quat": SCREW_LAMP_SPAWN_ROT,
                "pose_range": {
                    "x": (-0.02, 0.02),
                    "y": (-0.03, 0.03),
                    "z": (-0.01, 0.02),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "reset_height": 0.0,
                "table_block_name": None,
            },
        )

        self.events.randomize_bulb_mass = EventTerm(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="reset",
            min_step_count_between_reset=800,
            params={
                "asset_cfg": SceneEntityCfg("screw_lamp", body_names="body"),
                "mass_distribution_params": (0.5, 1.5),
                "operation": "abs",
                "distribution": "uniform",
            },
        )

        self.events.randomize_lamp_friction = EventTerm(
            func=isaac_mdp.randomize_joint_parameters,
            min_step_count_between_reset=200,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("screw_lamp"),
                "friction_distribution_params": (0.05, 0.2),
                "operation": "add",
                "distribution": "uniform",
            },
        )

        self.events.set_lamp_physics_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("screw_lamp", body_names="body"),
                "static_friction_range": (0.3, 0.5),
                "dynamic_friction_range": (0.3, 0.5),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )


@configclass
class ScrewLightbulbFrankaLeapJointAbsCfg(ScrewLightbulbFrankaLeapCfg):
    actions = franka_leap.FrankaLeapJointPositionAction()

    def warmup_action(self, env) -> torch.Tensor:
        return env.scene["robot"].data.joint_pos.clone()


@configclass
class ScrewLightbulbFrankaLeapHighFrictionJointAbsCfg(ScrewLightbulbFrankaLeapJointAbsCfg):
    """Variant with higher contact friction (0.5, 0.8) — easier grip with less squeezing force."""

    def __post_init__(self):
        super().__post_init__()

        self.events.set_lamp_physics_material.params["static_friction_range"] = (0.5, 0.8)
        self.events.set_lamp_physics_material.params["dynamic_friction_range"] = (0.5, 0.8)


@configclass
class ScrewLightbulbFrankaLeapLightBulbJointAbsCfg(ScrewLightbulbFrankaLeapJointAbsCfg):
    """Variant with lighter bulb mass (0.5 kg) to encourage gentler grasping forces."""

    def __post_init__(self):
        super().__post_init__()

        self.events.set_bulb_mass = EventTerm(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("screw_lamp", body_names="body"),
                "mass_distribution_params": (0.5, 0.5),
                "operation": "abs",
            },
        )


@configclass
class ScrewLightbulbFrankaLeapUnfixedJointAbsCfg(ScrewLightbulbFrankaLeapJointAbsCfg):
    """Variant with fix_root_link=False + table_block support + base stability penalty.

    Joint friction increased to 2.0 to prevent gravity-induced screw unwinding.
    Tune UNFIXED_JOINT_FRICTION if bulb still sinks under gravity (increase)
    or robot can't rotate it (decrease).
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.screw_lamp.spawn.articulation_props.fix_root_link = False
        self.scene.screw_lamp.actuators["screw_lamp"].friction = 2.0

        setattr(self.scene, "table_block", RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TableBlock",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(SCREW_LAMP_SPAWN_POS[0], SCREW_LAMP_SPAWN_POS[1], SCREW_LAMP_TABLE_BLOCK_Z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path="/workspace/uwlab/assets/table/table_block.usd",
                scale=(1.2, 1.0, 0.10),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=False,
                ),
            ),
        ))

        self.rewards.gravity_compensation = RewTerm(func=self._screw_rew.apply_gravity_compensation, weight=1.0)
        self.rewards.base_stability.weight = -5.0


@configclass
class ScrewLightbulbFrankaLeapTallBigBaseJointAbsCfg(ScrewLightbulbFrankaLeapJointAbsCfg):
    """Variant using the tall_big_base lightbulb asset."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.screw_lamp.spawn.usd_path = "/workspace/uwlab/assets/screw_lamp_tall_big_base/screw_lamp.usd"


@configclass
class ScrewLightbulbFrankaLeapRealLightbulbJointAbsCfg(ScrewLightbulbFrankaLeapJointAbsCfg):
    """Variant using the real_lightbulb USD (physics baked in, Blender export).

    Body names differ from screw_lamp_new: base=ObjectCapture, bulb=ObjectCapture_001.
    Joint axis=Y in local frame but base has 90deg X rotation, so world-space rotation is Z (correct).
    BULB_Z_OFFSET overridden to 0.115 (bulb translate Z from root).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.screw_lamp.spawn.usd_path = "/workspace/uwlab/assets/real_lightbulb/real_lightbulb.usd"

        # 90deg X rotation baked in USD (Blender Y-up -> Z-up) — must match spawn rot and reset rot
        _rot_90x = (0.7071, 0.7071, 0.0, 0.0)  # (w, x, y, z)
        self.scene.screw_lamp.init_state.rot = _rot_90x
        self.events.reset_lamp_pose.params["default_rot_quat"] = _rot_90x

        # Fix contact sensor filters: bulb body is ObjectCapture_001, not body
        for link_name in FINGERS_NAME_LIST:
            getattr(self.scene, f"{link_name}_contact").filter_prim_paths_expr = [
                "{ENV_REGEX_NS}/ScrewLamp/ObjectCapture_001"
            ]

        # Fix mass randomization and material body name
        self.events.randomize_bulb_mass.params["asset_cfg"] = SceneEntityCfg(
            "screw_lamp", body_names="ObjectCapture_001"
        )
        self.events.set_lamp_physics_material.params["asset_cfg"] = SceneEntityCfg(
            "screw_lamp", body_names="ObjectCapture_001"
        )

        # Fix bulb Z offset at instance level (bulb translate Z=0.115 from root)
        self._screw_rew._bulb_z_offset = 0.085


@configclass
class ScrewLightbulbFrankaLeapHighContactJointAbsCfg(ScrewLightbulbFrankaLeapJointAbsCfg):
    """Variant with higher contact reward weight (0.5) to encourage grasping."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.contact.weight = 0.5


@configclass
class ScrewLightbulbFrankaLeapCurriculumJointAbsCfg(ScrewLightbulbFrankaLeapJointAbsCfg):
    """Variant with curriculum-scheduled arm action rate and joint velocity penalties.

    Both penalties ramp linearly from their initial values up to a cap over 200M agent steps:
      - arm_action_rate: scale 5 -> 200 (replaces fixed weight=-5.0)
      - joint_vel:       scale 0.001 -> 0.025 (replaces fixed weight=-1e-3)
    """

    def __post_init__(self):
        super().__post_init__()

        self.rewards.arm_action_rate = RewTerm(func=self._screw_rew.rew_arm_action_rate_curriculum, weight=-1.0)
        self.rewards.joint_vel = RewTerm(func=self._screw_rew.rew_joint_vel_curriculum, weight=-1.0)
