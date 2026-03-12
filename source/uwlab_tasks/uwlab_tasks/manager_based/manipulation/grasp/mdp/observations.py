# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
import trimesh
import numpy as np

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from uwlab.utils.math import crop_points_to_bounds, fps_points

# For SamplePC: sample from USD (same geometry as sim)
from isaaclab.sim.utils import get_first_matching_child_prim
from pxr import UsdPhysics

from ...reset_states.mdp import utils as reset_states_utils

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False


def _benchmark_sync(env):
    device = env.device
    if (hasattr(device, "type") and device.type == "cuda") or device == "cuda":
        torch.cuda.synchronize()


##
# EE pose observation
# Exact port of config_rigids.py:_get_ee_pose_link8 (lines 210-223)
##

def ee_pose_w(env, asset_cfg: SceneEntityCfg, ee_body_name: str, ee_offset: tuple):
    robot = env.scene[asset_cfg.name]
    ee_body_idx = robot.body_names.index(ee_body_name)
    ee_pose = robot._data.body_state_w[:, ee_body_idx, :7].clone()
    ee_pose[:, :3] -= env.scene.env_origins
    device = ee_pose.device
    num_envs = ee_pose.shape[0]
    offset_pos = torch.tensor([list(ee_offset)], device=device).repeat(num_envs, 1)
    offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
    pos, quat = math_utils.combine_frame_transforms(
        ee_pose[:, :3], ee_pose[:, 3:7], offset_pos, offset_rot)
    return torch.cat([pos, quat], dim=-1)


##
# Joint position observation
##

def joint_pos_w(env, asset_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos


def arm_joint_pos_w(env, asset_cfg: SceneEntityCfg, num_arm_joints: int = 7):
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, :num_arm_joints]


def hand_joint_pos_w(env, asset_cfg: SceneEntityCfg, num_arm_joints: int = 7):
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, num_arm_joints:]



class CachedSamplePC:
    """Point cloud from USD geometry, sampled once and cached for fast subsequent transforms.

    Combines the accuracy of SamplePC (uses USD colliders, same geometry as sim) with the
    speed of SynthesizePC (caches points on first run, only transforms on later steps).
    Requires only the USD asset/object prim paths—no external mesh files.
    """

    def __init__(
        self,
        asset_name: str,
        object_names: list[str],
        pcd_crop_region: list[float, float, float, float, float, float] | None = None,
        num_arm_pcd: int = 64,
        num_hand_pcd: int = 64,
        num_object_pcd: list[int] = [512],
        num_downsample_points: int = 2048,
        object_prim_path_patterns: list[str] | None = None,
    ):
        if object_prim_path_patterns is not None and len(object_prim_path_patterns) != len(object_names):
            raise ValueError(
                f"object_prim_path_patterns must have length {len(object_names)} when provided, "
                f"got {len(object_prim_path_patterns)}"
            )
        self.asset_name = asset_name
        self.object_names = list(object_names)
        self.num_arm_pcd = num_arm_pcd
        self.num_hand_pcd = num_hand_pcd
        self.num_object_pcd = list(num_object_pcd)
        self.num_downsample_points = num_downsample_points
        self._object_prim_path_patterns = (
            list(object_prim_path_patterns) if object_prim_path_patterns is not None else [None] * len(object_names)
        )
        self.mesh_init = False
        self._printed_msg = False
        self.pcd_crop_region = pcd_crop_region

    def get_seg_pc(self, env):
        if not self.mesh_init:
            self.init_mesh(env)

        do_benchmark = False
        timings = {}

        robot_link_state = env.scene[self.asset_name]._data.body_pose_w.clone()
        robot_link_state[:, :, :3] -= env.scene.env_origins.unsqueeze(1).repeat_interleave(
            robot_link_state.shape[1], dim=1)

        arm_link_state = robot_link_state[:, : len(self.arm_names)]
        hand_link_state = robot_link_state[:, len(self.arm_names) : len(self.arm_names) + len(self._hand_prim_patterns)]

        if do_benchmark:
            _benchmark_sync(env)
            t0 = time.perf_counter()

        arm_vertices = math_utils.transform_points(
            self._arm_mesh.reshape(-1, self._arm_mesh.shape[-2], 3),
            arm_link_state[..., :3].reshape(-1, 3),
            arm_link_state[..., 3:7].reshape(-1, 4))
        arm_vertices = arm_vertices.reshape(env.num_envs, -1, 3)

        if do_benchmark:
            _benchmark_sync(env)
            timings["arm_transform_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        hand_vertices = math_utils.transform_points(
            self._hand_mesh.reshape(-1, self._hand_mesh.shape[-2], 3),
            hand_link_state[..., :3].reshape(-1, 3),
            hand_link_state[..., 3:7].reshape(-1, 4))
        hand_vertices = hand_vertices.reshape(env.num_envs, -1, 3)

        if do_benchmark:
            _benchmark_sync(env)
            timings["hand_transform_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        object_parts = []
        for k in range(len(self.object_names)):
            object_state = env.scene[self.object_names[k]]._data.root_pose_w.clone()
            object_state[:, :3] -= env.scene.env_origins
            object_vertices = math_utils.transform_points(
                self._object_meshes[k], object_state[..., :3], object_state[..., 3:7])
            object_parts.append(object_vertices)

        if do_benchmark:
            _benchmark_sync(env)
            timings["object_transform_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        object_vertices_all = torch.cat(object_parts, dim=1)
        all_pcd = torch.cat([arm_vertices, hand_vertices, object_vertices_all], dim=1)
        points_index = torch.randperm(all_pcd.shape[1]).to(env.device)
        sampled_pcd = all_pcd[:, points_index[: self.num_downsample_points * 10]]

        if do_benchmark:
            _benchmark_sync(env)
            timings["subsample_pcd_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        if self.pcd_crop_region is not None:
            sampled_pcd = crop_points_to_bounds(sampled_pcd, self.pcd_crop_region)

        if do_benchmark:
            _benchmark_sync(env)
            timings["crop_pcd_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()

        sampled_pcd = fps_points(sampled_pcd[None], self.num_downsample_points)

        if do_benchmark:
            _benchmark_sync(env)
            timings["fps_pcd_ms"] = (time.perf_counter() - t0) * 1000
            timings["total_ms"] = sum(timings.values())
            print("[INFO] CachedSamplePC get_seg_pc timings (ms):", timings)

        if not self._printed_msg:
            print("[INFO] Using cached sampled pointcloud (arm/hand/object from USD)")
            self._printed_msg = True

        return sampled_pcd.permute(0, 2, 1)

    def init_mesh(self, env):
        self.num_envs = env.num_envs
        robot_asset = env.scene[self.asset_name]
        self.arm_names = robot_asset.body_names[:8]
        hand_names_all = robot_asset.body_names[8:]
        robot_prim_env0 = robot_asset.cfg.prim_path.replace(".*", "0", 1)

        self._arm_prim_patterns = []
        for name in self.arm_names:
            prim = get_first_matching_child_prim(
                robot_prim_env0,
                predicate=lambda p, bn=name: p.GetName() == bn and p.HasAPI(UsdPhysics.RigidBodyAPI),
            )
            if prim is not None:
                self._arm_prim_patterns.append(str(prim.GetPath()).replace("env_0", "env_.*", 1))
            else:
                self._arm_prim_patterns.append(None)

        self._hand_prim_patterns = []
        for link_name in hand_names_all:
            if "sensor" in link_name.lower():
                print(f"CachedSamplePC: skipping sensor link {link_name} (no mesh)")
                self._hand_prim_patterns.append(None)
                continue
            prim = get_first_matching_child_prim(
                robot_prim_env0,
                predicate=lambda p, bn=link_name: p.GetName() == bn and p.HasAPI(UsdPhysics.RigidBodyAPI),
            )
            if prim is not None:
                self._hand_prim_patterns.append(str(prim.GetPath()).replace("env_0", "env_.*", 1))
            else:
                self._hand_prim_patterns.append(None)

        for k, name in enumerate(self.object_names):
            if self._object_prim_path_patterns[k] is None:
                self._object_prim_path_patterns[k] = env.scene[name].cfg.prim_path

        self._arm_mesh = torch.zeros(
            (env.num_envs, len(self.arm_names), self.num_arm_pcd, 3),
            device=env.device, dtype=torch.float32)
        for i, pattern in enumerate(self._arm_prim_patterns):
            if pattern is None:
                continue
            pc = reset_states_utils.sample_object_point_cloud(
                num_envs=env.num_envs,
                num_points=self.num_arm_pcd,
                prim_path_pattern=pattern,
                device=env.device,
            )
            if pc is not None:
                self._arm_mesh[:, i] = pc

        self._hand_mesh = torch.zeros(
            (env.num_envs, len(self._hand_prim_patterns), self.num_hand_pcd, 3),
            device=env.device, dtype=torch.float32)
        for j, pattern in enumerate(self._hand_prim_patterns):
            if pattern is None:
                continue
            pc = reset_states_utils.sample_object_point_cloud(
                num_envs=env.num_envs,
                num_points=self.num_hand_pcd,
                prim_path_pattern=pattern,
                device=env.device,
            )
            if pc is not None:
                self._hand_mesh[:, j] = pc

        self._object_meshes = []
        for k in range(len(self.object_names)):
            pc = reset_states_utils.sample_object_point_cloud(
                num_envs=env.num_envs,
                num_points=self.num_object_pcd[k],
                prim_path_pattern=self._object_prim_path_patterns[k],
                device=env.device,
            )
            self._object_meshes.append(pc)

        self.mesh_init = True

