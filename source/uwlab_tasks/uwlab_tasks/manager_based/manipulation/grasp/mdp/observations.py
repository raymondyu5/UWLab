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
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth

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
    ``object_names`` may be empty (arm + hand only).
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
        pcd_noise: float = 0.0,
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
        self.pcd_noise = pcd_noise
        # Set to True to enable per-call CUDA-event timing (printed every call).
        self.do_benchmark = False
        self._last_timing: dict = {}

    def get_seg_pc(self, env):
        if not self.mesh_init:
            self.init_mesh(env)

        def _evt():
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e

        if self.do_benchmark:
            e0 = _evt()

        robot_link_state = env.scene[self.asset_name]._data.body_pose_w.clone()
        robot_link_state[:, :, :3] -= env.scene.env_origins.unsqueeze(1).repeat_interleave(
            robot_link_state.shape[1], dim=1)

        arm_link_state = robot_link_state[:, : len(self.arm_names)]
        hand_link_state = robot_link_state[:, len(self.arm_names) : len(self.arm_names) + len(self._hand_prim_patterns)]

        arm_vertices = math_utils.transform_points(
            self._arm_mesh.reshape(-1, self._arm_mesh.shape[-2], 3),
            arm_link_state[..., :3].reshape(-1, 3),
            arm_link_state[..., 3:7].reshape(-1, 4))
        arm_vertices = arm_vertices.reshape(env.num_envs, -1, 3)

        if self.do_benchmark:
            e1 = _evt()

        hand_vertices = math_utils.transform_points(
            self._hand_mesh.reshape(-1, self._hand_mesh.shape[-2], 3),
            hand_link_state[..., :3].reshape(-1, 3),
            hand_link_state[..., 3:7].reshape(-1, 4))
        hand_vertices = hand_vertices.reshape(env.num_envs, -1, 3)

        if self.do_benchmark:
            e2 = _evt()

        object_parts = []
        for k in range(len(self.object_names)):
            if self._object_is_articulation[k]:
                obj = env.scene[self.object_names[k]]
                body_pos_w = obj._data.body_pos_w.clone()  # (B, num_bodies, 3)
                body_pos_w -= env.scene.env_origins.unsqueeze(1)
                body_quat_w = obj._data.body_quat_w.clone()  # (B, num_bodies, 4)
                # Root pose is well-defined regardless of fix_root_link body indexing.
                root_pos = obj._data.root_pos_w.clone() - env.scene.env_origins
                root_quat = obj._data.root_quat_w.clone()
                body_parts = []
                for local_idx, body_mesh in enumerate(self._object_meshes[k]):
                    if body_mesh is None:
                        continue
                    if local_idx == 0:
                        # whole-prim mesh: use root_pos_w so the base is always placed correctly
                        verts = math_utils.transform_points(body_mesh, root_pos, root_quat)
                    else:
                        # non-root body: use that body's own pose (local_idx maps to body index)
                        body_idx = min(local_idx, body_pos_w.shape[1] - 1)
                        verts = math_utils.transform_points(
                            body_mesh,
                            body_pos_w[:, body_idx, :],
                            body_quat_w[:, body_idx, :],
                        )
                    body_parts.append(verts)
                if body_parts:
                    object_parts.append(torch.cat(body_parts, dim=1))
            else:
                object_state = env.scene[self.object_names[k]]._data.root_pose_w.clone()
                object_state[:, :3] -= env.scene.env_origins
                object_vertices = math_utils.transform_points(
                    self._object_meshes[k], object_state[..., :3], object_state[..., 3:7])
                object_parts.append(object_vertices)

        if self.do_benchmark:
            e3 = _evt()

        if object_parts:
            object_vertices_all = torch.cat(object_parts, dim=1)
        else:
            object_vertices_all = torch.empty(
                env.num_envs, 0, 3, device=env.device, dtype=arm_vertices.dtype
            )
        all_pcd = torch.cat([arm_vertices, hand_vertices, object_vertices_all], dim=1)
        points_index = torch.randperm(all_pcd.shape[1]).to(env.device)
        sampled_pcd = all_pcd[:, points_index[: self.num_downsample_points * 10]]

        if self.do_benchmark:
            e4 = _evt()

        if self.pcd_crop_region is not None:
            sampled_pcd = crop_points_to_bounds(sampled_pcd, self.pcd_crop_region)

        if self.do_benchmark:
            e5 = _evt()

        perm = torch.randperm(sampled_pcd.shape[1], device=env.device)[:self.num_downsample_points]
        sampled_pcd = sampled_pcd[:, perm]

        if self.pcd_noise > 0:
            sampled_pcd = sampled_pcd + (torch.rand_like(sampled_pcd) * 2 - 1) * self.pcd_noise

        if self.do_benchmark:
            e6 = _evt()
            torch.cuda.synchronize()
            self._last_timing = {
                "arm_transform_ms":    e0.elapsed_time(e1),
                "hand_transform_ms":   e1.elapsed_time(e2),
                "obj_transform_ms":    e2.elapsed_time(e3),
                "subsample_ms":        e3.elapsed_time(e4),
                "crop_ms":             e4.elapsed_time(e5),
                "randperm_ms":         e5.elapsed_time(e6),
                "total_ms":            e0.elapsed_time(e6),
            }
            print("[PROFILE] CachedSamplePC.get_seg_pc (ms):", {k: f"{v:.2f}" for k, v in self._last_timing.items()})

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

        from isaaclab.assets import Articulation as IsaacArticulation
        self._object_is_articulation = []
        for k, name in enumerate(self.object_names):
            obj = env.scene[name]
            is_art = isinstance(obj, IsaacArticulation)
            self._object_is_articulation.append(is_art)
            if self._object_prim_path_patterns[k] is None:
                self._object_prim_path_patterns[k] = obj.cfg.prim_path

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
        for k, name in enumerate(self.object_names):
            if self._object_is_articulation[k]:
                obj = env.scene[name]
                art_prim_env0 = obj.cfg.prim_path.replace(".*", "0", 1)
                print(f"CachedSamplePC: articulation '{name}' body_names={obj.body_names}, num_bodies={len(obj.body_names)}")
                print(f"CachedSamplePC: whole-prim path pattern = '{self._object_prim_path_patterns[k]}'")
                # Body 0 (base/root): sample the whole articulation prim — guaranteed to find
                # all colliders including the base, which often can't be found by name alone.
                whole_pc = reset_states_utils.sample_object_point_cloud(
                    num_envs=env.num_envs,
                    num_points=self.num_object_pcd[k],
                    prim_path_pattern=self._object_prim_path_patterns[k],
                    device=env.device,
                )
                print(f"CachedSamplePC: whole_pc={'None (base will have no points!)' if whole_pc is None else f'shape {whole_pc.shape}'}")
                # Non-root bodies: try per-body sampling so they track their own pose.
                per_body_meshes = [whole_pc]  # index 0 = whole prim, transformed by body 0 pose
                for body_name in obj.body_names[1:]:
                    body_prim = get_first_matching_child_prim(
                        art_prim_env0,
                        predicate=lambda p, bn=body_name: p.GetName() == bn and p.HasAPI(UsdPhysics.RigidBodyAPI),
                    )
                    if body_prim is None:
                        print(f"CachedSamplePC: no prim found for articulation body '{body_name}', skipping")
                        per_body_meshes.append(None)
                        continue
                    body_pattern = str(body_prim.GetPath()).replace("env_0", "env_.*", 1)
                    pc = reset_states_utils.sample_object_point_cloud(
                        num_envs=env.num_envs,
                        num_points=self.num_object_pcd[k],
                        prim_path_pattern=body_pattern,
                        device=env.device,
                    )
                    per_body_meshes.append(pc)
                    if pc is not None:
                        print(f"CachedSamplePC: sampled {self.num_object_pcd[k]} pts from articulation body '{body_name}'")
                    else:
                        print(f"CachedSamplePC: no collider found for articulation body '{body_name}', skipping")
                self._object_meshes.append(per_body_meshes)
            else:
                pc = reset_states_utils.sample_object_point_cloud(
                    num_envs=env.num_envs,
                    num_points=self.num_object_pcd[k],
                    prim_path_pattern=self._object_prim_path_patterns[k],
                    device=env.device,
                )
                self._object_meshes.append(pc)

        self.mesh_init = True


MIN_POINTS_FOR_DOWNSAMPLE = 10


def _resolve_prim_path_for_env(prim_path_pattern: str, env_index: int) -> str:
    """Resolve {ENV_REGEX_NS} in prim_path to actual path for env_index."""
    suffix = prim_path_pattern.split("{ENV_REGEX_NS}", 1)[-1].strip("/")
    return f"/World/envs/env_{env_index}/{suffix}"


def _get_instance_id_to_labels(camera) -> dict:
    """Extract idToLabels from camera instance segmentation info."""
    info = camera.data.info
    if info is None:
        return {}
    if isinstance(info, dict):
        seg = info.get("instance_id_segmentation_fast", {})
    else:
        seg = {}
        for d in info:
            if isinstance(d, dict) and "instance_id_segmentation_fast" in d:
                seg = d["instance_id_segmentation_fast"]
                break
    return seg.get("idToLabels", {}) if isinstance(seg, dict) else {}


class RenderedSegPC:
    """Point cloud from depth camera unprojection. Used in distill_mode for policy distillation.

    Renders the scene from the train_camera (with depth) and unprojects to 3D points.
    Output format matches CachedSamplePC: (B, 3, num_downsample_points).

    When include_entity_names is set, depth is masked by instance segmentation so only
    pixels belonging to those scene entities (e.g. robot, grasp_object) are unprojected.
    This excludes ground plane, table, and other background geometry.
    """

    def __init__(
        self,
        camera_name: str,
        depth_key: str,
        pcd_crop_region: list[float] | None,
        num_downsample_points: int,
        focal_length: float,
        horizontal_aperture: float,
        include_entity_names: tuple[str, ...] | None = None,
        pcd_noise: float = 0.0,
    ):
        self.camera_name = camera_name
        self.depth_key = depth_key
        self.pcd_crop_region = pcd_crop_region
        self.num_downsample_points = num_downsample_points
        self.focal_length = focal_length
        self.horizontal_aperture = horizontal_aperture
        self.include_entity_names = include_entity_names or ()
        self.pcd_noise = pcd_noise
        self._printed_msg = False

    def get_seg_pc(self, env):
        camera = env.scene[self.camera_name]
        device = env.device
        B = env.num_envs

        if hasattr(env, "sim") and hasattr(env.sim, "render"):
            env.sim.render()
        if hasattr(camera, "_update_poses"):
            camera._update_poses(torch.arange(B, device=device))

        depth = camera.data.output[self.depth_key].clone()
        depth = depth.unsqueeze(0) if depth.dim() == 3 else depth
        depth = depth.unsqueeze(-1) if depth.shape[-1] != 1 else depth

        if self.include_entity_names:
            self._mask_depth_by_instance(env, camera, depth, B, device)

        cam_data = camera.data
        pos_w = getattr(cam_data, "pos_w", getattr(cam_data, "position_w", None))
        if pos_w is None:
            raise AttributeError(
                f"Camera {self.camera_name} must have pos_w/position_w in data"
            )
        pos_w = pos_w.unsqueeze(0) if pos_w.dim() == 1 else pos_w
        intrinsic_matrices = cam_data.intrinsic_matrices
        quat_w_ros = cam_data.quat_w_ros

        point_lists = []
        for i in range(B):
            pc_world = create_pointcloud_from_depth(
                intrinsic_matrix=intrinsic_matrices[i],
                depth=depth[i].squeeze(),
                position=pos_w[i],
                orientation=quat_w_ros[i],
                device=device,
            )
            pc_local = pc_world - env.scene.env_origins[i]
            valid = (pc_local.abs().sum(dim=1) > 1e-6) & torch.isfinite(pc_local).all(dim=1)
            point_lists.append(pc_local[valid])

        batch_pcd = []
        pad_template = torch.zeros(
            self.num_downsample_points, 3, device=device, dtype=torch.float32
        )
        for pts in point_lists:
            if self.pcd_crop_region is not None:
                cropped = crop_points_to_bounds(pts.unsqueeze(0), self.pcd_crop_region)[0]
                pts = cropped[cropped.abs().sum(dim=1) > 1e-6]
            if pts.shape[0] < MIN_POINTS_FOR_DOWNSAMPLE:
                batch_pcd.append(pad_template.clone().unsqueeze(0))
            else:
                batch_pcd.append(fps_points(pts.unsqueeze(0), self.num_downsample_points))

        if batch_pcd:
            out = torch.cat(batch_pcd, dim=0).permute(0, 2, 1)
        else:
            out = torch.zeros(1, 3, self.num_downsample_points, device=device, dtype=torch.float32)

        if self.pcd_noise > 0:
            out = out + (torch.rand_like(out) * 2 - 1) * self.pcd_noise

        if not self._printed_msg:
            filt = f", instance-filtered to {list(self.include_entity_names)}" if self.include_entity_names else ""
            print(f"[INFO] Rendered point cloud from {self.camera_name} (distill_mode){filt}")
            self._printed_msg = True

        return out

    def _mask_depth_by_instance(self, env, camera, depth, B, device):
        """Mask depth to inf for pixels not belonging to include_entity_names."""
        inst = camera.data.output["instance_id_segmentation_fast"]
        inst = inst.unsqueeze(0) if inst.dim() == 3 else inst
        inst = inst[..., :1] if inst.shape[-1] != 1 else inst
        id_to_labels = _get_instance_id_to_labels(camera)

        for i in range(B):
            prefixes = []
            for name in self.include_entity_names:
                if isinstance(name, str):
                    try:
                        prefixes.append(_resolve_prim_path_for_env(env.scene[name].cfg.prim_path, i))
                    except KeyError:
                        pass
            if not prefixes or not id_to_labels:
                continue
            allowed_ids = {
                int(k) for k, path in id_to_labels.items()
                if any(str(path).startswith(p) for p in prefixes)
            }
            if not allowed_ids:
                continue
            inst_i = inst[i].squeeze().to(torch.int64)
            allowed_t = torch.tensor(list(allowed_ids), device=device, dtype=torch.int64)
            mask = (inst_i.unsqueeze(-1) == allowed_t).any(dim=-1)
            depth[i].squeeze(-1)[~mask] = float("inf")

