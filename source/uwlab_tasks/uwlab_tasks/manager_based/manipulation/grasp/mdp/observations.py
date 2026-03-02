# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
import trimesh
import numpy as np

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from uwlab.utils.math import fps_points

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False


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


##
# Synthetic point cloud observation
# Exact port of SynthesizeEnvPC from synthesize_pcd.py
# Changes: scene entity names parameterized (right_hand -> asset_name, right_hand_object -> object_name)
##

class SynthesizePC:

    def __init__(
        self,
        asset_name: str,
        object_name: str,
        arm_mesh_dir: str,
        hand_mesh_dir: str,
        object_mesh_path: str,
        num_arm_pcd: int = 64,
        num_hand_pcd: int = 64,
        num_object_pcd: int = 512,
        num_downsample_points: int = 2048,
    ):
        self.asset_name = asset_name
        self.object_name = object_name
        self.arm_mesh_dir = arm_mesh_dir
        self.hand_mesh_dir = hand_mesh_dir
        self.object_mesh_path = object_mesh_path
        self.num_arm_pcd = num_arm_pcd
        self.num_hand_pcd = num_hand_pcd
        self.num_object_pcd = num_object_pcd
        self.num_downsample_points = num_downsample_points
        self.mesh_init = False
        self._printed_msg = False

    def synthesize_env(self, env):
        if not self.mesh_init:
            self.init_mesh(env)

        robot_link_state = env.scene[self.asset_name]._data.body_pose_w.clone()
        robot_link_state[:, :, :3] -= env.scene.env_origins.unsqueeze(1).repeat_interleave(
            robot_link_state.shape[1], dim=1)

        arm_link_state = robot_link_state[:, :8]
        hand_link_state = robot_link_state[:, 8:]

        arm_vertices = math_utils.transform_points(
            self.arm_mesh.reshape(-1, self.arm_mesh.shape[-2], 3),
            arm_link_state[..., :3].reshape(-1, 3),
            arm_link_state[..., 3:7].reshape(-1, 4))
        hand_vertices = math_utils.transform_points(
            self.hand_mesh.reshape(-1, self.hand_mesh.shape[-2], 3),
            hand_link_state[..., :3].reshape(-1, 3),
            hand_link_state[..., 3:7].reshape(-1, 4))

        arm_vertices = arm_vertices.reshape(env.num_envs, -1, 3)
        hand_vertices = hand_vertices.reshape(env.num_envs, -1, 3)

        object_state = env.scene[self.object_name]._data.root_pose_w.clone()
        object_state[:, :3] -= env.scene.env_origins
        object_vertices = math_utils.transform_points(
            self.object_mesh, object_state[..., :3], object_state[..., 3:7])

        all_pcd = torch.cat([arm_vertices, hand_vertices, object_vertices], dim=1)
        points_index = torch.randperm(all_pcd.shape[1]).to(env.device)
        sampled_pcd = all_pcd[:, points_index[:self.num_downsample_points]]

        if not self._printed_msg:
            print("[INFO] Using synthetic pointcloud")
            self._printed_msg = True

        return sampled_pcd.permute(0, 2, 1)

    def init_mesh(self, env):
        self.num_envs = env.num_envs
        self.arm_names = env.scene[self.asset_name].body_names[:8]
        self.hand_names = env.scene[self.asset_name].body_names[8:]
        self.load_arm_mesh(env)
        self.load_hand_mesh(env)
        self.load_object_mesh(env)
        self.mesh_init = True

    def load_arm_mesh(self, env):
        self.arm_mesh = torch.zeros(
            (env.num_envs, len(self.arm_names), self.num_arm_pcd, 3),
            device=env.device, dtype=torch.float32)
        for index, name in enumerate(self.arm_names):
            arm_mesh = trimesh.load(os.path.join(self.arm_mesh_dir, f"{name}.obj"))
            arm_mesh = trimesh.util.concatenate(arm_mesh)
            vertices = torch.tensor(arm_mesh.vertices, dtype=torch.float32).to(env.device)
            self.arm_mesh[:, index] = fps_points(vertices.unsqueeze(0), self.num_arm_pcd)

    def load_hand_mesh(self, env):
        self.hand_mesh = torch.zeros(
            (env.num_envs, len(self.hand_names), self.num_hand_pcd, 3),
            device=env.device, dtype=torch.float32)
        for index, link_name in enumerate(self.hand_names):
            if "sensor" in link_name.lower():
                print(f"Skipping sensor link: {link_name} (no mesh needed)")
                self.hand_mesh[:, index] = torch.zeros(
                    (1, self.num_hand_pcd, 3), device=env.device, dtype=torch.float32)
                continue

            if "palm_lower" in link_name:
                mesh_name = "palm_lower"
            elif "palm_upper" in link_name:
                mesh_name = "palm_upper"
            elif "thumb" in link_name:
                mesh_name = link_name
            else:
                parts = link_name.split("_")
                if parts[-1].isdigit():
                    mesh_name = "_".join(parts[:-1])
                else:
                    mesh_name = "_".join(parts)

            mesh_path = os.path.join(self.hand_mesh_dir, f"{mesh_name}.obj")
            print(f"Loading hand mesh: {link_name} -> {mesh_name} -> {mesh_path}")
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

            hand_mesh = trimesh.load(mesh_path)
            hand_mesh = trimesh.util.concatenate(hand_mesh)

            if HAS_PYMESHLAB:
                import pymeshlab
                mesh = pymeshlab.Mesh(
                    vertex_matrix=np.array(hand_mesh.vertices),
                    face_matrix=np.array(hand_mesh.faces, dtype=np.int32))
                ms = pymeshlab.MeshSet()
                ms.add_mesh(mesh, 'my_mesh')
                ms.meshing_remove_duplicate_faces()
                ms.meshing_repair_non_manifold_edges()
                ms.meshing_repair_non_manifold_vertices()
                ms.meshing_surface_subdivision_midpoint(iterations=3)
                current_mesh = ms.current_mesh()
                vertices = torch.tensor(current_mesh.vertex_matrix(), dtype=torch.float32).to(env.device)
            else:
                vertices = torch.tensor(hand_mesh.vertices, dtype=torch.float32).to(env.device)

            self.hand_mesh[:, index] = fps_points(vertices.unsqueeze(0), self.num_hand_pcd)

    def load_object_mesh(self, env):
        self.object_mesh = torch.zeros(
            (env.num_envs, self.num_object_pcd, 3), dtype=torch.float32).to(env.device)
        mesh_path = self.object_mesh_path
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.abspath(mesh_path)
        obj_mesh = trimesh.load(mesh_path)
        obj_mesh = trimesh.util.concatenate(obj_mesh)
        vertices = torch.tensor(obj_mesh.vertices, dtype=torch.float32).to(env.device)
        downsample_vertices = fps_points(vertices.unsqueeze(0), self.num_object_pcd)
        self.object_mesh[:] = downsample_vertices
