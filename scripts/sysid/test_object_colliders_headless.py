# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless script to test object collider discovery for PourBottle.

Creates the PourBottle env, then:
  A) Calls sample_object_point_cloud for the grasp_object prim path and prints whether
     it returns points or None.
  B) Walks the USD subtree under the object prim and prints each prim's path, type,
     and HasAPI(UsdPhysics.CollisionAPI).

Run headless (no viewer). Example:
  python scripts/sysid/test_object_colliders_headless.py
  python scripts/sysid/test_object_colliders_headless.py --device cpu
  (Use AppLauncher's --device for CPU/GPU; default is cuda.)
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test object collider discovery for PourBottle (headless).")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    EVAL_MODE,
    parse_franka_leap_env_cfg,
)
from uwlab_tasks.manager_based.manipulation.reset_states.mdp import utils as reset_states_utils
from pxr import UsdPhysics


def walk_usd_tree(stage, prim, indent: int = 0):
    """Recursively print prim path, type, and CollisionAPI. Returns count of prims with CollisionAPI."""
    path = str(prim.GetPath())
    typ = prim.GetTypeName()
    has_coll = prim.HasAPI(UsdPhysics.CollisionAPI)
    prefix = "  " * indent
    coll_tag = " [CollisionAPI]" if has_coll else ""
    print(f"{prefix}{path}  type={typ}{coll_tag}")
    n_coll = 1 if has_coll else 0
    for child in prim.GetChildren():
        n_coll += walk_usd_tree(stage, child, indent + 1)
    return n_coll


def main():
    task = "UW-FrankaLeap-PourBottle-JointAbs-v0"
    env_cfg = parse_franka_leap_env_cfg(
        task,
        EVAL_MODE,
        device=args.device,
        num_envs=1,
        use_fabric=not args.disable_fabric,
    )
    env = gym.make(task, cfg=env_cfg)

    breakpoint()
    
    obs, _ = env.reset()

    obj_asset = env.scene["grasp_object"]
    prim_path_pattern = obj_asset.cfg.prim_path
    object_prim_path_env0 = prim_path_pattern.replace(".*", "0", 1)

    print("=" * 60)
    print("Object (grasp_object) prim path (env 0):", object_prim_path_env0)
    print("Prim path pattern (for sample_object_point_cloud):", prim_path_pattern)
    print("=" * 60)

    # A) Call sample_object_point_cloud
    pc = reset_states_utils.sample_object_point_cloud(
        num_envs=1,
        num_points=512,
        prim_path_pattern=prim_path_pattern,
        device=env.device,
    )
    if pc is None:
        print("[A] sample_object_point_cloud: returned None (no colliders found).")
    else:
        print(f"[A] sample_object_point_cloud: OK, shape={pc.shape}")

    # B) Walk USD subtree under object root
    print()
    print("[B] USD subtree under object root (path, type, CollisionAPI):")
    print("    (sample_object_point_cloud only counts prims that are Mesh/Cube/Sphere/Cylinder/Capsule/Cone AND have CollisionAPI)")
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(object_prim_path_env0)
        if not root_prim.IsValid():
            print(f"  Prim not found at {object_prim_path_env0!r}")
        else:
            n_coll = walk_usd_tree(stage, root_prim)
            print()
            print(f"  Total prims with CollisionAPI under root: {n_coll}")
    except Exception as e:
        print(f"  Error walking USD: {e}")

    env.close()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
