"""Inspect the bounding box of a USD asset to find the origin offset from mesh center.

Usage:
    isaacpy scripts/tools/inspect_usd_bbox.py /workspace/uwlab/assets/poptart/rigid_object.usd --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect USD bounding box and origin offset")
parser.add_argument("usd_path", help="Path to the USD file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, Gf  # noqa: E402


def main():
    usd_path = args_cli.usd_path
    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    if not root:
        root = stage.GetPseudoRoot()

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox = bbox_cache.ComputeWorldBound(root)
    range_ = bbox.GetRange()

    mn = range_.GetMin()
    mx = range_.GetMax()
    center = (mn + mx) / 2
    size = mx - mn

    print(f"\nUSD: {usd_path}")
    print(f"  min:    ({mn[0]:.4f}, {mn[1]:.4f}, {mn[2]:.4f})")
    print(f"  max:    ({mx[0]:.4f}, {mx[1]:.4f}, {mx[2]:.4f})")
    print(f"  size:   ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})")
    print(f"  center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print()
    print("Interpretation:")
    print(f"  Origin is offset from mesh center by: ({-center[0]:.4f}, {-center[1]:.4f}, {-center[2]:.4f})")
    print(f"  Bottom of mesh (z_min) in local space: {mn[2]:.4f}")
    if abs(mn[2]) > 1e-4:
        print(f"  => Mesh bottom is NOT at origin. To rest on a surface, reset_height += {-mn[2]:.4f}")
    else:
        print(f"  => Mesh bottom is at origin; reset_height = half_height is correct")


if __name__ == "__main__":
    main()
    simulation_app.close()