"""Utility to set the center of mass on the bourbon USD and add a visible marker sphere.

Usage:
    python scripts/utils/set_bourbon_com.py --com_x 0.03 --com_y 0.04 --com_z 0.0

This writes `assets/bourbon/rigid_object_com.usd` with:
  - The specified CoM baked into the physics properties
  - A bright red sphere at the CoM position so you can see it in zero_agent.py

To preview in sim, temporarily point bottle.py at the _com variant:
    BOTTLE_USD = "/workspace/uwlab/assets/bourbon/rigid_object_com.usd"
then run:
    ./uwlab.sh -p scripts/environments/zero_agent.py \
        --task UW-FrankaLeap-GraspBottle-IkRel-v0 --num_envs 1 --headless

Coordinate frame (bottle lying on its side):
  X axis = long axis of bottle  (-0.132 = cap end, +0.132 = body/bottom end)
  Y axis = vertical when lying  (0.0 = bottom face, +0.081 = top face)
  Z axis = depth                (symmetric, +-0.042)

At spawn, BOTTLE_SPAWN_ROT = (0.707, 0, 0, -0.707) rotates the bottle
so local -X (cap) points in world +Y direction.

Typical starting guess: --com_x 0.03 --com_y 0.04 --com_z 0.0
  (3cm toward cap, centered in cross-section)

Last used command:
    python3 scripts/utils/set_bourbon_com.py --com_x 0.06 --com_y 0.04 --com_z 0.0 --no_marker
"""

import argparse
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ASSET_DIR = Path(__file__).parents[2] / "assets" / "bourbon"
SRC_USD = ASSET_DIR / "rigid_object.usd"
DST_USD = ASSET_DIR / "rigid_object_com.usd"

MARKER_RADIUS = 0.008  # 8mm sphere — visible but not huge


def main():
    parser = argparse.ArgumentParser(description="Set bourbon CoM and add visual marker.")
    parser.add_argument("--com_x", type=float, default=0.0, help="CoM X offset (m). Cap end = -0.132, body end = +0.132.")
    parser.add_argument("--com_y", type=float, default=0.04, help="CoM Y offset (m). Range 0.0–0.081.")
    parser.add_argument("--com_z", type=float, default=0.0, help="CoM Z offset (m). Range +-0.042.")
    parser.add_argument("--marker_radius", type=float, default=MARKER_RADIUS, help="Radius of the visual marker sphere (m).")
    parser.add_argument("--no_marker", action="store_true", help="Omit the visual marker sphere entirely.")
    args = parser.parse_args()

    com = (args.com_x, args.com_y, args.com_z)

    print(f"Source USD : {SRC_USD}")
    print(f"Output USD : {DST_USD}")
    print(f"CoM offset : X={com[0]:.4f}  Y={com[1]:.4f}  Z={com[2]:.4f}")
    print()
    print("Frame reminder:")
    print("  X: -0.132 (cap end) ... 0.0 (body center) ... +0.132 (bottom/body end)")
    print("  Y:  0.000 (bottom face) ... 0.040 (geometric center) ... +0.081 (top face)")
    print("  Z: -0.042 ... 0.0 ... +0.042 (symmetric)")

    shutil.copy(SRC_USD, DST_USD)
    stage = Usd.Stage.Open(str(DST_USD))

    # Set CoM on root prim
    root = stage.GetPrimAtPath("/bourbon")
    mass_api = UsdPhysics.MassAPI(root)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com))

    # Remove old marker if re-running
    for path in ["/bourbon/visual/com_marker", "/bourbon/visual/com_marker_mat",
                 "/bourbon/com_marker", "/bourbon/com_marker_mat"]:
        old = stage.GetPrimAtPath(path)
        if old.IsValid():
            stage.RemovePrim(path)

    if not args.no_marker:
        marker = UsdGeom.Sphere.Define(stage, "/bourbon/visual/com_marker")
        marker.GetRadiusAttr().Set(args.marker_radius)
        UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*com))

        mat_path = Sdf.Path("/bourbon/visual/com_marker_mat")
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(marker).Bind(mat)

    stage.GetRootLayer().Save()

    print(f"\nWrote {DST_USD}")
    print("\nTo preview:")
    print("  1. In bottle.py, change BOTTLE_USD to end in rigid_object_com.usd")
    print("  2. Run zero_agent.py with --task UW-FrankaLeap-GraspBottle-IkRel-v0 --num_envs 1 --enable_cameras --save_video /tmp/bourbon_com.mp4")
    print("  3. The red dot = CoM location")


if __name__ == "__main__":
    main()
