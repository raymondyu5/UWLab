"""Add a visual marker to the pink cup USD showing the pour target position.

The pour target is where the bottle cap should be: 3cm above the cup rim.
Cup mesh frame: Y goes 0 (bottom) -> 0.1507 (rim). Cup spawns upright via
PINK_CUP_POUR_ROT = (0.707, 0.707, 0, 0) — 90deg around X — so local Y maps to world Z.

This writes assets/pink_cup/rigid_object_marker.usd with a green sphere at
the pour target position in the cup's LOCAL frame.

Usage:
    python scripts/utils/set_cup_target_marker.py

Last used command:
    python3 scripts/utils/set_cup_target_marker.py  (POUR_TARGET_OFFSET=0.11)

To preview:
  1. In bottle_pour.py, temporarily change PINK_CUP_USD to rigid_object_marker.usd
  2. Run zero_agent.py --task UW-FrankaLeap-PourBottle-IkRel-v0 --num_envs 1 --enable_cameras --save_video ...
  3. Green sphere = where bottle cap tip should reach
"""

import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

ASSET_DIR = Path(__file__).parents[2] / "assets" / "pink_cup"
SRC_USD = ASSET_DIR / "rigid_object.usd"
DST_USD = ASSET_DIR / "rigid_object_marker.usd"

# Cup rim in local frame: Y=0.1507. Pour target = rim + 0.03m above.
# But the cap should be centered over the cup opening (XZ center = 0,0).
# In local frame: target = (0, 0.1507 + 0.03, 0) = (0, 0.1807, 0)
CUP_RIM_Y = 0.1507
POUR_TARGET_OFFSET = 0.11
TARGET_LOCAL = (0.0, CUP_RIM_Y + POUR_TARGET_OFFSET, 0.0)
MARKER_RADIUS = 0.015


def main():
    print(f"Source USD : {SRC_USD}")
    print(f"Output USD : {DST_USD}")
    print(f"Cup rim Y  : {CUP_RIM_Y:.4f}m")
    print(f"Pour target (local): {TARGET_LOCAL}")
    print()
    print("Cup frame reminder:")
    print("  Y: 0.0 (bottom) -> 0.1507 (rim)  [cup stands on Y=0]")
    print("  X/Z: symmetric, +-0.044 (cup radius)")
    print("  At spawn: 90deg around X rotates local Y -> world Z (cup stands upright)")

    shutil.copy(SRC_USD, DST_USD)
    stage = Usd.Stage.Open(str(DST_USD))

    for path in ["/pink_cup/visual/pour_target_marker", "/pink_cup/visual/pour_target_mat"]:
        old = stage.GetPrimAtPath(path)
        if old.IsValid():
            stage.RemovePrim(path)

    marker = UsdGeom.Sphere.Define(stage, "/pink_cup/visual/pour_target_marker")
    marker.GetRadiusAttr().Set(MARKER_RADIUS)
    UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*TARGET_LOCAL))

    mat = UsdShade.Material.Define(stage, "/pink_cup/visual/pour_target_mat")
    shader = UsdShade.Shader.Define(stage, "/pink_cup/visual/pour_target_mat/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(marker).Bind(mat)

    stage.GetRootLayer().Save()
    print(f"\nWrote {DST_USD}")
    print("\nTo preview:")
    print("  1. In bottle_pour.py, change PINK_CUP_USD to end in rigid_object_marker.usd")
    print("  2. Run zero_agent.py --task UW-FrankaLeap-PourBottle-IkRel-v0 --num_envs 1 --enable_cameras --save_video /workspace/uwlab/logs/cup_target.mp4")
    print("  3. Green sphere = pour target (where cap tip should reach)")


if __name__ == "__main__":
    main()
