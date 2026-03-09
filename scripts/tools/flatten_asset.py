import argparse
import asyncio

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Export flattened USD asset or USDZ (with embedded textures)."
)

parser.add_argument("--src", type=str, default='assets/bourbon/rigid_object_com.usd', help="Source USD path.")
parser.add_argument("--dst", type=str, default='assets/bourbon/rigid_object_com_flattened.usd', help="Output path (.usd or .usdz).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd


def flatten_usd(
    src_path: str,
    dst_path: str,
):
    """Flatten USD (resolve references, payloads, sublayers) into a single file.
    Textures remain external; use --usdz to embed them."""
    stage = Usd.Stage.Open(src_path)
    stage.Flatten()  # Resolves all composition arcs (refs, payloads, inherits, variants)
    stage.Export(dst_path)
    print(f"Flattened: {src_path} -> {dst_path}")


if __name__ == "__main__":
    src = args_cli.src or "/workspace/uwlab/assets/bourbon/rigid_object_com.usd"

    dst = args_cli.dst or src.replace(".usd", "_flattened.usd").replace(".usdc", "_flattened.usd")
    flatten_usd(src, dst)