"""CLI flags that override the RFS speed/precision keys in the rfs yaml (shared by train.py and play_rfs_wrapper.py).

Dependency-free on purpose: it is imported before the Isaac app is launched, when the argparse parser is built.
"""

import argparse


def add_speed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cfm_autocast", action="store_true", default=False,
                        help="Override rfs.cfm_autocast: bf16 CFM inference (~25%% faster CFM; lowers success on some tasks).")
    parser.add_argument("--cfm_steps", type=int, default=None,
                        help="Override rfs.cfm_num_steps: RK2 integration steps (policies are trained with 5).")
    parser.add_argument("--fast_substeps", action="store_true", default=False,
                        help="Force rfs.fast_substeps on (already the default; see grasp/mdp/fast_step.py).")
    parser.add_argument("--no_fast_substeps", action="store_true", default=False,
                        help="Disable fast_substeps (default on).")
    parser.add_argument("--skip_intermediate_obs", action="store_true", default=False,
                        help="Override rfs.skip_intermediate_obs: compute obs/point cloud only on the last env step of a PPO step.")


def apply_speed_overrides(rfs_cfg: dict, args) -> None:
    """Merge the flags from `add_speed_args` into the `rfs:` yaml dict (in place)."""
    if args.no_fast_substeps:
        rfs_cfg["fast_substeps"] = False
    if args.cfm_steps is not None:
        rfs_cfg["cfm_num_steps"] = args.cfm_steps
    for flag in ("cfm_autocast", "fast_substeps", "skip_intermediate_obs"):
        if getattr(args, flag):
            rfs_cfg[flag] = True
