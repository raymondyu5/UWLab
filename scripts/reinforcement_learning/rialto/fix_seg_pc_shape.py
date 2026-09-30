"""
Fix seg_pc shape in existing RL rollout zarr files.

collect_rl_rollouts.py incorrectly saved seg_pc as (T, 3, N) instead of (T, N, 3).
This script transposes data/seg_pc for every episode_*/episode_*.zarr found under
the given data directory, using multiprocessing for parallel I/O.

Usage:
    python scripts/reinforcement_learning/rialto/fix_seg_pc_shape.py \
        --data_dir data_storage/rialto_rl/rl/cube_grasp_rl_privileged \
        --workers 8

Add --dry_run to preview without writing anything.
"""

import argparse
import glob
import os
import shutil
import multiprocessing as mp

import numpy as np
import zarr
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

pattern = os.path.join(args.data_dir, "episode_*", "episode_*.zarr")
paths = sorted(glob.glob(pattern))

if not paths:
    raise FileNotFoundError(f"No episode zarr files found under {args.data_dir!r}")

print(f"Found {len(paths)} episodes  workers={args.workers}  dry_run={args.dry_run}")


def _open_zarr_ro(path):
    for fn in [
        lambda: zarr.open_group(path, mode="r"),
        lambda: zarr.open(path, mode="r"),
    ]:
        try:
            g = fn()
            _ = g["data"]
            return g
        except Exception:
            continue
    raise RuntimeError(f"Cannot open zarr (read-only) at {path}")


# Return values from worker: ("fixed"|"skipped"|"error", label, optional message)
def process_episode(args_tuple):
    path, dry_run = args_tuple
    label = os.path.basename(os.path.dirname(path))
    try:
        store = _open_zarr_ro(path)
        arrays = {k: store[f"data/{k}"][:] for k in store["data"].keys()}

        seg = arrays["seg_pc"]
        if seg.ndim != 3:
            return ("skipped", label, f"unexpected ndim {seg.ndim}")

        T, d1, d2 = seg.shape
        if d2 == 3 and d1 != 3:
            return ("skipped", label, None)  # already correct, don't spam output
        if d1 == 3 and d2 == 3:
            return ("skipped", label, f"ambiguous shape {seg.shape} (N=3?)")
        if d1 != 3:
            return ("skipped", label, f"unrecognised shape {seg.shape}")

        # (T, 3, N) → fix
        if dry_run:
            return ("fixed", label, f"WOULD fix {seg.shape} → ({T}, {d2}, {d1})")

        arrays["seg_pc"] = seg.transpose(0, 2, 1)
        shutil.rmtree(path)
        new_store = zarr.open(path, mode="w")
        for k, arr in arrays.items():
            new_store[f"data/{k}"] = arr

        return ("fixed", label, None)

    except Exception as e:
        return ("error", label, str(e))


if __name__ == "__main__":
    fixed = skipped = errors = 0

    work = [(p, args.dry_run) for p in paths]

    with mp.Pool(processes=args.workers) as pool:
        with tqdm(total=len(paths), unit="ep") as pbar:
            for status, label, msg in pool.imap_unordered(process_episode, work):
                if status == "fixed":
                    fixed += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errors += 1

                if msg:
                    tqdm.write(f"  {status.upper():7s} {label}: {msg}")

                pbar.set_postfix(fixed=fixed, skipped=skipped, errors=errors)
                pbar.update(1)

    print(f"\nDone. fixed={fixed}  already_ok={skipped}  errors={errors}")
