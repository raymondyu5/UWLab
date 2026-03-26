"""
Plot delta action distributions from a zarr dataset.

Computes deltas the same way ZarrDataset does:
  arm_delta = arm_joint_pos_target - arm_joint_pos  (7D)
  hand_action = hand_action                          (16D)
  total = 23D

Usage:
  python scripts/tools/plot_delta_actions.py \
    --data_path /gscratch/weirdlab/will/UWLab_Docker/data_storage/datasets/03_24_bourbon_pour \
    --out_path /gscratch/weirdlab/raymond/plots/delta_actions_0324.png
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zarr
import tqdm


def open_zarr(path):
    try:
        return zarr.open(path, mode="r")
    except Exception:
        return zarr.open_group(path, mode="r")


def load_key(store, key):
    if "data" in store and key in store["data"]:
        return np.array(store["data"][key])
    if key in store:
        return np.array(store[key])
    raise KeyError(key)


def load_dataset(data_path):
    entries = sorted(os.listdir(data_path))
    arm_deltas = []
    hand_actions = []

    for e in tqdm.tqdm(entries, desc="Loading episodes"):
        subdir = os.path.join(data_path, e)
        zarr_path = os.path.join(subdir, f"{e}.zarr")
        if not os.path.isdir(zarr_path):
            zarr_path = os.path.join(data_path, e) if e.endswith(".zarr") else None
        if zarr_path is None or not os.path.isdir(zarr_path):
            continue
        try:
            store = open_zarr(zarr_path)
            arm_target = load_key(store, "arm_joint_pos_target")
            arm_pos    = load_key(store, "arm_joint_pos")
            hand       = load_key(store, "hand_action")
            arm_deltas.append(arm_target - arm_pos)
            hand_actions.append(hand)
        except Exception as ex:
            print(f"Skipping {e}: {ex}")

    arm_deltas   = np.concatenate(arm_deltas, axis=0)    # (N, 7)
    hand_actions = np.concatenate(hand_actions, axis=0)  # (N, 16)
    return arm_deltas, hand_actions


def plot(arm_deltas, hand_actions, out_path):
    n_arm  = arm_deltas.shape[1]   # 7
    n_hand = hand_actions.shape[1] # 16
    n_cols = 8
    n_rows = (n_arm + n_hand + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
    axes = axes.flatten()

    for i in range(n_arm):
        ax = axes[i]
        data = arm_deltas[:, i]
        ax.hist(data, bins=100, color="steelblue", alpha=0.8)
        ax.set_title(f"arm_delta[{i}]")
        ax.set_xlabel(f"μ={data.mean():.4f}  σ={data.std():.4f}", fontsize=7)
        ax.tick_params(labelsize=7)

    for i in range(n_hand):
        ax = axes[n_arm + i]
        data = hand_actions[:, i]
        ax.hist(data, bins=100, color="darkorange", alpha=0.8)
        ax.set_title(f"hand[{i}]")
        ax.set_xlabel(f"μ={data.mean():.4f}  σ={data.std():.4f}", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(n_arm + n_hand, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Delta action distributions  (N={len(arm_deltas)} steps)", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_path", default="/gscratch/weirdlab/raymond/plots/delta_actions.png")
    args = parser.parse_args()

    arm_deltas, hand_actions = load_dataset(args.data_path)
    print(f"Loaded {len(arm_deltas)} steps")
    print(f"arm_delta  — min: {arm_deltas.min():.4f}  max: {arm_deltas.max():.4f}  std: {arm_deltas.std():.4f}")
    print(f"hand_action — min: {hand_actions.min():.4f}  max: {hand_actions.max():.4f}  std: {hand_actions.std():.4f}")
    plot(arm_deltas, hand_actions, args.out_path)
