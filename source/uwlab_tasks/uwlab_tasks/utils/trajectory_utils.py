import os
import numpy as np
from glob import glob

import zarr

def _open_zarr(path: str):
    """Open a zarr store, handling both v2 and v3."""
    if zarr is None:
        raise ImportError("zarr is required for load_real_episode_zarr")
    try:
        return zarr.open(path)
    except Exception:
        return zarr.open_group(path, mode="r")


def _load_zarr_key(store, key: str) -> np.ndarray:
    """Read a key from zarr, trying data/{key} then {key}."""
    if "data" in store and key in store["data"]:
        return np.array(store["data"][key])
    if key in store:
        return np.array(store[key])
    raise KeyError(f"Key '{key}' not found in zarr store")


def load_real_episode_npy(episode_path: str):
    """Load real robot episode data from legacy .npy format."""
    episode_name = os.path.basename(episode_path.rstrip("/"))

    if episode_path.endswith(".npy"):
        episode_file = episode_path
        episode_id = episode_name.split("_")[1].split(".")[0]
    else:
        episode_id = episode_name.split("_")[1]
        episode_file = os.path.join(episode_path, f"episode_{episode_id}.npy")

    print(f"\nLoading episode (npy): {episode_file}")
    data = np.load(episode_file, allow_pickle=True).item()

    obs_list = data["obs"]
    actions = data.get("actions", [])

    search_dir = os.path.dirname(episode_file) if episode_path.endswith(".npy") else episode_path
    pcd_files = sorted(glob(os.path.join(search_dir, "CL*.npy")))
    pointclouds = [np.load(f) for f in pcd_files] if pcd_files else None

    print(f"  Observations: {len(obs_list)}")
    print(f"  Actions: {len(actions)}")
    if pointclouds:
        print(f"  Pointclouds: {len(pointclouds)}")

    if obs_list:
        print(f"\n  Observation keys: {list(obs_list[0].keys())}")

    if actions:
        a0 = np.array(actions[0])
        print(f"  Action shape: {a0.shape}")

    return {
        "obs": obs_list,
        "actions": actions,
        "pointclouds": pointclouds,
        "episode_id": episode_id,
    }


def load_real_episode_zarr(episode_path: str):
    """Load real robot episode data from zarr format.

    Expects episode_N.zarr/ with data/ containing:
      arm_joint_pos        (T, 7)
      hand_joint_pos       (T, 16)
      ee_pose              (T, 7)
      arm_delta_action     (T, 6)
      hand_action          (T, 16)
      actions              (T, 22)  # arm_delta_action + hand_action
      ee_pose_cmd          (T, 7)
      arm_joint_pos_target (T, 7)
      seg_pc               (T, N, 3)
      rewards              (T,)
      dones                (T,)
    """
    episode_name = os.path.basename(episode_path.rstrip("/"))
    if episode_path.endswith(".zarr"):
        zarr_path = episode_path
    else:
        zarr_path = os.path.join(episode_path, episode_name)

    if not os.path.isdir(zarr_path):
        raise FileNotFoundError(f"Zarr episode not found: {zarr_path}")

    episode_id = episode_name.replace(".zarr", "").split("_")[-1]
    print(f"\nLoading episode (zarr): {zarr_path}")

    store = _open_zarr(zarr_path)

    arm_joint_pos = _load_zarr_key(store, "arm_joint_pos")
    hand_joint_pos = _load_zarr_key(store, "hand_joint_pos")
    actions_arr = _load_zarr_key(store, "actions")
    arm_joint_pos_target = _load_zarr_key(store, "arm_joint_pos_target")
    ee_pose = _load_zarr_key(store, "ee_pose")
    seg_pc = _load_zarr_key(store, "seg_pc")
    ee_pose_cmd = _load_zarr_key(store, "ee_pose_cmd")

    T = actions_arr.shape[0]
    obs_list = []
    for t in range(T):
        obs_list.append({
            "joint_positions": arm_joint_pos[t],
            "gripper_position": hand_joint_pos[t],
            "cartesian_position": ee_pose[t],
            "ik_joint_pos_desired": arm_joint_pos_target[t],
            "commanded_ee_position": ee_pose_cmd[t],
            "commanded_joint_positions": arm_joint_pos_target[t],
            "seg_pc": seg_pc[t],
        })

    actions = [actions_arr[t] for t in range(T)]
    pointclouds = [seg_pc[t] for t in range(T)]

    print(f"  Observations: {T}")
    print(f"  Actions: {T}")
    print(f"  Pointclouds: {T} (inline)")
    print(f"  Action shape: {actions_arr.shape}")

    return {
        "obs": obs_list,
        "actions": actions,
        "pointclouds": pointclouds,
        "episode_id": episode_id,
    }


def load_real_episode(episode_path: str):
    """Load real robot episode data. Auto-detects format: .zarr -> zarr, else -> npy."""
    path = episode_path.rstrip("/")
    name = os.path.basename(path)
    if name.endswith(".zarr") or path.endswith(".zarr"):
        return load_real_episode_zarr(episode_path)
    zarr_child = os.path.join(path, f"{name}.zarr")
    if os.path.isdir(zarr_child):
        return load_real_episode_zarr(zarr_child)
    return load_real_episode_npy(episode_path)

    