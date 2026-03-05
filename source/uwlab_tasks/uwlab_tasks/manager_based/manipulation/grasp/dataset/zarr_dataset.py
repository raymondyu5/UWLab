from typing import Dict, List, Optional
import os
import copy
import numpy as np
import zarr
import torch
from torch.utils.data import Dataset

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

from uwlab_tasks.manager_based.manipulation.grasp.dataset.pcd_sampler import (
    PCDSequenceSampler,
    get_val_mask,
    downsample_mask,
)


def _get_pcd_range_normalizer() -> SingleFieldLinearNormalizer:
    """Fixed [-10, 10] -> [-1, 1] normalizer for point clouds."""
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0.0], dtype=np.float32)
    stat = {
        'min': np.array([-10], dtype=np.float32),
        'max': np.array([10], dtype=np.float32),
        'mean': np.array([0.0], dtype=np.float32),
        'std': np.array([1], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


class _ReplayBuffer:
    """
    Simple in-memory buffer that holds all episodes concatenated along the time axis.

    Exposes:
      - buffer[key]        -> np.ndarray of shape (total_steps, ...)
      - buffer.episode_ends -> np.ndarray of shape (n_episodes,), cumulative step counts
      - buffer.keys()
      - buffer.n_episodes
    """

    def __init__(self):
        self._data: Dict[str, np.ndarray] = {}
        self._episode_ends: List[int] = []

    def add_episode(self, episode_data: Dict[str, np.ndarray]):
        """episode_data: dict of key -> (T, ...) arrays, all same T."""
        lengths = [v.shape[0] for v in episode_data.values()]
        assert len(set(lengths)) == 1, f"Mismatched episode lengths: {lengths}"
        T = lengths[0]

        for key, arr in episode_data.items():
            if key not in self._data:
                self._data[key] = arr
            else:
                self._data[key] = np.concatenate([self._data[key], arr], axis=0)

        offset = self._episode_ends[-1] if self._episode_ends else 0
        self._episode_ends.append(offset + T)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    @property
    def episode_ends(self) -> np.ndarray:
        return np.array(self._episode_ends, dtype=np.int64)

    @property
    def n_episodes(self) -> int:
        return len(self._episode_ends)


def _open_zarr(path: str):
    """Open a zarr store, handling both v2 and v3."""
    try:
        return zarr.open(path)
    except Exception:
        return zarr.open_group(path, mode='r')


def _load_zarr_key(store, key: str) -> np.ndarray:
    """Read a key from zarr, trying data/{key} then {key}."""
    if 'data' in store and key in store['data']:
        return np.array(store['data'][key])
    if key in store:
        return np.array(store[key])
    raise KeyError(f"Key '{key}' not found in zarr store")


class SimZarrDataset(Dataset):
    """
    Dataset for loading sim zarr episodes for BC training.

    Each zarr episode stores arrays of shape (T, ...) under data/{key}.
    This dataset:
      1. Loads all episodes into RAM at init (concatenated numpy arrays)
      2. Concatenates obs_keys -> agent_pos
      3. Exposes a sequence sampler for sliding-window batches
      4. Returns {"obs": {"agent_pos": (1, D), "seg_pc": (1, 3, N)}, "action": (H, A)}

    Args:
        data_path:   directory containing episode_*.zarr subdirectories
        load_list:   list of episode folder names, or ["all"] to load everything
        num_demo:    max episodes to load
        obs_keys:    zarr keys to concatenate into agent_pos, e.g. ["right_hand_joint_pos", "right_ee_pose"]
        action_key:  zarr key for actions
        image_keys:  zarr keys for point clouds, e.g. ["seg_pc"]
        horizon:     prediction horizon (sequence length)
        pad_before:  timesteps to pad at episode start
        pad_after:   timesteps to pad at episode end
        val_ratio:   fraction of episodes to hold out for validation
        seed:        random seed for val split and noise
        downsample_points: number of points to sample from each PCD frame
        pcd_noise:   std of Gaussian noise added to xyz
        noise_extrinsic: whether to apply random rotation+translation augmentation
        noise_extrinsic_parameter: [translation_scale, rotation_scale_rad]
        obs_noise:   dict of obs_key -> noise std for proprio augmentation
    """

    def __init__(
        self,
        data_path: str,
        load_list: List[str],
        num_demo: int = 150,
        obs_keys: List[str] = None,
        action_key: str = "actions",
        image_keys: List[str] = None,
        horizon: int = 4,
        pad_before: int = 0,
        pad_after: int = 0,
        val_ratio: float = 0.05,
        seed: int = 42,
        downsample_points: int = 2048,
        pcd_noise: float = 0.02,
        noise_extrinsic: bool = False,
        noise_extrinsic_parameter: Optional[List[float]] = None,
        obs_noise: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        if obs_keys is None:
            obs_keys = []
        if image_keys is None:
            image_keys = []
        if noise_extrinsic_parameter is None:
            noise_extrinsic_parameter = [0.05, 0.2]
        if obs_noise is None:
            obs_noise = {}

        self.obs_keys = obs_keys
        self.action_key = action_key
        self.image_keys = image_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.pcd_noise = pcd_noise
        self.obs_noise = obs_noise

        # --- resolve episode list ---
        if "all" in load_list:
            entries = sorted(os.listdir(data_path))
        else:
            entries = sorted(load_list)

        zarr_episodes = [
            e for e in entries
            if e.endswith('.zarr') and os.path.isdir(os.path.join(data_path, e))
        ][:num_demo]

        if len(zarr_episodes) == 0:
            raise ValueError(f"No .zarr episodes found in {data_path}")

        # --- load into RAM ---
        replay_buffer = _ReplayBuffer()
        for ep_name in zarr_episodes:
            ep_path = os.path.join(data_path, ep_name)
            try:
                store = _open_zarr(ep_path)
            except Exception as e:
                print(f"Warning: skipping {ep_name}: {e}")
                continue

            try:
                episode = {}

                # actions
                episode["action"] = _load_zarr_key(store, action_key)

                # obs keys -> will be concatenated to agent_pos in __getitem__
                for key in obs_keys:
                    episode[key] = _load_zarr_key(store, key)

                # image keys (PCDs): store raw (T, N, 3) — sampler will downsample
                for key in image_keys:
                    episode[key] = _load_zarr_key(store, key)

                replay_buffer.add_episode(episode)
            except Exception as e:
                print(f"Warning: skipping {ep_name}: {e}")
                continue

        self.replay_buffer = replay_buffer
        self.action_dim = replay_buffer["action"].shape[-1]
        self.low_obs_dim = sum(
            replay_buffer[k].shape[-1] for k in obs_keys if k in replay_buffer
        )

        print(
            f"Loaded {replay_buffer.n_episodes} episodes, "
            f"{replay_buffer.episode_ends[-1]} total steps. "
            f"action_dim={self.action_dim}, obs_dim={self.low_obs_dim}"
        )

        # --- train/val split ---
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = downsample_mask(~val_mask, max_n=None, seed=seed)
        self.train_mask = train_mask

        self.sampler = PCDSequenceSampler(
            replay_buffer=replay_buffer,
            image_keys=image_keys,
            downsample_points=downsample_points,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            noise_extrinsic=noise_extrinsic,
            noise_extrinsic_parameter=noise_extrinsic_parameter,
        )

    def get_validation_dataset(self) -> "SimZarrDataset":
        val_set = copy.copy(self)
        val_set.sampler = PCDSequenceSampler(
            replay_buffer=self.replay_buffer,
            image_keys=self.image_keys,
            downsample_points=self.sampler.downsample_points,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            noise_extrinsic=self.sampler.noise_extrinsic,
            noise_extrinsic_parameter=self.sampler.noise_extrinsic_parameter,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode: str = "limits") -> LinearNormalizer:
        # Concatenate obs_keys to form agent_pos, same as __getitem__
        obs_list = [self.replay_buffer[k] for k in self.obs_keys if k in self.replay_buffer]
        agent_pos = np.concatenate(obs_list, axis=-1) if obs_list else np.zeros((1, 1))

        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": agent_pos,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode)

        for key in self.image_keys:
            normalizer[key] = _get_pcd_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # concatenate obs_keys -> agent_pos, optionally add noise
        obs_list = []
        for key in self.obs_keys:
            arr = sample[key].astype(np.float32)
            if key in self.obs_noise:
                arr = arr + np.random.randn(*arr.shape).astype(np.float32) * self.obs_noise[key]
            obs_list.append(arr)

        agent_pos = np.concatenate(obs_list, axis=-1) if obs_list else np.zeros((self.horizon, 1), dtype=np.float32)

        # add Gaussian noise to pcd xyz
        image_obs = {}
        for key in self.image_keys:
            pcd = sample[key].astype(np.float32)  # (T, 3, N)
            noise = (np.random.rand(*pcd.shape) * 2 - 1) * self.pcd_noise
            pcd[:, :3, :] += noise[:, :3, :]
            image_obs[key] = pcd[0:1]  # take first timestep: (1, 3, N)

        data = {
            "obs": {"agent_pos": agent_pos[0:1]} | image_obs,  # agent_pos: (1, D)
            "action": sample["action"].astype(np.float32),      # (H, A)
        }

        return {
            "obs": {k: torch.from_numpy(v) for k, v in data["obs"].items()},
            "action": torch.from_numpy(data["action"]),
        }
