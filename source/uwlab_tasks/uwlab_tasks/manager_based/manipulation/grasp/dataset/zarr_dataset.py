from typing import Dict, List, Optional
import os
import copy
import numpy as np
import zarr
import torch
import tqdm
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


class ZarrDataset(Dataset):
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
        obs_noise:   dict of obs_key -> noise std for proprio augmentation
    """

    def __init__(
        self,
        data_path: str,
        load_list: List[str],
        num_demo: Optional[int] = None,
        obs_keys: List[str] = None,
        action_key: str = "actions",
        action_base_keys: Optional[List[str]] = None,
        image_keys: List[str] = None,
        horizon: int = 4,
        n_obs_steps: int = 1,
        pad_after: int = 0,
        val_ratio: float = 0.05,
        seed: int = 42,
        downsample_points: int = 2048,
        pcd_noise: float = 0.02,
        noise_extrinsic: bool = False,
        noise_extrinsic_parameter: Optional[List[float]] = None,
        obs_noise: Optional[Dict[str, float]] = None,
        hand_dropout_prob: float = 0.0,
        chunk_relative: bool = False,
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
        self.action_keys = [action_key] if isinstance(action_key, str) else list(action_key)
        self.action_key = self.action_keys[0]  # kept for backward compat
        self.action_base_keys = list(action_base_keys) if action_base_keys is not None else None
        self.image_keys = image_keys
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        # pad_before must be exactly n_obs_steps-1: each window starts n_obs_steps-1
        # steps before the action sequence, so early-episode samples get zero-padded history.
        self.pad_before = n_obs_steps - 1
        self.pad_after = pad_after
        self.pcd_noise = pcd_noise
        self.obs_noise = obs_noise
        self.hand_dropout_prob = hand_dropout_prob
        self.chunk_relative = chunk_relative
        # total window = history frames + action frames
        self._window_size = n_obs_steps - 1 + horizon

        # --- resolve episode list ---
        if "all" in load_list:
            entries = sorted(os.listdir(data_path))
        else:
            entries = sorted(load_list)

        # Flat format: data_path/episode_0.zarr, episode_2.zarr, ...
        zarr_paths = [
            os.path.join(data_path, e) for e in entries
            if e.endswith('.zarr') and os.path.isdir(os.path.join(data_path, e))
        ]

        # Nested format: data_path/episode_0/episode_0.zarr, episode_2/episode_2.zarr, ...
        nested_paths = []
        for e in entries:
            subdir = os.path.join(data_path, e)
            nested = os.path.join(subdir, f"{e}.zarr")
            if os.path.isdir(subdir) and os.path.isdir(nested):
                nested_paths.append(nested)
        nested_paths = sorted(nested_paths)
        if len(zarr_paths) == 0 or (len(nested_paths) > len(zarr_paths)):
            zarr_paths = nested_paths

        if num_demo is not None:
            zarr_paths = zarr_paths[:num_demo]

        if len(zarr_paths) == 0:
            raise ValueError(f"No .zarr episodes found in {data_path}")

        # --- load into RAM ---
        replay_buffer = _ReplayBuffer()
        for ep_path in tqdm.tqdm(zarr_paths, desc=f"Loading episodes from {os.path.basename(data_path)}"):
            try:
                store = _open_zarr(ep_path)
            except Exception as e:
                print(f"Warning: skipping {ep_path}: {e}")
                continue

            try:
                episode = {}

                # actions
                action_arrays = []
                for i, k in enumerate(self.action_keys):
                    arr = _load_zarr_key(store, k)
                    if self.action_base_keys is not None and self.action_base_keys[i] is not None:
                        arr = arr - _load_zarr_key(store, self.action_base_keys[i])
                    action_arrays.append(arr)
                episode["action"] = np.concatenate(action_arrays, axis=-1) if len(action_arrays) > 1 else action_arrays[0]

                # obs keys -> will be concatenated to agent_pos in __getitem__
                for key in obs_keys:
                    episode[key] = _load_zarr_key(store, key)

                # image keys (PCDs): store raw (T, N, 3) — sampler will downsample
                for key in image_keys:
                    episode[key] = _load_zarr_key(store, key)

                replay_buffer.add_episode(episode)
            except Exception as e:
                print(f"Warning: skipping {ep_path}: {e}")
                continue

        self.replay_buffer = replay_buffer
        self.action_dim = replay_buffer["action"].shape[-1]
        self.low_obs_dim = sum(
            replay_buffer[k].shape[-1] for k in obs_keys if k in replay_buffer
        )

        print(
            f"Loaded {replay_buffer.n_episodes} episodes, "
            f"{replay_buffer.episode_ends[-1]} total steps "
            f"from {data_path} "
            f"(action_dim={self.action_dim}, obs_dim={self.low_obs_dim})"
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
            sequence_length=self._window_size,
            pad_before=self.pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            noise_extrinsic=noise_extrinsic,
            noise_extrinsic_parameter=noise_extrinsic_parameter,
            n_obs_steps=n_obs_steps,
        )

    def get_validation_dataset(self) -> "ZarrDataset":
        val_set = copy.copy(self)
        val_set.sampler = PCDSequenceSampler(
            replay_buffer=self.replay_buffer,
            image_keys=self.image_keys,
            downsample_points=self.sampler.downsample_points,
            sequence_length=self._window_size,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            noise_extrinsic=self.sampler.noise_extrinsic,
            noise_extrinsic_parameter=self.sampler.noise_extrinsic_parameter,
            n_obs_steps=self.sampler.n_obs_steps,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def _get_chunk_relative_action_samples(self) -> np.ndarray:
        """Collect representative chunk-relative action samples for normalizer fitting.

        Iterates over all episodes, extracts full H-step windows (no cross-episode
        boundary), zero-pads partial windows at episode end, applies cumsum, and
        returns a flat (N, A) array suitable for LinearNormalizer.fit.
        """
        raw = self.replay_buffer["action"]  # (total_steps, A)
        ends = self.replay_buffer.episode_ends
        chunks = []
        prev_end = 0
        for end in ends:
            ep = raw[prev_end:end]
            T = len(ep)
            stride = max(1, (T - self.horizon) // 200)
            for start in range(0, max(1, T - self.horizon + 1), stride):
                chunk = ep[start:start + self.horizon].copy().astype(np.float32)
                if len(chunk) < self.horizon:
                    pad = np.zeros((self.horizon - len(chunk), chunk.shape[-1]), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=0)
                chunks.append(np.cumsum(chunk, axis=0))
            prev_end = end
        if not chunks:
            return raw
        return np.concatenate(chunks, axis=0)  # (N*horizon, A)

    def get_normalizer(self, mode: str = "limits") -> LinearNormalizer:
        # Concatenate obs_keys to form agent_pos, same as __getitem__
        obs_list = [self.replay_buffer[k] for k in self.obs_keys if k in self.replay_buffer]
        agent_pos = np.concatenate(obs_list, axis=-1) if obs_list else np.zeros((1, 1))

        # action normalizer: chunk-relative samples if in that mode, otherwise raw step-wise.
        # past_actions are always the executed step-wise deltas regardless of mode.
        action_for_norm = (
            self._get_chunk_relative_action_samples()
            if self.chunk_relative
            else self.replay_buffer["action"]
        )
        data = {
            "action": action_for_norm,
            "agent_pos": agent_pos,
            "past_actions": self.replay_buffer["action"],
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
            if key == 'hand_joint_pos' and self.hand_dropout_prob > 0:
                mask = (np.random.rand(arr.shape[-1]) > self.hand_dropout_prob).astype(np.float32)
                arr = arr * mask
            obs_list.append(arr)

        agent_pos = np.concatenate(obs_list, axis=-1) if obs_list else np.zeros((self._window_size, 1), dtype=np.float32)

        # add Gaussian noise to pcd xyz
        image_obs = {}
        for key in self.image_keys:
            pcd = sample[key].astype(np.float32)  # (window_size, 3, N)
            noise = (np.random.rand(*pcd.shape) * 2 - 1) * self.pcd_noise
            pcd[:, :3, :] += noise[:, :3, :]
            # pcd: current frame only (last obs step in the window = index n_obs_steps-1)
            image_obs[key] = pcd[self.n_obs_steps - 1 : self.n_obs_steps]  # (1, 3, N)

        # obs: first n_obs_steps frames [t-n_obs_steps+1 .. t]
        # past_actions: actions at [t-n_obs_steps+1 .. t-1] (n_obs_steps-1 frames, empty when n_obs_steps=1)
        # action: frames [n_obs_steps-1 .. n_obs_steps-1+horizon] = actions at [t .. t+horizon-1]
        obs_dict = {"agent_pos": agent_pos[0:self.n_obs_steps]} | image_obs
        if self.n_obs_steps > 1:
            past_act = sample["action"][0:self.n_obs_steps - 1].astype(np.float32)
            _, _, sample_start_idx, _ = self.sampler.indices[idx]
            if sample_start_idx > 0:
                past_act[:sample_start_idx] = 0.0
            obs_dict["past_actions"] = past_act

        action_chunk = sample["action"][self.n_obs_steps - 1:].astype(np.float32)  # (horizon, A)
        if self.chunk_relative:
            # Zero out any padded frames (repeated last action) before cumsum so they
            # contribute zero displacement rather than a growing ramp.
            # sampler.indices[idx] = [buf_start, buf_end, sample_start, sample_end]
            # sample_end_idx is where real data ends in the full window.
            _, _, _, sample_end_idx = self.sampler.indices[idx]
            real_in_chunk = sample_end_idx - (self.n_obs_steps - 1)
            if real_in_chunk < self.horizon:
                action_chunk[real_in_chunk:] = 0.0
            action_chunk = np.cumsum(action_chunk, axis=0)

        data = {
            "obs": obs_dict,
            "action": action_chunk,
        }

        return {
            "obs": {k: torch.from_numpy(v) for k, v in data["obs"].items()},
            "action": torch.from_numpy(data["action"]),
        }
