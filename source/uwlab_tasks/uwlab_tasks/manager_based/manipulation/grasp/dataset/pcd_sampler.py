from typing import Optional
import numpy as np
import numba
from scipy.spatial.transform import Rotation as R


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx
            ])
    return np.array(indices)


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask: np.ndarray, max_n: Optional[int], seed: int = 0) -> np.ndarray:
    if (max_n is not None) and (np.sum(mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        mask = np.zeros_like(mask)
        mask[train_idxs] = True
    return mask


class PCDSequenceSampler:
    """
    Samples fixed-length windows from a concatenated episode buffer.

    The replay_buffer must expose:
      - replay_buffer[key] -> np.ndarray of shape (total_steps, ...)
      - replay_buffer.episode_ends -> np.ndarray of shape (n_episodes,)
        containing cumulative step counts, e.g. [120, 240, 360, ...]

    image_keys are loaded with PCD augmentation (random downsample + optional
    extrinsic noise). All other keys are loaded as plain slices.
    """

    def __init__(
        self,
        replay_buffer,
        image_keys: list,
        downsample_points: int,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        episode_mask: Optional[np.ndarray] = None,
        noise_extrinsic: bool = False,
        noise_extrinsic_parameter: Optional[list] = None,
    ):
        assert sequence_length >= 1
        self.image_keys = image_keys
        self.downsample_points = downsample_points
        self.noise_extrinsic = noise_extrinsic
        self.noise_extrinsic_parameter = noise_extrinsic_parameter

        episode_ends = replay_buffer.episode_ends
        if episode_mask is None:
            episode_mask = np.ones(len(episode_ends), dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(
                episode_ends,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        self.indices = indices
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer

    def __len__(self):
        return len(self.indices)

    def _load_pcd(self, data: np.ndarray) -> np.ndarray:
        """
        data: (T, N, 3) raw point cloud sequence from buffer.
        Returns: (T, 3, downsample_points) — transposed, randomly downsampled,
                 with optional extrinsic noise applied to xyz.
        """
        data = data.astype(np.float32)
        # random downsample: pick downsample_points columns per frame
        perm = np.random.permutation(data.shape[1])[:self.downsample_points]
        data = data[:, perm, :]  # (T, downsample_points, 3)

        if self.noise_extrinsic and self.noise_extrinsic_parameter is not None:
            translation_scale, rotation_scale = self.noise_extrinsic_parameter
            euler_angles = (np.random.rand(3) * 2 - 1) * rotation_scale
            rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()
            translation = (np.random.rand(3) * 2 - 1) * translation_scale
            data[0, :, :3] = data[0, :, :3] @ rotation_matrix + translation

        return data.transpose(0, 2, 1)  # (T, 3, downsample_points)

    def sample_sequence(self, idx: int) -> dict:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        result = {}
        for key in self.replay_buffer.keys():
            input_arr = self.replay_buffer[key]
            sample = input_arr[buffer_start_idx:buffer_end_idx]

            if key in self.image_keys:
                sample = self._load_pcd(sample)

            # pad with zeros at start, repeat last frame at end
            if sample_start_idx > 0 or sample_end_idx < self.sequence_length:
                data = np.zeros(
                    (self.sequence_length,) + sample.shape[1:], dtype=sample.dtype
                )
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            else:
                data = sample

            result[key] = data

        return result
