"""
RealDatasetLoader: loads real robot zarr episodes and precomputes a GPU pool
of actor obs for KL regularization of the PPO noise policy.

Matches absjoint BC training data format exactly:
  obs_keys:   [arm_joint_pos (7), hand_joint_pos (16)]  -> agent_pos (23D)
  action_key: [arm_joint_pos_target (7), hand_action (16)] -> action (23D, absolute)
  image_keys: [seg_pc]

Precomputed pool (built once at setup via precompute_pool()):
  Each valid (episode, timestep) window is replicated n_augmentations times
  with the same PCD augmentations used during BC training:
    1. Random point downsampling (N -> downsample_points)
    2. Extrinsic noise: random rotation + translation on XYZ
    3. PCD XYZ noise: uniform [-pcd_noise, pcd_noise]

  actor_agent_pos_history:    (P, n_obs_steps * 23)
  actor_past_actions_history: (P, (n_obs_steps-1) * 23)
  actor_pcd_emb:              (P, 256)
  Sampling is pure GPU indexing — zero overhead.
"""

import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Bypass the diffusion_policy zarr stub (third_party/diffusion_policy/zarr/__init__.py).
# The stub may already be cached in sys.modules (loaded when diffusion_policy was imported),
# so we must evict it before importing the real zarr.
_dp_paths = [p for p in sys.path if "diffusion_policy" in p]
for _p in _dp_paths:
    sys.path.remove(_p)
_cached_zarr = sys.modules.pop("zarr", None)
try:
    import zarr as _zarr
except ImportError as e:
    raise ImportError("zarr not importable in container.") from e
finally:
    if _cached_zarr is not None:
        sys.modules["zarr"] = _cached_zarr
    for _p in reversed(_dp_paths):
        sys.path.insert(0, _p)


class RealDatasetLoader:
    """Loads real robot episodes from zarr for noise-policy KL regularization.

    At init time all episodes are loaded into CPU memory.  Call
    precompute_pool(rfs_env) once before training to encode all PCDs through
    the frozen PointNet and build GPU tensors.  After that, sample_from_pool()
    returns random subsets with zero PointNet overhead.

    Args:
        dataset_path:      Directory containing episode_N/episode_N.zarr subdirs.
        n_obs_steps:       History length (must match CFM checkpoint, e.g. 4).
        action_dim:        Action dimension (23 for JointAbs).
        downsample_points: PCD points to keep after random downsampling (2048).
        device:            Torch device for output tensors.
    """

    def __init__(
        self,
        dataset_path: str,
        n_obs_steps: int,
        action_dim: int,
        downsample_points: int,
        device: torch.device,
    ):
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.downsample_points = downsample_points
        self.device = device
        self.episodes: list[dict] = []
        self._load_episodes(dataset_path)

        self._pool_agent_pos: torch.Tensor | None = None
        self._pool_past_actions: torch.Tensor | None = None
        self._pool_pcd_emb: torch.Tensor | None = None
        self._pool_size: int = 0

    def _load_episodes(self, dataset_path: str) -> None:
        ep_dirs = sorted(
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith("episode_")
        )
        skipped = 0
        for ep_dir in ep_dirs:
            zarr_path = os.path.join(dataset_path, ep_dir, ep_dir + ".zarr")
            if not os.path.exists(zarr_path):
                skipped += 1
                continue
            try:
                z = _zarr.open(zarr_path)
                data = z["data"]
                agent_pos = np.concatenate([
                    data["arm_joint_pos"][:],
                    data["hand_joint_pos"][:],
                ], axis=-1).astype(np.float32)                    # (T, 23)
                action = np.concatenate([
                    data["arm_joint_pos_target"][:],
                    data["hand_action"][:],
                ], axis=-1).astype(np.float32)                    # (T, 23)
                seg_pc = data["seg_pc"][:].astype(np.float32)     # (T, N, 3)
                T = len(agent_pos)
                if T < self.n_obs_steps:
                    skipped += 1
                    continue
                self.episodes.append({
                    "agent_pos": agent_pos,
                    "action": action,
                    "seg_pc": seg_pc,
                    "T": T,
                })
            except Exception as e:
                print(f"[RealDatasetLoader] Warning: skipping {zarr_path}: {e}")
                skipped += 1

        print(f"[RealDatasetLoader] Loaded {len(self.episodes)} real episodes "
              f"({skipped} skipped) from {dataset_path}")
        if len(self.episodes) == 0:
            raise RuntimeError(f"No valid episodes found in {dataset_path}")

    def precompute_pool(
        self,
        rfs_env,
        n_augmentations: int = 10,
        pcd_noise: float = 0.02,
        noise_extrinsic: bool = False,
        noise_extrinsic_parameter: list | None = None,
        encode_batch_size: int = 256,
    ) -> None:
        """Enumerate all valid windows, apply BC-matching PCD augmentations,
        encode through frozen PointNet, and store as GPU tensors.

        Each window is replicated n_augmentations times with independent:
          1. Random point downsampling  (matches PCDSequenceSampler._load_pcd)
          2. Extrinsic noise: rotation + translation  (matches _load_pcd)
          3. PCD XYZ noise: uniform [-pcd_noise, pcd_noise]  (matches ZarrDataset.__getitem__)
        """
        import time
        t_start = time.time()
        n_episodes = len(self.episodes)

        all_agent_pos = []
        all_past_actions = []
        all_pcd_embs = []

        for ep_idx, ep in enumerate(self.episodes):
            print(f"\r[RealDatasetLoader] Encoding episode {ep_idx + 1}/{n_episodes} "
                  f"(T={ep['T']}, {n_augmentations} augs)...", end="", flush=True)
            T = ep["T"]
            valid_start = self.n_obs_steps - 1

            ep_agent_pos = []
            ep_past_actions = []
            for t in range(valid_start, T):
                window = ep["agent_pos"][t - self.n_obs_steps + 1 : t + 1]
                ep_agent_pos.append(window.reshape(-1))
                if self.n_obs_steps > 1:
                    pa = ep["action"][t - self.n_obs_steps + 1 : t]
                    ep_past_actions.append(pa.reshape(-1))

            seg_pc_raw = ep["seg_pc"][valid_start:]  # (valid_T, N, 3)
            N_raw = seg_pc_raw.shape[1]
            valid_T = len(seg_pc_raw)

            for aug in range(n_augmentations):
                all_agent_pos.extend(ep_agent_pos)
                if ep_past_actions:
                    all_past_actions.extend(ep_past_actions)

                aug_frames = np.empty(
                    (valid_T, self.downsample_points, 3), dtype=np.float32
                )
                for j in range(valid_T):
                    frame = seg_pc_raw[j]  # (N, 3)
                    perm = np.random.permutation(N_raw)[:self.downsample_points]
                    frame = frame[perm, :]  # (K, 3)

                    if noise_extrinsic and noise_extrinsic_parameter is not None:
                        trans_scale, rot_scale = noise_extrinsic_parameter
                        euler = (np.random.rand(3) * 2 - 1) * rot_scale
                        rot_mat = R.from_euler("xyz", euler).as_matrix().astype(np.float32)
                        trans = ((np.random.rand(3) * 2 - 1) * trans_scale).astype(np.float32)
                        frame[:, :3] = frame[:, :3] @ rot_mat + trans

                    if pcd_noise > 0:
                        noise = (np.random.rand(*frame.shape).astype(np.float32) * 2 - 1) * pcd_noise
                        frame[:, :3] += noise[:, :3]

                    aug_frames[j] = frame

                seg_pc_t = torch.from_numpy(aug_frames).to(self.device).float()
                seg_pc_t = seg_pc_t.permute(0, 2, 1)  # (valid_T, 3, K)

                for i in range(0, len(seg_pc_t), encode_batch_size):
                    batch = seg_pc_t[i : i + encode_batch_size]
                    nobs = rfs_env.policy.normalizer["seg_pc"].normalize(batch.unsqueeze(1))
                    with torch.no_grad():
                        emb = rfs_env.policy.obs_encoder.encode_pcd_only(
                            {"seg_pc": nobs[:, 0]}
                        ).detach()
                    all_pcd_embs.append(emb)

                del seg_pc_t

        pool_size = len(all_agent_pos)
        self._pool_agent_pos = torch.from_numpy(
            np.stack(all_agent_pos)
        ).to(self.device)
        if all_past_actions:
            self._pool_past_actions = torch.from_numpy(
                np.stack(all_past_actions)
            ).to(self.device)
        else:
            self._pool_past_actions = None
        self._pool_pcd_emb = torch.cat(all_pcd_embs, dim=0)
        self._pool_size = pool_size

        assert self._pool_pcd_emb.shape[0] == pool_size, (
            f"PCD pool size {self._pool_pcd_emb.shape[0]} != window count {pool_size}"
        )

        elapsed = time.time() - t_start
        n_base = pool_size // n_augmentations
        mem_bytes = (
            self._pool_agent_pos.nelement() * self._pool_agent_pos.element_size()
            + (self._pool_past_actions.nelement() * self._pool_past_actions.element_size()
               if self._pool_past_actions is not None else 0)
            + self._pool_pcd_emb.nelement() * self._pool_pcd_emb.element_size()
        )
        print(f"\n[RealDatasetLoader] Precomputed pool in {elapsed:.1f}s: "
              f"{n_base} windows × {n_augmentations} augmentations = {pool_size} entries, "
              f"{mem_bytes / 1e6:.1f} MB on GPU"
              f" (pcd_noise={pcd_noise}, extrinsic={noise_extrinsic})")

        self.episodes.clear()

    def sample_from_pool(self, batch_size: int) -> dict:
        """Random-index into the precomputed pool. Pure GPU op, zero overhead."""
        idx = torch.randint(0, self._pool_size, (batch_size,), device=self.device)
        obs = {
            "actor_agent_pos_history": self._pool_agent_pos[idx],
            "actor_pcd_emb": self._pool_pcd_emb[idx],
        }
        if self._pool_past_actions is not None:
            obs["actor_past_actions_history"] = self._pool_past_actions[idx]
        return obs
