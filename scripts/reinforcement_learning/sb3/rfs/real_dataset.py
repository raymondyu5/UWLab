"""
RealDatasetLoader: loads real robot zarr episodes and samples actor obs
batches for KL regularization of the PPO noise policy.

Matches absjoint BC training data format exactly:
  obs_keys:   [arm_joint_pos (7), hand_joint_pos (16)]  -> agent_pos (23D)
  action_key: [arm_joint_pos_target (7), hand_action (16)] -> action (23D, absolute)
  image_keys: [seg_pc]

Actor obs built matches AsymmetricActorCriticPolicy actor input:
  actor_agent_pos_history:    (B, n_obs_steps * 23)
  actor_past_actions_history: (B, (n_obs_steps-1) * 23)  [when n_obs_steps > 1]
  actor_pcd_emb:        (B, pcd_feat_dim)  [frozen CFM PointNet, no_grad]
"""

import os
import sys

import numpy as np
import torch

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

    At init time all episodes are loaded into CPU memory. During training,
    sample_actor_obs() draws random windows and returns tensors ready for the
    PPO actor forward pass.

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

    def sample_actor_obs(self, batch_size: int, rfs_env) -> dict:
        """Sample a batch of actor obs from real data.

        Returns a dict of tensors matching the AsymmetricActorCriticPolicy
        actor input keys (actor_pcd_emb, actor_agent_pos_history, actor_past_actions_history).

        Args:
            batch_size: Number of samples to draw.
            rfs_env:    RFSWrapper instance (provides frozen CFM policy for PCD encoding).
        """
        ep_indices = np.random.randint(0, len(self.episodes), size=batch_size)

        agent_pos_windows = []
        past_action_windows = []
        seg_pc_frames = []

        for ep_idx in ep_indices:
            ep = self.episodes[ep_idx]
            T = ep["T"]
            # t must give a full n_obs_steps history without going before episode start.
            t = np.random.randint(self.n_obs_steps - 1, T)

            # agent_pos history: frames [t-(n-1), ..., t], shape (n_obs_steps, 23)
            agent_pos_windows.append(ep["agent_pos"][t - self.n_obs_steps + 1 : t + 1])

            # past actions: frames [t-(n-1), ..., t-1], shape (n_obs_steps-1, 23)
            if self.n_obs_steps > 1:
                past_action_windows.append(ep["action"][t - self.n_obs_steps + 1 : t])

            # PCD at current frame t, shape (N, 3)
            seg_pc_frames.append(ep["seg_pc"][t])

        # --- actor_agent_pos_history ---
        agent_pos = torch.from_numpy(
            np.stack(agent_pos_windows)     # (B, n_obs_steps, 23)
        ).to(self.device)
        obs_dict = {"actor_agent_pos_history": agent_pos.flatten(1)}   # (B, n_obs_steps*23)

        # --- actor_past_actions_history ---
        if self.n_obs_steps > 1:
            past_actions = torch.from_numpy(
                np.stack(past_action_windows)   # (B, n_obs_steps-1, 23)
            ).to(self.device)
            obs_dict["actor_past_actions_history"] = past_actions.flatten(1)

        # --- actor_pcd_emb (frozen CFM PointNet, no_grad) ---
        # Dataset stores seg_pc as (N, 3); CFM expects (B, 3, N).
        seg_pc = torch.from_numpy(
            np.stack(seg_pc_frames)     # (B, N, 3)
        ).to(self.device)
        seg_pc_t = seg_pc.permute(0, 2, 1)    # (B, 3, N)
        N = seg_pc_t.shape[-1]
        if N > self.downsample_points:
            perm = torch.randperm(N, device=self.device)[:self.downsample_points]
            seg_pc_t = seg_pc_t[:, :, perm]
        nobs_pcd = rfs_env.policy.normalizer["seg_pc"].normalize(seg_pc_t.unsqueeze(1))
        pcd_input = {"seg_pc": nobs_pcd[:, 0]}
        with torch.no_grad():
            pcd_emb = rfs_env.policy.obs_encoder.encode_pcd_only(pcd_input).detach()
        obs_dict["actor_pcd_emb"] = pcd_emb    # (B, 256)

        return obs_dict
