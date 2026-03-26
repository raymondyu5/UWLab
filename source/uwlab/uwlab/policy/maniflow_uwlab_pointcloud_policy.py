"""
ManiFlowTransformerPointcloudPolicy adapted for UWLab Zarr batches.

UWLab dataloaders emit ``seg_pc`` (B, 1, 3, N) for the current frame and ``agent_pos``
(B, T, D). With ``single_frame_point_cloud``, PointNet runs once; the state MLP takes a
**single flattened** vector: all ``T`` proprio steps, plus flattened ``past_actions`` when
``fuse_past_actions_in_state`` is true. DiTX conditioning uses ``N`` tokens (one per point),
not ``T * N``.

Use :meth:`ZarrDataset.get_maniflow_normalizer` and :func:`shape_meta_for_maniflow` with
matching ``single_frame_point_cloud``, ``fuse_past_actions_in_state``, and ``n_obs_steps``.
"""

from __future__ import annotations

from typing import Dict

import torch
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy


class ManiFlowUWPointcloudPolicy(ManiFlowTransformerPointcloudPolicy):
    """Same as ManiFlow point-cloud policy with UWLab observation key/layout handling."""

    def __init__(self, *args, pcd_key: str = "seg_pc", **kwargs):
        self._uwlab_pcd_key = pcd_key
        super().__init__(*args, **kwargs)

    def _pcd_b_t_3_n_to_b_t_n_3(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected PCD (B,T,3,N), got shape {tuple(x.shape)}")
        if (
            x.shape[1] == 1
            and self.n_obs_steps > 1
            and not getattr(self, "single_frame_point_cloud", False)
        ):
            x = x.expand(-1, self.n_obs_steps, -1, -1)
        return x.permute(0, 1, 3, 2).contiguous()

    def _remap_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            if k == "past_actions":
                if (
                    getattr(self, "fuse_past_actions_in_state", False)
                    and self.n_obs_steps > 1
                ):
                    out["past_actions"] = v
                continue
            if k == self._uwlab_pcd_key:
                out["point_cloud"] = self._pcd_b_t_3_n_to_b_t_n_3(v)
            else:
                out[k] = v
        return out

    def compute_loss(self, batch, ema_model=None, **kwargs):
        b = dict(batch)
        b["obs"] = self._remap_obs(b["obs"])
        return super().compute_loss(b, ema_model=ema_model, **kwargs)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super().predict_action(self._remap_obs(obs_dict))
