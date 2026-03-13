import copy
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class MultiPCDObsEncoder(ModuleAttrMixin):
    """
    Obs encoder for policies with point cloud + low-dim inputs.

    shape_meta example:
        {
          "obs": {
            "seg_pc":    {"shape": [3, 2048], "type": "pcd"},
            "agent_pos": {"shape": [30],      "type": "low_dim"},
          }
        }

    Forward input: obs_dict with matching keys.
      - pcd keys:     (B, 3, N) tensors -> passed through pcd_model
      - low_dim keys: (B, D) tensors -> concatenated directly

    Output: (B, pcd_feature_dim + low_dim_total)
    """

    def __init__(self, shape_meta: dict, pcd_model: nn.Module):
        super().__init__()

        pcd_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_shape_map = {}

        for key, attr in shape_meta["obs"].items():
            shape = tuple(attr["shape"])
            obs_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape

            if obs_type == "pcd":
                pcd_keys.append(key)
                key_model_map[key] = copy.deepcopy(pcd_model).to(self.device)
            elif obs_type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise ValueError(f"Unsupported obs type: {obs_type}")

        self.pcd_keys = sorted(pcd_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_model_map = key_model_map
        self.key_shape_map = key_shape_map
        self.shape_meta = shape_meta

    def encode_pcd_only(self, obs_dict: dict) -> torch.Tensor:
        """Encode only pcd keys. Input: {key: (B, 3, N)}. Returns (B, pcd_feat_dim)."""
        device = next(self.parameters()).device
        features = []
        for key in self.pcd_keys:
            pcd = obs_dict[key][:, :3].to(device, non_blocking=True)
            features.append(self.key_model_map[key](pcd))
        if not features:
            batch = next(iter(obs_dict.values()))
            return torch.zeros(batch.shape[0], 0, device=device)
        return torch.cat(features, dim=-1)

    def forward(self, obs_dict: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        features = []

        for key in self.pcd_keys:
            pcd = obs_dict[key][:, :3].to(device, non_blocking=True)  # (B, 3, N)
            features.append(self.key_model_map[key](pcd))

        for key in self.low_dim_keys:
            features.append(obs_dict[key].to(device, non_blocking=True))

        if not features:
            raise RuntimeError("No features to concatenate — check obs_dict keys.")

        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self) -> tuple:
        batch_size = 1
        dummy = {
            key: torch.zeros((batch_size,) + shape, dtype=self.dtype, device=self.device)
            for key, shape in self.key_shape_map.items()
        }
        return self.forward(dummy).shape[1:]
