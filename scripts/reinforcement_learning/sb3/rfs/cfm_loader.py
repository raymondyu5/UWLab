"""CFM checkpoint loading and small helpers for RFSWrapper (split out of wrapper.py)."""

import os

from uwlab.utils.paths import setup_third_party_paths

setup_third_party_paths()  # dill, diffusion_policy (Isaac's python.sh resets PYTHONPATH); must precede imports below

import dill
import torch
import torch.nn as nn
import yaml
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy
from uwlab.utils.checkpoint import extract_ckpt_metadata, format_ckpt_metadata


class LPFilter(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.y = None

    def reset(self):
        self.y = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.y is None:
            self.y = x.clone()
            return self.y.clone()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.clone()


def _resolve_hydra_refs(cfg: dict) -> dict:
    """Resolve simple ${key} Hydra references using top-level scalar values."""
    import re
    lookup = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}

    def resolve(val):
        if isinstance(val, str):
            def replacer(m):
                key = m.group(1)
                return str(lookup.get(key, m.group(0)))
            return re.sub(r'\$\{(\w+)\}', replacer, val)
        elif isinstance(val, dict):
            return {k: resolve(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [resolve(v) for v in val]
        return val

    return resolve(cfg)


def _load_cfm_checkpoint(diffusion_path: str, device: torch.device):
    ckpt_path = os.path.join(diffusion_path, "checkpoints", "best.ckpt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(diffusion_path, "best.ckpt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(diffusion_path, "checkpoints", "latest.ckpt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found. Looked for best.ckpt, checkpoints/best.ckpt, latest.ckpt in {diffusion_path}"
        )

    print(f"[RFSWrapper] Loading CFM checkpoint from {ckpt_path}")
    resolved_ckpt_path = os.path.realpath(ckpt_path)
    if resolved_ckpt_path != ckpt_path:
        print(f"[RFSWrapper] Resolved checkpoint path: {resolved_ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, pickle_module=dill)
    ckpt_meta = extract_ckpt_metadata(ckpt)
    print(f"[RFSWrapper] Checkpoint metadata: {format_ckpt_metadata(ckpt_meta)}")

    if "state_dicts" in ckpt and "ema_model" in ckpt["state_dicts"]:
        # Format A: state_dicts wrapper with OmegaConf cfg (older training code).
        state_dict = ckpt["state_dicts"]["ema_model"]
        cfg = ckpt["cfg"]
        shape_meta = {
            "action": {"shape": list(cfg.shape_meta.action.shape)},
            "obs": {
                "agent_pos": {"shape": list(cfg.shape_meta.obs.agent_pos.shape), "type": "low_dim"},
                "seg_pc": {"shape": list(cfg.shape_meta.obs.seg_pc.shape), "type": "pcd"},
            },
        }
        sigma = float(cfg.policy.noise_scheduler.sigma)
        horizon = int(cfg.horizon)
        n_action_steps = int(cfg.n_action_steps) + int(getattr(cfg, "n_latency_steps", 0))
        n_obs_steps = int(cfg.n_obs_steps)
        down_dims = tuple(cfg.policy.down_dims)
        obs_keys = list(cfg.dataset.obs_keys)
        image_keys = list(getattr(cfg.dataset, "image_keys", ["seg_pc"]))
        downsample_points = int(getattr(cfg.dataset, "downsample_points", 2048))
        pnet_cfg = {}  # Format A checkpoint: use PointNet defaults
    else:
        # Format B: flat checkpoint — cfg embedded in checkpoint or from config.yaml.
        state_dict = ckpt["ema_model"]
        if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
            train_cfg = ckpt["cfg"]
        else:
            config_path = os.path.join(diffusion_path, "config.yaml")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(
                    f"BC checkpoint format detected but config.yaml not found at {config_path}"
                )
            with open(config_path) as f:
                train_cfg = _resolve_hydra_refs(yaml.safe_load(f))
        agent_pos_dim = int(state_dict["normalizer.params_dict.agent_pos.scale"].shape[0])
        action_dim = int(state_dict["normalizer.params_dict.action.scale"].shape[0])
        ds = train_cfg["dataset"]
        pol = train_cfg["policy"]
        downsample_points = int(ds.get("downsample_points", 2048))
        obs_keys = ds.get("obs_keys", ds.get("obs_key", []))
        image_keys = ds.get("image_keys", ["seg_pc"])
        shape_meta = {
            "action": {"shape": [action_dim]},
            "obs": {
                "agent_pos": {"shape": [agent_pos_dim], "type": "low_dim"},
                "seg_pc": {"shape": [3, downsample_points], "type": "pcd"},
            },
        }
        sigma = float(pol.get("sigma", 0.0))
        horizon = int(train_cfg.get("horizon", 4))
        n_action_steps = int(train_cfg.get("n_action_steps", 8))
        n_obs_steps = int(train_cfg.get("n_obs_steps", 1))
        down_dims = tuple(pol["down_dims"])
        pnet_cfg = train_cfg.get("pointnet", {})

    use_action_history = "normalizer.params_dict.past_actions.scale" in state_dict

    # PointNet architecture: read from checkpoint cfg (Format B) with fallback to
    # legacy hardcoded defaults so old checkpoints continue to load correctly.
    _default_local  = (64, 64, 64, 128, 1024)
    _default_global = (512, 256)
    pcd_model = PointNet(
        in_channels=3,
        local_channels=tuple(pnet_cfg.get("local_channels", _default_local)),
        global_channels=tuple(pnet_cfg.get("global_channels", _default_global)),
        use_bn=bool(pnet_cfg.get("use_bn", False)),
    )
    obs_encoder = MultiPCDObsEncoder(shape_meta=shape_meta, pcd_model=pcd_model)
    noise_scheduler = ConditionalFlowMatcher(sigma=sigma)

    policy = CFMPCDPolicy(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=5,
        diffusion_step_embed_dim=256,
        down_dims=down_dims,
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        use_action_history=use_action_history,
    )
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()

    # agent_pos_flat_dim = n_obs_steps * per_step_dim (total flat size of history tensor)
    agent_pos_flat_dim = n_obs_steps * int(state_dict["normalizer.params_dict.agent_pos.scale"].shape[0])

    metadata = {
        "obs_keys": obs_keys,
        "image_keys": image_keys,
        "downsample_points": downsample_points,
        "n_obs_steps": n_obs_steps,
        "use_action_history": use_action_history,
        "agent_pos_flat_dim": agent_pos_flat_dim,
        "ckpt_path": ckpt_path,
        "resolved_ckpt_path": resolved_ckpt_path,
        "ckpt_meta": ckpt_meta,
    }

    print(f"[RFSWrapper] CFM policy loaded: horizon={policy.horizon}, "
          f"action_dim={policy.action_dim}, n_obs_steps={n_obs_steps}, "
          f"use_action_history={use_action_history}")
    return policy, metadata
