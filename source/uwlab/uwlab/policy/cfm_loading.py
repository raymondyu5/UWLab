"""Load a trained CFMPCDPolicy (and its training cfg) from a checkpoint directory. Same logic as `load_cfm_policy` in scripts/eval/play_bc.py
(which is a script and cannot be imported without launching the simulator). Call uwlab.utils.paths.setup_third_party_paths() first."""
from __future__ import annotations

import os

import torch
import yaml
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy


def find_checkpoint(checkpoint_dir: str, ckpt_name: str | None = None) -> str:
    for name in ([ckpt_name] if ckpt_name else ["best.ckpt", "latest.ckpt"]):
        path = os.path.join(checkpoint_dir, "checkpoints", name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}/checkpoints/")


def load_cfm_policy_dir(checkpoint_dir: str, device, num_inference_steps: int | None = None):
    """Returns (policy, train_cfg). The training cfg is embedded in the checkpoint (or .hydra/config.yaml for old ones)."""
    ckpt = torch.load(find_checkpoint(checkpoint_dir), map_location="cpu", weights_only=False)
    train_cfg = ckpt.get("cfg")
    if train_cfg is None:
        with open(os.path.join(checkpoint_dir, ".hydra", "config.yaml")) as f:
            train_cfg = yaml.safe_load(f)
    ds_cfg, pol_cfg, pn_cfg = train_cfg["dataset"], dict(train_cfg["policy"]), train_cfg["pointnet"]
    if num_inference_steps is not None:
        pol_cfg["num_inference_steps"] = int(num_inference_steps)

    sd = ckpt["ema_model"]
    low_obs_dim = sd["normalizer.params_dict.agent_pos.scale"].shape[0]
    action_dim = sd["normalizer.params_dict.action.scale"].shape[0]
    use_action_history = "normalizer.params_dict.past_actions.scale" in sd
    shape_meta = {"action": {"shape": [action_dim]}, "obs": {"agent_pos": {"shape": [low_obs_dim], "type": "low_dim"}}}
    for key in list(ds_cfg["image_keys"]):
        shape_meta["obs"][key] = {"shape": [3, int(ds_cfg["downsample_points"])], "type": "pcd"}

    pcd_model = PointNet(in_channels=int(pn_cfg["in_channels"]), local_channels=tuple(pn_cfg["local_channels"]),
                         global_channels=tuple(pn_cfg["global_channels"]), use_bn=bool(pn_cfg["use_bn"]))
    policy = CFMPCDPolicy(
        shape_meta=shape_meta,
        obs_encoder=MultiPCDObsEncoder(shape_meta=shape_meta, pcd_model=pcd_model),
        noise_scheduler=ConditionalFlowMatcher(sigma=float(pol_cfg["sigma"])),
        horizon=int(train_cfg["horizon"]),
        n_action_steps=int(train_cfg["n_action_steps"]),
        n_obs_steps=int(train_cfg["n_obs_steps"]),
        num_inference_steps=int(pol_cfg["num_inference_steps"]),
        diffusion_step_embed_dim=int(pol_cfg["diffusion_step_embed_dim"]),
        down_dims=tuple(pol_cfg["down_dims"]),
        kernel_size=int(pol_cfg["kernel_size"]),
        n_groups=int(pol_cfg["n_groups"]),
        cond_predict_scale=bool(pol_cfg["cond_predict_scale"]),
        use_action_history=use_action_history,
    )
    policy.load_state_dict(sd)
    return policy.to(device).eval(), train_cfg
