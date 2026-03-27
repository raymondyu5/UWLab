from __future__ import annotations

import numpy as np
import torch


def _to_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return None
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return None


def extract_ckpt_metadata(ckpt: dict) -> dict:
    candidates = {
        "epoch": ["epoch", "current_epoch"],
        "global_step": ["global_step", "step"],
        "best_val_mse": ["best_val_mse", "val_action_mse", "val_mse"],
        "best_model_score": ["best_model_score", "monitor_best", "best_score"],
    }
    meta = {}
    for out_key, in_keys in candidates.items():
        for key in in_keys:
            if key in ckpt:
                scalar = _to_scalar(ckpt.get(key))
                if scalar is not None:
                    meta[out_key] = scalar
                    break
    return meta


def format_ckpt_metadata(meta: dict) -> str:
    if not meta:
        return "unavailable"
    parts = []
    for key in ("epoch", "global_step", "best_val_mse", "best_model_score"):
        if key in meta:
            value = meta[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.6g}")
            else:
                parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "unavailable"

