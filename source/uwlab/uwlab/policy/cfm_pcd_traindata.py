"""Shared Zarr dataset + shape_meta builders for CFM PCD training scripts."""

from __future__ import annotations

from typing import Any, Dict, List, Union

from omegaconf import DictConfig

from uwlab_tasks.manager_based.manipulation.grasp.dataset.cotrain_zarr_dataset import (
    CotrainZarrDataset,
)
from uwlab_tasks.manager_based.manipulation.grasp.dataset.zarr_dataset import ZarrDataset


def build_zarr_dataset(cfg: DictConfig) -> Union[ZarrDataset, CotrainZarrDataset]:
    real_data_path = cfg.dataset.get("real_data_path", None)
    if real_data_path:
        sim_dataset = _one_zarr(cfg, cfg.dataset.data_path, cfg.training.seed)
        real_action_key = cfg.dataset.get("real_action_key", None) or cfg.dataset.action_key
        real_dataset = _one_zarr(cfg, real_data_path, cfg.training.seed, action_key=real_action_key)
        sim_ratio = cfg.dataset.get("sim_ratio", 0.95)
        return CotrainZarrDataset(
            sim_dataset=sim_dataset, real_dataset=real_dataset, sim_ratio=sim_ratio
        )
    return _one_zarr(cfg, cfg.dataset.data_path, cfg.training.seed)


def _one_zarr(cfg: DictConfig, data_path: str, seed: int, action_key=None) -> ZarrDataset:
    return ZarrDataset(
        data_path=data_path,
        load_list=list(cfg.dataset.load_list),
        obs_keys=list(cfg.dataset.obs_keys),
        action_key=action_key if action_key is not None else cfg.dataset.action_key,
        action_base_keys=list(cfg.dataset.action_base_keys)
        if cfg.dataset.get("action_base_keys")
        else None,
        image_keys=list(cfg.dataset.image_keys),
        horizon=cfg.horizon,
        n_obs_steps=cfg.n_obs_steps,
        pad_after=cfg.dataset.pad_after,
        val_ratio=cfg.dataset.val_ratio,
        seed=seed,
        downsample_points=cfg.dataset.downsample_points,
        pcd_noise=cfg.dataset.pcd_noise,
        noise_extrinsic=cfg.dataset.get("noise_extrinsic", False),
        noise_extrinsic_parameter=list(
            cfg.dataset.get("noise_extrinsic_parameter", [0.05, 0.2])
        ),
        obs_noise=dict(cfg.dataset.get("obs_noise", {})),
        hand_dropout_prob=cfg.dataset.get("hand_dropout_prob", 0.0),
        chunk_relative=bool(cfg.dataset.get("chunk_relative", False)),
    )


def shape_meta_for_unet(
    dataset: Union[ZarrDataset, CotrainZarrDataset],
    image_keys: List[str],
    downsample_points: int,
) -> Dict[str, Any]:
    obs: Dict[str, Any] = {
        "agent_pos": {"shape": [dataset.low_obs_dim], "type": "low_dim"},
    }
    for key in image_keys:
        obs[key] = {"shape": [3, downsample_points], "type": "pcd"}
    return {"action": {"shape": [dataset.action_dim]}, "obs": obs}


def shape_meta_for_maniflow(
    dataset: Union[ZarrDataset, CotrainZarrDataset],
    downsample_points: int,
    *,
    fuse_past_actions_in_state: bool = False,
    n_obs_steps: int = 1,
    concat_history_state: bool = True,
) -> Dict[str, Any]:
    """Build ``shape_meta`` for ``ManiFlowTransformerPointcloudPolicy``.

    ManiFlow does ``obs_dict = dict_apply(shape_meta['obs'], lambda x: x['shape'])``. Their
    ``dict_apply`` recurses into **nested** dicts; values like ``{"shape": [...], "type": "..."}``
    hit the ``type`` string and crash. YAML-style obs entries must therefore be **plain shape
    lists** at the top level of ``obs`` (not ``{"shape": ...}`` wrappers): each value is a list of
    ints so ``dict_apply`` only walks integer elements (no ``func`` on strings).

    When ``concat_history_state`` (UW ``single_frame_point_cloud``), the state MLP takes one
    vector: ``T * low_dim`` plus optionally ``(T - 1) * action_dim`` if fusing past actions.
    Otherwise (legacy multi-step PCD path) each step uses ``low_dim`` or ``low_dim + action_dim``.
    """
    low = dataset.low_obs_dim
    act = dataset.action_dim
    if concat_history_state:
        if n_obs_steps <= 1:
            agent_dim = low
        else:
            agent_dim = n_obs_steps * low
            if fuse_past_actions_in_state:
                agent_dim += (n_obs_steps - 1) * act
    else:
        agent_dim = low
        if fuse_past_actions_in_state and n_obs_steps > 1:
            agent_dim = low + act
    return {
        "action": {"shape": [dataset.action_dim]},
        "obs": {
            "agent_pos": [agent_dim],
            "point_cloud": [downsample_points, 3],
        },
    }
