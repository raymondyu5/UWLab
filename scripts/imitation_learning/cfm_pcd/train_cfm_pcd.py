"""
Train a CFM PCD policy with Hydra config.

Usage:
    ./isaaclab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
        --config-path ../../../configs/bc \
        --config-name train_cfm_pcd \
        dataset.data_path=/path/to/zarr/episodes \
        training.device=cuda:0
"""
import os
import sys

from uwlab.utils.paths import setup_third_party_paths
setup_third_party_paths()

import hydra
from omegaconf import DictConfig
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy
from uwlab.policy.train_cfm_workspace import TrainCFMWorkspace
from uwlab_tasks.manager_based.manipulation.grasp.dataset.zarr_dataset import ZarrDataset
from uwlab_tasks.manager_based.manipulation.grasp.dataset.cotrain_zarr_dataset import CotrainZarrDataset


def _build_zarr_dataset(cfg, data_path, seed):
    return ZarrDataset(
        data_path=data_path,
        load_list=list(cfg.dataset.load_list),
        obs_keys=list(cfg.dataset.obs_keys),
        action_key=cfg.dataset.action_key,
        action_base_keys=list(cfg.dataset.action_base_keys) if cfg.dataset.get("action_base_keys") else None,
        image_keys=list(cfg.dataset.image_keys),
        horizon=cfg.horizon,
        n_obs_steps=cfg.n_obs_steps,
        pad_after=cfg.dataset.pad_after,
        val_ratio=cfg.dataset.val_ratio,
        seed=seed,
        downsample_points=cfg.dataset.downsample_points,
        pcd_noise=cfg.dataset.pcd_noise,
        noise_extrinsic=cfg.dataset.noise_extrinsic,
        noise_extrinsic_parameter=list(cfg.dataset.noise_extrinsic_parameter),
        obs_noise=dict(cfg.dataset.get("obs_noise", {})),
    )


@hydra.main(config_path="../../../configs/bc", config_name="train_cfm_pcd", version_base=None)
def main(cfg: DictConfig):
    # dataset
    real_data_path = cfg.dataset.get("real_data_path", None)
    if real_data_path:
        sim_dataset = _build_zarr_dataset(cfg, cfg.dataset.data_path, cfg.training.seed)
        real_dataset = _build_zarr_dataset(cfg, real_data_path, cfg.training.seed)
        sim_ratio = cfg.dataset.get("sim_ratio", 0.95)
        dataset = CotrainZarrDataset(sim_dataset=sim_dataset, real_dataset=real_dataset, sim_ratio=sim_ratio)
    else:
        dataset = _build_zarr_dataset(cfg, cfg.dataset.data_path, cfg.training.seed)

    # shape_meta: infer action/obs dims from dataset
    shape_meta = {
        "action": {"shape": [dataset.action_dim]},
        "obs": {
            "agent_pos": {"shape": [dataset.low_obs_dim], "type": "low_dim"},
        },
    }
    for key in cfg.dataset.image_keys:
        shape_meta["obs"][key] = {
            "shape": [3, cfg.dataset.downsample_points],
            "type": "pcd",
        }

    # obs encoder
    pcd_model = PointNet(
        in_channels=cfg.pointnet.in_channels,
        local_channels=tuple(cfg.pointnet.local_channels),
        global_channels=tuple(cfg.pointnet.global_channels),
        use_bn=cfg.pointnet.use_bn,
    )
    obs_encoder = MultiPCDObsEncoder(shape_meta=shape_meta, pcd_model=pcd_model)

    # policy
    noise_scheduler = ConditionalFlowMatcher(sigma=cfg.policy.sigma)
    policy = CFMPCDPolicy(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        noise_scheduler=noise_scheduler,
        horizon=cfg.horizon,
        n_action_steps=cfg.n_action_steps,
        n_obs_steps=cfg.n_obs_steps,
        num_inference_steps=cfg.policy.num_inference_steps,
        diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
        down_dims=tuple(cfg.policy.down_dims),
        kernel_size=cfg.policy.kernel_size,
        n_groups=cfg.policy.n_groups,
        cond_predict_scale=cfg.policy.cond_predict_scale,
    )

    workspace = TrainCFMWorkspace(cfg=cfg, dataset=dataset, policy=policy)
    workspace.run()


if __name__ == "__main__":
    main()
