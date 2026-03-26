"""
Train a CFM PCD policy with Hydra config.

Usage:
    ./isaaclab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
        --config-path ../../../configs/bc \
        --config-name train_cfm_pcd \
        dataset.data_path=/path/to/zarr/episodes \
        training.device=cuda:0

ManiFlow (DP3 + DiTX): ``--config-name train_cfm_pcd_maniflow`` or see ``train_cfm_pcd_maniflow.py``.
"""
from uwlab.utils.paths import setup_third_party_paths

setup_third_party_paths()

import hydra
from omegaconf import DictConfig
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy
from uwlab.policy.cfm_pcd_traindata import (
    build_zarr_dataset,
    shape_meta_for_unet,
)
from uwlab.policy.train_cfm_workspace import TrainCFMWorkspace


@hydra.main(config_path="../../../configs/bc", config_name="train_cfm_pcd", version_base=None)
def main(cfg: DictConfig):
    dataset = build_zarr_dataset(cfg)
    shape_meta = shape_meta_for_unet(
        dataset, list(cfg.dataset.image_keys), cfg.dataset.downsample_points
    )

    pcd_model = PointNet(
        in_channels=cfg.pointnet.in_channels,
        local_channels=tuple(cfg.pointnet.local_channels),
        global_channels=tuple(cfg.pointnet.global_channels),
        use_bn=cfg.pointnet.use_bn,
    )
    obs_encoder = MultiPCDObsEncoder(shape_meta=shape_meta, pcd_model=pcd_model)

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
        use_action_history=bool(cfg.policy.get("use_action_history", False)),
    )

    workspace = TrainCFMWorkspace(cfg=cfg, dataset=dataset, policy=policy)
    workspace.run()


if __name__ == "__main__":
    main()
