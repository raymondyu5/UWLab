"""
Train a CFM PCD policy with Hydra config.

Usage:
    ./isaaclab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
        --config-path ../../../configs/bc \
        --config-name train_cfm_pcd \
        dataset.data_path=/path/to/zarr/episodes \
        training.device=cuda:0
"""
import hydra
from omegaconf import DictConfig
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy
from uwlab.policy.train_cfm_workspace import TrainCFMWorkspace
from uwlab_tasks.manager_based.manipulation.grasp.dataset.zarr_dataset import ZarrDataset


@hydra.main(config_path="../../../configs/bc", config_name="train_cfm_pcd", version_base=None)
def main(cfg: DictConfig):
    # dataset
    dataset = ZarrDataset(
        data_path=cfg.dataset.data_path,
        load_list=list(cfg.dataset.load_list),
        num_demo=cfg.dataset.num_demo,
        obs_keys=list(cfg.dataset.obs_keys),
        action_key=cfg.dataset.action_key,
        image_keys=list(cfg.dataset.image_keys),
        horizon=cfg.horizon,
        pad_before=cfg.dataset.pad_before,
        pad_after=cfg.dataset.pad_after,
        val_ratio=cfg.dataset.val_ratio,
        seed=cfg.training.seed,
        downsample_points=cfg.dataset.downsample_points,
        pcd_noise=cfg.dataset.pcd_noise,
        noise_extrinsic=cfg.dataset.noise_extrinsic,
        noise_extrinsic_parameter=list(cfg.dataset.noise_extrinsic_parameter),
        obs_noise=dict(cfg.dataset.get("obs_noise", {})),
    )

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
