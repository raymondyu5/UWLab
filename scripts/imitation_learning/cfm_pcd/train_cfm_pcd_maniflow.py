"""
Train ManiFlow (DP3Encoder + DiTX) on UWLab Zarr data.

Usage:
    ./isaaclab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd_maniflow.py \
        --config-path ../../../configs/bc \
        --config-name train_cfm_pcd_maniflow \
        dataset.data_path=/path/to/zarr/episodes \
        training.device=cuda:0

Switch vs Unet+torchcfm: use ``train_cfm_pcd.py`` with ``--config-name train_cfm_pcd`` (defaults).

With ``policy.single_frame_point_cloud: true`` (default), obs history is **flattened** into one
state vector (plus flattened ``past_actions`` if ``policy.fuse_past_actions_in_state: true``).
DiTX cross-attention length is ``downsample_points``, not ``n_obs_steps * downsample_points``.
"""
from uwlab.utils.paths import setup_maniflow_path, setup_third_party_paths

setup_third_party_paths()
setup_maniflow_path()

import hydra
from omegaconf import DictConfig, OmegaConf

from uwlab.policy.cfm_pcd_traindata import build_zarr_dataset, shape_meta_for_maniflow
from uwlab.policy.maniflow_uwlab_pointcloud_policy import ManiFlowUWPointcloudPolicy
from uwlab.policy.train_cfm_workspace import TrainCFMWorkspace


def _build_maniflow_policy(cfg: DictConfig, shape_meta: dict) -> ManiFlowUWPointcloudPolicy:
    """Construct policy from YAML without hydra.utils.instantiate (no _target_ import resolution)."""
    kwargs = OmegaConf.to_container(cfg.policy, resolve=True)
    kwargs.pop("_target_", None)
    # DP3Encoder uses attribute access on pointcloud_encoder_cfg (e.g. .state_mlp_size, .in_channels =).
    # Plain dicts from to_container() break that; OmegaConf supports both attr and key access.
    pec = kwargs.get("pointcloud_encoder_cfg")
    if isinstance(pec, dict):
        kwargs["pointcloud_encoder_cfg"] = OmegaConf.create(pec)
    return ManiFlowUWPointcloudPolicy(shape_meta=shape_meta, **kwargs)


@hydra.main(
    config_path="../../../configs/bc",
    config_name="train_cfm_pcd_maniflow",
    version_base=None,
)
def main(cfg: DictConfig):
    fuse_pa = bool(cfg.policy.get("fuse_past_actions_in_state", False))
    if fuse_pa and cfg.n_obs_steps <= 1:
        raise ValueError("fuse_past_actions_in_state requires n_obs_steps > 1")
    single_sf = bool(cfg.policy.get("single_frame_point_cloud", True))
    dataset = build_zarr_dataset(cfg)
    shape_meta = shape_meta_for_maniflow(
        dataset,
        cfg.dataset.downsample_points,
        fuse_past_actions_in_state=fuse_pa,
        n_obs_steps=cfg.n_obs_steps,
        concat_history_state=single_sf,
    )
    policy = _build_maniflow_policy(cfg, shape_meta)
    normalizer = dataset.get_maniflow_normalizer()
    workspace = TrainCFMWorkspace(
        cfg=cfg, dataset=dataset, policy=policy, normalizer=normalizer
    )
    workspace.run()


if __name__ == "__main__":
    main()
