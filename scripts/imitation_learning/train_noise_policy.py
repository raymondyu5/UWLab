"""
Train a supervised MLP noise policy on rendered PCD distillation data.

Maps (rendered seg_pc embedding, ee_pose, hand_joint_pos) -> noise (horizon * action_dim flat).
Uses a frozen PointNet from the CFM diffusion checkpoint to encode rendered seg_pc.
Supervises against sampled noise stored during distillation data collection (Gaussian policy loss).
Initialized from PPO actor weights — fine-tunes the existing noise policy for rendered PCD.
At eval time: sample N(predicted_mean, exp(log_std)) where log_std is copied from the PPO checkpoint.

Usage (inside container):
    ./uwlab.sh -p scripts/imitation_learning/train_noise_policy.py \\
        --config-path ../../../configs/bc \\
        --config-name train_noise_policy \\
        diffusion_checkpoint=logs/bc_cfm_pcd_bourbon_0312 \\
        ppo_checkpoint=logs/rfs/PourBottle_0315_1939/model_000400.zip \\
        dataset.data_path=logs/distill_collection/pour_bottle_0316 \\
        training.use_wandb=true
"""

import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
import zarr
from torch.utils.data import Dataset, DataLoader

from uwlab.utils.paths import setup_third_party_paths
setup_third_party_paths()

_RFS_PATH = "/workspace/uwlab/scripts/reinforcement_learning/sb3/rfs"
if _RFS_PATH not in sys.path:
    sys.path.insert(0, _RFS_PATH)

import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

from wrapper import _load_cfm_checkpoint
from asymmetric_policy import AsymmetricActorCriticPolicy


_ACT_FNS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class _RolloutBuffer(DictRolloutBuffer):
    def __init__(self, *args, gpu_buffer=False, **kwargs):
        super().__init__(*args, **kwargs)


class NoiseDataset(Dataset):
    """
    Per-step dataset from zarr distillation episodes.
    Returns per step: seg_pc (2048, 3), ee_pose (7,), hand_joint_pos (16,), noise (4, 22).
    noise is the sampled PPO output — used as Gaussian policy target.
    All data is loaded into RAM at init.
    """

    def __init__(self, data_path: str, val_ratio: float = 0.05, seed: int = 42, split: str = "train",
                 max_episodes: int = -1):
        episodes = sorted(
            [d for d in os.listdir(data_path) if d.startswith("episode_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if max_episodes > 0:
            episodes = episodes[:max_episodes]
        rng = random.Random(seed)
        rng.shuffle(episodes)
        n_val = max(1, int(len(episodes) * val_ratio))
        episodes = episodes[:n_val] if split == "val" else episodes[n_val:]

        seg_pc_list, ee_pose_list, hand_list, noise_list = [], [], [], []
        for ep_name in tqdm.tqdm(episodes, desc=f"Loading {split} episodes"):
            zarr_path = os.path.join(data_path, ep_name, f"{ep_name}.zarr")
            data = zarr.open(zarr_path, mode="r")["data"]
            seg_pc_list.append(data["seg_pc"][:].astype(np.float32))          # (T, 2048, 3)
            ee_pose_list.append(data["ee_pose"][:].astype(np.float32))         # (T, 7)
            hand_list.append(data["hand_joint_pos"][:].astype(np.float32))    # (T, 16)
            noise_list.append(data["noise"][:].astype(np.float32))             # (T, 4, 22)

        self.seg_pc         = np.concatenate(seg_pc_list,  axis=0)  # (N, 2048, 3)
        self.ee_pose        = np.concatenate(ee_pose_list, axis=0)  # (N, 7)
        self.hand_joint_pos = np.concatenate(hand_list,    axis=0)  # (N, 16)
        self.noise          = np.concatenate(noise_list,   axis=0)  # (N, 4, 22)

    def __len__(self):
        return len(self.seg_pc)

    def __getitem__(self, idx):
        return {
            "seg_pc":          self.seg_pc[idx],
            "ee_pose":         self.ee_pose[idx],
            "hand_joint_pos":  self.hand_joint_pos[idx],
            "noise":           self.noise[idx],
        }


class NoiseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, net_arch: list, activation_fn: type):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in net_arch:
            layers += [nn.Linear(in_dim, out_dim), activation_fn()]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _init_mlp_from_ppo(mlp: "NoiseMLP", ppo: PPO) -> None:
    """Copy PPO actor weights into NoiseMLP.

    PPO actor layout:
        mlp_extractor.policy_net: Sequential [Linear, ELU, Linear, ELU, Linear, ELU]  (indices 0-5)
        action_net:                Linear(64, 88)

    NoiseMLP.net layout:
        Sequential [Linear, ELU, Linear, ELU, Linear, ELU, Linear]  (indices 0-6)
    """
    ppo_net    = ppo.policy.mlp_extractor.policy_net
    action_net = ppo.policy.action_net
    with torch.no_grad():
        for i, module in enumerate(ppo_net):
            mlp.net[i].load_state_dict(module.state_dict())
        mlp.net[-1].load_state_dict(action_net.state_dict())
    print("[train_noise_policy] initialized MLP from PPO actor weights")


def _encode_pcd(seg_pc: torch.Tensor, diffusion, downsample_points: int) -> torch.Tensor:
    """Encode rendered seg_pc with the frozen CFM PointNet.

    Args:
        seg_pc: (B, N, 3) float tensor on device
    Returns:
        (B, pcd_feat_dim)
    """
    pcd = seg_pc.permute(0, 2, 1)  # (B, 3, N)
    N = pcd.shape[-1]
    if N > downsample_points:
        perm = torch.randperm(N, device=pcd.device)[:downsample_points]
        pcd = pcd[:, :, perm]
    # normalizer expects (B, T, 3, N) — add obs-step dim then remove after
    nobs = diffusion.normalizer.normalize({"seg_pc": pcd.unsqueeze(1)})
    pcd_normalized = nobs["seg_pc"][:, 0]  # (B, 3, N)
    return diffusion.obs_encoder.encode_pcd_only({"seg_pc": pcd_normalized})  # (B, pcd_feat_dim)


@hydra.main(config_path="../../configs/bc", config_name="train_noise_policy", version_base=None)
def main(cfg: DictConfig):
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(cfg.training.device)

    # --- frozen diffusion / PointNet ---
    diffusion, _ = _load_cfm_checkpoint(cfg.diffusion_checkpoint, device)
    for p in diffusion.parameters():
        p.requires_grad_(False)
    diffusion.eval()
    downsample_points = diffusion.obs_encoder.shape_meta["obs"]["seg_pc"]["shape"][-1]
    diffusion_horizon = diffusion.horizon
    action_dim        = diffusion.action_dim

    # --- PPO: source of log_std and actor weight initialization ---
    ppo = PPO.load(
        cfg.ppo_checkpoint,
        device=device,
        custom_objects={"rollout_buffer_class": _RolloutBuffer, "policy_class": AsymmetricActorCriticPolicy},
    )
    log_std = ppo.policy.log_std.detach().cpu()  # (output_dim,)

    # --- datasets ---
    max_episodes = cfg.dataset.get("max_episodes", -1)
    train_dataset = NoiseDataset(cfg.dataset.data_path, val_ratio=cfg.dataset.val_ratio, seed=seed, split="train", max_episodes=max_episodes)
    val_dataset   = NoiseDataset(cfg.dataset.data_path, val_ratio=cfg.dataset.val_ratio, seed=seed, split="val",   max_episodes=max_episodes)
    print(f"[train_noise_policy] train={len(train_dataset)} val={len(val_dataset)} steps")

    # --- infer dims from a dummy forward ---
    with torch.no_grad():
        dummy_emb = _encode_pcd(torch.zeros(1, downsample_points, 3, device=device), diffusion, downsample_points)
    pcd_feat_dim = dummy_emb.shape[-1]
    input_dim  = pcd_feat_dim + 7 + 16          # pcd_emb + ee_pose + hand_joint_pos
    output_dim = diffusion_horizon * action_dim  # 4 * 22 = 88
    print(f"[train_noise_policy] input_dim={input_dim} output_dim={output_dim} pcd_feat_dim={pcd_feat_dim}")

    # --- model: init from PPO actor weights, then free PPO ---
    activation_fn = _ACT_FNS[cfg.activation_fn]
    mlp = NoiseMLP(input_dim, output_dim, list(cfg.net_arch), activation_fn).to(device)
    _init_mlp_from_ppo(mlp, ppo)
    del ppo
    ema_mlp = copy.deepcopy(mlp)

    optimizer = torch.optim.AdamW(
        mlp.parameters(),
        lr=cfg.training.lr,
        betas=tuple(cfg.training.get("betas", [0.95, 0.999])),
        weight_decay=cfg.training.weight_decay,
    )
    optimizer_to(optimizer, device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=True,
        pin_memory=cfg.dataloader.get("pin_memory", True),
        persistent_workers=cfg.dataloader.get("persistent_workers", False),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=False,
        pin_memory=cfg.dataloader.get("pin_memory", True),
        persistent_workers=cfg.dataloader.get("persistent_workers", False),
    )

    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * cfg.training.num_epochs,
        last_epoch=-1,
    )

    ema_cfg = cfg.get("ema", {})
    ema = EMAModel(
        model=ema_mlp,
        update_after_step=ema_cfg.get("update_after_step", 0),
        inv_gamma=ema_cfg.get("inv_gamma", 1.0),
        power=ema_cfg.get("power", 0.75),
        min_value=ema_cfg.get("min_value", 0.0),
        max_value=ema_cfg.get("max_value", 0.9999),
    )

    ckpt_dir = cfg.training.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    use_wandb = cfg.training.get("use_wandb", False)
    if use_wandb:
        wandb.init(
            project=cfg.training.wandb_project,
            name=cfg.training.get("wandb_run_name", None),
            config=dict(cfg),
        )

    def _save_checkpoint(path):
        torch.save({
            "model":            mlp.state_dict(),
            "ema_model":        ema_mlp.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "global_step":      global_step,
            "epoch":            epoch,
            "best_val_mse":     best_val_mse,
            # everything needed to reconstruct + use the model outside this script
            "log_std":          log_std,
            "net_arch":         list(cfg.net_arch),
            "activation_fn":    cfg.activation_fn,
            "input_dim":        input_dim,
            "output_dim":       output_dim,
            "diffusion_horizon": diffusion_horizon,
            "action_dim":       action_dim,
        }, path)

    global_step  = 0
    epoch        = 0
    best_val_mse = float("inf")

    for _ in range(cfg.training.num_epochs):
        # --- train ---
        mlp.train()
        train_losses = []
        with tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False) as pbar:
            for batch in pbar:
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                with torch.no_grad():
                    pcd_emb = _encode_pcd(batch["seg_pc"], diffusion, downsample_points)
                x      = torch.cat([pcd_emb, batch["ee_pose"], batch["hand_joint_pos"]], dim=-1)
                pred   = mlp(x)
                target = batch["noise"].reshape(pred.shape[0], -1)
                loss   = nn.functional.mse_loss(pred, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                ema.step(mlp)
                train_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
                global_step += 1

        log = {
            "train_loss": np.mean(train_losses),
            "epoch":      epoch,
            "lr":         lr_scheduler.get_last_lr()[0],
        }

        # --- val ---
        if epoch % cfg.training.val_every == 0:
            ema_mlp.eval()
            val_mses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    batch  = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    pcd_emb = _encode_pcd(batch["seg_pc"], diffusion, downsample_points)
                    x      = torch.cat([pcd_emb, batch["ee_pose"], batch["hand_joint_pos"]], dim=-1)
                    pred   = ema_mlp(x)
                    target = batch["noise"].reshape(pred.shape[0], -1)
                    val_mses.append(nn.functional.mse_loss(pred, target).item())
            val_mse = np.mean(val_mses)
            log["val_mse"] = val_mse
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                _save_checkpoint(os.path.join(ckpt_dir, "best.ckpt"))

        # --- checkpoint ---
        if epoch % cfg.training.checkpoint_every == 0:
            _save_checkpoint(os.path.join(ckpt_dir, "latest.ckpt"))
            _save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.ckpt"))

        if use_wandb:
            wandb.log(log, step=global_step)

        epoch += 1

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
