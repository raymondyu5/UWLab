"""
RialTo Stage 1a: Standalone BC pretraining on zarr demos (no Isaac Sim required).

Trains a state-based MLP on offline demonstrations and saves a .pt checkpoint
that can be loaded by train_stage1_bc_ppo.py (via --bc_checkpoint) or
evaluated in sim with eval_bc_policy.py.

Usage:
    python scripts/reinforcement_learning/rialto/train_stage1_bc.py \
        --data_path data_storage/rialto/bottle_grasp \
        --log_dir logs/rialto/bc
"""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import zarr

parser = argparse.ArgumentParser(description="RialTo standalone BC pretraining.")
parser.add_argument("--data_path", type=str, required=True, help="Path to zarr demo directory.")
parser.add_argument(
    "--obs_keys", nargs="+", default=["arm_joint_pos", "hand_joint_pos", "manipulated_object_pose"],
    help="Zarr obs keys to concatenate into state.",
)
parser.add_argument("--action_key", type=str, default="actions")
parser.add_argument("--bc_epochs", type=int, default=500)
parser.add_argument("--bc_lr", type=float, default=3e-4)
parser.add_argument("--bc_batch_size", type=int, default=256)
parser.add_argument("--net_arch", nargs="+", type=int, default=[512, 512, 512],
                    help="Hidden layer sizes for SimpleMLP.")
parser.add_argument("--log_dir", type=str, default="logs/rialto/bc")
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_zarr_demos(data_path: str, obs_keys: list, action_key: str):
    root = Path(data_path)
    zarr_files = sorted(root.glob("episode_*/episode_*.zarr"))
    if not zarr_files:
        zarr_files = sorted(root.glob("*.zarr"))
    if not zarr_files:
        raise FileNotFoundError(f"No zarr files found under {root}")

    print(f"[BC] Loading {len(zarr_files)} episodes from {root}")
    obs_list, act_list = [], []

    for zpath in zarr_files:
        z = zarr.open(str(zpath), mode="r")
        obs_parts = []
        for key in obs_keys:
            arr = None
            for candidate in (f"data/{key}", key):
                if candidate in z:
                    arr = z[candidate][:].astype(np.float32)
                    break
            if arr is None:
                print(f"  [warn] '{key}' missing in {zpath.name}, skipping episode.")
                break
            obs_parts.append(arr)
        else:
            act = None
            for candidate in (f"data/{action_key}", action_key):
                if candidate in z:
                    act = z[candidate][:].astype(np.float32)
                    break
            if act is None:
                print(f"  [warn] '{action_key}' missing in {zpath.name}, skipping episode.")
                continue
            obs = np.concatenate(obs_parts, axis=-1)
            T = min(len(obs), len(act))
            obs_list.append(obs[:T])
            act_list.append(act[:T])

    if not obs_list:
        raise RuntimeError("No valid episodes loaded.")

    all_obs = np.concatenate(obs_list, axis=0)
    all_acts = np.concatenate(act_list, axis=0)
    print(f"[BC] {len(all_obs)} timesteps — obs {all_obs.shape}, acts {all_acts.shape}")
    return all_obs, all_acts


# ── Policy network ────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        layers, in_dim = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.shared_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(in_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.shared_net(x))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.join(args.log_dir, f"bc_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[BC] Logging to {log_dir}")

    demo_obs_np, demo_acts_np = load_zarr_demos(args.data_path, args.obs_keys, args.action_key)
    obs_dim = demo_obs_np.shape[-1]
    action_dim = demo_acts_np.shape[-1]
    print(f"[BC] obs_dim={obs_dim}  action_dim={action_dim}  net_arch={args.net_arch}")

    demo_obs_t = torch.tensor(demo_obs_np, dtype=torch.float32)
    demo_acts_t = torch.tensor(demo_acts_np, dtype=torch.float32)

    net = SimpleMLP(obs_dim, action_dim, hidden=tuple(args.net_arch)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.bc_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.bc_epochs, eta_min=1e-5
    )
    dataset = TensorDataset(demo_obs_t, demo_acts_t)
    loader = DataLoader(dataset, batch_size=args.bc_batch_size, shuffle=True, drop_last=True)

    import wandb
    if args.wandb_project:
        run_name = args.wandb_run_name or f"rialto-bc-{timestamp}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={k: v for k, v in vars(args).items()},
        )
        print(f"[BC] wandb run: {wandb.run.get_url()}")

    print(f"[BC] Training for {args.bc_epochs} epochs ...")
    net.train()
    for epoch in range(1, args.bc_epochs + 1):
        total_loss = 0.0
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            loss = F.mse_loss(net(obs_b), act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        if epoch % 50 == 0:
            print(f"  epoch {epoch:4d}/{args.bc_epochs}  loss={avg_loss:.5f}")
        if args.wandb_project and wandb.run is not None:
            wandb.log({"bc/loss": avg_loss, "bc/lr": scheduler.get_last_lr()[0]}, step=epoch)

    bc_ckpt = os.path.join(log_dir, "bc_pretrained.pt")
    torch.save({
        "state_dict": net.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "net_arch": args.net_arch,
        "obs_keys": args.obs_keys,
        "action_key": args.action_key,
    }, bc_ckpt)
    print(f"[BC] Checkpoint saved → {bc_ckpt}")

    if args.wandb_project and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
