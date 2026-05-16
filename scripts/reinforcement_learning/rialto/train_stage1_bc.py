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
parser.add_argument("--val_frac", type=float, default=0.1,
                    help="Fraction of episodes held out for validation.")
parser.add_argument("--log_dir", type=str, default="logs/rialto/bc")
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_zarr_episodes(data_path: str, obs_keys: list, action_key: str):
    """Return list of (obs_array, act_array) per valid episode."""
    root = Path(data_path)
    zarr_files = sorted(root.glob("episode_*/episode_*.zarr"))
    if not zarr_files:
        zarr_files = sorted(root.glob("*.zarr"))
    if not zarr_files:
        raise FileNotFoundError(f"No zarr files found under {root}")

    print(f"[BC] Found {len(zarr_files)} episodes in {root}")
    episodes = []

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
            min_T = min(arr.shape[0] for arr in obs_parts)
            obs_parts = [arr[:min_T] for arr in obs_parts]
            obs = np.concatenate(obs_parts, axis=-1)
            T = min(len(obs), len(act))
            episodes.append((obs[:T], act[:T]))

    if not episodes:
        raise RuntimeError("No valid episodes loaded.")
    return episodes


def episodes_to_tensors(episodes):
    obs = np.concatenate([e[0] for e in episodes], axis=0)
    acts = np.concatenate([e[1] for e in episodes], axis=0)
    return torch.tensor(obs, dtype=torch.float32), torch.tensor(acts, dtype=torch.float32)


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


# ── Checkpoint helper ─────────────────────────────────────────────────────────

def make_ckpt(net, obs_dim, action_dim, epoch, val_loss):
    return {
        "state_dict": net.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "net_arch": args.net_arch,
        "obs_keys": args.obs_keys,
        "action_key": args.action_key,
        "epoch": epoch,
        "val_loss": val_loss,
    }


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

    # ── Load and split by episode ──────────────────────────────────────────────
    all_episodes = load_zarr_episodes(args.data_path, args.obs_keys, args.action_key)
    random.shuffle(all_episodes)
    n_val = max(1, int(len(all_episodes) * args.val_frac))
    val_episodes = all_episodes[:n_val]
    train_episodes = all_episodes[n_val:]
    print(f"[BC] {len(train_episodes)} train episodes / {len(val_episodes)} val episodes")

    train_obs_t, train_acts_t = episodes_to_tensors(train_episodes)
    val_obs_t, val_acts_t = episodes_to_tensors(val_episodes)

    obs_dim = train_obs_t.shape[-1]
    action_dim = train_acts_t.shape[-1]
    print(f"[BC] obs_dim={obs_dim}  action_dim={action_dim}  net_arch={args.net_arch}")
    print(f"[BC] {len(train_obs_t)} train steps / {len(val_obs_t)} val steps")

    net = SimpleMLP(obs_dim, action_dim, hidden=tuple(args.net_arch)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.bc_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.bc_epochs, eta_min=1e-5
    )
    train_loader = DataLoader(
        TensorDataset(train_obs_t, train_acts_t),
        batch_size=args.bc_batch_size, shuffle=True, drop_last=True, num_workers=4,
    )
    val_obs_dev = val_obs_t.to(device)
    val_acts_dev = val_acts_t.to(device)

    import wandb
    if args.wandb_project:
        run_name = args.wandb_run_name or f"rialto-bc-{timestamp}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={k: v for k, v in vars(args).items()},
        )
        print(f"[BC] wandb run: {wandb.run.get_url()}")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"[BC] Training for {args.bc_epochs} epochs ...")
    net.train()
    for epoch in range(1, args.bc_epochs + 1):
        total_loss = 0.0
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            loss = F.mse_loss(net(obs_b), act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        net.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(net(val_obs_dev), val_acts_dev).item()
        net.train()
        val_losses.append(val_loss)

        if epoch % 50 == 0:
            print(f"  epoch {epoch:4d}/{args.bc_epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(make_ckpt(net, obs_dim, action_dim, epoch, val_loss),
                       os.path.join(log_dir, "bc_best.pt"))

        if args.wandb_project and wandb.run is not None:
            wandb.log({"bc/train_loss": train_loss, "bc/val_loss": val_loss,
                       "bc/lr": scheduler.get_last_lr()[0]}, step=epoch)

    # ── Save final checkpoint ──────────────────────────────────────────────────
    torch.save(make_ckpt(net, obs_dim, action_dim, args.bc_epochs, val_losses[-1]),
               os.path.join(log_dir, "bc_final.pt"))
    print(f"[BC] Final checkpoint → {log_dir}/bc_final.pt")
    print(f"[BC] Best checkpoint  → {log_dir}/bc_best.pt  (val={best_val_loss:.5f} at epoch {best_epoch})")

    # ── Plot loss curves ───────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, args.bc_epochs + 1))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, train_losses, label="train")
        ax.plot(epochs, val_losses, label="val")
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8, label=f"best val (ep {best_epoch})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title("BC pretraining loss")
        ax.legend()
        ax.set_yscale("log")
        fig.tight_layout()
        plot_path = os.path.join(log_dir, "loss_curve.png")
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[BC] Loss curve → {plot_path}")
    except Exception as e:
        print(f"[BC] Could not save loss plot: {e}")

    if args.wandb_project and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
