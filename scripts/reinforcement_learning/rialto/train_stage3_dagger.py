"""
RialTo Stage 3: DAgger in sim with teacher labels + cotraining against zarr real demos.

Loads the Stage 1 teacher MLP, rolls out a student MLP in sim, labels visited states
with teacher actions, and trains with a weighted cotraining loss:

    loss = real_coef * MSE(student(real_obs), real_acts)
         + (1 - real_coef) * MSE(student(dagger_obs), teacher_acts)

Usage:
    ./uwlab.sh -p scripts/reinforcement_learning/rialto/train_stage3_dagger.py \
        --task UW-FrankaLeap-GraspBottleRandomResets-JointAbs-v0 \
        --data_path /path/to/zarr/episodes \
        --teacher_checkpoint logs/rialto/stage1/stage1_MMDD_HHMM/stage1_final.zip \
        --num_envs 4 \
        --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RialTo Stage 3: DAgger + cotraining.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True, help="Path to zarr real demo directory.")
parser.add_argument("--teacher_checkpoint", type=str, required=True,
                    help="Stage 1 SB3 PPO checkpoint (.zip path).")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--obs_keys", nargs="+", default=["ee_pose", "hand_joint_pos", "manipulated_object_pose"],
    help="Must match the obs_keys used in Stage 1 (including privileged object pose).",
)
parser.add_argument("--action_key", type=str, default="actions")
# DAgger loop
parser.add_argument("--dagger_iterations", type=int, default=50,
                    help="Number of collect → train cycles.")
parser.add_argument("--collection_steps", type=int, default=2000,
                    help="Env steps per iteration (num_envs parallel, so wall-clock steps = this).")
parser.add_argument("--train_steps_per_iter", type=int, default=500,
                    help="Gradient steps on student per iteration.")
parser.add_argument("--student_lr", type=float, default=1e-3)
parser.add_argument("--student_batch_size", type=int, default=256)
parser.add_argument("--real_coef", type=float, default=0.5,
                    help="Weight on real-demo loss (1-real_coef goes to DAgger loss).")
parser.add_argument("--sampling_expert", type=float, default=0.0,
                    help="Prob of taking teacher action during rollout (0=pure student, β in DAgger).")
parser.add_argument("--warm_start_from_teacher", action="store_true", default=True,
                    help="Init student weights from teacher checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume student from this .pt file (overrides --warm_start_from_teacher).")
parser.add_argument("--log_dir", type=str, default="logs/rialto/stage3")
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── all Isaac-dependent imports after AppLauncher ──────────────────────────

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
import gymnasium as gym

from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap import (
    RL_MODE,
    parse_franka_leap_env_cfg,
)


# ── Shared utilities (duplicated from Stage 1 to keep scripts self-contained) ──

def load_zarr_demos(data_path: str, obs_keys: list, action_key: str):
    root = Path(data_path)
    zarr_files = sorted(root.glob("episode_*/episode_*.zarr"))
    if not zarr_files:
        zarr_files = sorted(root.glob("*.zarr"))
    if not zarr_files:
        raise FileNotFoundError(f"No zarr files found under {root}")

    print(f"[Stage3] Loading {len(zarr_files)} episodes from {root}")
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
                break
            obs_parts.append(arr)
        else:
            act = None
            for candidate in (f"data/{action_key}", action_key):
                if candidate in z:
                    act = z[candidate][:].astype(np.float32)
                    break
            if act is None:
                continue
            obs = np.concatenate(obs_parts, axis=-1)
            T = min(len(obs), len(act))
            obs_list.append(obs[:T])
            act_list.append(act[:T])

    if not obs_list:
        raise RuntimeError("No valid episodes loaded.")

    all_obs = np.concatenate(obs_list, axis=0)
    all_acts = np.concatenate(act_list, axis=0)
    print(f"[Stage3] {len(all_obs)} real-demo timesteps — obs {all_obs.shape}, acts {all_acts.shape}")
    return all_obs, all_acts


class SimpleMLP(nn.Module):
    """Matches SB3 MlpPolicy with net_arch=[256, 256]: shared_net + action_net."""

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


class ProprioObsWrapper(gym.Wrapper):
    """Extracts obs_keys from 'policy' obs group → flat (num_envs, obs_dim) float32 array."""

    def __init__(self, env, obs_keys: list):
        super().__init__(env)
        self._obs_keys = obs_keys
        obs_dim = self._infer_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        print(f"[ProprioObsWrapper] obs_dim={obs_dim}  keys={obs_keys}")

    def _infer_obs_dim(self) -> int:
        obs_space = self.env.observation_space
        # Unwrap to the policy group's spaces dict.
        spaces = getattr(obs_space, "spaces", obs_space)
        if "policy" in spaces:
            spaces = getattr(spaces["policy"], "spaces", spaces["policy"])
        dim = 0
        for key in self._obs_keys:
            if key not in spaces:
                raise KeyError(f"obs key '{key}' not in env policy observation_space: {list(spaces.keys())}")
            shape = spaces[key].shape
            # Isaac gymnasium env reports batched shapes (num_envs, obs_dim).
            # Use shape[1:] to get the per-env dimensions only.
            per_env_shape = shape[1:] if len(shape) > 1 else shape
            dim += int(np.prod(per_env_shape))
        return dim

    def _extract(self, obs_raw: dict) -> np.ndarray:
        policy_obs = obs_raw.get("policy", obs_raw)
        parts = []
        for key in self._obs_keys:
            val = policy_obs[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            parts.append(val.reshape(val.shape[0], -1) if val.ndim > 1 else val)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def reset(self, **kwargs):
        obs_raw, info = self.env.reset(**kwargs)
        return self._extract(obs_raw), info

    def step(self, action):
        obs_raw, reward, terminated, truncated, info = self.env.step(action)
        return self._extract(obs_raw), reward, terminated, truncated, info


# ── DAgger data collection ─────────────────────────────────────────────────

def collect_dagger_rollouts(
    gym_env: ProprioObsWrapper,
    student: SimpleMLP,
    teacher: SimpleMLP,
    collection_steps: int,
    sampling_expert: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the student (optionally mixing in teacher actions), label every
    visited state with the teacher's deterministic action.

    Returns:
        dagger_obs:  (N, obs_dim)  — states visited by the student
        dagger_acts: (N, action_dim) — teacher's action at each state
    """
    student.eval()
    teacher.eval()

    obs_buf, act_buf = [], []
    obs, _ = gym_env.reset()  # (num_envs, obs_dim)

    for _ in range(collection_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            teacher_acts = teacher(obs_t)                      # (num_envs, action_dim)
            student_acts = student(obs_t)                      # (num_envs, action_dim)

        # β-mixing: with prob sampling_expert the teacher acts, otherwise the student
        if sampling_expert > 0.0 and np.random.rand() < sampling_expert:
            exec_acts = teacher_acts.cpu().numpy()
        else:
            exec_acts = student_acts.cpu().numpy()

        obs_buf.append(obs)                          # record current obs
        act_buf.append(teacher_acts.cpu().numpy())   # label with teacher

        obs, _reward, _terminated, _truncated, _info = gym_env.step(exec_acts)
        # Isaac Sim auto-resets individual envs on termination; no manual reset needed

    dagger_obs = np.vstack(obs_buf)   # (collection_steps * num_envs, obs_dim)
    dagger_acts = np.vstack(act_buf)
    return dagger_obs, dagger_acts


# ── Cotraining step ────────────────────────────────────────────────────────

def train_student(
    student: SimpleMLP,
    optimizer: torch.optim.Optimizer,
    real_obs: torch.Tensor,
    real_acts: torch.Tensor,
    dagger_obs: torch.Tensor,
    dagger_acts: torch.Tensor,
    train_steps: int,
    batch_size: int,
    real_coef: float,
    device: str,
) -> dict:
    """Gradient steps on student with weighted real-demo + DAgger loss."""
    student.train()
    n_real = len(real_obs)
    n_dag = len(dagger_obs)
    half = batch_size // 2

    real_obs_dev = real_obs.to(device)
    real_acts_dev = real_acts.to(device)
    dagger_obs_dev = dagger_obs.to(device)
    dagger_acts_dev = dagger_acts.to(device)

    real_losses, dag_losses = [], []

    for _ in range(train_steps):
        real_idx = torch.randint(0, n_real, (half,))
        dag_idx = torch.randint(0, n_dag, (half,))

        pred_real = student(real_obs_dev[real_idx])
        pred_dag = student(dagger_obs_dev[dag_idx])

        real_loss = F.mse_loss(pred_real, real_acts_dev[real_idx])
        dag_loss = F.mse_loss(pred_dag, dagger_acts_dev[dag_idx])
        loss = real_coef * real_loss + (1.0 - real_coef) * dag_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        real_losses.append(real_loss.item())
        dag_losses.append(dag_loss.item())

    return {
        "real_loss": float(np.mean(real_losses)),
        "dagger_loss": float(np.mean(dag_losses)),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    device = args_cli.device or "cuda:0"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.join(args_cli.log_dir, f"stage3_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[Stage3] Logging to {log_dir}")

    # ── 1. Load real zarr demonstrations ─────────────────────────────────
    real_obs_np, real_acts_np = load_zarr_demos(
        args_cli.data_path, args_cli.obs_keys, args_cli.action_key
    )
    obs_dim = real_obs_np.shape[-1]
    action_dim = real_acts_np.shape[-1]

    real_obs_t = torch.tensor(real_obs_np, dtype=torch.float32)
    real_acts_t = torch.tensor(real_acts_np, dtype=torch.float32)

    # ── 2. Load teacher from Stage 1 checkpoint ───────────────────────────
    print(f"[Stage3] Loading teacher from {args_cli.teacher_checkpoint} ...")
    teacher_ppo = PPO.load(args_cli.teacher_checkpoint, device=device)
    teacher = SimpleMLP(obs_dim, action_dim).to(device)
    teacher.shared_net.load_state_dict(teacher_ppo.policy.mlp_extractor.policy_net.state_dict())
    teacher.action_net.load_state_dict(teacher_ppo.policy.action_net.state_dict())
    teacher.eval()
    print("[Stage3] Teacher loaded.")

    # ── 3. Init student ───────────────────────────────────────────────────
    student = SimpleMLP(obs_dim, action_dim).to(device)

    if args_cli.checkpoint:
        student.load_state_dict(torch.load(args_cli.checkpoint, map_location=device))
        print(f"[Stage3] Student loaded from {args_cli.checkpoint}")
    elif args_cli.warm_start_from_teacher:
        student.shared_net.load_state_dict(teacher.shared_net.state_dict())
        student.action_net.load_state_dict(teacher.action_net.state_dict())
        print("[Stage3] Student warm-started from teacher weights.")

    optimizer = torch.optim.Adam(student.parameters(), lr=args_cli.student_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args_cli.dagger_iterations, eta_min=1e-5
    )

    # ── 4. Create Isaac Sim environment ───────────────────────────────────
    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        RL_MODE,
        device=device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    gym_env_raw = gym.make(args_cli.task, cfg=env_cfg)
    gym_env = ProprioObsWrapper(gym_env_raw, args_cli.obs_keys)

    # ── 5. Wandb (optional) ───────────────────────────────────────────────
    if args_cli.wandb_project:
        import wandb
        wandb.init(project=args_cli.wandb_project, name=f"stage3_{timestamp}",
                   config=vars(args_cli))

    # ── 6. DAgger loop ────────────────────────────────────────────────────
    print(f"[Stage3] Starting DAgger for {args_cli.dagger_iterations} iterations ...")
    print(f"         collection_steps={args_cli.collection_steps}  "
          f"train_steps={args_cli.train_steps_per_iter}  "
          f"real_coef={args_cli.real_coef}  "
          f"sampling_expert={args_cli.sampling_expert}")

    for iteration in range(1, args_cli.dagger_iterations + 1):
        # ── Collect DAgger rollouts ───────────────────────────────────────
        dagger_obs_np, dagger_acts_np = collect_dagger_rollouts(
            gym_env=gym_env,
            student=student,
            teacher=teacher,
            collection_steps=args_cli.collection_steps,
            sampling_expert=args_cli.sampling_expert,
            device=device,
        )
        dagger_obs_t = torch.tensor(dagger_obs_np, dtype=torch.float32)
        dagger_acts_t = torch.tensor(dagger_acts_np, dtype=torch.float32)

        # ── Cotraining ────────────────────────────────────────────────────
        metrics = train_student(
            student=student,
            optimizer=optimizer,
            real_obs=real_obs_t,
            real_acts=real_acts_t,
            dagger_obs=dagger_obs_t,
            dagger_acts=dagger_acts_t,
            train_steps=args_cli.train_steps_per_iter,
            batch_size=args_cli.student_batch_size,
            real_coef=args_cli.real_coef,
            device=device,
        )
        scheduler.step()

        # ── Eval: query env metrics if available ──────────────────────────
        success_rate = None
        isaac_env = gym_env_raw.unwrapped
        if hasattr(isaac_env, "metrics") and hasattr(isaac_env.metrics, "get_metrics"):
            m = isaac_env.metrics.get_metrics()
            success_rate = float(m.get("is_success", torch.tensor(float("nan"))).float().mean())

        log_str = (
            f"[Stage3] iter {iteration:3d}/{args_cli.dagger_iterations}  "
            f"real_loss={metrics['real_loss']:.4f}  "
            f"dagger_loss={metrics['dagger_loss']:.4f}  "
            f"dagger_pts={len(dagger_obs_np)}"
        )
        if success_rate is not None:
            log_str += f"  success={success_rate:.3f}"
        print(log_str)

        if args_cli.wandb_project:
            import wandb
            log_dict = {
                "iteration": iteration,
                "real_loss": metrics["real_loss"],
                "dagger_loss": metrics["dagger_loss"],
                "dagger_points": len(dagger_obs_np),
            }
            if success_rate is not None:
                log_dict["success_rate"] = success_rate
            wandb.log(log_dict)

        # ── Save checkpoint ───────────────────────────────────────────────
        if iteration % 10 == 0 or iteration == args_cli.dagger_iterations:
            ckpt_path = os.path.join(log_dir, f"student_iter{iteration:04d}.pt")
            torch.save(student.state_dict(), ckpt_path)
            print(f"[Stage3] Saved → {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, "stage3_final.pt")
    torch.save(student.state_dict(), final_path)
    print(f"[Stage3] Final student → {final_path}")

    if args_cli.wandb_project:
        wandb.finish()

    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
