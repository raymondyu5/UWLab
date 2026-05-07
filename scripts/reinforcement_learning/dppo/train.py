"""
DPPO (Diffusion Policy Policy Optimization) baseline for Isaac Lab environments.

Fine-tunes a BC-pretrained DiffusionMLP diffusion policy using PPO directly over
the denoising chain, on the same tasks as RFS.

Algorithm reference: third_party/dppo/model/diffusion/diffusion_ppo.py
Training loop adapted from: third_party/dppo/agent/finetune/train_ppo_diffusion_agent.py

Usage (inside container):
    ./uwlab.sh -p scripts/reinforcement_learning/dppo/train.py \\
        --task UW-FrankaLeap-GraspPinkCup-IkRel-v0 \\
        --num_envs 1024 \\
        --network_path logs/dppo_pretrain/best.pt \\
        --headless

    # Without pretrained model (random init — less fair vs RFS):
    ./uwlab.sh -p scripts/reinforcement_learning/dppo/train.py \\
        --task UW-FrankaLeap-GraspPinkCup-IkRel-v0 \\
        --num_envs 1024 \\
        --headless
"""

import argparse
import contextlib
import os
import pickle
import signal
import sys
import math
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train DPPO policy on Isaac Lab tasks.")

# Env / infra
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume DPPO from a saved model dict (.pt file).")

# DPPO
parser.add_argument("--network_path", type=str, default=None,
                    help="Path to BC-pretrained DiffusionMLP checkpoint (.pt). "
                         "Checkpoint must contain 'ema' or 'model' key. "
                         "If omitted, the policy is initialised randomly.")
parser.add_argument("--cfg", type=str, default="configs/rl/dppo_cfg.yaml",
                    help="Path to DPPO config YAML.")
parser.add_argument("--eval_interval", type=int, default=None,
                    help="Override eval.interval (env steps).")
parser.add_argument("--no_eval_video", action="store_true", default=False)

# Wandb
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Ensure dppo/ is importable from third_party
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UWLAB_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../../"))
_DPPO_DIR = os.path.join(_UWLAB_DIR, "third_party", "dppo")
_PIP_PACKAGES = os.path.join(_UWLAB_DIR, "third_party", "pip_packages")
for _p in [_SCRIPT_DIR, _DPPO_DIR, _PIP_PACKAGES]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    import gc
    for obj in gc.get_objects():
        if "tqdm_rich" in type(obj).__name__:
            obj.close()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest of imports after AppLauncher (Isaac Sim requirement)."""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml

from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import uwlab_tasks  # noqa: F401

# DPPO model classes
from model.diffusion.diffusion_ppo import PPODiffusion
from model.diffusion.mlp_diffusion import DiffusionMLP
from model.common.critic import CriticObs
from model.diffusion.eta import EtaFixed
from util.scheduler import CosineAnnealingWarmupRestarts

from wrapper import IsaacLabDPPOWrapper


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _short_task(task: str) -> str:
    import re
    name = task
    if name.startswith("UW-FrankaLeap-"):
        name = name[len("UW-FrankaLeap-"):]
    name = re.sub(r"-(IkRel-)?v\d+$", "", name)
    return name


def _compute_episode_stats(reward_trajs, firsts_trajs, n_envs, act_steps,
                            best_reward_threshold=0.5):
    """Compute success rate and average episode reward from rollout data.

    Args:
        reward_trajs: (n_steps, n_envs) tensor
        firsts_trajs: (n_steps+1, n_envs) bool tensor — True at episode starts
        act_steps: number of env steps per policy step (for per-step reward scaling)
    """
    n_steps = reward_trajs.shape[0]
    reward_np = reward_trajs.cpu().numpy()
    firsts_np = firsts_trajs.cpu().numpy()

    episodes = []
    for env_ind in range(n_envs):
        env_starts = np.where(firsts_np[:, env_ind] == 1)[0]
        for i in range(len(env_starts) - 1):
            start = env_starts[i]
            end = env_starts[i + 1]
            if end - start > 1:
                episodes.append(reward_np[start:end, env_ind])

    if not episodes:
        return 0.0, 0.0, 0.0, 0

    episode_rewards = np.array([ep.sum() for ep in episodes])
    # Best-step reward: max step reward normalised by act_steps
    episode_best = np.array([ep.max() / act_steps for ep in episodes])
    avg_reward = float(np.mean(episode_rewards))
    avg_best = float(np.mean(episode_best))
    success_rate = float(np.mean(episode_best >= best_reward_threshold))
    return avg_reward, avg_best, success_rate, len(episodes)


@torch.no_grad()
def _eval_rollout(model, wrapper, n_steps, act_steps, device):
    """Run a deterministic eval rollout. Returns avg episode reward."""
    model.eval()
    obs = wrapper.reset()
    n_envs = wrapper.n_envs
    total_reward = torch.zeros(n_envs, device=device)
    for _ in range(n_steps):
        cond = {k: v.to(device) for k, v in obs.items()}
        samples = model(cond=cond, deterministic=True, return_chain=False)
        actions = samples.trajectories[:, :act_steps]
        obs, reward, terminated, truncated, _ = wrapper.step(actions)
        total_reward += reward
    model.train()
    return total_reward.mean().item()


def main():
    cfg = _load_cfg(args_cli.cfg)
    ppo_cfg = cfg["ppo"]
    diff_cfg = cfg["diffusion"]
    eval_cfg = cfg["eval"]

    device = torch.device(args_cli.device if args_cli.device else "cuda:0")

    # Derived training params
    n_steps = ppo_cfg["n_steps"]
    batch_size = ppo_cfg["batch_size"]
    update_epochs = ppo_cfg["update_epochs"]
    gamma = ppo_cfg["gamma"]
    gae_lambda = ppo_cfg["gae_lambda"]
    target_kl = ppo_cfg["target_kl"]
    max_grad_norm = ppo_cfg["max_grad_norm"]
    actor_lr = ppo_cfg["actor_lr"]
    critic_lr = ppo_cfg["critic_lr"]
    actor_wd = ppo_cfg["actor_weight_decay"]
    critic_wd = ppo_cfg["critic_weight_decay"]
    n_critic_warmup_itr = ppo_cfg["n_critic_warmup_itr"]
    ent_coef = ppo_cfg["ent_coef"]
    vf_coef = ppo_cfg["vf_coef"]
    logprob_batch_size = ppo_cfg["logprob_batch_size"]
    use_bc_loss = ppo_cfg["use_bc_loss"]
    reward_horizon = ppo_cfg["reward_horizon"]

    horizon_steps = diff_cfg["horizon_steps"]
    act_steps = diff_cfg["act_steps"]
    ft_denoising_steps = diff_cfg["ft_denoising_steps"]

    eval_interval_steps = args_cli.eval_interval or eval_cfg["interval"]
    # Convert env-step eval interval to policy-step interval
    eval_interval_policy_steps = eval_interval_steps // act_steps

    # Logging / run dir
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = args_cli.wandb_run_name or f"dppo_{_short_task(args_cli.task)}_{timestamp}_{uuid4().hex[:6]}"
    log_dir = os.path.abspath(os.path.join("logs", "dppo", run_name))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)
    print(f"[INFO] Run: {run_name}")
    print(f"[INFO] Logging to: {log_dir}")

    # Isaac Lab env
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=str(device),
        num_envs=args_cli.num_envs,
    )
    env_cfg.run_mode = "rl_mode"
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg, "table_z_range"):
        env_cfg.table_z_range = (0.0, 0.0)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    need_render = args_cli.video or (eval_cfg["record_video"] and not args_cli.no_eval_video)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if need_render else None)

    control_hz = 1 / (env_cfg.sim.dt * env_cfg.decimation)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, **{
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "fps": control_hz,
        })

    wrapper = IsaacLabDPPOWrapper(
        env,
        act_steps=act_steps,
        horizon_steps=horizon_steps,
    )

    # Run a reset to infer obs_dim / action_dim / n_envs
    init_obs = wrapper.reset()
    obs_dim = wrapper.obs_dim
    action_dim = wrapper.action_dim
    n_envs = wrapper.n_envs
    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}, n_envs={n_envs}")

    # Instantiate DPPO model
    actor = DiffusionMLP(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=obs_dim,  # cond_steps=1 so cond_dim == obs_dim
        time_dim=diff_cfg["actor_time_dim"],
        mlp_dims=diff_cfg["actor_mlp_dims"],
        activation_type=diff_cfg["actor_activation_type"],
        residual_style=diff_cfg["actor_residual_style"],
    )
    critic = CriticObs(
        cond_dim=obs_dim,
        mlp_dims=diff_cfg["critic_mlp_dims"],
        activation_type=diff_cfg["critic_activation_type"],
    )
    eta = EtaFixed(
        base_eta=diff_cfg["eta_base"],
        min_eta=diff_cfg["eta_min"],
        max_eta=diff_cfg["eta_max"],
    )
    model = PPODiffusion(
        # VPGDiffusion args
        actor=actor,
        critic=critic,
        ft_denoising_steps=ft_denoising_steps,
        network_path=args_cli.network_path,
        min_sampling_denoising_std=diff_cfg["min_sampling_denoising_std"],
        min_logprob_denoising_std=diff_cfg["min_logprob_denoising_std"],
        eta=eta,
        learn_eta=False,
        # DiffusionModel args
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        denoising_steps=diff_cfg["denoising_steps"],
        predict_epsilon=diff_cfg["predict_epsilon"],
        use_ddim=diff_cfg["use_ddim"],
        ddim_steps=diff_cfg["ddim_steps"],
        device=str(device),
        # PPODiffusion args
        gamma_denoising=diff_cfg["gamma_denoising"],
        clip_ploss_coef=diff_cfg["clip_ploss_coef"],
        clip_ploss_coef_base=diff_cfg["clip_ploss_coef_base"],
        clip_ploss_coef_rate=diff_cfg["clip_ploss_coef_rate"],
        clip_vloss_coef=diff_cfg["clip_vloss_coef"],
        norm_adv=diff_cfg["norm_adv"],
        clip_advantage_lower_quantile=diff_cfg["clip_advantage_lower_quantile"],
        clip_advantage_upper_quantile=diff_cfg["clip_advantage_upper_quantile"],
    ).to(device)

    if args_cli.checkpoint is not None:
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Resumed from checkpoint: {args_cli.checkpoint}")

    n_params_actor = sum(p.numel() for p in model.actor_ft.parameters() if p.requires_grad)
    n_params_critic = sum(p.numel() for p in model.critic.parameters() if p.requires_grad)
    print(f"[INFO] actor_ft parameters: {n_params_actor:,}  critic parameters: {n_params_critic:,}")

    # Optimizers
    actor_optimizer = torch.optim.AdamW(
        model.actor_ft.parameters(), lr=actor_lr, weight_decay=actor_wd
    )
    critic_optimizer = torch.optim.AdamW(
        model.critic.parameters(), lr=critic_lr, weight_decay=critic_wd
    )

    actor_lr_sch_cfg = ppo_cfg["actor_lr_scheduler"]
    critic_lr_sch_cfg = ppo_cfg["critic_lr_scheduler"]
    actor_lr_scheduler = CosineAnnealingWarmupRestarts(
        actor_optimizer,
        first_cycle_steps=actor_lr_sch_cfg["first_cycle_steps"],
        cycle_mult=1.0,
        max_lr=actor_lr,
        min_lr=actor_lr_sch_cfg["min_lr"],
        warmup_steps=actor_lr_sch_cfg["warmup_steps"],
        gamma=1.0,
    )
    critic_lr_scheduler = CosineAnnealingWarmupRestarts(
        critic_optimizer,
        first_cycle_steps=critic_lr_sch_cfg["first_cycle_steps"],
        cycle_mult=1.0,
        max_lr=critic_lr,
        min_lr=critic_lr_sch_cfg["min_lr"],
        warmup_steps=critic_lr_sch_cfg["warmup_steps"],
        gamma=1.0,
    )

    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), {**ppo_cfg, **diff_cfg})

    # Wandb init
    use_wandb = bool(args_cli.wandb_project)
    if use_wandb:
        wandb.init(
            project=args_cli.wandb_project,
            name=run_name,
            notes=command,
            config={**vars(args_cli), **cfg},
        )

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    model.train()
    itr = 0
    cnt_train_step = 0  # total env control steps
    run_results = []

    prev_obs = init_obs
    done_venv = torch.zeros(n_envs, dtype=torch.bool, device=device)

    # Pre-allocate rollout buffers (stays on GPU throughout)
    obs_buf = torch.zeros(n_steps, n_envs, 1, obs_dim, device=device)
    chains_buf = torch.zeros(
        n_steps, n_envs, ft_denoising_steps + 1, horizon_steps, action_dim, device=device
    )
    reward_buf = torch.zeros(n_steps, n_envs, device=device)
    terminated_buf = torch.zeros(n_steps, n_envs, dtype=torch.bool, device=device)
    firsts_buf = torch.zeros(n_steps + 1, n_envs, dtype=torch.bool, device=device)

    n_train_itr = 600_000_000 // (n_steps * n_envs * act_steps)  # effectively infinite

    with contextlib.suppress(KeyboardInterrupt):
        while itr < n_train_itr:
            eval_mode = (
                eval_interval_policy_steps > 0
                and itr % eval_interval_policy_steps == 0
            )

            # ---- Eval ----
            if eval_mode:
                eval_reward = _eval_rollout(
                    model, wrapper, n_steps=50, act_steps=act_steps, device=device
                )
                print(f"[eval itr={itr}] avg_reward={eval_reward:.4f}")
                if use_wandb:
                    wandb.log({"eval/avg_episode_reward": eval_reward}, step=itr)
                # Reset env after eval so training continues cleanly
                prev_obs = wrapper.reset()
                model.train()

            # ---- Rollout ----
            firsts_buf[0] = done_venv  # carry over done from previous itr
            firsts_buf[0] = True  # always treat start of rollout as fresh

            for step in range(n_steps):
                cond = {k: v.to(device) for k, v in prev_obs.items()}
                with torch.no_grad():
                    samples = model(cond=cond, deterministic=False, return_chain=True)

                # samples.trajectories: (n_envs, horizon_steps, action_dim)
                # samples.chains:       (n_envs, ft_denoising_steps+1, horizon_steps, action_dim)
                actions = samples.trajectories[:, :act_steps]  # (n_envs, act_steps, action_dim)
                obs, reward, terminated, truncated, _ = wrapper.step(actions)
                done_venv = terminated | truncated

                obs_buf[step] = cond["state"]
                chains_buf[step] = samples.chains
                reward_buf[step] = reward
                terminated_buf[step] = terminated
                firsts_buf[step + 1] = done_venv

                prev_obs = obs
                cnt_train_step += n_envs * act_steps

            # ---- Compute old values + logprobs ----
            N = n_steps * n_envs
            obs_k = obs_buf.reshape(N, 1, obs_dim)                          # (N, 1, obs_dim)
            chains_k = chains_buf.reshape(N, ft_denoising_steps + 1, horizon_steps, action_dim)

            with torch.no_grad():
                # Batched to avoid OOM
                values_list = []
                logprobs_list = []
                for i in range(0, N, logprob_batch_size):
                    obs_b = {"state": obs_k[i:i + logprob_batch_size]}
                    chains_b = chains_k[i:i + logprob_batch_size]
                    values_list.append(model.critic(obs_b).view(-1))
                    logprobs_list.append(
                        model.get_logprobs(obs_b, chains_b)
                    )  # (batch, K, Ta, Da)
                values_k = torch.cat(values_list, dim=0)           # (N,)
                logprobs_k = torch.cat(logprobs_list, dim=0)       # (N, K, Ta, Da)
                values_trajs = values_k.reshape(n_steps, n_envs)

                # Bootstrap next value
                next_obs_state = prev_obs["state"].to(device)  # (n_envs, 1, obs_dim)
                next_values = model.critic({"state": next_obs_state}).reshape(1, n_envs)

            # ---- GAE advantage computation ----
            advantages_trajs = torch.zeros(n_steps, n_envs, device=device)
            lastgaelam = torch.zeros(n_envs, device=device)
            for t in reversed(range(n_steps)):
                nv = next_values if t == n_steps - 1 else values_trajs[t + 1].unsqueeze(0)
                nonterminal = (~terminated_buf[t]).float()
                delta = reward_buf[t] + gamma * nv.squeeze(0) * nonterminal - values_trajs[t]
                lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
                advantages_trajs[t] = lastgaelam
            returns_trajs = advantages_trajs + values_trajs

            # ---- PPO mini-batch updates ----
            advantages_k = advantages_trajs.reshape(-1)  # (N,)
            returns_k = returns_trajs.reshape(-1)         # (N,)

            total_steps = N * ft_denoising_steps
            clipfracs = []
            flag_break_outer = False

            for update_epoch in range(update_epochs):
                inds_k = torch.randperm(total_steps, device=device)
                num_batch = max(1, total_steps // batch_size)
                flag_break = False

                for batch_idx in range(num_batch):
                    start = batch_idx * batch_size
                    end = start + batch_size
                    inds_b = inds_k[start:end]

                    # Unravel to (env_step_index, denoising_step_index)
                    batch_inds_b, denoising_inds_b = torch.unravel_index(
                        inds_b, (N, ft_denoising_steps)
                    )

                    obs_b = {"state": obs_k[batch_inds_b]}
                    chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                    chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                    returns_b = returns_k[batch_inds_b]
                    values_b = values_k[batch_inds_b]
                    advantages_b = advantages_k[batch_inds_b]
                    logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                    (
                        pg_loss,
                        entropy_loss,
                        v_loss,
                        clipfrac,
                        approx_kl,
                        ratio,
                        bc_loss,
                        eta_val,
                    ) = model.loss(
                        obs_b,
                        chains_prev_b,
                        chains_next_b,
                        denoising_inds_b,
                        returns_b,
                        values_b,
                        advantages_b,
                        logprobs_b,
                        use_bc_loss=use_bc_loss,
                        reward_horizon=reward_horizon,
                    )

                    loss = pg_loss + ent_coef * entropy_loss + vf_coef * v_loss
                    clipfracs.append(clipfrac)

                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    loss.backward()

                    if itr >= n_critic_warmup_itr:
                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                model.actor_ft.parameters(), max_grad_norm
                            )
                        actor_optimizer.step()

                    critic_optimizer.step()

                    if target_kl is not None and approx_kl > target_kl:
                        flag_break = True
                        break

                if flag_break:
                    flag_break_outer = True
                    break

            # ---- LR step ----
            if itr >= n_critic_warmup_itr:
                actor_lr_scheduler.step()
            critic_lr_scheduler.step()
            model.step()  # anneal min_sampling_denoising_std if scheduled

            # ---- Episode stats (for logging) ----
            avg_reward, avg_best, success_rate, n_ep = _compute_episode_stats(
                reward_buf, firsts_buf, n_envs, act_steps
            )

            # ---- Explained variance ----
            y_pred = values_k.cpu().numpy()
            y_true = returns_k.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = float(
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # ---- Logging ----
            run_results.append({"itr": itr, "step": cnt_train_step})
            print(
                f"[itr {itr:5d} | step {cnt_train_step:10d}] "
                f"loss={loss.item():.4f} pg={pg_loss.item():.4f} "
                f"v={v_loss.item():.4f} kl={approx_kl:.4f} "
                f"reward={avg_reward:.4f} success={success_rate:.3f}"
            )
            if use_wandb:
                wandb.log(
                    {
                        "train/total_env_step": cnt_train_step,
                        "train/loss": loss.item(),
                        "train/pg_loss": pg_loss.item(),
                        "train/v_loss": v_loss.item(),
                        "train/bc_loss": bc_loss if isinstance(bc_loss, float) else bc_loss.item(),
                        "train/eta": eta_val,
                        "train/approx_kl": approx_kl,
                        "train/ratio": ratio,
                        "train/clipfrac": float(np.mean(clipfracs)),
                        "train/explained_variance": explained_var,
                        "train/avg_episode_reward": avg_reward,
                        "train/avg_best_reward": avg_best,
                        "train/success_rate": success_rate,
                        "train/n_episodes": n_ep,
                        "train/actor_lr": actor_optimizer.param_groups[0]["lr"],
                        "train/critic_lr": critic_optimizer.param_groups[0]["lr"],
                        "train/diffusion_min_sampling_std": model.get_min_sampling_denoising_std(),
                        "train/kl_early_stop": int(flag_break_outer),
                    },
                    step=itr,
                )

            # ---- Save ----
            if itr % 100 == 0 or itr == n_train_itr - 1:
                ckpt_path = os.path.join(log_dir, f"model_{itr:06d}.pt")
                torch.save({"model": model.state_dict(), "itr": itr}, ckpt_path)
                # also keep a latest symlink
                latest = os.path.join(log_dir, "model_latest.pt")
                if os.path.lexists(latest):
                    os.remove(latest)
                os.symlink(ckpt_path, latest)

            with open(os.path.join(log_dir, "results.pkl"), "wb") as f:
                pickle.dump(run_results, f)

            itr += 1

    # Final save
    torch.save({"model": model.state_dict(), "itr": itr}, os.path.join(log_dir, "model_final.pt"))
    print(f"[INFO] Model saved to {log_dir}/model_final.pt")

    if wandb.run is not None:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
