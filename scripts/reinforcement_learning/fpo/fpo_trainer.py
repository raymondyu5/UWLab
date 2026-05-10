"""
FPO trainer: rollout collection, GAE, and FPO minibatch updates.

The FPO policy ratio is:
    rho = exp(clamp(initial_cfm_loss - current_cfm_loss, -3, 3))

This is the exact analogue of exp(log_p_new - log_p_old) in standard PPO,
applied to the conditional flow matching loss instead of log probabilities.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fpo_wrapper import FPOWrapper, FpoStepData


@dataclass
class RolloutStep:
    ppo_obs: dict           # actor_* + critic_* obs at time t
    fpo_data: FpoStepData   # CFM loss info from time t
    reward: torch.Tensor    # (B,)
    done: torch.Tensor      # (B,) bool
    value: torch.Tensor     # (B,) critic value estimate at time t
    info: list              # env info list, may contain episode stats for done envs


class FPOTrainer:
    """
    Implements the FPO training loop for vectorized IsaacLab environments.

    Actor (policy.model): updated with FPO clipped surrogate loss.
    Critic (separate MLP): updated with MSE value loss.
    """

    def __init__(
        self,
        env: FPOWrapper,
        critic: nn.Module,
        cfg: dict,
        device: torch.device,
        log_fn=None,
        bc_loader=None,
    ):
        self.env    = env
        self.critic = critic
        self.policy = env.policy
        self.device = device
        self.log_fn = log_fn or (lambda metrics: None)

        fpo_cfg = cfg.get("fpo", {})
        ppo_cfg = cfg["ppo"]

        self.n_steps     = int(ppo_cfg["n_steps"])
        self.n_epochs    = int(ppo_cfg["n_epochs"])
        self.batch_size  = int(ppo_cfg["batch_size"])
        self.gamma       = float(ppo_cfg["gamma"])
        self.gae_lambda  = float(ppo_cfg["gae_lambda"])
        self.clip_range  = float(ppo_cfg["clip_range"])
        self.vf_coef     = float(ppo_cfg["vf_coef"])
        self.max_grad_norm = float(ppo_cfg["max_grad_norm"])
        self.target_kl   = float(ppo_cfg.get("target_kl", 1e9))
        self.critic_warmup_iters = int(fpo_cfg.get("critic_warmup_iters", 0))
        self.reward_scale = float(fpo_cfg.get("reward_scale", 1.0))
        self.bc_coef = float(fpo_cfg.get("bc_coef", 0.0))
        self.bc_loader = bc_loader

        actor_lr  = float(ppo_cfg.get("actor_lr",  ppo_cfg["learning_rate"]))
        critic_lr = float(ppo_cfg.get("critic_lr", ppo_cfg["learning_rate"]))
        self.actor_optim  = torch.optim.Adam(self.policy.model.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.num_envs = env.num_envs
        self._iteration = 0
        self._t_so_far  = 0

        # Rolling buffers for training-time success rate (same maxlen as RFS).
        self._train_success_buf = deque(maxlen=400)
        self._train_extra_bufs: dict[str, deque] = {}  # populated lazily per metric

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_rollout(self, current_obs: dict):
        """
        Collect n_steps of experience across all envs.

        Args:
            current_obs: ppo_obs dict from previous step (or reset).
        Returns:
            steps:       list of RolloutStep
            last_value:  (B,) critic bootstrap value for the final obs
            next_obs:    ppo_obs dict to carry into the next rollout
        """
        steps: List[RolloutStep] = []
        obs = current_obs

        isaac_env = self.env.unwrapped

        for _ in range(self.n_steps):
            crit_flat = self.env.flatten_critic_obs(obs)
            value = self.critic(crit_flat).squeeze(-1)  # (B,)

            # Read metrics BEFORE step (pre-auto-reset state for done envs).
            try:
                metrics_pre = isaac_env.metrics.get_metrics()
            except Exception:
                metrics_pre = None

            next_obs, reward, done, info = self.env.step()
            reward = reward * self.reward_scale
            fpo_data = self.env.last_fpo_data

            # Accumulate success/grasp for episodes that just ended.
            if metrics_pre is not None and done.any():
                is_success = metrics_pre.get("is_success", np.zeros(self.num_envs, dtype=bool))
                for i in range(self.num_envs):
                    if done[i]:
                        self._train_success_buf.append(float(is_success[i]))
                for key, arr in metrics_pre.items():
                    if key == "is_success":
                        continue
                    if key not in self._train_extra_bufs:
                        self._train_extra_bufs[key] = deque(maxlen=400)
                    for i in range(self.num_envs):
                        if done[i]:
                            self._train_extra_bufs[key].append(float(arr[i]))

            steps.append(RolloutStep(
                ppo_obs=obs,
                fpo_data=fpo_data,
                reward=reward,
                done=done,
                value=value,
                info=info,
            ))
            obs = next_obs

        # Bootstrap value for the obs after the last step.
        last_value = self.critic(self.env.flatten_critic_obs(obs)).squeeze(-1)
        return steps, last_value, obs

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self, steps: List[RolloutStep], last_value: torch.Tensor
    ):
        """
        Compute GAE advantages and returns.
        Returns: advantages (T*B,), returns (T*B,)
        """
        T = len(steps)
        B = self.num_envs

        rewards = torch.stack([s.reward          for s in steps])        # (T, B)
        dones   = torch.stack([s.done.float()    for s in steps])        # (T, B)
        values  = torch.stack([s.value           for s in steps])        # (T, B)

        advantages = torch.zeros_like(rewards)
        last_gae   = torch.zeros(B, device=self.device)

        for t in reversed(range(T)):
            next_val  = last_value if t == T - 1 else values[t + 1]
            next_done = dones[t]
            delta     = rewards[t] + self.gamma * next_val * (1 - next_done) - values[t]
            last_gae  = delta + self.gamma * self.gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages.reshape(-1), returns.reshape(-1)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        steps: List[RolloutStep],
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict:
        """
        Run n_epochs of minibatch FPO + critic updates.
        Returns dict of logging metrics.
        """
        # Flatten rollout tensors: (T*B, ...)
        global_cond  = torch.cat([s.fpo_data.global_cond        for s in steps], dim=0)
        chunk_norm   = torch.cat([s.fpo_data.chunk_norm          for s in steps], dim=0)
        eps          = torch.cat([s.fpo_data.eps                 for s in steps], dim=0)
        t_samp       = torch.cat([s.fpo_data.t_samp              for s in steps], dim=0)
        initial_loss = torch.cat([s.fpo_data.initial_cfm_loss    for s in steps], dim=0)
        critic_flat  = torch.cat([
            self.env.flatten_critic_obs(s.ppo_obs) for s in steps
        ], dim=0)

        # Normalize advantages once over the full batch.
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = global_cond.shape[0]
        actor_losses, critic_losses, rho_means, kl_approxs, bc_losses = [], [], [], [], []
        early_stop = False

        for epoch in range(self.n_epochs):
            if early_stop:
                break
            perm = torch.randperm(N, device=self.device)

            for start in range(0, N, self.batch_size):
                idx = perm[start: start + self.batch_size]
                if len(idx) == 0:
                    continue

                mb_gc    = global_cond[idx]
                mb_chunk = chunk_norm[idx]
                mb_eps   = eps[idx]
                mb_t     = t_samp[idx]
                mb_init  = initial_loss[idx]
                mb_adv   = adv[idx]
                mb_ret   = returns[idx]
                mb_crit  = critic_flat[idx]

                # Critic MSE loss.
                v_pred      = self.critic(mb_crit).squeeze(-1)
                critic_loss = F.mse_loss(v_pred, mb_ret)

                warming_up = (self._iteration <= self.critic_warmup_iters)
                if warming_up:
                    # Critic-only update: let value function calibrate before touching actor.
                    self.critic_optim.zero_grad()
                    (self.vf_coef * critic_loss).backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()
                    actor_losses.append(0.0)
                    critic_losses.append(critic_loss.item())
                    rho_means.append(1.0)
                    kl_approxs.append(0.0)
                    bc_losses.append(0.0)
                    continue

                # FPO++ per-sample ratio: one ratio per (tau_i, eps_i) pair rather than
                # averaging losses before exp, giving finer-grained trust region clipping.
                cfm_loss_curr = self.env.compute_cfm_loss(mb_gc, mb_chunk, mb_eps, mb_t)
                cfm_diff      = mb_init - cfm_loss_curr                    # (B, N)
                rho           = torch.exp(cfm_diff)                        # (B, N)

                adv_n = mb_adv.unsqueeze(1)                                # (B, 1) → broadcasts over N
                surr1 = rho * adv_n                                        # (B, N)
                surr2 = torch.clamp(rho, 1 - self.clip_range, 1 + self.clip_range) * adv_n
                ppo_obj = torch.min(surr1, surr2)

                # ASPO: for negative-advantage samples outside the trust region (rho > 1+eps),
                # replace the unclipped rho*adv with the log-ratio objective. This bounds the
                # gradient amplitude to |adv| regardless of rho, preventing the ~100x gradient
                # amplification observed when rho spikes during bad training runs.
                neg_outside = (adv_n < 0) & (rho > 1 + self.clip_range)
                aspo_obj = torch.where(neg_outside, cfm_diff * adv_n, ppo_obj)
                actor_loss = -aspo_obj.mean()

                # BC regularization: anchor UNet toward frozen BC weights on real data.
                bc_loss_val = 0.0
                if self.bc_loader is not None and self.bc_coef > 0.0:
                    real_batch = self.bc_loader.sample_from_pool(len(idx))
                    real_gc = torch.cat([
                        real_batch["actor_pcd_emb"],
                        real_batch["actor_agent_pos_history"],
                    ] + ([real_batch["actor_past_actions_history"]]
                         if "actor_past_actions_history" in real_batch else []), dim=-1)
                    real_chunk = real_batch["chunk_norm"]          # (B, H, A)
                    B_r, H_r, A_r = real_chunk.shape
                    real_eps   = torch.randn(B_r, 1, H_r, A_r, device=self.device)
                    real_t     = torch.rand(B_r, 1, device=self.device)
                    cfm_curr   = self.env.compute_cfm_loss(real_gc, real_chunk, real_eps, real_t)
                    cfm_frozen = self.env.compute_cfm_loss_frozen(real_gc, real_chunk, real_eps, real_t)
                    bc_loss    = self.bc_coef * (cfm_curr - cfm_frozen).mean()
                    actor_loss = actor_loss + bc_loss
                    bc_loss_val = bc_loss.item()
                bc_losses.append(bc_loss_val)

                # Combined backward (actor and critic have disjoint params).
                total_loss = actor_loss + self.vf_coef * critic_loss
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                rho_means.append(rho.mean().item())

                # Approximate KL for early stopping.
                with torch.no_grad():
                    approx_kl = ((rho - 1) - cfm_diff).mean().item()
                kl_approxs.append(approx_kl)
                if approx_kl > self.target_kl:
                    early_stop = True
                    break

        return {
            "actor_loss":      float(np.mean(actor_losses))  if actor_losses  else 0.0,
            "critic_loss":     float(np.mean(critic_losses)) if critic_losses else 0.0,
            "rho_mean":        float(np.mean(rho_means))     if rho_means     else 1.0,
            "approx_kl":       float(np.mean(kl_approxs))   if kl_approxs   else 0.0,
            "bc_reg_loss":     float(np.mean(bc_losses))     if bc_losses     else 0.0,
            "advantages_mean": advantages.mean().item(),
            "advantages_std":  advantages.std().item(),
            "returns_mean":    returns.mean().item(),
            "early_stop":      int(early_stop),
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _save(self, log_dir: str, iteration: int, t_so_far: int):
        import os
        # Full policy state dict in BC-compatible format (Format B).
        # PointNet + normalizer weights come from the frozen BC checkpoint;
        # UNet weights are updated by FPO. Saving the full policy means this
        # checkpoint can be loaded directly by _load_cfm_checkpoint without
        # needing the original BC checkpoint.
        ckpt = {
            "ema_model":       self.policy.state_dict(),   # full CFMPCDPolicy
            "critic":          self.critic.state_dict(),
            "actor_optim":     self.actor_optim.state_dict(),
            "critic_optim":    self.critic_optim.state_dict(),
            "iteration":       iteration,
            "timesteps":       t_so_far,
        }
        path = os.path.join(log_dir, "checkpoints", f"ckpt_{iteration:06d}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ckpt, path)
        # best.ckpt symlink so _load_cfm_checkpoint finds it automatically.
        best = os.path.join(log_dir, "checkpoints", "best.ckpt")
        if os.path.islink(best):
            os.remove(best)
        os.symlink(os.path.abspath(path), best)
        print(f"[FPO] Saved checkpoint: {path}")

    def train(self, total_timesteps: int, eval_fn=None, eval_interval: int = 50,
              log_dir: str = ".", save_interval: int = 50):
        import os
        from pathlib import Path
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        current_obs, _ = self.env.reset()
        t_so_far  = 0
        iteration = 0

        if eval_fn is not None:
            current_obs = eval_fn(t_so_far, iteration)

        while t_so_far < total_timesteps:
            steps, last_value, current_obs = self.collect_rollout(current_obs)
            t_so_far  += self.n_steps * self.num_envs
            iteration += 1
            self._iteration = iteration
            self._t_so_far  = t_so_far

            advantages, returns = self.compute_gae(steps, last_value)
            metrics = self.update(steps, advantages, returns)
            metrics["timesteps"] = t_so_far
            metrics["iteration"] = iteration

            # Episode reward from step info (populated by FPOWrapper for done envs).
            ep_rewards = []
            for s in steps:
                if isinstance(s.info, list):
                    for entry in s.info:
                        if isinstance(entry, dict) and "episode" in entry:
                            ep_rewards.append(entry["episode"]["r"])
            if ep_rewards:
                metrics["ep_rew_mean"] = float(np.mean(ep_rewards))

            # Training-time success rate (rolling over last 400 episodes).
            if len(self._train_success_buf) == self._train_success_buf.maxlen:
                metrics["train/success_rate"] = float(np.mean(self._train_success_buf))
                for key, buf in self._train_extra_bufs.items():
                    if len(buf) == buf.maxlen:
                        metrics[f"train/{key}_rate"] = float(np.mean(buf))

            self.log_fn(metrics)

            if iteration % 10 == 0:
                sr_str = f"  sr={metrics['train/success_rate']:.3f}" if "train/success_rate" in metrics else ""
                print(
                    f"[FPO] iter={iteration:4d}  t={t_so_far:10,d}  "
                    f"actor_loss={metrics['actor_loss']:7.4f}  "
                    f"critic_loss={metrics['critic_loss']:7.4f}  "
                    f"rho={metrics['rho_mean']:.3f}  "
                    f"adv={metrics['advantages_mean']:+.3f}±{metrics['advantages_std']:.3f}  "
                    f"kl={metrics['approx_kl']:.4f}{sr_str}",
                    flush=True,
                )

            # Periodic eval — eval_fn returns new current_obs after reset.
            if eval_fn is not None and iteration % eval_interval == 0:
                current_obs = eval_fn(t_so_far, iteration)

            # Save checkpoint.
            if iteration % save_interval == 0:
                self._save(log_dir, iteration, t_so_far)
