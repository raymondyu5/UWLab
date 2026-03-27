"""
RegularizedPPO: PPO subclass with real-state KL regularization on the actor.

Integrates KL(N(actor_mean, actor_std²) || N(0, 1)) on real robot states into
each PPO minibatch update step, so the regularization gradient is combined with
the clipped surrogate objective in a single backward pass / optimizer step.

Motivation: the noise policy (PPO actor) is only trained on sim states. On real
states it is OOD. Regularizing toward N(0,1) on real inputs ensures that in the
worst case the actor outputs standard Gaussian noise — equivalent to running the
base CFM policy without any noise steering, which is the conservative fallback.

The KL loss is:
    KL(N(μ, σ²) || N(0,1)) = 0.5 * (σ² + μ² - 1 - 2 log σ)   [per dim]

Gradients flow only through the PPO actor (pi_features_extractor, mlp_extractor
policy_net, action_net, log_std). The CFM PointNet used to encode PCD inputs is
called under torch.no_grad() inside RealDatasetLoader.sample_actor_obs().
"""

import time
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

from buffers import GpuDictRolloutBuffer


class RegularizedPPO(PPO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._real_loader = None
        self._rfs_env = None
        self._reg_coef: float = 0.0
        self._reg_batch_size: int = 256

    def set_real_regularization(
        self,
        loader,
        rfs_env,
        reg_coef: float,
        reg_batch_size: int = 256,
        n_augmentations: int = 10,
        pcd_noise: float = 0.02,
        noise_extrinsic: bool = False,
        noise_extrinsic_parameter: list | None = None,
    ) -> None:
        """Configure real-state KL regularization.

        Precomputes a GPU pool of all valid real-data windows with
        BC-matching PCD augmentations (random downsample, extrinsic noise,
        XYZ noise).  During training each minibatch samples a fresh random
        subset from this pool — zero PointNet overhead, no batch reuse.

        Args:
            loader:        RealDatasetLoader instance.
            rfs_env:       RFSWrapper (provides frozen CFM policy for PCD encoding).
            reg_coef:      Weight on the KL regularization loss.
            reg_batch_size: Number of real states to sample per minibatch.
            n_augmentations: Copies per window with different PCD augmentations.
            pcd_noise:     Uniform XYZ noise magnitude (matches BC pcd_noise).
            noise_extrinsic: Apply random rotation+translation (matches BC).
            noise_extrinsic_parameter: [translation_scale, rotation_scale].
        """
        self._real_loader = loader
        self._rfs_env = rfs_env
        self._reg_coef = reg_coef
        self._reg_batch_size = reg_batch_size
        loader.precompute_pool(
            rfs_env,
            n_augmentations=n_augmentations,
            pcd_noise=pcd_noise,
            noise_extrinsic=noise_extrinsic,
            noise_extrinsic_parameter=noise_extrinsic_parameter,
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        """GPU-resident rollout collection.

        Overrides OnPolicyAlgorithm.collect_rollouts() when the rollout buffer is
        a GpuDictRolloutBuffer with gpu_buffer=True. Observations, rewards, and
        done flags stay on GPU throughout — no CPU<->GPU transfers per step.

        Falls back to the standard SB3 implementation for any other buffer type.
        """
        if not (isinstance(rollout_buffer, GpuDictRolloutBuffer) and rollout_buffer.gpu_buffer):
            _t = time.perf_counter()
            result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
            _secs = time.perf_counter() - _t
            _sps = int(n_rollout_steps * env.num_envs / _secs)
            print(f"[collect_rollouts] {_sps} steps/sec  "
                  f"({n_rollout_steps} steps × {env.num_envs} envs in {_secs:.2f}s)  "
                  f"gpu_buffer=False")
            return result

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        _t_rollout_start = time.perf_counter()
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # _last_obs is already a dict of GPU tensors from GpuSb3VecEnvWrapper.
                actions, values, log_probs = self.policy(self._last_obs)

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = th.clamp(
                        actions,
                        th.as_tensor(self.action_space.low, device=self.device),
                        th.as_tensor(self.action_space.high, device=self.device),
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            # Populate ep_info_buffer for done envs so SB3's _dump_logs works correctly.
            ep_info = infos["episode"]
            reset_mask = ep_info["mask"]
            if reset_mask.any():
                for idx in reset_mask.nonzero(as_tuple=True)[0]:
                    self.ep_info_buffer.extend([{
                        "r": ep_info["r"][idx].item(),
                        "l": int(ep_info["l"][idx].item()),
                    }])

            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Vectorized bootstrap for time-limit truncations.
            # In practice RFSWrapper always sets truncated=zeros so this never fires,
            # but it's kept correct for generality.
            bootstrap_mask = infos["terminal_mask"] & infos["TimeLimit.truncated"]
            bootstrap_indices = bootstrap_mask.nonzero(as_tuple=True)[0]
            if bootstrap_indices.numel() > 0:
                terminal_obs = {
                    k: v[bootstrap_indices]
                    for k, v in infos["terminal_observation"].items()
                }
                with th.no_grad():
                    terminal_values = self.policy.predict_values(terminal_obs).squeeze(-1)
                rewards[bootstrap_indices] += self.gamma * terminal_values

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        _rollout_secs = time.perf_counter() - _t_rollout_start
        _steps_per_sec = int(n_rollout_steps * env.num_envs / _rollout_secs)
        print(f"[collect_rollouts] {_steps_per_sec} steps/sec  "
              f"({n_rollout_steps} steps × {env.num_envs} envs in {_rollout_secs:.2f}s)  "
              f"gpu_buffer={rollout_buffer.gpu_buffer}")

        with th.no_grad():
            values = self.policy.predict_values(new_obs)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def _compute_kl_reg(self) -> th.Tensor:
        """Sample fresh real obs from the precomputed pool and compute
        KL(N(actor_mean, actor_std²) || N(0,1)).

        Returns scalar loss (already weighted by _reg_coef).
        """
        real_obs = self._real_loader.sample_from_pool(self._reg_batch_size)
        pi_features = self.policy.pi_features_extractor(real_obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        mean = self.policy.action_net(latent_pi)
        log_std = self.policy.log_std

        kl = 0.5 * (log_std.exp() ** 2 + mean ** 2 - 1.0 - 2.0 * log_std)
        return self._reg_coef * kl.mean()

    def train(self) -> None:
        """PPO update with per-minibatch KL regularization on real states.

        Overrides PPO.train() to inject the KL term into each minibatch loss
        so that policy_loss + entropy_loss + value_loss + kl_reg_loss are all
        combined in a single backward() / optimizer.step().

        Each minibatch samples a fresh random subset from the precomputed
        real-data pool (GPU indexing only, zero PointNet overhead).
        """
        use_reg = self._real_loader is not None and self._reg_coef > 0.0

        # --- Exact replica of PPO.train() from here, with KL injection ---

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        reg_losses = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # --- KL regularization: added to the same loss ---
                if use_reg:
                    kl_reg = self._compute_kl_reg()
                    loss = loss + kl_reg
                    reg_losses.append(kl_reg.item())

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        vals = self.rollout_buffer.values.flatten()
        rets = self.rollout_buffer.returns.flatten()
        if isinstance(vals, th.Tensor):
            vals = vals.cpu().numpy()
            rets = rets.cpu().numpy()
        explained_var = explained_variance(vals, rets)

        # --- Logging (matches SB3 PPO exactly) ---
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        if use_reg and reg_losses and wandb.run is not None:
            wandb.log({"train/real_kl_reg": np.mean(reg_losses)}, step=self.num_timesteps)
