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

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


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

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

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
