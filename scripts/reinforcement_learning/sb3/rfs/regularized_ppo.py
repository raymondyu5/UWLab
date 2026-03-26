"""
RegularizedPPO: PPO subclass with real-state KL regularization on the actor.

After each standard PPO train() call, samples a batch of real robot states and
penalizes KL(N(actor_mean, actor_std²) || N(0, 1)) on those states.

Motivation: the noise policy (PPO actor) is only trained on sim states. On real
states it is OOD. Regularizing toward N(0,1) on real inputs ensures that in the
worst case the actor outputs standard Gaussian noise — equivalent to running the
base CFM policy without any noise steering, which is the conservative fallback.

The KL loss is:
    KL(N(μ, σ²) || N(0,1)) = 0.5 * (σ² + μ² - 1 - 2 log σ)   [per dim]
This is identical to the VAE KL term.

Gradients flow only through the PPO actor (pi_features_extractor, mlp_extractor
policy_net, action_net, log_std). The CFM PointNet used to encode PCD inputs is
called under torch.no_grad() inside RealDatasetLoader.sample_actor_obs().
"""

import torch as th
import torch.nn as nn
import wandb
from stable_baselines3 import PPO


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
    ) -> None:
        """Configure real-state KL regularization.

        Args:
            loader:        RealDatasetLoader instance.
            rfs_env:       RFSWrapper (provides frozen CFM policy for PCD encoding).
            reg_coef:      Weight on the KL regularization loss.
            reg_batch_size: Number of real states to sample per train() call.
        """
        self._real_loader = loader
        self._rfs_env = rfs_env
        self._reg_coef = reg_coef
        self._reg_batch_size = reg_batch_size

    def train(self) -> None:
        super().train()
        if self._real_loader is not None and self._reg_coef > 0.0:
            self._real_reg_step()

    def _real_reg_step(self) -> None:
        """One gradient step of real-state KL regularization on the actor."""
        obs_dict = self._real_loader.sample_actor_obs(self._reg_batch_size, self._rfs_env)

        self.policy.set_training_mode(True)

        # Actor forward pass: features → latent_pi → mean actions
        pi_features = self.policy.pi_features_extractor(obs_dict)
        latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        mean = self.policy.action_net(latent_pi)    # (B, action_dim)
        log_std = self.policy.log_std               # (action_dim,) learned param

        # KL(N(mean, std²) || N(0,1)) = 0.5 * (std² + mean² - 1 - 2*log_std)
        kl = 0.5 * (log_std.exp() ** 2 + mean ** 2 - 1.0 - 2.0 * log_std)
        reg_loss = self._reg_coef * kl.mean()

        self.policy.optimizer.zero_grad(set_to_none=True)
        reg_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self.policy.set_training_mode(False)

        if wandb.run is not None:
            wandb.log({"train/real_kl_reg": reg_loss.item()}, step=self.num_timesteps)
