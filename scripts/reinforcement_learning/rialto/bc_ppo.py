"""
RialTo-style PPO with BC loss fused into the combined optimization step.

Adapted from third_party/RialToPolicyLearning/rialto/algo/sb3_ppo.py.

Changes from the original:
  - bc_buffer (OnlineBuffer, discrete)  →  bc_obs / bc_acts tensors (continuous)
  - CrossEntropyLoss / logits           →  MSELoss / distribution.mean
  - removed: from_vision, state-doubling, 3D/Sparse3D policy aliases
  - wandb.log removed; all logging goes through self.logger.record
"""

import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
import gymnasium.spaces as spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

PPOSelf = TypeVar("PPOSelf", bound="BcPPO")


class BcPPO(OnPolicyAlgorithm):
    """PPO with behavioural-cloning loss fused into every minibatch update.

    For each PPO minibatch the combined loss is:

        loss = policy_loss + ent_coef * entropy_loss
             + vf_coef   * value_loss
             + bc_coef   * MSE(policy_mean(bc_obs), bc_acts)

    bc_obs / bc_acts are flat tensors of demonstration transitions sampled once
    per minibatch from the pre-loaded demo buffer.
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # BC args
        bc_obs: Optional[th.Tensor] = None,
        bc_acts: Optional[th.Tensor] = None,
        bc_coef: float = 0.0,
        bc_batch_size: int = 256,
        # Warmup: collect this many rollouts deterministically before enabling exploration.
        warmup_rollouts: int = 0,
        # Critic warmup: train only the value head for this many rollouts after
        # behavioral warmup before allowing policy gradient updates.
        critic_warmup_rollouts: int = 0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.bc_obs = bc_obs
        self.bc_acts = bc_acts
        self.bc_coef = bc_coef
        self.bc_batch_size = bc_batch_size
        self.warmup_rollouts = warmup_rollouts
        self._warmup_rollouts_done = 0
        self._buffer_has_warmup_data = False
        self.critic_warmup_rollouts = critic_warmup_rollouts
        self._critic_warmup_done = 0

        if normalize_advantage:
            assert batch_size > 1, (
                "`batch_size` must be greater than 1. "
                "See https://github.com/DLR-RM/stable-baselines3/issues/440"
            )
        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or not normalize_advantage, (
                f"`n_steps * n_envs` must be greater than 1. "
                f"Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            )
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"Mini-batch size {batch_size} does not evenly divide rollout buffer "
                    f"size {buffer_size} (n_steps={self.n_steps}, n_envs={self.env.num_envs}). "
                    f"The last {buffer_size % batch_size} samples will be dropped each epoch."
                )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"
                )
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        in_warmup = self._warmup_rollouts_done < self.warmup_rollouts
        if in_warmup and hasattr(self.policy, "log_std"):
            orig_log_std = self.policy.log_std.data.clone()
            self.policy.log_std.data.fill_(-10.0)

        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        if in_warmup and hasattr(self.policy, "log_std"):
            self.policy.log_std.data.copy_(orig_log_std)
            self._warmup_rollouts_done += 1
            # old_log_probs in this buffer were computed under std≈0 and are
            # incompatible with PPO ratio computation. Flag train() to stay in
            # critic-only mode (which never touches log_probs or the actor).
            self._buffer_has_warmup_data = True
            if self._warmup_rollouts_done == self.warmup_rollouts:
                print(f"[BcPPO] Behavioral warmup complete ({self.warmup_rollouts} rollouts). Enabling exploration.")
        else:
            self._buffer_has_warmup_data = False

        return result

    def compute_bc_loss(self) -> th.Tensor:
        """MSE between policy action mean and a random batch of demo actions."""
        if self.bc_obs is None or self.bc_coef == 0.0:
            return th.tensor(0.0, device=self.device)

        n = len(self.bc_obs)
        idx = th.randint(0, n, (self.bc_batch_size,))
        obs_b  = self.bc_obs[idx].to(self.device)
        acts_b = self.bc_acts[idx].to(self.device)

        dist = self.policy.get_distribution(obs_b)
        pred_mean = dist.distribution.mean
        return F.mse_loss(pred_mean, acts_b)

    def train(self) -> None:
        """Update policy using the currently gathered rollout buffer."""
        # Merged warmup: behavioral-warmup buffers (std≈0) also train the critic.
        # The old_log_probs in those buffers are incompatible with PPO ratio
        # computation, but the critic path never uses them, so it's safe.
        in_critic_warmup = self._buffer_has_warmup_data or (
            self._critic_warmup_done < self.critic_warmup_rollouts
        )

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, bc_losses, pg_losses, value_losses, clip_fractions = [], [], [], [], []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if in_critic_warmup:
                    # Only train the value head; actor params get zero gradient
                    # because value_loss flows only through mlp_extractor.value_net.
                    loss = self.vf_coef * value_loss
                    pg_losses.append(0.0)
                    clip_fractions.append(0.0)
                    entropy_losses.append(0.0)
                    bc_losses.append(0.0)
                else:
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

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)
                    entropy_losses.append(entropy_loss.item())

                    bc_loss = self.compute_bc_loss()
                    bc_losses.append(bc_loss.item())

                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.bc_coef * bc_loss
                    )

                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        if not self._buffer_has_warmup_data and self._critic_warmup_done < self.critic_warmup_rollouts:
            self._critic_warmup_done += 1
            if self._critic_warmup_done == self.critic_warmup_rollouts:
                print(f"[BcPPO] Critic warmup complete ({self.critic_warmup_rollouts} rollouts). Enabling policy gradient.")

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/in_critic_warmup", int(in_critic_warmup))
        self.logger.record("train/bc_loss", np.mean(bc_losses) if bc_losses else 0.0)
        self.logger.record("train/entropy_loss", np.mean(entropy_losses) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses) if pg_losses else 0.0)
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions) if clip_fractions else 0.0)
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
