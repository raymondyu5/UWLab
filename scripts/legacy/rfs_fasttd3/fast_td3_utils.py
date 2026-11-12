"""
Utilities for FastTD3 training.

- EmpiricalNormalization : online running mean/variance obs normalizer
- RewardNormalizer       : running reward-scale normalization
- SimpleReplayBuffer     : GPU-resident circular replay buffer (flat obs)
- save_params            : checkpoint saving
- mark_step              : no-op framework hook
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from tensordict import TensorDict


# ---------------------------------------------------------------------------
# Observation normalizer
# ---------------------------------------------------------------------------

class EmpiricalNormalization(nn.Module):
    """Online Welford mean/variance normalizer for flat observation tensors.

    Args:
        shape: Number of observation dimensions.
        device: Torch device.
        epsilon: Small constant added to std for numerical stability.
    """

    def __init__(self, shape: int, device: torch.device, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape, device=device))
        self.register_buffer("var", torch.ones(shape, device=device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        """Update running stats with a batch of observations (Welford online)."""
        batch_count = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count.float()
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * (self.count.float() * batch_count / total)
        new_var = m2 / total

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Normalize x.  When update=True, also update running stats."""
        if update:
            self._update(x)
        return (x - self.mean) / (self.var.sqrt() + self.epsilon)


# ---------------------------------------------------------------------------
# Reward normalizer
# ---------------------------------------------------------------------------

class RewardNormalizer(nn.Module):
    """Normalize rewards by a running estimate of the return scale.

    Uses a running discounted-return estimate to track reward magnitude, then
    normalizes so the effective return range matches [-g_max, g_max].

    Args:
        gamma: Discount factor (used for return accumulation).
        device: Torch device.
        g_max: Target return scale; clamps the denominator.
        epsilon: Numerical stability.
    """

    def __init__(
        self,
        gamma: float,
        device: torch.device,
        g_max: float,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.register_buffer("_ret", torch.tensor(0.0, device=device))
        self.register_buffer("_mean", torch.tensor(0.0, device=device))
        self.register_buffer("_m2", torch.tensor(0.0, device=device))
        self.register_buffer("_n", torch.tensor(0, dtype=torch.long, device=device))

    @torch.no_grad()
    def update_stats(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Update running return estimate.  Call once per env step."""
        self._ret.mul_(self.gamma * (1.0 - dones.float().mean())).add_(rewards.mean())
        self._n.add_(1)
        delta = self._ret - self._mean
        self._mean.add_(delta / self._n.float())
        self._m2.add_(delta * (self._ret - self._mean))

    def forward(self, rewards: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._n < 2:
            return rewards
        var = self._m2 / (self._n.float() - 1.0)
        std = (var + self.epsilon).sqrt().clamp(max=self.g_max)
        return rewards / std


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class SimpleReplayBuffer:
    """GPU-resident circular replay buffer.

    Stores flat float observation tensors.  Each environment has its own
    contiguous slice of the flat storage (size = capacity).  Sampling is
    fully vectorised on GPU.

    Args:
        n_env: Number of parallel environments.
        buffer_size: Capacity per environment.
        n_obs: Flat actor-observation dimension.
        n_act: Action dimension.
        n_critic_obs: Flat critic-observation dimension (ignored if not asymmetric).
        asymmetric_obs: Store separate critic observations.
        n_steps: (Reserved for future n-step returns; currently uses 1-step TD.)
        gamma: Discount factor (used in n-step returns when n_steps > 1).
        device: Torch device.
    """

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool,
        n_steps: int,
        gamma: float,
        device: torch.device,
    ):
        self.n_env = n_env
        self.capacity = buffer_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device
        self.asymmetric_obs = asymmetric_obs

        total = n_env * buffer_size
        self._obs = torch.zeros(total, n_obs, device=device)
        self._actions = torch.zeros(total, n_act, device=device)
        self._next_obs = torch.zeros(total, n_obs, device=device)
        self._rewards = torch.zeros(total, device=device)
        self._dones = torch.zeros(total, device=device)
        self._truncations = torch.zeros(total, device=device)
        self._eff_n = torch.ones(total, device=device)  # effective_n_steps

        if asymmetric_obs:
            self._critic_obs = torch.zeros(total, n_critic_obs, device=device)
            self._next_critic_obs = torch.zeros(total, n_critic_obs, device=device)

        # Per-env write pointers (int64) and full-flags (bool)
        self._ptr = torch.zeros(n_env, dtype=torch.long, device=device)
        self._full = torch.zeros(n_env, dtype=torch.bool, device=device)
        self._total_added = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _offsets(self) -> torch.Tensor:
        """Flat starting index for each env: shape (n_env,)."""
        return torch.arange(self.n_env, device=self.device) * self.capacity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extend(self, transition: TensorDict) -> None:
        """Add one transition from every environment.

        Expected TensorDict keys::

            "observations"          : (n_env, n_obs)
            "actions"               : (n_env, n_act)
            "critic_observations"   : (n_env, n_critic_obs)  [if asymmetric]
            "next" / "observations" : (n_env, n_obs)
            "next" / "rewards"      : (n_env,)
            "next" / "dones"        : (n_env,)
            "next" / "truncations"  : (n_env,)
            "next" / "effective_n_steps" : (n_env,)          [optional]
        """
        idxs = (self._offsets + self._ptr).long()

        self._obs[idxs] = transition["observations"].float()
        self._actions[idxs] = transition["actions"].float()
        self._next_obs[idxs] = transition["next"]["observations"].float()
        self._rewards[idxs] = transition["next"]["rewards"].float()
        self._dones[idxs] = transition["next"]["dones"].float()
        self._truncations[idxs] = transition["next"]["truncations"].float()

        if "effective_n_steps" in transition["next"].keys():
            self._eff_n[idxs] = transition["next"]["effective_n_steps"].float()
        else:
            self._eff_n[idxs] = 1.0

        if self.asymmetric_obs:
            self._critic_obs[idxs] = transition["critic_observations"].float()
            self._next_critic_obs[idxs] = transition["next"]["critic_observations"].float()

        # Advance per-env write pointers
        self._ptr = (self._ptr + 1) % self.capacity
        self._full |= self._ptr == 0
        self._total_added += self.n_env

    def sample(self, per_env_batch: int) -> TensorDict:
        """Return a random batch of *per_env_batch* transitions per environment.

        Total batch size = n_env * per_env_batch.
        """
        valid = torch.where(
            self._full,
            torch.full((self.n_env,), self.capacity, device=self.device, dtype=torch.long),
            self._ptr,
        )  # (n_env,)

        # Vectorised uniform sampling in [0, valid[e]) for each env
        rand_frac = torch.rand(self.n_env, per_env_batch, device=self.device)
        rand_local = (rand_frac * valid.unsqueeze(1).float()).long()
        rand_local = rand_local.clamp(max=self.capacity - 1)

        flat_idxs = (self._offsets.unsqueeze(1) + rand_local).reshape(-1)

        data = TensorDict(
            {
                "observations": self._obs[flat_idxs],
                "actions": self._actions[flat_idxs],
                "next": {
                    "observations": self._next_obs[flat_idxs],
                    "rewards": self._rewards[flat_idxs],
                    "dones": self._dones[flat_idxs],
                    "truncations": self._truncations[flat_idxs],
                    "effective_n_steps": self._eff_n[flat_idxs],
                },
            },
            batch_size=(self.n_env * per_env_batch,),
            device=self.device,
        )
        if self.asymmetric_obs:
            data["critic_observations"] = self._critic_obs[flat_idxs]
            data["next"]["critic_observations"] = self._next_critic_obs[flat_idxs]
        return data

    @property
    def size(self) -> int:
        """Total number of transitions added (including overwrites)."""
        return self._total_added


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_params(
    global_step: int,
    actor: nn.Module,
    qnet: nn.Module,
    qnet_target: nn.Module,
    obs_normalizer: nn.Module,
    critic_obs_normalizer: nn.Module,
    args,
    path: str,
) -> None:
    """Save a FastTD3 checkpoint to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": global_step,
            "actor_state_dict": actor.state_dict(),
            "qnet_state_dict": qnet.state_dict(),
            "qnet_target_state_dict": qnet_target.state_dict(),
            "obs_normalizer_state": obs_normalizer.state_dict(),
            "critic_obs_normalizer_state": critic_obs_normalizer.state_dict(),
        },
        path,
    )


def mark_step() -> None:
    """No-op framework hook (placeholder for XLA / JAX mark_step compatibility)."""
    pass
