"""
IsaacLabDPPOWrapper: adapts a batched Isaac Lab ManagerBasedRLEnv to the
interface expected by the DPPO training loop.

DPPO expects:
  reset()  -> {"state": Tensor(n_envs, cond_steps, obs_dim)}
  step(actions: Tensor(n_envs, act_steps, action_dim))
           -> obs_dict, reward, terminated, truncated, info

Observations mirror RFS symmetric mode (no point cloud):
  - Skip keys: {"seg_pc", "rgb"}
  - ee_pose -> stripped to xyz [:3]
  - All remaining obs["policy"] keys concatenated into a flat state vector
  - Wrapped in a cond_steps=1 history dimension
"""

import torch
from typing import Optional


_SKIP_OBS_KEYS = frozenset({"seg_pc", "rgb"})


class IsaacLabDPPOWrapper:
    """Wraps an Isaac Lab gymnasium env for DPPO fine-tuning.

    Args:
        env: Isaac Lab gymnasium environment (already made via gym.make).
        act_steps: number of env control steps to execute per DPPO policy step.
            Matches diffusion.act_steps in dppo_cfg.yaml.
        horizon_steps: action chunk size from diffusion model (unused at env
            level, stored for reference / shape checks).
        skip_obs_keys: obs keys to exclude from the flat state vector (PCD, RGB).
    """

    def __init__(
        self,
        env,
        act_steps: int = 4,
        horizon_steps: int = 4,
        skip_obs_keys: Optional[frozenset] = None,
    ):
        self.env = env
        self.act_steps = act_steps
        self.horizon_steps = horizon_steps
        self._skip_keys = skip_obs_keys if skip_obs_keys is not None else _SKIP_OBS_KEYS

        # Inferred on first reset
        self._obs_dim: Optional[int] = None
        self._obs_key_order: Optional[list] = None  # ordered keys for stable concat

        # Cached from env
        self._n_envs: Optional[int] = None
        self._action_dim: Optional[int] = None
        self._device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is None:
            raise RuntimeError("Call reset() first to infer obs_dim.")
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        if self._action_dim is None:
            raise RuntimeError("Call reset() first to infer action_dim.")
        return self._action_dim

    @property
    def n_envs(self) -> int:
        if self._n_envs is None:
            raise RuntimeError("Call reset() first to infer n_envs.")
        return self._n_envs

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Call reset() first to infer device.")
        return self._device

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset all envs and return initial DPPO observation."""
        obs, _ = self.env.reset()
        self._init_from_obs(obs)
        return self._obs_to_state(obs)

    def step(self, actions: torch.Tensor):
        """Execute act_steps control steps, accumulating rewards.

        Args:
            actions: (n_envs, act_steps, action_dim) or (n_envs, action_dim)
                     If 2-D, treated as a single step (act_steps=1).
        Returns:
            obs_dict:  {"state": Tensor(n_envs, 1, obs_dim)}
            reward:    Tensor(n_envs,)  — sum over act_steps
            terminated: Tensor(n_envs,) bool — OR over act_steps
            truncated:  Tensor(n_envs,) bool — OR over act_steps
            info:       dict from the last env step
        """
        actions = actions.to(self._device)
        if actions.ndim == 2:
            # single step shortcut
            actions = actions.unsqueeze(1)

        n_envs = actions.shape[0]
        total_reward = torch.zeros(n_envs, device=self._device)
        any_terminated = torch.zeros(n_envs, dtype=torch.bool, device=self._device)
        any_truncated = torch.zeros(n_envs, dtype=torch.bool, device=self._device)
        # active[i] is False once env i has terminated/truncated — stop adding reward
        active = torch.ones(n_envs, dtype=torch.bool, device=self._device)

        obs = info = None
        for i in range(self.act_steps):
            obs, reward, terminated, truncated, info = self.env.step(actions[:, i])

            # Coerce to GPU tensor
            reward = _to_tensor(reward, self._device)
            terminated = _to_bool_tensor(terminated, self._device)
            truncated = _to_bool_tensor(truncated, self._device)

            total_reward += reward * active.float()
            done = terminated | truncated
            # Freeze reward accumulation for envs that just finished
            active = active & ~done
            any_terminated |= terminated
            any_truncated |= truncated

        return self._obs_to_state(obs), total_reward, any_terminated, any_truncated, info

    # ------------------------------------------------------------------
    # Obs helpers
    # ------------------------------------------------------------------

    def _init_from_obs(self, obs: dict):
        """Infer and cache shapes from the first observation."""
        policy = obs["policy"]
        # Determine stable key order: sort for reproducibility, skip excluded keys
        keys = sorted(
            k for k in policy if k not in self._skip_keys
        )
        # Pull tensors, strip ee_pose to xyz
        parts = []
        for k in keys:
            v = policy[k].float()
            if k == "ee_pose":
                v = v[..., :3]
            parts.append(v)

        self._obs_key_order = keys
        flat = torch.cat([p.reshape(p.shape[0], -1) for p in parts], dim=-1)
        self._obs_dim = flat.shape[-1]
        self._n_envs = flat.shape[0]
        self._device = flat.device

        # Infer action dim from env action space
        import numpy as np
        act_space = self.env.action_space
        if hasattr(act_space, "shape"):
            self._action_dim = int(act_space.shape[-1])
        else:
            raise ValueError(f"Cannot infer action_dim from action space: {act_space}")

    def _obs_to_state(self, obs: dict) -> dict:
        """Convert Isaac Lab obs dict to {"state": (n_envs, 1, obs_dim)}."""
        policy = obs["policy"]
        parts = []
        for k in self._obs_key_order:
            v = policy[k].float()
            if k == "ee_pose":
                v = v[..., :3]
            parts.append(v.reshape(v.shape[0], -1))
        flat = torch.cat(parts, dim=-1)  # (n_envs, obs_dim)
        return {"state": flat.unsqueeze(1)}  # (n_envs, 1, obs_dim)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _to_tensor(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    import numpy as np
    return torch.tensor(x, dtype=torch.float32, device=device)


def _to_bool_tensor(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.bool().to(device)
    import numpy as np
    return torch.tensor(x, dtype=torch.bool, device=device)
