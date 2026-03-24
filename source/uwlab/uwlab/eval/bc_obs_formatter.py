from collections import deque
from typing import Dict, List
import numpy as np
import torch


class BCObsFormatter:
    """
    Formats raw Isaac Sim env observations into the dict expected by CFMPCDPolicy.

    Low-dim state (agent_pos) and past actions carry a rolling history of
    n_obs_steps frames, matching how ZarrDataset presents obs at train time.
    Point clouds always use the current frame only.

    Temporal ordering: history buffer index 0 is oldest, index -1 is current.
    Episode boundaries: call reset() at the start of each episode. Buffers are
    pre-filled with zeros, matching training's zero pad_before at episode start.

    Call update_action(action) after each env.step() to push the executed action
    into the past-action buffer so the next policy query sees it.

    Args:
        obs_keys:          list of env obs keys to concatenate into agent_pos
        image_keys:        list of pcd keys, e.g. ["seg_pc"]
        downsample_points: number of points to randomly sample from each PCD
        device:            torch device for output tensors
        n_obs_steps:       number of history steps (matches training config)
        action_dim:        action dimension, required when n_obs_steps > 1
    """

    def __init__(
        self,
        obs_keys: List[str],
        image_keys: List[str],
        downsample_points: int,
        device: torch.device,
        n_obs_steps: int = 1,
        action_dim: int = 0,
    ):
        self.obs_keys = obs_keys
        self.image_keys = image_keys
        self.downsample_points = downsample_points
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self._agent_pos_buf: deque | None = None
        self._past_action_buf: deque | None = None
        self._past_action_first: bool = False

    def reset(self):
        """Call at the start of each episode to clear all history buffers."""
        self._agent_pos_buf = None
        self._past_action_buf = None
        self._past_action_first = self.n_obs_steps > 1 and self.action_dim > 0

    def reset_envs(self, reset_mask: torch.Tensor, current_obs: dict):
        """Reinitialize history for specific envs after a mid-episode auto-reset.

        Called when some envs in a vectorized batch terminate while others continue.
        Fills the agent_pos history for reset envs with their current obs (boundary
        repetition, matching episode-start behavior). Zeros out their past_actions.

        reset_mask: (B,) bool tensor — True for envs that just reset
        current_obs: raw policy obs dict (same format as format() input), post-reset
        """
        if not reset_mask.any() or self._agent_pos_buf is None:
            return

        obs_parts = []
        for key in self.obs_keys:
            val = current_obs[key]
            if isinstance(val, torch.Tensor):
                val = val.float()
            else:
                val = torch.from_numpy(val).to(self.device).float()
            obs_parts.append(val)
        current_agent_pos = torch.cat(obs_parts, dim=-1)  # (B, D)

        for frame in self._agent_pos_buf:
            frame[reset_mask] = current_agent_pos[reset_mask]

        if self._past_action_buf is not None:
            for frame in self._past_action_buf:
                frame[reset_mask] = 0.0

    def update_action(self, action: torch.Tensor):
        """Push the executed action into the past-action buffer.
        Call this after env.step() and before the next format() call.
        action: (B, A)
        """
        if not (self.n_obs_steps > 1 and self.action_dim > 0):
            return
        if self._past_action_first:
            # First real action: fill entire buffer by repeating it to match
            # training's boundary-frame repetition at episode start.
            self._past_action_buf = deque(
                [action] * (self.n_obs_steps - 1), maxlen=self.n_obs_steps - 1
            )
            self._past_action_first = False
        else:
            self._past_action_buf.append(action)

    def format(self, raw_obs: dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            raw_obs: env observation dict (obs["policy"] from Isaac Sim).
                     Values are (B, D) or (B, 3, N) tensors already on device.

        Returns:
            {
              "agent_pos":    (B, n_obs_steps, D_total),        # low-dim history, oldest first
              "past_actions": (B, n_obs_steps-1, A),            # only when n_obs_steps > 1
              <image_key>:    (B, 1, 3, downsample_points),     # current frame only
            }
        """
        policy_obs = raw_obs

        # --- agent_pos: concatenate obs_keys for current step ---
        obs_parts = []
        for key in self.obs_keys:
            val = policy_obs[key]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).to(self.device)
            obs_parts.append(val.float())
        agent_pos = torch.cat(obs_parts, dim=-1)  # (B, D_total)

        if self._agent_pos_buf is None:
            self._agent_pos_buf = deque([agent_pos] * self.n_obs_steps, maxlen=self.n_obs_steps)
        else:
            self._agent_pos_buf.append(agent_pos)

        result = {"agent_pos": torch.stack(list(self._agent_pos_buf), dim=1)}

        # --- past actions ---
        if self.n_obs_steps > 1 and self.action_dim > 0:
            if self._past_action_buf is not None:
                result["past_actions"] = torch.stack(list(self._past_action_buf), dim=1)
            else:
                # Before first action is taken: use zeros as placeholder
                B = agent_pos.shape[0]
                result["past_actions"] = torch.zeros(
                    B, self.n_obs_steps - 1, self.action_dim, device=self.device
                )

        # --- pcd: current frame only, no history ---
        for key in self.image_keys:
            pcd = policy_obs[key]  # (B, 3, N)
            if isinstance(pcd, np.ndarray):
                pcd = torch.from_numpy(pcd).to(self.device)
            pcd = pcd.float()

            N = pcd.shape[-1]
            if N > self.downsample_points:
                perm = torch.randperm(N, device=self.device)[:self.downsample_points]
                pcd = pcd[:, :, perm]

            result[key] = pcd.unsqueeze(1)  # (B, 1, 3, downsample_points)

        return result
