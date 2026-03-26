"""
RFS (Residual Flow-matching RL) environment wrapper.

Wraps a ManagerBasedRLEnv (IkRel-v0) with a frozen CFMPCDPolicy as base policy.
PPO acts in a compound space:
  [residual (n_residual_dims) | noise (n_noise_dims * horizon)]

The noise is injected as the starting trajectory for CFM denoising (noise-space RL / DSRL style).
The residual is added on top of the denoised base action for specified action dims.

Usage:
    env = gym.make("UW-FrankaLeap-GraspPinkCup-IkRel-v0", cfg=env_cfg)
    rfs_env = RFSWrapper(
        env,
        diffusion_path="/path/to/cfm/checkpoint_dir",
        noise_dims=(0, 22),     # full action gets noise
        residual_dims=(6, 22),  # hand dims (6:22) get residual
        residual_scale=0.1,
    )
    sb3_env = Sb3VecEnvWrapper(rfs_env)
"""

import os
import sys

import yaml

# Ensure third_party packages (dill, diffusion_policy) are importable.
# isaac-sim's python.sh resets PYTHONPATH so we inject the paths here.
_RFS_DIR = os.path.dirname(os.path.abspath(__file__))
_UWLAB_DIR = os.path.abspath(os.path.join(_RFS_DIR, "../../../../"))
for _p in [
    os.path.join(_UWLAB_DIR, "third_party", "pip_packages"),
    os.path.join(_UWLAB_DIR, "third_party", "diffusion_policy"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dill
from collections import deque
import gymnasium as gym
import numpy as np
import torch

import torch.nn as nn
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.eval.bc_obs_formatter import BCObsFormatter


class LPFilter(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.y = None

    def reset(self):
        self.y = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.y is None:
            self.y = x.clone()
            return self.y.clone()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.clone()

from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy


def _resolve_hydra_refs(cfg: dict) -> dict:
    """Resolve simple ${key} Hydra references using top-level scalar values."""
    import re
    lookup = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}

    def resolve(val):
        if isinstance(val, str):
            def replacer(m):
                key = m.group(1)
                return str(lookup.get(key, m.group(0)))
            return re.sub(r'\$\{(\w+)\}', replacer, val)
        elif isinstance(val, dict):
            return {k: resolve(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [resolve(v) for v in val]
        return val

    return resolve(cfg)


def _load_cfm_checkpoint(diffusion_path: str, device: torch.device):
    ckpt_path = os.path.join(diffusion_path, "checkpoints", "best.ckpt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(diffusion_path, "checkpoints", "latest.ckpt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(diffusion_path, "latest.ckpt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found. Looked for best.ckpt, checkpoints/latest.ckpt, latest.ckpt in {diffusion_path}"
        )

    print(f"[RFSWrapper] Loading CFM checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, pickle_module=dill)

    if "state_dicts" in ckpt and "ema_model" in ckpt["state_dicts"]:
        state_dict = ckpt["state_dicts"]["ema_model"]
        cfg = ckpt["cfg"]
        shape_meta = {
            "action": {"shape": list(cfg.shape_meta.action.shape)},
            "obs": {
                "agent_pos": {"shape": list(cfg.shape_meta.obs.agent_pos.shape), "type": "low_dim"},
                "seg_pc": {"shape": list(cfg.shape_meta.obs.seg_pc.shape), "type": "pcd"},
            },
        }
        sigma = float(cfg.policy.noise_scheduler.sigma)
        horizon = int(cfg.horizon)
        n_action_steps = int(cfg.n_action_steps) + int(getattr(cfg, "n_latency_steps", 0))
        n_obs_steps = int(cfg.n_obs_steps)
        down_dims = tuple(cfg.policy.down_dims)
        obs_keys = list(cfg.dataset.obs_keys)
        image_keys = list(getattr(cfg.dataset, "image_keys", ["seg_pc"]))
        downsample_points = int(getattr(cfg.dataset, "downsample_points", 2048))
    else:
        state_dict = ckpt["ema_model"]
        config_path = os.path.join(diffusion_path, "config.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"BC checkpoint format detected but config.yaml not found at {config_path}"
            )
        with open(config_path) as f:
            train_cfg = _resolve_hydra_refs(yaml.safe_load(f))
        agent_pos_dim = int(state_dict["normalizer.params_dict.agent_pos.scale"].shape[0])
        action_dim = int(state_dict["normalizer.params_dict.action.scale"].shape[0])
        ds = train_cfg["dataset"]
        downsample_points = int(ds["downsample_points"])
        obs_keys = ds.get("obs_keys", ds.get("obs_key", []))
        image_keys = ds.get("image_keys", ["seg_pc"])
        shape_meta = {
            "action": {"shape": [action_dim]},
            "obs": {
                "agent_pos": {"shape": [agent_pos_dim], "type": "low_dim"},
                "seg_pc": {"shape": [3, downsample_points], "type": "pcd"},
            },
        }
        pol = train_cfg["policy"]
        sigma = float(pol.get("sigma", 0.0))
        horizon = int(train_cfg.get("horizon", 4))
        n_action_steps = int(train_cfg.get("n_action_steps", 8))
        n_obs_steps = int(train_cfg.get("n_obs_steps", 1))
        down_dims = tuple(pol["down_dims"])

    use_action_history = "normalizer.params_dict.past_actions.scale" in state_dict

    pcd_model = PointNet(
        in_channels=3,
        local_channels=(64, 64, 64, 128, 1024),
        global_channels=(512, 256),
        use_bn=False,
    )
    obs_encoder = MultiPCDObsEncoder(shape_meta=shape_meta, pcd_model=pcd_model)
    noise_scheduler = ConditionalFlowMatcher(sigma=sigma)

    policy = CFMPCDPolicy(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=5,
        diffusion_step_embed_dim=256,
        down_dims=down_dims,
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        use_action_history=use_action_history,
    )
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()

    # agent_pos_flat_dim = n_obs_steps * per_step_dim (total flat size seen by CFM normalizer)
    agent_pos_flat_dim = int(state_dict["normalizer.params_dict.agent_pos.scale"].shape[0])

    metadata = {
        "obs_keys": obs_keys,
        "image_keys": image_keys,
        "downsample_points": downsample_points,
        "n_obs_steps": n_obs_steps,
        "use_action_history": use_action_history,
        "agent_pos_flat_dim": agent_pos_flat_dim,
    }

    print(f"[RFSWrapper] CFM policy loaded: horizon={policy.horizon}, "
          f"action_dim={policy.action_dim}, n_obs_steps={n_obs_steps}, "
          f"use_action_history={use_action_history}")
    return policy, metadata


class RFSWrapper:
    """
    Wraps a ManagerBasedRLEnv for noise-space RL with a frozen CFMPCDPolicy base.

    PPO action space layout (flat, per-env):
        [residual (n_residual_dims) | noise (n_noise_dims * horizon)]

    Args:
        env: Isaac Lab gymnasium env (IkRel-v0 recommended).
        diffusion_path: Directory containing checkpoints/latest.ckpt.
        residual_step: Number of env substeps per PPO step. Rewards accumulate.
        noise_dims: (start, end) slice of cfm_action_dim that receives PPO noise.
        residual_dims: (start, end) slice of cfm_action_dim that receives PPO residual.
        residual_scale: Scale applied to PPO residual output (in [-1,1]) before adding.
        clip_actions: Clip arm dims to ±0.03m/±0.05rad after scaling.
        finger_smooth_alpha: LPFilter alpha for hand dims. 1.0 = disabled.
        num_warmup_steps: Zero-action warmup steps after each reset.
    """

    def __init__(
        self,
        env,
        diffusion_path: str,
        residual_step: int = 1,
        noise_dims: tuple = (0, 22),
        residual_dims: tuple = (6, 22),
        residual_scale: float = 0.1,
        clip_actions: bool = True,
        finger_smooth_alpha: float = 0.3,
        finger_start_dim: int = 6,
        num_warmup_steps: int = 0,
        asymmetric_ac: bool = False,
        gamma: float = 0.99,
        ppo_history: bool = False,
    ):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.residual_step = residual_step
        self.clip_actions = clip_actions
        self.residual_scale = residual_scale
        self.gamma = gamma
        self.ppo_history = ppo_history
        self.finger_start_dim = finger_start_dim

        self.asymmetric_ac = asymmetric_ac
        self.noise_dims = noise_dims
        self.residual_dims = residual_dims
        self.noise_slice = slice(*noise_dims) if noise_dims else None
        self.residual_slice = slice(*residual_dims) if residual_dims else None
        self.n_noise = (noise_dims[1] - noise_dims[0]) if noise_dims else 0
        self.n_residual = (residual_dims[1] - residual_dims[0]) if residual_dims else 0

        # Whether to maintain PPO history + past_action buffers.
        # Always true for asymmetric_ac (actor must match BC obs); else follows ppo_history flag.
        self._use_ppo_history: bool = ppo_history or asymmetric_ac

        # PPO history buffers (separate from formatter's buffers, same update cadence).
        self._ppo_history_buf: deque | None = None       # agent_pos frames
        self._ppo_past_action_buf: deque | None = None   # past executed actions

        # Episode reward tracking for SB3 ep_rew_mean logging.
        self._ep_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Last composite action (base + residual) sent to env. Set in step().
        self.last_action: torch.Tensor | None = None
        self.last_noise_flat: torch.Tensor | None = None
        self.last_residual: torch.Tensor | None = None
        self.last_pcd_embedding: torch.Tensor | None = None

        self.policy, metadata = _load_cfm_checkpoint(diffusion_path, self.device)
        self.horizon = self.policy.horizon
        self.cfm_action_dim = self.policy.action_dim

        # Keys that CFM conditions on (used as PPO history when _use_ppo_history=True).
        self._ppo_history_obs_keys: list[str] = metadata["obs_keys"]
        self._ppo_n_obs_steps: int = metadata["n_obs_steps"]
        self._agent_pos_flat_dim: int = metadata["agent_pos_flat_dim"]
        self._use_action_history: bool = metadata["use_action_history"]

        self.formatter = BCObsFormatter(
            obs_keys=metadata["obs_keys"],
            image_keys=metadata["image_keys"],
            downsample_points=metadata["downsample_points"],
            device=self.device,
            n_obs_steps=metadata["n_obs_steps"],
            action_dim=self.policy.action_dim if metadata["use_action_history"] else 0,
        )

        print(f"[RFSWrapper] Diffusion obs keys from checkpoint: {metadata['obs_keys']}")
        print(f"[RFSWrapper] n_obs_steps={metadata['n_obs_steps']}, "
              f"use_action_history={metadata['use_action_history']}, "
              f"ppo_history={ppo_history}, residual_step={residual_step}")
        print(f"[RFSWrapper] noise_dims={noise_dims}, residual_dims={residual_dims}")

        total_ppo_dim = self.n_residual + self.n_noise * self.horizon
        print(f"[RFSWrapper] PPO action dim: {self.n_residual} residual + "
              f"{self.n_noise}*{self.horizon} noise = {total_ppo_dim}")

        # Override env action/observation spaces for SB3 compatibility.
        # Sb3VecEnvWrapper reads single_action_space at __init__ time.
        single_as = gym.spaces.Box(low=-10.0, high=10.0, shape=(total_ppo_dim,), dtype=np.float32)
        self.env.unwrapped.single_action_space = single_as
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(single_as, self.num_envs)

        # Patch single_observation_space["policy"] for SB3 compatibility.
        _SKIP_OBS_KEYS = {"seg_pc", "rgb"}
        policy_obs_space = self.env.unwrapped.single_observation_space.get("policy")
        if isinstance(policy_obs_space, gym.spaces.Dict):
            if self.asymmetric_ac:
                self._setup_asymmetric_obs_space(policy_obs_space)
            else:
                # Symmetric (default): remove seg_pc/rgb, strip ee_pose to 3D.
                for k in _SKIP_OBS_KEYS:
                    policy_obs_space.spaces.pop(k, None)
                if "ee_pose" in policy_obs_space.spaces:
                    old = policy_obs_space.spaces["ee_pose"]
                    policy_obs_space.spaces["ee_pose"] = gym.spaces.Box(
                        low=old.low[..., :3], high=old.high[..., :3],
                        shape=(3,), dtype=old.dtype,
                    )
                if self.ppo_history:
                    # Replace raw per-step obs_keys with flattened history tensors.
                    for k in self._ppo_history_obs_keys:
                        policy_obs_space.spaces.pop(k, None)
                    policy_obs_space.spaces["agent_pos_history"] = gym.spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=(self._agent_pos_flat_dim,),
                        dtype=np.float32,
                    )
                    if self._use_action_history and self._ppo_n_obs_steps > 1:
                        past_dim = (self._ppo_n_obs_steps - 1) * self.cfm_action_dim
                        policy_obs_space.spaces["past_actions_history"] = gym.spaces.Box(
                            low=-np.inf, high=np.inf,
                            shape=(past_dim,),
                            dtype=np.float32,
                        )

        if finger_smooth_alpha < 1.0:
            self.finger_filter = LPFilter(alpha=finger_smooth_alpha).to(self.device)
        else:
            self.finger_filter = None

        self.num_warmup_steps = num_warmup_steps
        self.last_obs = None

    _SKIP_PPO_KEYS = {"seg_pc", "rgb"}

    def _setup_asymmetric_obs_space(self, policy_obs_space: gym.spaces.Dict):
        """Replace the policy obs space with actor_*/critic_* keys for asymmetric AC.

        Actor (non-privileged, matches BC training obs):
          - actor_pcd_emb:              PointNet embedding of current seg_pc
          - actor_agent_pos_history:    flattened n_obs_steps joint-pos history
          - actor_past_actions_history: flattened (n_obs_steps-1) past action history (if checkpoint uses it)

        Critic (privileged sim state):
          - critic_* for all non-PCD policy obs keys
        """
        _SKIP = self._SKIP_PPO_KEYS
        actor_spaces = {
            "actor_pcd_emb": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.policy.pcd_feat_dim,),
                dtype=np.float32,
            ),
            "actor_agent_pos_history": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._agent_pos_flat_dim,),
                dtype=np.float32,
            ),
        }
        if self._use_action_history and self._ppo_n_obs_steps > 1:
            past_dim = (self._ppo_n_obs_steps - 1) * self.cfm_action_dim
            actor_spaces["actor_past_actions_history"] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(past_dim,),
                dtype=np.float32,
            )
        critic_spaces = {}
        for k, space in policy_obs_space.spaces.items():
            if k in _SKIP:
                continue
            if k == "ee_pose":
                space = gym.spaces.Box(
                    low=space.low[..., :3], high=space.high[..., :3],
                    shape=(3,), dtype=space.dtype,
                )
            critic_spaces[f"critic_{k}"] = space
        policy_obs_space.spaces.clear()
        policy_obs_space.spaces.update({**actor_spaces, **critic_spaces})

    def _update_ppo_history(self, raw_policy_obs: dict):
        """Push current obs into the PPO history buffer.

        Called once per chunk (at the PPO decision boundary, after all substeps).
        History advances at the same cadence PPO acts — one obs per chunk —
        matching the cadence at which CFM's formatter buffer is updated.
        Buffer has maxlen=n_obs_steps; oldest frame is dropped on each append.
        """
        obs_parts = []
        for key in self._ppo_history_obs_keys:
            val = raw_policy_obs[key]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).to(self.device)
            obs_parts.append(val.float())
        agent_pos = torch.cat(obs_parts, dim=-1)  # (B, D)

        if self._ppo_history_buf is None:
            # Initialize: fill all frames with current obs (episode-start boundary).
            self._ppo_history_buf = deque(
                [agent_pos.clone() for _ in range(self._ppo_n_obs_steps)],
                maxlen=self._ppo_n_obs_steps,
            )
        else:
            self._ppo_history_buf.append(agent_pos)

    def _get_ppo_agent_pos_history(self) -> torch.Tensor:
        """Return flattened PPO joint-pos history (B, n_obs_steps * D), oldest first."""
        return torch.stack(list(self._ppo_history_buf), dim=1).flatten(1)

    def _update_ppo_past_actions(self, action: torch.Tensor):
        """Push last executed action of chunk into PPO past-actions buffer.
        Mirrors formatter.update_action() but at chunk-boundary cadence for PPO.
        """
        if not (self._use_action_history and self._ppo_n_obs_steps > 1):
            return
        n_past = self._ppo_n_obs_steps - 1
        if self._ppo_past_action_buf is None:
            self._ppo_past_action_buf = deque(
                [torch.zeros_like(action) for _ in range(n_past)],
                maxlen=n_past,
            )
        self._ppo_past_action_buf.append(action.clone())

    def _get_ppo_past_actions(self) -> torch.Tensor:
        """Return flattened PPO past-actions history (B, (n_obs_steps-1) * A), oldest first."""
        return torch.stack(list(self._ppo_past_action_buf), dim=1).flatten(1)

    def _reset_ppo_history_for_envs(self, reset_mask: torch.Tensor, raw_policy_obs: dict):
        """Fill all history frames for reset envs with their current (post-reset) obs,
        and zero out their past-actions history."""
        if self._ppo_history_buf is not None:
            obs_parts = []
            for key in self._ppo_history_obs_keys:
                val = raw_policy_obs[key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).to(self.device)
                obs_parts.append(val.float())
            current_agent_pos = torch.cat(obs_parts, dim=-1)  # (B, D)
            for frame in self._ppo_history_buf:
                frame[reset_mask] = current_agent_pos[reset_mask]
        if self._ppo_past_action_buf is not None:
            for frame in self._ppo_past_action_buf:
                frame[reset_mask] = 0.0

    def _strip_ppo_obs(self, obs: dict) -> dict:
        """Return obs stripped for PPO: remove seg_pc/rgb, strip ee_pose to 3D xyz.
        When ppo_history=True (symmetric) or asymmetric_ac: replaces raw obs_keys
        with flattened agent_pos_history and past_actions_history tensors.
        self.last_obs is NOT modified — CFM still reads the full obs including seg_pc."""
        if self.asymmetric_ac:
            return self._asymmetric_ppo_obs(obs)
        policy = obs["policy"]
        stripped = {k: v for k, v in policy.items() if k not in self._SKIP_PPO_KEYS}
        if "ee_pose" in stripped:
            stripped["ee_pose"] = stripped["ee_pose"][..., :3]
        if self.ppo_history:
            for k in self._ppo_history_obs_keys:
                stripped.pop(k, None)
            stripped["agent_pos_history"] = self._get_ppo_agent_pos_history()
            if self._use_action_history and self._ppo_past_action_buf is not None:
                stripped["past_actions_history"] = self._get_ppo_past_actions()
        return {"policy": stripped}

    def _compute_pcd_embedding(self) -> torch.Tensor:
        """Encode the current frame's PCD from self.last_obs without touching formatter state."""
        policy_obs = self.last_obs["policy"]
        for key in self.policy.obs_encoder.pcd_keys:
            pcd = policy_obs[key].float()  # (B, 3, N)
            N = pcd.shape[-1]
            if N > self.formatter.downsample_points:
                perm = torch.randperm(N, device=self.device)[:self.formatter.downsample_points]
                pcd = pcd[:, :, perm]
            # normalizer expects (B, 1, 3, N); index back to (B, 3, N) after
            nobs_pcd = self.policy.normalizer[key].normalize(pcd.unsqueeze(1))
            pcd_obs = {key: nobs_pcd[:, 0]}
        with torch.no_grad():
            return self.policy.obs_encoder.encode_pcd_only(pcd_obs).detach()

    def _asymmetric_ppo_obs(self, obs: dict) -> dict:
        """Build actor/critic obs dict for asymmetric AC.
        Actor sees what BC sees (non-privileged): PCD embedding + joint history + past actions.
        Critic sees privileged sim state (all non-PCD policy obs).
        """
        policy = obs["policy"]
        result = {
            "actor_pcd_emb":             self.last_pcd_embedding,
            "actor_agent_pos_history":   self._get_ppo_agent_pos_history(),
        }
        if self._use_action_history and self._ppo_past_action_buf is not None:
            result["actor_past_actions_history"] = self._get_ppo_past_actions()
        for k, v in policy.items():
            if k in self._SKIP_PPO_KEYS:
                continue
            if k == "ee_pose":
                v = v[..., :3]
            result[f"critic_{k}"] = v
        return {"policy": result}

    def step(self, ppo_actions: torch.Tensor):
        ppo_out = ppo_actions.clamp(-1.0, 1.0)

        residual = ppo_out[:, :self.n_residual]
        noise_flat = ppo_out[:, self.n_residual:]

        self.last_noise_flat = noise_flat.detach()
        self.last_residual = residual.detach() if self.n_residual > 0 else None

        noise = torch.zeros(self.num_envs, self.horizon, self.cfm_action_dim, device=self.device)
        if self.noise_slice is not None and self.n_noise > 0:
            noise[:, :, self.noise_slice] = noise_flat.reshape(self.num_envs, self.horizon, self.n_noise)

        diffusion_obs = self.formatter.format(self.last_obs["policy"])
        with torch.no_grad():
            cfm_result = self.policy.predict_action(diffusion_obs, noise=noise)
        base_actions = cfm_result["action_pred"]

        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        any_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Apply γ^k discounting within the chunk so the accumulated reward is the
        # correct multi-step discounted return, not an undiscounted sum.
        discount = 1.0

        for substep in range(self.residual_step):
            action = base_actions[:, substep].clone()

            if self.n_residual > 0 and self.residual_slice is not None:
                action[:, self.residual_slice] += residual * self.residual_scale

            if self.clip_actions:
                action[:, :3].clamp_(-0.03, 0.03)
                action[:, 3:6].clamp_(-0.05, 0.05)

            if self.finger_filter is not None:
                action[:, self.finger_start_dim:] = self.finger_filter(action[:, self.finger_start_dim:])

            self.last_action = action.detach()
            obs, step_rewards, terminated, truncated, info = self.env.step(action)
            rewards += discount * step_rewards
            discount *= self.gamma
            self.last_obs = obs
            any_reset |= terminated | truncated

        # Reset formatter history for envs that auto-reset during this PPO step.
        # Use zeros for their past_actions update to match episode-start behavior.
        if any_reset.any():
            self.formatter.reset_envs(any_reset, self.last_obs["policy"])
            masked_action = self.last_action.clone()
            masked_action[any_reset] = 0.0
            self.formatter.update_action(masked_action)
            if self._use_ppo_history:
                self._reset_ppo_history_for_envs(any_reset, self.last_obs["policy"])
        else:
            self.formatter.update_action(self.last_action)

        # Advance PPO history and past-action buffers once per chunk.
        # Both update at chunk-boundary cadence, matching CFM's formatter cadence.
        # For reset envs, _reset_ppo_history_for_envs already filled their buffers;
        # appending one more copy of the same obs/zero is a no-op for those envs.
        # Use masked action for reset envs to match formatter.update_action() behavior.
        if self._use_ppo_history:
            self._update_ppo_history(self.last_obs["policy"])
            ppo_action = self.last_action
            if any_reset.any():
                ppo_action = self.last_action.clone()
                ppo_action[any_reset] = 0.0
            self._update_ppo_past_actions(ppo_action)

        # Update pcd embedding from the final obs so it is consistent with the
        # state returned to the agent — no lag between pcd_emb and state.
        if self.asymmetric_ac:
            self.last_pcd_embedding = self._compute_pcd_embedding()

        self._ep_rewards += rewards
        # Use any_reset (not just last-substep dones) so mid-chunk resets are logged.
        # rewards[i] may be slightly contaminated by post-reset substeps for mid-chunk
        # resets, but this is the standard frame-skip approximation and far better than
        # silently dropping completed episodes from ep_rew_mean.
        log_done = any_reset | terminated | truncated
        if log_done.any():
            if not isinstance(info, list):
                info = [{} for _ in range(self.num_envs)]
            for i in range(self.num_envs):
                if log_done[i]:
                    info[i]["episode"] = {"r": self._ep_rewards[i].item(), "l": 0}
                    self._ep_rewards[i] = 0.0

        return self._strip_ppo_obs(self.last_obs), rewards, terminated, truncated, info

    def _do_warmup(self):
        if self.num_warmup_steps <= 0:
            return
        warmup_act = self.env.unwrapped.cfg.warmup_action(self.env.unwrapped)
        for _ in range(self.num_warmup_steps):
            obs, _, _, _, _ = self.env.step(warmup_act)
            self.last_obs = obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self._ep_rewards.zero_()
        self.formatter.reset()
        self._ppo_history_buf = None      # clear so _update_ppo_history re-initializes below
        self._ppo_past_action_buf = None  # clear so _update_ppo_past_actions re-initializes below

        if self.finger_filter is not None:
            self.finger_filter.reset()
            current_finger = self.env.unwrapped.scene["robot"].data.joint_pos[:, 7:]
            self.finger_filter(current_finger)

        self._do_warmup()
        if self.asymmetric_ac:
            self.last_pcd_embedding = self._compute_pcd_embedding()
        # Initialize PPO history with post-warmup obs (fills all n_obs_steps frames).
        # Also initialize past_action_buf with zeros so _strip_ppo_obs always finds it.
        if self._use_ppo_history:
            self._update_ppo_history(self.last_obs["policy"])
            if self._use_action_history and self._ppo_n_obs_steps > 1:
                n_past = self._ppo_n_obs_steps - 1
                self._ppo_past_action_buf = deque(
                    [torch.zeros(self.num_envs, self.cfm_action_dim, device=self.device)
                     for _ in range(n_past)],
                    maxlen=n_past,
                )
        return self._strip_ppo_obs(self.last_obs), info

    def reset_to_spawn(self, spawn_pose, info=None):
        obs, info = self.reset()  # includes _do_warmup()

        isaac_env = self.env.unwrapped
        obj = isaac_env.scene["grasp_object"]
        defaults = isaac_env.cfg.object_spawn_defaults

        default_pos = torch.tensor(defaults["default_pos"], dtype=torch.float32, device=self.device)
        default_rot = torch.tensor(defaults["default_rot"], dtype=torch.float32, device=self.device)

        new_pos = default_pos.clone()
        new_pos[0] += spawn_pose.x
        new_pos[1] += spawn_pose.y

        root_state = obj.data.default_root_state.clone()
        root_state[:, :3] = new_pos.unsqueeze(0) + isaac_env.scene.env_origins
        root_state[:, 3:7] = default_rot.unsqueeze(0)
        root_state[:, 7:] = 0.0

        obj.write_root_state_to_sim(root_state)
        isaac_env.sim.step()

        obs = {"policy": isaac_env.observation_manager.compute()["policy"]}
        self.last_obs = obs
        if self._use_ppo_history:
            self._ppo_history_buf = None
            self._ppo_past_action_buf = None
            self._update_ppo_history(self.last_obs["policy"])
            if self._use_action_history and self._ppo_n_obs_steps > 1:
                n_past = self._ppo_n_obs_steps - 1
                self._ppo_past_action_buf = deque(
                    [torch.zeros(self.num_envs, self.cfm_action_dim, device=self.device)
                     for _ in range(n_past)],
                    maxlen=n_past,
                )
        if self.asymmetric_ac:
            self.last_pcd_embedding = self._compute_pcd_embedding()
        return self._strip_ppo_obs(self.last_obs), info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
