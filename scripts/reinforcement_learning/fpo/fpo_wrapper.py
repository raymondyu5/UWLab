"""
FPO (Flow Policy Optimization) environment wrapper.

Wraps a ManagerBasedRLEnv with an unfrozen CFMPCDPolicy trained end-to-end
with FPO policy gradients.

Unlike RFS, there is no residual or noise injection — the flow model itself is
the policy being optimized.  The PointNet obs encoder is frozen; only the UNet
denoising network (policy.model) receives RL gradients.

Action executed per step: chunk[:, 0]  (first of the H-step predicted chunk).
"""

import os
import sys
import yaml

_FPO_DIR = os.path.dirname(os.path.abspath(__file__))
_UWLAB_DIR = os.path.abspath(os.path.join(_FPO_DIR, "../../../"))
for _p in [
    os.path.join(_UWLAB_DIR, "third_party", "pip_packages"),
    os.path.join(_UWLAB_DIR, "third_party", "diffusion_policy"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import copy
import dill
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.eval.bc_obs_formatter import BCObsFormatter
from uwlab.utils.checkpoint import extract_ckpt_metadata, format_ckpt_metadata
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy


@dataclass
class FpoStepData:
    """FPO loss info collected at rollout time (all tensors are detached)."""
    global_cond: torch.Tensor       # (B, global_cond_dim)
    chunk_norm: torch.Tensor        # (B, H, A) normalized action chunk
    eps: torch.Tensor               # (B, N, H, A) CFM noise samples
    t_samp: torch.Tensor            # (B, N)    CFM interpolation timesteps
    initial_cfm_loss: torch.Tensor  # (B, N)    CFM loss at rollout-time weights


def _resolve_hydra_refs(cfg: dict) -> dict:
    import re
    lookup = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}
    def resolve(val):
        if isinstance(val, str):
            return re.sub(r'\$\{(\w+)\}', lambda m: str(lookup.get(m.group(1), m.group(0))), val)
        elif isinstance(val, dict):
            return {k: resolve(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [resolve(v) for v in val]
        return val
    return resolve(cfg)


def _load_cfm_checkpoint(diffusion_path: str, device: torch.device):
    """Load CFM checkpoint — identical to rfs/wrapper.py."""
    ckpt_path = os.path.join(diffusion_path, "checkpoints", "best.ckpt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(diffusion_path, "best.ckpt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(diffusion_path, "checkpoints", "latest.ckpt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found. Looked for best.ckpt, checkpoints/best.ckpt, "
            f"latest.ckpt in {diffusion_path}"
        )

    print(f"[FPOWrapper] Loading CFM checkpoint from {ckpt_path}")
    resolved_ckpt_path = os.path.realpath(ckpt_path)
    if resolved_ckpt_path != ckpt_path:
        print(f"[FPOWrapper] Resolved checkpoint path: {resolved_ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, pickle_module=dill)
    ckpt_meta = extract_ckpt_metadata(ckpt)
    print(f"[FPOWrapper] Checkpoint metadata: {format_ckpt_metadata(ckpt_meta)}")

    if "state_dicts" in ckpt and "ema_model" in ckpt["state_dicts"]:
        state_dict = ckpt["state_dicts"]["ema_model"]
        cfg = ckpt["cfg"]
        shape_meta = {
            "action": {"shape": list(cfg.shape_meta.action.shape)},
            "obs": {
                "agent_pos": {"shape": list(cfg.shape_meta.obs.agent_pos.shape), "type": "low_dim"},
                "seg_pc":    {"shape": list(cfg.shape_meta.obs.seg_pc.shape),    "type": "pcd"},
            },
        }
        sigma          = float(cfg.policy.noise_scheduler.sigma)
        horizon        = int(cfg.horizon)
        n_action_steps = int(cfg.n_action_steps) + int(getattr(cfg, "n_latency_steps", 0))
        n_obs_steps    = int(cfg.n_obs_steps)
        down_dims      = tuple(cfg.policy.down_dims)
        obs_keys       = list(cfg.dataset.obs_keys)
        image_keys     = list(getattr(cfg.dataset, "image_keys", ["seg_pc"]))
        downsample_points = int(getattr(cfg.dataset, "downsample_points", 2048))
        pnet_cfg = {}
    else:
        state_dict = ckpt["ema_model"]
        if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
            train_cfg = ckpt["cfg"]
        else:
            config_path = os.path.join(diffusion_path, "config.yaml")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"config.yaml not found at {config_path}")
            with open(config_path) as f:
                train_cfg = _resolve_hydra_refs(yaml.safe_load(f))
        agent_pos_dim = int(state_dict["normalizer.params_dict.agent_pos.scale"].shape[0])
        action_dim    = int(state_dict["normalizer.params_dict.action.scale"].shape[0])
        ds, pol = train_cfg["dataset"], train_cfg["policy"]
        downsample_points = int(ds.get("downsample_points", 2048))
        obs_keys          = ds.get("obs_keys", ds.get("obs_key", []))
        image_keys        = ds.get("image_keys", ["seg_pc"])
        shape_meta = {
            "action": {"shape": [action_dim]},
            "obs": {
                "agent_pos": {"shape": [agent_pos_dim],              "type": "low_dim"},
                "seg_pc":    {"shape": [3, downsample_points],       "type": "pcd"},
            },
        }
        sigma          = float(pol.get("sigma", 0.0))
        horizon        = int(train_cfg.get("horizon", 4))
        n_action_steps = int(train_cfg.get("n_action_steps", 8))
        n_obs_steps    = int(train_cfg.get("n_obs_steps", 1))
        down_dims      = tuple(pol["down_dims"])
        pnet_cfg       = train_cfg.get("pointnet", {})

    use_action_history = "normalizer.params_dict.past_actions.scale" in state_dict

    _default_local  = (64, 64, 64, 128, 1024)
    _default_global = (512, 256)
    pcd_model = PointNet(
        in_channels=3,
        local_channels=tuple(pnet_cfg.get("local_channels", _default_local)),
        global_channels=tuple(pnet_cfg.get("global_channels", _default_global)),
        use_bn=bool(pnet_cfg.get("use_bn", False)),
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

    agent_pos_flat_dim = n_obs_steps * int(state_dict["normalizer.params_dict.agent_pos.scale"].shape[0])
    metadata = {
        "obs_keys": obs_keys,
        "image_keys": image_keys,
        "downsample_points": downsample_points,
        "n_obs_steps": n_obs_steps,
        "use_action_history": use_action_history,
        "agent_pos_flat_dim": agent_pos_flat_dim,
        "ckpt_path": ckpt_path,
        "resolved_ckpt_path": resolved_ckpt_path,
        "ckpt_meta": ckpt_meta,
    }
    print(f"[FPOWrapper] CFM policy loaded: horizon={policy.horizon}, "
          f"action_dim={policy.action_dim}, n_obs_steps={n_obs_steps}, "
          f"use_action_history={use_action_history}")
    return policy, metadata


class FPOWrapper:
    """
    Wraps a ManagerBasedRLEnv for FPO training.

    The CFMPCDPolicy is loaded from a BC checkpoint.
    - PointNet obs encoder: FROZEN (no RL gradients)
    - UNet denoising network (policy.model): UNFROZEN (receives RL gradients)

    The wrapper exposes asymmetric AC obs (same layout as RFSWrapper):
      actor_*: pcd_emb + agent_pos_history [+ past_actions_history]
      critic_*: all non-PCD policy obs keys (privileged sim state)

    call reset() once, then step() in a loop.
    FPO loss info for each step is stored in self.last_fpo_data.
    """

    _SKIP_PPO_KEYS = {"seg_pc", "rgb"}

    def __init__(
        self,
        env,
        diffusion_path: str,
        num_warmup_steps: int = 0,
        gamma: float = 0.99,
        n_cfm_samples: int = 1,
    ):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.gamma = gamma
        self.num_warmup_steps = num_warmup_steps
        self.n_cfm_samples = n_cfm_samples

        self._ep_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.policy, metadata = _load_cfm_checkpoint(diffusion_path, self.device)

        # Freeze PointNet + normalizer; UNet (policy.model) gets RL gradients.
        for param in self.policy.obs_encoder.parameters():
            param.requires_grad_(False)
        for param in self.policy.normalizer.parameters():
            param.requires_grad_(False)
        self.policy.obs_encoder.eval()
        self.policy.model.train()

        # Frozen copy of the original BC UNet for BC regularization.
        self.frozen_model = copy.deepcopy(self.policy.model)
        self.frozen_model.eval()
        for param in self.frozen_model.parameters():
            param.requires_grad_(False)

        self.policy_horizon  = self.policy.horizon
        self.cfm_action_dim  = self.policy.action_dim

        self._ppo_history_obs_keys: list = metadata["obs_keys"]
        self._ppo_n_obs_steps: int       = metadata["n_obs_steps"]
        self._agent_pos_flat_dim: int    = metadata["agent_pos_flat_dim"]
        self._use_action_history: bool   = metadata["use_action_history"]
        self.diffusion_ckpt_path         = metadata.get("ckpt_path")
        self.diffusion_ckpt_resolved_path= metadata.get("resolved_ckpt_path")
        self.diffusion_ckpt_meta: dict   = metadata.get("ckpt_meta", {})

        self.formatter = BCObsFormatter(
            obs_keys=metadata["obs_keys"],
            image_keys=metadata["image_keys"],
            downsample_points=metadata["downsample_points"],
            device=self.device,
            n_obs_steps=metadata["n_obs_steps"],
            action_dim=self.policy.action_dim if metadata["use_action_history"] else 0,
        )

        # PPO history buffers (same cadence as RFSWrapper: one update per chunk).
        self._ppo_history_buf: deque | None     = None
        self._ppo_past_action_buf: deque | None = None
        self._last_pcd_embedding: torch.Tensor | None = None

        # Patch obs space with actor_* / critic_* keys.
        policy_obs_space = self.env.unwrapped.single_observation_space.get("policy")
        if isinstance(policy_obs_space, gym.spaces.Dict):
            self._setup_asymmetric_obs_space(policy_obs_space)

        # Critic obs: all critic_* keys flattened.
        self.critic_obs_keys = sorted(
            k for k in policy_obs_space.spaces if k.startswith("critic_")
        )
        self.critic_obs_dim = sum(
            int(np.prod(policy_obs_space.spaces[k].shape))
            for k in self.critic_obs_keys
        )

        self.last_obs: dict | None             = None
        self.last_fpo_data: FpoStepData | None = None
        self.last_action: torch.Tensor | None  = None

        # Video frame capture (enabled only during eval).
        self._collect_substep_frames: bool  = False
        self.last_substep_frames: list | None = None

        n_unet_params = sum(p.numel() for p in self.policy.model.parameters())
        print(f"[FPOWrapper] critic_obs_dim={self.critic_obs_dim}, "
              f"global_cond_dim={self.policy.global_cond_dim}, "
              f"UNet trainable params={n_unet_params:,}")

    # ------------------------------------------------------------------
    # Obs space setup (mirrors RFSWrapper._setup_asymmetric_obs_space)
    # ------------------------------------------------------------------

    def _setup_asymmetric_obs_space(self, policy_obs_space: gym.spaces.Dict):
        _SKIP = self._SKIP_PPO_KEYS
        actor_spaces = {
            "actor_pcd_emb": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.policy.pcd_feat_dim,), dtype=np.float32,
            ),
            "actor_agent_pos_history": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._agent_pos_flat_dim,), dtype=np.float32,
            ),
        }
        if self._use_action_history and self._ppo_n_obs_steps > 1:
            past_dim = (self._ppo_n_obs_steps - 1) * self.cfm_action_dim
            actor_spaces["actor_past_actions_history"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(past_dim,), dtype=np.float32,
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

    # ------------------------------------------------------------------
    # History management (mirrors RFSWrapper)
    # ------------------------------------------------------------------

    def _update_ppo_history(self, raw_policy_obs: dict):
        parts = []
        for key in self._ppo_history_obs_keys:
            val = raw_policy_obs[key]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).to(self.device)
            parts.append(val.float())
        agent_pos = torch.cat(parts, dim=-1)
        if self._ppo_history_buf is None:
            self._ppo_history_buf = deque(
                [agent_pos.clone() for _ in range(self._ppo_n_obs_steps)],
                maxlen=self._ppo_n_obs_steps,
            )
        else:
            self._ppo_history_buf.append(agent_pos)

    def _get_ppo_agent_pos_history(self) -> torch.Tensor:
        return torch.stack(list(self._ppo_history_buf), dim=1).flatten(1)

    def _update_ppo_past_actions(self, action: torch.Tensor):
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
        return torch.stack(list(self._ppo_past_action_buf), dim=1).flatten(1)

    def _reset_ppo_history_for_envs(self, reset_mask: torch.Tensor, raw_policy_obs: dict):
        if self._ppo_history_buf is not None:
            parts = []
            for key in self._ppo_history_obs_keys:
                val = raw_policy_obs[key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).to(self.device)
                parts.append(val.float())
            cur = torch.cat(parts, dim=-1)
            for frame in self._ppo_history_buf:
                frame[reset_mask] = cur[reset_mask]
        if self._ppo_past_action_buf is not None:
            for frame in self._ppo_past_action_buf:
                frame[reset_mask] = 0.0

    # ------------------------------------------------------------------
    # Obs construction
    # ------------------------------------------------------------------

    def _compute_pcd_embedding(self) -> torch.Tensor:
        policy_obs = self.last_obs["policy"]
        pcd_obs = {}
        for key in self.policy.obs_encoder.pcd_keys:
            pcd = policy_obs[key].float()
            N = pcd.shape[-1]
            if N > self.formatter.downsample_points:
                perm = torch.randperm(N, device=self.device)[:self.formatter.downsample_points]
                pcd = pcd[:, :, perm]
            nobs_pcd = self.policy.normalizer[key].normalize(pcd.unsqueeze(1))
            pcd_obs[key] = nobs_pcd[:, 0]
        with torch.no_grad():
            return self.policy.obs_encoder.encode_pcd_only(pcd_obs).detach()

    def _build_ppo_obs(self, obs: dict) -> dict:
        """Build actor_* / critic_* obs dict from raw IsaacLab obs."""
        policy = obs["policy"]
        result = {
            "actor_pcd_emb":           self._last_pcd_embedding,
            "actor_agent_pos_history": self._get_ppo_agent_pos_history(),
        }
        if self._use_action_history and self._ppo_past_action_buf is not None:
            result["actor_past_actions_history"] = self._get_ppo_past_actions()
        for k, v in policy.items():
            if k in self._SKIP_PPO_KEYS:
                continue
            if k == "ee_pose":
                v = v[..., :3]
            result[f"critic_{k}"] = v
        return result

    def flatten_critic_obs(self, ppo_obs: dict) -> torch.Tensor:
        """Flatten critic_* keys into (B, critic_obs_dim)."""
        parts = []
        for k in self.critic_obs_keys:
            v = ppo_obs[k]
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).to(self.device)
            parts.append(v.float().reshape(self.num_envs, -1))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _predict_action_with_info(self) -> Tuple[torch.Tensor, FpoStepData]:
        """
        Run CFM inference (no grad) + compute initial CFM loss for FPO.

        Returns:
            action  (B, A) — first step of denoised chunk, unnormalized
            fpo_data       — FpoStepData (all tensors detached)
        """
        diffusion_obs = self.formatter.format(self.last_obs["policy"])

        # predict_action already has @torch.no_grad()
        result    = self.policy.predict_action(diffusion_obs)
        global_cond = result["global_cond"].detach()  # (B, G)
        chunk       = result["action_pred"].detach()  # (B, H, A) unnormalized

        with torch.no_grad():
            chunk_norm = self.policy.normalizer["action"].normalize(chunk)  # (B, H, A)

            B, H, A = chunk_norm.shape
            N = self.n_cfm_samples
            eps    = torch.randn(B, N, H, A, device=self.device)     # (B, N, H, A)
            t_samp = torch.rand(B, N, device=self.device)            # (B, N)

            # Loop over N samples to keep UNet batch size at B (avoid OOM).
            all_losses = []
            for n in range(N):
                t_bc = t_samp[:, n, None, None]
                x_t  = (1 - t_bc) * eps[:, n] + t_bc * chunk_norm
                pred = self.policy.model(x_t, t_samp[:, n], local_cond=None, global_cond=global_cond)
                all_losses.append(F.mse_loss(pred, chunk_norm - eps[:, n], reduction="none").mean(dim=[-2, -1]))
            initial_cfm_loss = torch.stack(all_losses, dim=1)  # (B, N)

        fpo_data = FpoStepData(
            global_cond=global_cond,
            chunk_norm=chunk_norm.detach(),
            eps=eps.detach(),
            t_samp=t_samp.detach(),
            initial_cfm_loss=initial_cfm_loss.detach(),
        )
        return chunk[:, 0].clone(), fpo_data  # execute first step of chunk

    def step(self) -> Tuple[dict, torch.Tensor, torch.Tensor, list]:
        """
        One FPO step: predict action, step env, update obs history.

        Returns: (ppo_obs, reward, done, info)
          ppo_obs: dict with actor_* and critic_* keys
          reward:  (B,) float32
          done:    (B,) bool
        Also sets self.last_fpo_data for the trainer to read.
        """
        action, fpo_data = self._predict_action_with_info()
        self.last_action   = action
        self.last_fpo_data = fpo_data

        if self._collect_substep_frames:
            frame = self.env.render()
            self.last_substep_frames = [frame] if frame is not None else []
        else:
            self.last_substep_frames = None

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        done = terminated | truncated

        # Update formatter history.
        if done.any():
            self.formatter.reset_envs(done, self.last_obs["policy"])
            masked = action.clone(); masked[done] = 0.0
            self.formatter.update_action(masked)
            self._reset_ppo_history_for_envs(done, self.last_obs["policy"])
        else:
            self.formatter.update_action(action)

        self._update_ppo_history(self.last_obs["policy"])
        if self._use_action_history:
            ppo_act = action.clone()
            if done.any():
                ppo_act[done] = 0.0
            self._update_ppo_past_actions(ppo_act)

        # PCD embedding for next obs: reuse from this step's global_cond.
        self._last_pcd_embedding = fpo_data.global_cond[:, :self.policy.pcd_feat_dim].detach()

        # Episode reward logging.
        self._ep_rewards += reward
        if done.any():
            if not isinstance(info, list):
                info = [{} for _ in range(self.num_envs)]
            for i in range(self.num_envs):
                if done[i]:
                    info[i]["episode"] = {"r": self._ep_rewards[i].item(), "l": 0}
                    self._ep_rewards[i] = 0.0

        return self._build_ppo_obs(self.last_obs), reward, done, info

    def _do_warmup(self):
        if self.num_warmup_steps <= 0:
            return
        warmup_act = self.env.unwrapped.cfg.warmup_action(self.env.unwrapped)
        for _ in range(self.num_warmup_steps):
            obs, _, _, _, _ = self.env.step(warmup_act)
            self.last_obs = obs

    def reset(self, **kwargs) -> Tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self._ep_rewards.zero_()
        self.formatter.reset()
        self._ppo_history_buf     = None
        self._ppo_past_action_buf = None
        self.last_fpo_data        = None

        self._do_warmup()

        # Seed history from post-warmup obs.
        self._last_pcd_embedding = self._compute_pcd_embedding()
        self._update_ppo_history(self.last_obs["policy"])
        if self._use_action_history and self._ppo_n_obs_steps > 1:
            n_past = self._ppo_n_obs_steps - 1
            self._ppo_past_action_buf = deque(
                [torch.zeros(self.num_envs, self.cfm_action_dim, device=self.device)
                 for _ in range(n_past)],
                maxlen=n_past,
            )

        return self._build_ppo_obs(self.last_obs), info

    def reset_to_spawn(self, spawn_pose) -> Tuple[dict, dict]:
        """Reset all envs and place grasp_object at a fixed spawn offset."""
        obs, info = self.reset()

        isaac_env = self.env.unwrapped
        obj       = isaac_env.scene["grasp_object"]
        defaults  = isaac_env.cfg.object_spawn_defaults

        default_pos = torch.tensor(defaults["default_pos"], dtype=torch.float32, device=self.device)
        default_rot = torch.tensor(defaults["default_rot"], dtype=torch.float32, device=self.device)

        new_pos    = default_pos.clone()
        new_pos[0] += spawn_pose.x
        new_pos[1] += spawn_pose.y

        root_state            = obj.data.default_root_state.clone()
        root_state[:, :3]     = new_pos.unsqueeze(0) + isaac_env.scene.env_origins
        root_state[:, 3:7]    = default_rot.unsqueeze(0)
        root_state[:, 7:]     = 0.0
        obj.write_root_state_to_sim(root_state)
        isaac_env.sim.step()

        raw_obs       = {"policy": isaac_env.observation_manager.compute()["policy"]}
        self.last_obs = raw_obs

        # Rebuild history from post-spawn obs.
        self._ppo_history_buf     = None
        self._ppo_past_action_buf = None
        self._last_pcd_embedding  = self._compute_pcd_embedding()
        self._update_ppo_history(self.last_obs["policy"])
        if self._use_action_history and self._ppo_n_obs_steps > 1:
            n_past = self._ppo_n_obs_steps - 1
            self._ppo_past_action_buf = deque(
                [torch.zeros(self.num_envs, self.cfm_action_dim, device=self.device)
                 for _ in range(n_past)],
                maxlen=n_past,
            )

        return self._build_ppo_obs(self.last_obs), info

    # ------------------------------------------------------------------
    # FPO update-time CFM loss (gradients flow through policy.model only)
    # ------------------------------------------------------------------

    def compute_cfm_loss(
        self,
        global_cond: torch.Tensor,  # (B, G) detached — from rollout
        chunk_norm:  torch.Tensor,  # (B, H, A) detached
        eps:         torch.Tensor,  # (B, N, H, A) detached
        t_samp:      torch.Tensor,  # (B, N) detached
    ) -> torch.Tensor:
        """
        Recompute CFM loss with current UNet weights.
        Only policy.model receives gradients (global_cond is .detach()-ed).
        Returns per-sample loss (B, N).
        """
        B, N, H, A = eps.shape
        # Loop over N samples to keep UNet batch size at B (avoid OOM).
        losses = []
        for n in range(N):
            t_bc = t_samp[:, n, None, None]
            x_t  = (1 - t_bc) * eps[:, n] + t_bc * chunk_norm
            pred = self.policy.model(x_t, t_samp[:, n], local_cond=None, global_cond=global_cond.detach())
            losses.append(F.mse_loss(pred, chunk_norm - eps[:, n], reduction="none").mean(dim=[-2, -1]))
        return torch.stack(losses, dim=1)  # (B, N)

    def compute_cfm_loss_frozen(
        self,
        global_cond: torch.Tensor,  # (B, G) — from real data pool, already no-grad
        chunk_norm:  torch.Tensor,  # (B, H, A)
        eps:         torch.Tensor,  # (B, N, H, A)
        t_samp:      torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        """
        Recompute CFM loss with frozen BC UNet weights. No gradients flow.
        Returns per-sample loss (B, N).
        """
        B, N, H, A = eps.shape
        losses = []
        with torch.no_grad():
            for n in range(N):
                t_bc = t_samp[:, n, None, None]
                x_t  = (1 - t_bc) * eps[:, n] + t_bc * chunk_norm
                pred = self.frozen_model(x_t, t_samp[:, n], local_cond=None, global_cond=global_cond)
                losses.append(F.mse_loss(pred, chunk_norm - eps[:, n], reduction="none").mean(dim=[-2, -1]))
        return torch.stack(losses, dim=1)  # (B, N)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
