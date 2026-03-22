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

import copy
import os
import sys
from types import SimpleNamespace

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
import gymnasium as gym
import numpy as np
import torch

import torch.nn as nn
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher


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


# TODO: remove _OBS_KEY_MAP once UWLab obs keys are aligned with checkpoint format.
# Maps checkpoint obs_key names → (env_obs_dict_key, slice).
# "joint_pos" in the UWLab env is 23D (arm7 + hand16 concatenated).
_OBS_KEY_MAP = {
    "joint_positions":    ("joint_pos", slice(0, 7)),    # arm
    "gripper_position":   ("joint_pos", slice(7, 23)),   # hand
    "cartesian_position": ("ee_pose",   slice(None)),
    "arm_joint_pos":      ("joint_pos", slice(0, 7)),
    "hand_joint_pos":     ("joint_pos", slice(7, 23)),
    "ee_pose":            ("ee_pose",   slice(None)),
    "joint_pos":          ("joint_pos", slice(None)),    # 23D combined
}


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
        downsample_points = int(train_cfg["dataset"]["downsample_points"])
        shape_meta = {
            "action": {"shape": [action_dim]},
            "obs": {
                "agent_pos": {"shape": [agent_pos_dim], "type": "low_dim"},
                "seg_pc": {"shape": [3, downsample_points], "type": "pcd"},
            },
        }
        pol = train_cfg["policy"]
        ds = train_cfg["dataset"]
        sigma = float(pol.get("sigma", 0.0))
        horizon = int(train_cfg.get("horizon", 4))
        n_action_steps = int(train_cfg.get("n_action_steps", 8))
        n_obs_steps = int(train_cfg.get("n_obs_steps", 1))
        down_dims = tuple(pol["down_dims"])
        obs_keys = ds.get("obs_keys", ds.get("obs_key", []))
        cfg = SimpleNamespace(
            dataset=SimpleNamespace(obs_keys=obs_keys),
        )

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
    )
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()

    print(f"[RFSWrapper] CFM policy loaded: horizon={policy.horizon}, "
          f"action_dim={policy.action_dim}, n_obs_steps={policy.n_obs_steps}")
    return policy, cfg


def _parse_obs_key(obs_key: str, obs_dict: dict) -> torch.Tensor:
    if obs_key not in _OBS_KEY_MAP:
        raise KeyError(
            f"Unknown obs_key '{obs_key}'. Add it to _OBS_KEY_MAP in rfs/wrapper.py."
        )
    env_key, slc = _OBS_KEY_MAP[obs_key]
    if env_key not in obs_dict:
        raise KeyError(
            f"obs_key '{obs_key}' maps to env key '{env_key}' which is not in obs_dict. "
            f"Available keys: {list(obs_dict.keys())}"
        )
    return obs_dict[env_key][:, slc] if slc != slice(None) else obs_dict[env_key]


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
        num_warmup_steps: int = 0,
        asymmetric_ac: bool = False,
    ):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.residual_step = residual_step
        self.clip_actions = clip_actions
        self.residual_scale = residual_scale

        self.asymmetric_ac = asymmetric_ac
        self.noise_dims = noise_dims
        self.residual_dims = residual_dims
        self.noise_slice = slice(*noise_dims) if noise_dims else None
        self.residual_slice = slice(*residual_dims) if residual_dims else None
        self.n_noise = (noise_dims[1] - noise_dims[0]) if noise_dims else 0
        self.n_residual = (residual_dims[1] - residual_dims[0]) if residual_dims else 0

        # Episode reward tracking for SB3 ep_rew_mean logging.
        self._ep_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Last composite action (base + residual) sent to env. Set in step().
        self.last_action: torch.Tensor | None = None
        self.last_noise_flat: torch.Tensor | None = None
        self.last_residual: torch.Tensor | None = None
        self.last_pcd_embedding: torch.Tensor | None = None

        self.policy, self.policy_cfg = _load_cfm_checkpoint(diffusion_path, self.device)
        self.horizon = self.policy.horizon
        self.cfm_action_dim = self.policy.action_dim
        dataset_cfg = self.policy_cfg.dataset
        obs_key_attr = "obs_keys" if hasattr(dataset_cfg, "obs_keys") else "obs_key"
        self.diffusion_obs_keys = list(getattr(dataset_cfg, obs_key_attr))

        print(f"[RFSWrapper] Diffusion obs keys from checkpoint: {self.diffusion_obs_keys}")
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

        if finger_smooth_alpha < 1.0:
            self.finger_filter = LPFilter(alpha=finger_smooth_alpha).to(self.device)
        else:
            self.finger_filter = None

        self.num_warmup_steps = num_warmup_steps
        self.last_obs = None

    _SKIP_PPO_KEYS = {"seg_pc", "rgb"}

    def _setup_asymmetric_obs_space(self, policy_obs_space: gym.spaces.Dict):
        """Replace the policy obs space with actor_*/critic_* keys for asymmetric AC.
        Actor: pcd_feat embedding (256D) + abs ee_pose (7D) + hand_joint_pos (16D).
        Critic: all current stripped keys (privileged sim state).
        """
        _SKIP = self._SKIP_PPO_KEYS
        actor_spaces = {
            "actor_pcd_emb": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.policy.pcd_feat_dim,),
                dtype=np.float32,
            ),
            "actor_ee_pose":        policy_obs_space.spaces["ee_pose"],
            "actor_hand_joint_pos": policy_obs_space.spaces["hand_joint_pos"],
        }
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

    def _strip_ppo_obs(self, obs: dict) -> dict:
        """Return obs stripped for PPO: remove seg_pc/rgb, strip ee_pose to 3D xyz.
        self.last_obs is NOT modified — CFM still reads the full obs including seg_pc."""
        if self.asymmetric_ac:
            return self._asymmetric_ppo_obs(obs)
        policy = obs["policy"]
        stripped = {k: v for k, v in policy.items() if k not in self._SKIP_PPO_KEYS}
        if "ee_pose" in stripped:
            stripped["ee_pose"] = stripped["ee_pose"][..., :3]
        return {"policy": stripped}

    def _compute_pcd_embedding(self) -> torch.Tensor:
        """Run the frozen CFM PointNet on self.last_obs and return pcd_feat."""
        diffusion_obs = self._get_diffusion_obs(self.last_obs["policy"])
        with torch.no_grad():
            nobs = self.policy.normalizer.normalize(diffusion_obs)
            pcd_obs = {k: nobs[k][:, 0] for k in self.policy.obs_encoder.pcd_keys}
            return self.policy.obs_encoder.encode_pcd_only(pcd_obs).detach()

    def _asymmetric_ppo_obs(self, obs: dict) -> dict:
        """Build actor_*/critic_* obs dict for asymmetric AC."""
        policy = obs["policy"]
        result = {
            "actor_pcd_emb":        self.last_pcd_embedding,
            "actor_ee_pose":        policy["ee_pose"],
            "actor_hand_joint_pos": policy["hand_joint_pos"],
        }
        for k, v in policy.items():
            if k in self._SKIP_PPO_KEYS:
                continue
            if k == "ee_pose":
                v = v[..., :3]
            result[f"critic_{k}"] = v
        return {"policy": result}

    def _get_diffusion_obs(self, obs_dict: dict) -> dict:
        parts = [_parse_obs_key(k, obs_dict) for k in self.diffusion_obs_keys]
        agent_pos = torch.cat(parts, dim=-1)  # (B, D)
        seg_pc = obs_dict["seg_pc"]           # (B, 3, N)
        return {
            "agent_pos": agent_pos.unsqueeze(1),  # (B, 1, D)
            "seg_pc": seg_pc.unsqueeze(1),         # (B, 1, 3, N)
        }

    def step(self, ppo_actions: torch.Tensor):
        ppo_out = ppo_actions.clamp(-1.0, 1.0)

        residual = ppo_out[:, :self.n_residual]
        noise_flat = ppo_out[:, self.n_residual:]

        self.last_noise_flat = noise_flat.detach()
        self.last_residual = residual.detach() if self.n_residual > 0 else None

        noise = torch.zeros(self.num_envs, self.horizon, self.cfm_action_dim, device=self.device)
        if self.noise_slice is not None and self.n_noise > 0:
            noise[:, :, self.noise_slice] = noise_flat.reshape(self.num_envs, self.horizon, self.n_noise)

        diffusion_obs = self._get_diffusion_obs(self.last_obs["policy"])
        with torch.no_grad():
            cfm_result = self.policy.predict_action(diffusion_obs, noise=noise)
        base_actions = cfm_result["action_pred"]

        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for substep in range(self.residual_step):
            action = base_actions[:, substep].clone()

            if self.n_residual > 0 and self.residual_slice is not None:
                action[:, self.residual_slice] += residual * self.residual_scale

            if self.clip_actions:
                action[:, :3].clamp_(-0.03, 0.03)
                action[:, 3:6].clamp_(-0.05, 0.05)

            if self.finger_filter is not None:
                action[:, 6:] = self.finger_filter(action[:, 6:])

            self.last_action = action.detach()
            obs, step_rewards, terminated, truncated, info = self.env.step(action)
            rewards += step_rewards
            self.last_obs = obs

            if self.env.unwrapped.episode_length_buf[0] == self.env.unwrapped.max_episode_length - 1:
                break

        # Update pcd embedding from the final obs so it is consistent with the
        # state returned to the agent — no lag between pcd_emb and state.
        if self.asymmetric_ac:
            self.last_pcd_embedding = self._compute_pcd_embedding()

        self._ep_rewards += rewards
        dones = terminated | truncated
        if dones.any():
            if not isinstance(info, list):
                info = [{} for _ in range(self.num_envs)]
            for i in range(self.num_envs):
                if dones[i]:
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

        if self.finger_filter is not None:
            self.finger_filter.reset()
            current_finger = self.env.unwrapped.scene["robot"].data.joint_pos[:, 7:]
            self.finger_filter(current_finger)

        self._do_warmup()
        if self.asymmetric_ac:
            self.last_pcd_embedding = self._compute_pcd_embedding()
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
        return self._strip_ppo_obs(self.last_obs), info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
