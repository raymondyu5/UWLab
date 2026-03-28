"""
Asymmetric Actor-Critic policy for SB3 PPO.

Actor sees only 'actor_*' obs keys (abs ee_pose + hand_joint_pos — real-world deployable).
Critic sees only 'critic_*' obs keys (privileged sim state: object pose, contacts, etc.).
No gradient flow between the two sets.

The trick: CombinedExtractor.forward only iterates its registered keys, silently
ignoring any extras. So pi_features_extractor (registered with actor_* keys) ignores
critic_* keys from the full obs dict, and vice versa — with no extra overrides needed.
"""

import os
from typing import Dict, List, Optional, Type, Union

import gymnasium as gym
import torch.nn as nn

from stable_baselines3.common.policies import MultiInputActorCriticPolicy as MultiInputPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule

ACTOR_PREFIX = "actor_"
CRITIC_PREFIX = "critic_"


def infer_asymmetric_ac_from_ppo_zip(path: str) -> bool:
    """True if the saved SB3 model's observation space uses actor_/critic_ keys (asymmetric AC)."""
    from stable_baselines3.common.save_util import load_from_zip_file

    path = os.path.expanduser(path)
    data, _, _ = load_from_zip_file(path, load_data=True)
    if not data:
        return False
    obs = data.get("observation_space")
    if obs is None or not hasattr(obs, "spaces"):
        return False
    return any(str(k).startswith(ACTOR_PREFIX) for k in obs.spaces)


def resolve_asymmetric_ac(cli_asymmetric: bool, rfs_cfg: dict, ckpt_path: Optional[str]) -> bool:
    """Prefer explicit ``--asymmetric_ac``; else infer from PPO .zip; else ``rfs_cfg['asymmetric_ac']``."""
    if cli_asymmetric:
        return True
    expanded = os.path.expanduser(ckpt_path) if ckpt_path else ""
    if expanded and os.path.isfile(expanded):
        use = infer_asymmetric_ac_from_ppo_zip(expanded)
        if use:
            print(
                "[RFS] Inferred asymmetric_ac=True from PPO checkpoint observation_space "
                f"({ACTOR_PREFIX}*/{CRITIC_PREFIX}* keys). RFSWrapper will match training."
            )
        return use
    return bool(rfs_cfg.get("asymmetric_ac", False))


class AsymmetricMlpExtractor(nn.Module):
    """MlpExtractor with separate input dims for actor and critic networks."""

    def __init__(
        self,
        pi_feature_dim: int,
        vf_feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
    ):
        super().__init__()
        if isinstance(net_arch, dict):
            pi_layers = net_arch.get("pi", [])
            vf_layers = net_arch.get("vf", [])
        else:
            pi_layers = vf_layers = net_arch

        policy_net: list[nn.Module] = []
        in_dim = pi_feature_dim
        for out_dim in pi_layers:
            policy_net += [nn.Linear(in_dim, out_dim), activation_fn()]
            in_dim = out_dim
        self.policy_net = nn.Sequential(*policy_net)
        self.latent_dim_pi = in_dim

        value_net: list[nn.Module] = []
        in_dim = vf_feature_dim
        for out_dim in vf_layers:
            value_net += [nn.Linear(in_dim, out_dim), activation_fn()]
            in_dim = out_dim
        self.value_net = nn.Sequential(*value_net)
        self.latent_dim_vf = in_dim

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class AsymmetricActorCriticPolicy(MultiInputPolicy):
    """
    Asymmetric actor-critic for SB3 PPO.

    Expects the obs dict to have keys prefixed with 'actor_' and 'critic_'.
    The actor only processes 'actor_*' keys; the critic only processes 'critic_*' keys.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        **kwargs,
    ):
        actor_keys = sorted(k for k in observation_space.spaces if k.startswith(ACTOR_PREFIX))
        critic_keys = sorted(k for k in observation_space.spaces if k.startswith(CRITIC_PREFIX))
        assert actor_keys, f"No '{ACTOR_PREFIX}*' keys found in observation space."
        assert critic_keys, f"No '{CRITIC_PREFIX}*' keys found in observation space."

        actor_space = gym.spaces.Dict({k: observation_space[k] for k in actor_keys})
        critic_space = gym.spaces.Dict({k: observation_space[k] for k in critic_keys})

        # Pre-compute feature dims so _build_mlp_extractor (called inside super().__init__)
        # uses the correct asymmetric dims rather than the full obs space dims.
        self._pi_feat_dim = CombinedExtractor(actor_space).features_dim
        self._vf_feat_dim = CombinedExtractor(critic_space).features_dim
        self._actor_space = actor_space
        self._critic_space = critic_space

        kwargs["share_features_extractor"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Replace the SB3-built extractors (which use the full obs space) with
        # filtered ones. mlp_extractor is already correct from our _build_mlp_extractor
        # override called during super().__init__.
        self.pi_features_extractor = CombinedExtractor(actor_space)
        self.vf_features_extractor = CombinedExtractor(critic_space)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            pi_feature_dim=self._pi_feat_dim,
            vf_feature_dim=self._vf_feat_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )
