from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder


def _rk2(model, num_steps: int, trajectory: torch.Tensor,
         local_cond, global_cond) -> torch.Tensor:
    for t in range(num_steps):
        t0 = t / num_steps
        t1 = (t + 1) / num_steps
        dt = t1 - t0

        ts = torch.full((trajectory.shape[0],), t0, device=trajectory.device, dtype=trajectory.dtype)
        tm = torch.full((trajectory.shape[0],), t0 + dt / 2, device=trajectory.device, dtype=trajectory.dtype)

        v0 = model(trajectory, ts, local_cond=local_cond, global_cond=global_cond)
        x_mid = trajectory + (dt / 2) * v0
        v_mid = model(x_mid, tm, local_cond=local_cond, global_cond=global_cond)
        trajectory = trajectory + dt * v_mid

    return trajectory


class CFMPCDPolicy(BaseImagePolicy):
    """
    Conditional Flow Matching policy with point cloud + low-dim observations.

    Obs encoder (MultiPCDObsEncoder) produces a flat feature vector used as
    global conditioning for a ConditionalUnet1D flow field.

    Inputs to forward/compute_loss:
        batch["obs"]["agent_pos"]: (B, 1, D_proprio)
        batch["obs"]["seg_pc"]:    (B, 1, 3, N)   (or other image_keys)
        batch["action"]:           (B, H, A)

    Outputs of predict_action:
        {"action": (B, n_action_steps, A), "action_pred": (B, H, A)}
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiPCDObsEncoder,
        noise_scheduler: ConditionalFlowMatcher,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 5,
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        use_action_history: bool = True,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        assert horizon >= 2, f"horizon must be >= 2, got {horizon}"
        down_dims = list(down_dims)[: horizon // 2 + 1]
        assert len(down_dims) >= 1, f"down_dims truncated to empty for horizon={horizon}"

        obs_feature_dim = obs_encoder.output_shape()[0]

        # Low-dim and pcd features are encoded with different history depths:
        #   pcd  -> 1 frame (current only)
        #   low_dim -> n_obs_steps frames concatenated flat
        low_dim_dim = sum(
            obs_encoder.key_shape_map[k][-1] for k in obs_encoder.low_dim_keys
        )
        pcd_feat_dim = obs_feature_dim - low_dim_dim
        n_past_actions = (n_obs_steps - 1) if use_action_history else 0
        global_cond_dim = pcd_feat_dim + n_obs_steps * low_dim_dim + n_past_actions * action_dim

        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.use_action_history = use_action_history

    def _encode_obs(self, nobs: dict, batch_size: int) -> torch.Tensor:
        # PCD: encode current frame only. nobs[pcd_key] is (B, 1, 3, N); index 0 = current.
        pcd_obs = {k: nobs[k][:, 0] for k in self.obs_encoder.pcd_keys}
        pcd_feat = self.obs_encoder.encode_pcd_only(pcd_obs)   # (B, pcd_feat_dim)

        # Low-dim: flatten all n_obs_steps frames into one vector (oldest first).
        # nobs[low_dim_key] is (B, n_obs_steps, D); reshape -> (B, n_obs_steps*D).
        low_dim_parts = [
            nobs[k].reshape(batch_size, -1)
            for k in self.obs_encoder.low_dim_keys
        ]
        low_dim_flat = torch.cat(low_dim_parts, dim=-1) if low_dim_parts else pcd_feat.new_zeros(batch_size, 0)

        # Past actions: (B, n_obs_steps-1, A) -> (B, (n_obs_steps-1)*A). Only present when n_obs_steps > 1.
        if "past_actions" in nobs:
            past_act_flat = nobs["past_actions"].reshape(batch_size, -1)
            return torch.cat([pcd_feat, low_dim_flat, past_act_flat], dim=-1)

        return torch.cat([pcd_feat, low_dim_flat], dim=-1)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: dict) -> torch.Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        B = nactions.shape[0]
        global_cond = self._encode_obs(nobs, B)

        device = nactions.device
        noise = torch.randn_like(nactions)
        t, x_t, ut = self.noise_scheduler.sample_location_and_conditional_flow(noise, nactions)
        t, x_t, ut = t.to(device), x_t.to(device), ut.to(device)

        pred = self.model(x_t, t.reshape(-1), local_cond=None, global_cond=global_cond)
        return F.mse_loss(pred, ut)

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], noise=None) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)

        B = next(iter(nobs.values())).shape[0]
        global_cond = self._encode_obs(nobs, B)

        if noise is not None:
            trajectory = noise.to(device=self.device, dtype=self.dtype)
        else:
            trajectory = torch.randn(
                (B, self.horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )

        trajectory = _rk2(
            self.model,
            self.num_inference_steps,
            trajectory,
            local_cond=None,
            global_cond=global_cond,
        )

        action_pred = self.normalizer["action"].unnormalize(trajectory)
        action = action_pred[:, : self.n_action_steps]

        return {"action": action, "action_pred": action_pred}
