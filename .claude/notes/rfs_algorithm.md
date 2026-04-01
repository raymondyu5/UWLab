# RFS Algorithm (Residual Flow-Matching RL)

## High-Level Concept

RFS = **Noise-Space RL + Residual** on top of a frozen diffusion (CFM) base policy.

The idea: instead of learning a policy from scratch, we take a pre-trained Conditional Flow Matching (CFM) behavior cloning policy and use PPO to:
1. **Steer the noise** fed into the CFM denoiser (DSRL-style noise-space RL)
2. **Add a residual** on top of the denoised action for specific joint dims

The CFM policy acts as a strong prior. PPO only needs to learn small corrections.

## PPO Action Space Layout

```
PPO action = [residual (n_residual_dims * residual_step) | noise (n_noise_dims * horizon)]
```

For the command's config (`chunk_4_arm_rfs_joint_cfg.yaml`):
- `noise_dims = "0:23"` → all 23 dims (7 arm + 16 hand) get noise steering
- `residual_dims = "0:23"` → all 23 dims get residual
- `residual_step = 4` → 4 env substeps per PPO step
- `policy horizon = 16` (from BC model `h16`)

So total PPO action dim = `23*4 residual + 23*16 noise = 92 + 368 = 460`.

## Per-Step Execution (inside `RFSWrapper.step()`)

For each PPO step:
1. PPO outputs `ppo_action = [residual_flat | noise_flat]`
2. Noise is reshaped to `(B, horizon, noise_dims)` and injected as the CFM starting trajectory
3. CFM denoises the noise using RK2 ODE integration (5 steps) conditioned on current obs
4. This gives `base_actions` of shape `(B, horizon, action_dim)`
5. For each of `residual_step` substeps:
   - Take `base_action = base_actions[:, substep, :]`
   - Add `residual[:, substep, :]` (scaled by `residual_scale=0.01`) to `residual_dims`
   - Optionally apply low-pass filter to finger dims (`finger_smooth_alpha`)
   - Send composite action to Isaac Lab env
   - Accumulate reward (with discount = `gamma^substep`)

## CFM Policy (BCObsFormatter + CFMPCDPolicy)

The base policy is a **Conditional Flow Matching** policy:
- **Obs encoder**: PointNet on segmented point cloud (seg_pc, 2048 pts) + low-dim joint state
- **Backbone**: ConditionalUnet1D flow field, conditioned on obs embedding
- **Inference**: RK2 integration with `num_inference_steps=5` from noise → action trajectory
- **Output**: action trajectory of shape `(B, horizon, action_dim)`

### BCObsFormatter
Manages observation history for the CFM model:
- Maintains a ring buffer of `n_obs_steps` frames
- Formats raw Isaac Lab obs dict into BC training format
- Normalizes obs/actions to match training distribution

## Asymmetric Actor-Critic (`--asymmetric_ac`)

Used in the training command. Two separate networks:

**Actor** (deployed to real world):
- Sees non-privileged info (matches BC training obs):
  - `actor_pcd_emb`: PointNet embedding of current seg_pc
  - `actor_agent_pos_history`: flattened n_obs_steps joint-pos history
  - `actor_past_actions_history`: flattened (n_obs_steps-1) past action history

**Critic** (sim-only, never deployed):
- Sees privileged sim state (`critic_*` prefix for all non-PCD obs):
  - `critic_joint_pos`, `critic_arm_joint_pos`, `critic_hand_joint_pos`
  - `critic_ee_pose` (3D position only)
  - `critic_cup_pose`, `critic_manipulated_object_pose`, `critic_target_object_pose`

The trick: `CombinedExtractor.forward` only processes its registered keys, so actor/critic obs are naturally split.

## Why This Works

- CFM prior provides motion quality and safety from demonstrations
- Noise-space steering lets PPO change the "intent" without breaking motion smoothness
- Small residual on top handles fine-grained corrections
- Asymmetric AC: critic can see ground-truth object poses (sim-privileged) while actor only uses observable info → better value estimates during training
