# Training/Eval/Visualization Setup

## Command Breakdown

```bash
scripts/reinforcement_learning/sb3/rfs/train.py \
  --task UW-FrankaLeap-PourBottle-JointAbs-v0 \
  --num_envs 1024 \
  --diffusion_path logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise \
  --cfg configs/rl/chunk_4_arm_rfs_joint_cfg.yaml \
  --eval_spawn random_1_trial \
  --wandb_project rfs_uwlab \
  --asymmetric_ac \
  --enable_cameras \
  --headless
```

| Argument | Value | Meaning |
|----------|-------|---------|
| `--task` | `UW-FrankaLeap-PourBottle-JointAbs-v0` | Isaac Lab env: pour task, absolute joint control |
| `--num_envs` | 1024 | Parallel environments for vectorized training |
| `--diffusion_path` | `logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise` | Pre-trained CFM BC policy checkpoint dir |
| `--cfg` | `configs/rl/chunk_4_arm_rfs_joint_cfg.yaml` | RFS algorithm config (PPO params + RFS params) |
| `--eval_spawn` | `random_1_trial` | Eval spawn config: 1 random trial per eval |
| `--wandb_project` | `rfs_uwlab` | W&B project name for logging |
| `--asymmetric_ac` | flag | Asymmetric actor-critic: actor=BC obs, critic=privileged sim state |
| `--enable_cameras` | flag | Needed for eval video recording |
| `--headless` | flag | No GUI (cluster mode) |

## Config (`chunk_4_arm_rfs_joint_cfg.yaml`)

```yaml
ppo:
  n_steps: 200         # Rollout steps per env before update
  batch_size: 4096     # Minibatch size for gradient updates
  n_epochs: 8          # PPO epochs per rollout
  gamma: 0.99          # Discount factor
  gae_lambda: 0.95     # GAE lambda
  clip_range: 0.2      # PPO clip
  learning_rate: 0.0003
  vf_coef: 0.0001      # Very small value function loss weight
  ent_coef: 0.0        # No entropy bonus
  target_kl: 6.0       # Large KL threshold (allows aggressive updates)
  net_arch: {pi: [256, 128, 64], vf: [256, 128, 64]}
  activation_fn: elu

rfs:
  noise_dims: "0:23"        # All 23 dims get noise steering
  residual_dims: "0:23"     # All 23 dims get residual
  residual_step: 4           # 4 env steps per PPO step (action chunk of 4)
  residual_scale: 0.01       # Residual output scaled by 0.01 before adding
  clip_actions: false        # Don't clip (BoundedAction already clamps arm)
  finger_smooth_alpha: 1.0   # No low-pass filter on fingers
  num_warmup_steps: 5        # 5 extra zero-action warmup steps after env reset

eval:
  interval: 50         # Eval every 50 PPO iterations
  spawn: "random_1_trial"  # Default spawn (overridden by CLI to "random_1_trial")
  record_video: true
```

**Key derived values**:
- PPO action dim = 23*4 + 23*16 = 460
- Effective eval interval = 50 // 4 = 12 PPO steps (due to residual_step scaling)
- Steps per update batch = 200 * 1024 = 204,800 env steps per rollout

## Training Loop

1. `gym.make()` creates `FrankaLeapGraspEnv` with `rl_mode` (cameras disabled)
2. `RFSWrapper` wraps it, loads frozen CFM checkpoint, sets up obs/action spaces
3. `GpuSb3VecEnvWrapper` makes it SB3-compatible (GPU tensors throughout)
4. `RegularizedPPO` (SB3 PPO subclass) trains for 200M timesteps

### Key Feature: `residual_step`
PPO acts once per `residual_step=4` env steps. This means:
- PPO "time" is 4x slower than env time
- Eval interval is divided by `residual_step` to account for this
- PPO n_steps=200 → 200*4 = 800 env steps per env per rollout

## Eval Callback (`RFSEvalCallback`)

Runs at `eval_interval` PPO steps:
- Loads spawn config: `random_1_trial.yaml` → `poses: []`, `num_trials: 1`
  - Empty poses = use env's random reset (default randomization)
  - 1 trial = 1 episode per spawn pose
- Runs deterministic rollouts across all envs with `policy.set_training_mode(False)`
- Records video (fixed camera) if `record_video=True`
- Logs to W&B: success rate, is_grasped rate, ep_rew_mean, scatter plots

## Callbacks During Training

1. `WandbRewardTermCallback`: logs individual reward term means to W&B each iteration
2. `WandbNoisePredCallback`: logs noise prediction stats (mean, std of PPO noise output)
3. `RFSEvalCallback`: periodic eval with spawn poses

## Checkpoint Loading (BC Policy)

The BC policy is loaded from:
```
logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise/
  checkpoints/best.ckpt   (preferred)
  best.ckpt               (fallback)
  config.yaml             (training config, read if cfg not embedded in ckpt)
```

The checkpoint name encodes:
- `cfm_pcd`: CFM policy with point cloud obs
- `bourbon`: bourbon bottle task
- `0324`: date (March 24)
- `absjoint_h16`: absolute joint action space, horizon=16
- `hist4`: n_obs_steps=4 (4-frame history)
- `extnoise`: extended noise perturbation during training

## Output Structure

```
logs/rfs/{task}_{timestamp}_{uuid}/
  command.txt          # Full training command
  params/env.yaml      # Serialized env config
  params/agent.yaml    # Serialized PPO config
  model.zip            # Final SB3 PPO checkpoint
  eval/                # Eval rollout videos and plots
  videos/train/        # Training videos (if --video)
```

## Wandb Logging

Project: `rfs_uwlab`
- Logs all PPO metrics (loss, clip fraction, explained variance, etc.)
- Per-reward-term means (via `WandbRewardTermCallback`)
- Noise statistics (via `WandbNoisePredCallback`)
- Eval success/grasp rates and videos at each eval

## Infrastructure

- Uses `GpuDictRolloutBuffer`: keeps rollout buffer on GPU (faster)
- Uses `GpuSb3VecEnvWrapper` instead of standard `Sb3VecEnvWrapper` (GPU tensors throughout)
- Isaac Sim `AppLauncher` must be initialized before any Isaac imports
- `signal.SIGINT` handler closes tqdm progress bars cleanly

## File Structure (RFS-specific)

```
scripts/reinforcement_learning/sb3/rfs/
  train.py              # Main training script
  wrapper.py            # RFSWrapper: CFM + noise-space + residual
  asymmetric_policy.py  # AsymmetricActorCriticPolicy for SB3
  eval_callback.py      # RFSEvalCallback
  regularized_ppo.py    # PPO subclass with optional real-data KL regularization
  buffers.py            # GpuDictRolloutBuffer
  gpu_vec_env.py        # GpuSb3VecEnvWrapper
  callbacks.py          # WandbNoisePredCallback, WandbRewardTermCallback
  real_dataset.py       # RealDatasetLoader for optional sim2real regularization
  play_rfs.py           # Playback script for trained RFS policies
```
