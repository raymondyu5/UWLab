#!/bin/bash
#SBATCH --job-name=rl_rfs_v1
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rl_rfs_v1_%j.out
#SBATCH --error=logs/rl_rfs_v1_%j.err

# RFS RL training with PPO + frozen CFM base policy.
# Algorithm: noise-space RL (PPO controls CFM starting noise) + hand residual.
#
# Algorithm params (noise_dims, residual_dims, residual_scale, PPO hyperparams, etc.)
# live in configs/rl/rfs_cfg.yaml — edit there, not here.
#
# CLI overrides available: --noise_dims, --residual_dims, --eval_interval, --eval_spawn

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

export UWLAB_BASE=/gscratch/weirdlab/raymond/uwlab_docker

apptainer exec --nv \
  --bind /gscratch/weirdlab/raymond/UWLab/source:/workspace/uwlab/source \
  --bind /gscratch/weirdlab/raymond/UWLab/scripts:/workspace/uwlab/scripts \
  --bind /gscratch/weirdlab/raymond/UWLab/assets:/workspace/uwlab/assets \
  --bind /gscratch/weirdlab/raymond/UWLab/configs:/workspace/uwlab/configs \
  --bind /gscratch/weirdlab/raymond/UWLab/third_party:/workspace/uwlab/third_party \
  --bind /gscratch/weirdlab/raymond/UWLab/logs:/workspace/uwlab/logs \
  --bind /gscratch/weirdlab/raymond/IsaacLab_assets/assets:/workspace/uwlab/assets/ycb \
  --bind $UWLAB_BASE/isaac-cache-kit:/isaac-sim/kit/cache \
  --bind $UWLAB_BASE/isaac-sim-data:/isaac-sim/kit/data \
  --bind $UWLAB_BASE/isaac-cache-ov:/root/.cache/ov \
  --bind $UWLAB_BASE/isaac-cache-pip:/root/.cache/pip \
  --bind $UWLAB_BASE/isaac-cache-gl:/root/.cache/nvidia/GLCache \
  --bind $UWLAB_BASE/isaac-cache-compute:/root/.nv/ComputeCache \
  --bind $UWLAB_BASE/outputs:/workspace/uwlab/outputs \
  --bind $UWLAB_BASE/data_storage:/workspace/uwlab/data_storage \
  --bind $UWLAB_BASE/isaac-cache-ov:/gscratch/weirdlab/raymond/.cache/ov \
  --pwd /workspace/uwlab \
  $UWLAB_BASE/uw-lab_latest.sif \
  /bin/bash -c '
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    cd /workspace/uwlab

    export WANDB_INSECURE_DISABLE_SSL=true

    ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py \
      --task UW-FrankaLeap-GraspPinkCup-IkRel-v0 \
      --num_envs 1024 \
      --diffusion_path logs/real/mini_18/cfm/pcd_cfm/horizon_4_nobs_1 \
      --wandb_project rfs_uwlab \
      --enable_cameras \
      --headless
  '

echo "End time: $(date)"
