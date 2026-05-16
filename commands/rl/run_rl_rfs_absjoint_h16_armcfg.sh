#!/bin/bash
#SBATCH --job-name=rl_rfs_absjoint_h16_armcfg
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/rl_rfs_absjoint_h16_armcfg_%j.out
#SBATCH --error=logs/rl_rfs_absjoint_h16_armcfg_%j.err

# RFS RL training: JointAbs action space, horizon=16, n_obs_steps=4.
# CFM base policy: absjoint_h16_hist4 checkpoint.
# Uses arm_rfs_joint_cfg: full 23D residual, no KL regularization.
# asymmetric_ac enables ppo_history automatically.

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

mkdir -p /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab

export UWLAB_BASE=/gscratch/weirdlab/raymond/uwlab_docker

apptainer exec --nv \
  --bind /gscratch/weirdlab/will:/gscratch/weirdlab/will \
  --bind /gscratch/weirdlab/raymond/UWLab/source:/workspace/uwlab/source \
  --bind /gscratch/weirdlab/raymond/UWLab/scripts:/workspace/uwlab/scripts \
  --bind /gscratch/weirdlab/raymond/UWLab/assets:/workspace/uwlab/assets \
  --bind /gscratch/weirdlab/raymond/UWLab/configs:/workspace/uwlab/configs \
  --bind /gscratch/weirdlab/raymond/UWLab/third_party:/workspace/uwlab/third_party \
  --bind /gscratch/weirdlab/raymond/UWLab/logs:/workspace/uwlab/logs \
  --bind /gscratch/weirdlab/raymond/IsaacLab_assets/assets:/workspace/uwlab/assets/ycb \
  --bind /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab:/tmp/isaaclab \
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
      --task UW-FrankaLeap-PourBottle-JointAbs-v0 \
      --num_envs 1024 \
      --diffusion_path /gscratch/weirdlab/will/UWLab_Docker/logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise \
      --cfg configs/rl/arm_rfs_joint_cfg.yaml \
      --eval_spawn random_1_trial \
      --wandb_project rfs_uwlab \
      --asymmetric_ac \
      --enable_cameras \
      --headless
  '

echo "End time: $(date)"
