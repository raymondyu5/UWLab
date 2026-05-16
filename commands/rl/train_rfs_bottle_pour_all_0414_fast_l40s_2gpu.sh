#!/bin/bash
#SBATCH --job-name=uwlab_rfs_pour_all_0414_l40s_2gpu
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/uwlab_rfs_pour_all_0414_fast_l40s_2gpu_%j.out
#SBATCH --error=logs/uwlab_rfs_pour_all_0414_fast_l40s_2gpu_%j.err

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

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
  --bind /gscratch/weirdlab/raymond/reset_poses_bottle_pour_ep1.json:/workspace/uwlab/assets/reset_poses_bottle_pour.json \
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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

    ./uwlab.sh -p -m torch.distributed.run \
      --nnodes=1 \
      --nproc_per_node=2 \
      --master_port $MASTER_PORT \
      scripts/reinforcement_learning/sb3/rfs/train.py \
      --task UW-FrankaLeap-PourBottleRandomResets-JointAbs-v0 \
      --num_envs 4096 \
      --diffusion_path logs/bc_cfm_pcd_bottle_pour_all_0414_absjoint_h16_hist4_noextnoise_fast \
      --cfg configs/rl/arm_rfs_joint_cfg.yaml \
      --eval_spawn random_1_trial \
      --wandb_project rfs_uwlab \
      --asymmetric_ac \
      --enable_cameras \
      --headless \
      --distributed
  '

echo "End: $(date)"
