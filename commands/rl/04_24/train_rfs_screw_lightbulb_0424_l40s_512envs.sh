#!/bin/bash
#SBATCH --job-name=uwlab_rfs_screw_0424_512
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/raymond/logs/rfs_screw_0424_512_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/raymond/logs/rfs_screw_0424_512_%j.err

mkdir -p /mmfs1/gscratch/weirdlab/raymond/logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab

export UWLAB_BASE=/gscratch/weirdlab/raymond/uwlab_docker

apptainer exec --nv \
  --bind /gscratch/weirdlab/raymond/UWLab/source:/workspace/uwlab/source \
  --bind /gscratch/weirdlab/raymond/UWLab/scripts:/workspace/uwlab/scripts \
  --bind /gscratch/weirdlab/raymond/UWLab/assets:/workspace/uwlab/assets \
  --bind /gscratch/weirdlab/raymond/UWLab/configs:/workspace/uwlab/configs \
  --bind /gscratch/weirdlab/raymond/UWLab/third_party:/workspace/uwlab/third_party \
  --bind /gscratch/weirdlab/raymond/UWLab/logs:/workspace/uwlab/logs \
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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py \
      --task UW-FrankaLeap-ScrewLightbulb-JointAbs-v0 \
      --num_envs 64 \
      --diffusion_path logs/bc_cfm_pcd_screw_lightbulb_0421_absjoint_h16_hist4_noextnoise_fast \
      --cfg configs/rl/arm_rfs_joint_cfg.yaml \
      --eval_spawn random_1_trial \
      --wandb_project rfs_uwlab \
      --asymmetric_ac \
      --enable_cameras \
      --headless
  '

echo "End: $(date)"
