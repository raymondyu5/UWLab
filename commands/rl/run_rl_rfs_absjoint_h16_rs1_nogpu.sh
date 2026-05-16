#!/bin/bash
#SBATCH --job-name=rfs_rs1_nogpu
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rfs_rs1_nogpu_%j.out
#SBATCH --error=logs/rfs_rs1_nogpu_%j.err

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

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
      --diffusion_path logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise \
      --cfg configs/rl/rfs_cfg_absjoint_h16_rs1_nogpu.yaml \
      --eval_spawn random_1_trial \
      --wandb_project rfs_uwlab \
      --wandb_run_name rfs_rs1_nogpu \
      --asymmetric_ac \
      --enable_cameras \
      --headless
  '

echo "End time: $(date)"
