#!/bin/bash
#SBATCH --job-name=rl_dsrl_v1
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rl_dsrl_v1_%j.out
#SBATCH --error=logs/rl_dsrl_v1_%j.err

# DSRL training: noise-space RL only, no hand residual.
# Algorithm params live in configs/rl/dsrl_cfg.yaml.

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab
export EXTRA_APPTAINER_BINDS="--bind /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab:/tmp/isaaclab"

/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py \
    --task UW-FrankaLeap-PourBottle-IkRel-v0 \
    --num_envs 1024 \
    --diffusion_path logs/bc_cfm_pcd_bourbon_0312 \
    --cfg configs/rl/dsrl_cfg.yaml \
    --eval_spawn random \
    --wandb_project rfs_uwlab \
    --enable_cameras \
    --headless
'

echo "End: $(date)"
