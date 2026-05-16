#!/bin/bash
# RFS fine-tunes a BC policy. Run BC training first:
#   sbatch commands/uwlab/bc/run_train_cfm_pcd_bourbon_0311.sh
#SBATCH --job-name=uwlab_rfs_bottle_pour
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/uwlab_rfs_bottle_pour_%j.out
#SBATCH --error=logs/uwlab_rfs_bottle_pour_%j.err

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

# /tmp inside the container fills up during long RL runs; redirect to scratch.
mkdir -p /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab
export EXTRA_APPTAINER_BINDS="--bind /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab:/tmp/isaaclab"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$SCRIPT_DIR/../run_in_container.sh" '
  ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py \
    --task UW-FrankaLeap-PourBottle-IkRel-v0 \
    --num_envs 1024 \
    --diffusion_path logs/bc_cfm_pcd_bourbon_0312 \
    --eval_spawn random \
    --wandb_project rfs_uwlab \
    --enable_cameras \
    --headless
'

echo "End: $(date)"
