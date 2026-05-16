#!/bin/bash
#SBATCH --job-name=test_l40s
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/test_l40s_%j.out
#SBATCH --error=logs/test_l40s_%j.err

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"
nvidia-smi
echo "Done"
