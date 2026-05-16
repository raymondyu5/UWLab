#!/bin/bash
#SBATCH --job-name=uwlab_train_cfm_cotrain_0312
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_cfm_cotrain_0312_%j.out
#SBATCH --error=logs/train_cfm_cotrain_0312_%j.err

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
    dataset.data_path=/gscratch/weirdlab/raymond/UWLab/logs/distill_collection/pour_bottle_0316 \
    +dataset.num_demo=1150 \
    dataset.real_data_path=/gscratch/weirdlab/raymond/03_12/bourbon \
    dataset.sim_ratio=0.95 \
    training.use_wandb=true \
    training.wandb_run_name=cfm_pcd_cotrain_0312 \
    hydra.run.dir=/workspace/uwlab/logs/bc_cfm_pcd_cotrain_0312
'

echo "End: $(date)"
