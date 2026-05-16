#!/bin/bash
#SBATCH --job-name=uwlab_train_cfm
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_cfm_%j.out
#SBATCH --error=logs/train_cfm_%j.err

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
    --config-name train_cfm_pcd_deltaee_h16_hist4 \
    dataset.data_path=/mmfs1/gscratch/weirdlab/raymond/03_12/bourbon \
    training.use_wandb=true \
    training.wandb_run_name=cfm_pcd_bourbon_0312_deltaee_h16_hist4 \
    hydra.run.dir=/workspace/uwlab/logs/bc_cfm_pcd_bourbon_0312_deltaee_h16_hist4
'

echo "End: $(date)"
