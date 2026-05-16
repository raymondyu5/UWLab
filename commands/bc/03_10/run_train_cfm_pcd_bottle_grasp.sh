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
#SBATCH --output=logs/train_cfm_bottle_grasp_%j.out
#SBATCH --error=logs/train_cfm_bottle_grasp_%j.err

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"


/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
    dataset.data_path=/gscratch/weirdlab/raymond/03_10_bottle_grasp/bourbon \
    training.use_wandb=true \
    training.wandb_run_name=cfm_pcd_bottle_grasp \
    hydra.run.dir=/workspace/uwlab/logs/bc_cfm_pcd_grasp
'

echo "End: $(date)"
