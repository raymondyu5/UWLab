#!/bin/bash
#SBATCH --job-name=uwlab_train_cfm
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/raymond/logs/train_cfm_plate_dishrack_full_0429_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/raymond/logs/train_cfm_plate_dishrack_full_0429_%j.err

mkdir -p /mmfs1/gscratch/weirdlab/raymond/logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
    --config-name train_cfm_pcd_absjoint_h16_hist4 \
    "+dataset.data_paths=[/gscratch/weirdlab/raymond/plate_dishrack,/gscratch/weirdlab/raymond/plate_dishrack_2]" \
    training.use_wandb=true \
    training.wandb_run_name=cfm_pcd_plate_dishrack_0429_absjoint_h16_hist4 \
    hydra.run.dir=/workspace/uwlab/logs/bc_cfm_pcd_plate_dishrack_0429_absjoint_h16_hist4
'

echo "End: $(date)"
