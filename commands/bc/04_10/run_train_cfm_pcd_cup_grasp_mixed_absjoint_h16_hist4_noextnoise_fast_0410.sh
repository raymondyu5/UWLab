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
#SBATCH --output=/mmfs1/gscratch/weirdlab/raymond/logs/train_cfm_cup_grasp_mixed_absjoint_h16_hist4_noextnoise_fast_0410_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/raymond/logs/train_cfm_cup_grasp_mixed_absjoint_h16_hist4_noextnoise_fast_0410_%j.err

mkdir -p /mmfs1/gscratch/weirdlab/raymond/logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
    --config-name train_cfm_pcd_absjoint_h16_hist4_fast \
    "+dataset.data_paths=[/gscratch/weirdlab/raymond/04_08/cup_grasp_random_resets,/gscratch/weirdlab/raymond/04_04/grasp_cup]" \
    training.use_wandb=true \
    training.wandb_run_name=cfm_pcd_cup_grasp_mixed_0410_absjoint_h16_hist4_noextnoise_fast \
    hydra.run.dir=/workspace/uwlab/logs/bc_cfm_pcd_cup_grasp_mixed_0410_absjoint_h16_hist4_noextnoise_fast
'

echo "End: $(date)"
