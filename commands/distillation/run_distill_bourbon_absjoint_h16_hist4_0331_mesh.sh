#!/bin/bash
#SBATCH --job-name=uwlab_distill_0331_mesh
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/raymond/logs/distill_bourbon_absjoint_h16_hist4_0331_mesh_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/raymond/logs/distill_bourbon_absjoint_h16_hist4_0331_mesh_%j.err

# Distill-only training on multi-env mesh PCD data (pour_bottle_0327_multienv).
# seg_pc stores synthetic mesh PCD (collected via RL_MODE, 1024 envs).

mkdir -p /mmfs1/gscratch/weirdlab/raymond/logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
    --config-name train_cfm_pcd_absjoint_h16_hist4 \
    dataset.data_path=/workspace/uwlab/logs/distill_collection/pour_bottle_0327_multienv \
    dataset.action_key=actions \
    training.use_wandb=true \
    training.wandb_run_name=distill_bourbon_0331_absjoint_h16_hist4_mesh \
    hydra.run.dir=/workspace/uwlab/logs/distill_bourbon_0331_absjoint_h16_hist4_mesh
'

echo "End: $(date)"
