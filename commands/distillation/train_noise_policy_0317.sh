#!/bin/bash
#SBATCH --job-name=train_noise_policy_0317
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_noise_policy_0317_%j.out
#SBATCH --error=logs/train_noise_policy_0317_%j.err

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/train_noise_policy.py \
    --config-path ../../configs/bc \
    --config-name train_noise_policy \
    diffusion_checkpoint=logs/bc_cfm_pcd_bourbon_0312 \
    ppo_checkpoint=logs/rfs/PourBottle_0315_1939/model_000400.zip \
    dataset.data_path=logs/distill_collection/pour_bottle_0316 \
    +dataset.max_episodes=900 \
    training.use_wandb=true \
    training.wandb_run_name=noise_policy_0317 \
    hydra.run.dir=/workspace/uwlab/logs/noise_policy_0317
'

echo "End: $(date)"
