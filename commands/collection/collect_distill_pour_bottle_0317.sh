#!/bin/bash
#SBATCH --job-name=collect_distill_pour_0317
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/collect_distill_pour_0317_%j.out
#SBATCH --error=logs/collect_distill_pour_0317_%j.err

# Collect distillation trajectories for PourBottle using the asymmetric DSRL policy.
# Stores noise_mean (distribution mean) alongside noise (sample) for BC noise policy training.

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab
export EXTRA_APPTAINER_BINDS="--bind /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab:/tmp/isaaclab"

/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/collect_distill_trajectories.py \
    --task UW-FrankaLeap-PourBottle-IkRel-v0 \
    --diffusion_checkpoint logs/bc_cfm_pcd_bourbon_0312 \
    --ppo_checkpoint logs/rfs/PourBottle_0316_2133/model_000300.zip \
    --asymmetric_ac \
    --n_residual 0 \
    --horizon 180 \
    --finger_smooth_alpha 0.7 \
    --num_episodes 2000 \
    --num_warmup_steps 10 \
    --output_dir logs/distill_collection/pour_bottle_0318 \
    --headless
'

echo "End: $(date)"
