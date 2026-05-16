#!/bin/bash
#SBATCH --job-name=collect_distill_pour_0327
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/collect_distill_pour_0327_%j.out
#SBATCH --error=logs/collect_distill_pour_0327_%j.err

# Collect distillation trajectories for PourBottle-JointAbs using the 0326 asymmetric DSRL policy.
#
# Mirrors RFSWrapper conditions from arm_rfs_joint_cfg.yaml exactly:
#   n_residual=23, residual_scale=0.01 (residual_dims="0:23")
#   finger_smooth_alpha=1.0       (disabled)
#   num_warmup_steps=5
#
# Diffusion ckpt: bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise (from will's path)
# PPO ckpt:       PourBottle-JointAbs_0326_2312_28e64e/model_000600.zip (from will's path)

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab
export EXTRA_APPTAINER_BINDS="--bind /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab:/tmp/isaaclab \
  --bind /gscratch/weirdlab/will:/gscratch/weirdlab/will"

/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/collect_distill_trajectories.py \
    --task UW-FrankaLeap-PourBottle-JointAbs-v0 \
    --diffusion_checkpoint /gscratch/weirdlab/will/UWLab_Docker/logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise \
    --ppo_checkpoint /gscratch/weirdlab/will/UWLab_Docker/logs/rfs/PourBottle-JointAbs_0326_2312_28e64e/model_000600.zip \
    --asymmetric_ac \
    --n_residual 23 \
    --residual_scale 0.01 \
    --finger_smooth_alpha 1.0 \
    --num_warmup_steps 5 \
    --horizon 112 \
    --num_episodes 2000 \
    --output_dir logs/distill_collection/pour_bottle_0327 \
    --headless
'

echo "End: $(date)"
