#!/bin/bash
#SBATCH --job-name=eval_rl_0326
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_rl_0326_%j.out
#SBATCH --error=logs/eval_rl_0326_%j.err

# Eval the 0326 PPO checkpoint with synthetic (mesh) PCD, rl_mode, multi-env.
# Fast — no cameras, 16 envs in parallel.

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab
export EXTRA_APPTAINER_BINDS="--bind /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab:/tmp/isaaclab \
  --bind /gscratch/weirdlab/will:/gscratch/weirdlab/will"

/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/eval_rl.py \
    --task UW-FrankaLeap-PourBottle-JointAbs-v0 \
    --num_envs 1024 \
    --diffusion_path /gscratch/weirdlab/will/UWLab_Docker/logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise \
    --checkpoint /gscratch/weirdlab/will/UWLab_Docker/logs/rfs/PourBottle-JointAbs_0326_2312_28e64e/model_000600.zip \
    --cfg configs/rl/arm_rfs_joint_cfg.yaml \
    --asymmetric_ac \
    --eval_spawn random_1_trial \
    --output_dir logs/eval/eval_rl_0326 \
    --no_eval_video \
    --headless
'

echo "End: $(date)"
