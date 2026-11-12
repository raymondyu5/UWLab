#!/bin/bash
# Validate fast_substeps: cube RFS eval with --fast_substeps (compare to fp32 baseline 225/320 = 0.703125, same seed/spawns).
# Submit: scripts/cluster/submit.sh scripts/cluster/eval_fast_check.sbatch
#SBATCH --job-name=eval_fast_check
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/cube_grasp_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml --fast_substeps --output_dir logs/eval/cube_fast --headless --num_envs 64 record_video=false"
