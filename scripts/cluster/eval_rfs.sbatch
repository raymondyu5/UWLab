#!/bin/bash
# Usage: submit.sh scripts/cluster/eval_rfs.sbatch --job-name=eval_<name> --partition=... --gres=... --export=ALL,NAME=<name>
# Eval cfg configs/eval/${NAME}_rfs_joint_abs.yaml (its rfs_checkpoint is used).
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/${NAME}_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml --output_dir logs/eval/${NAME}_rfs --headless --enable_cameras --num_envs 64"
