#!/bin/bash
#SBATCH --job-name=uwlab_eval_rfs
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
RFS=logs/rfs/GraspCube-JointAbs_0509_0950_e7765f/model_000150.zip
bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/cube_grasp_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml --rfs_checkpoint $RFS --output_dir logs/eval/cube_rfs --headless --enable_cameras --num_envs 64"
