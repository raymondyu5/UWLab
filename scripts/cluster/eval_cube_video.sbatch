#!/bin/bash
#SBATCH --job-name=uwlab_eval_cube
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/eval/play_bc.py --eval_cfg configs/eval/cube_grasp_bc_joint_abs.yaml --output_dir logs/eval/cube_grasp_video --headless --enable_cameras"
