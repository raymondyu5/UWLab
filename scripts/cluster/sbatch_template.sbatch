#!/bin/bash
# Template for UWLab experiments. #SBATCH lines can't expand variables, so the remote path is hardcoded.
# Submit: scripts/cluster/submit.sh <this file>. Prefer partition gpu-l40 (account weirdlab), then ckpt.
#SBATCH --job-name=uwlab_exp
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out

cd /mmfs1/gscratch/weirdlab/will/UWLab
bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py --task <TASK> --num_envs 4096 --headless"
