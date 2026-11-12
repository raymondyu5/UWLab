#!/bin/bash
#SBATCH --job-name=uwlab_smoke
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
nvidia-smi -L
bash scripts/cluster/run_in_container.sh "nvidia-smi -L; ./uwlab.sh -p -c 'import isaaclab, uwlab; print(isaaclab.__file__, uwlab.__file__)'"
