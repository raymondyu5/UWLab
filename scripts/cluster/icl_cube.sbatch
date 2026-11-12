#!/bin/bash
# ICL student on the cube grasp task. Usage: scripts/cluster/submit.sh scripts/cluster/icl_cube.sbatch -- ARGS   (e.g. --refs_only --debug_checks)
# Extra args after the sbatch file are passed to sbatch, so set ARGS via the ICL_ARGS env var instead:  ICL_ARGS="--exp_name x" submit.sh ...
#SBATCH --job-name=uwlab_icl_cube
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --export=ALL
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/imitation_learning/icl/train_icl_cube.py --headless --enable_cameras $ICL_ARGS"
