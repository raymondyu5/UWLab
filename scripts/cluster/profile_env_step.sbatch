#!/bin/bash
# Per-phase env.step timing + behavior checksum, baseline then --fast_substeps.
# Submit: scripts/cluster/submit.sh scripts/cluster/profile_env_step.sbatch
#SBATCH --job-name=env_step_prof
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
for FLAG in "" "--fast_substeps"; do
  echo "=== PROFILE flag='$FLAG' ==="
  bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/profile/profile_env_step.py --task UW-FrankaLeap-GraspCube-JointAbs-v0 --num_envs 512 --headless $FLAG"
done
