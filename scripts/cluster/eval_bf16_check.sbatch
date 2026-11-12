#!/bin/bash
# Validate bf16 CFM autocast: cube RFS eval, fp32 vs bf16 (same seed/spawns). Submit: scripts/cluster/submit.sh scripts/cluster/eval_bf16_check.sbatch
#SBATCH --job-name=eval_bf16_check
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
for V in fp32 bf16; do
  FLAG=""; [ $V = bf16 ] && FLAG="--cfm_autocast"
  echo "=== EVAL $V ==="
  bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/cube_grasp_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml $FLAG --output_dir logs/eval/cube_$V --headless --num_envs 64 record_video=false"
done
