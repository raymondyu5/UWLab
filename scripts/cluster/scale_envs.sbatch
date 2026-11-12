#!/bin/bash
# Training throughput vs num_envs (python per-step overhead is constant, so more envs amortize it).
# Reads PROFILE blocks; envs/sec = num_envs * 4 / total_step_ms (residual_step=4 env steps per PPO step).
# Submit: scripts/cluster/submit.sh scripts/cluster/scale_envs.sbatch
#SBATCH --job-name=scale_envs
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
for N in 512 1024 2048 4096; do
  echo "=== SCALE num_envs=$N ==="
  bash scripts/cluster/run_in_container.sh "timeout 480 ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py --task UW-FrankaLeap-GraspCube-JointAbs-v0 --num_envs $N --diffusion_path logs/base/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast --cfg configs/rl/arm_rfs_joint_cfg.yaml --fast_substeps --cfm_autocast --skip_intermediate_obs --asymmetric_ac --profile_interval 20 --no_eval_at_start --no_eval_video --eval_interval 100000000 --headless"
done
