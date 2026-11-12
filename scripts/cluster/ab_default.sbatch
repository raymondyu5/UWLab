#!/bin/bash
# Same-node A/B of PPO step time: baseline (--no_fast_substeps) vs the default (fast_substeps, fp32), interleaved (profile_interval 20, 300 s each).
# Submit: scripts/cluster/submit.sh scripts/cluster/ab_speed.sbatch
#SBATCH --job-name=ab_default
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
# fast_substeps is default on; baseline needs --no_fast_substeps
for FLAGS in "--no_fast_substeps" "" "--no_fast_substeps" ""; do
  echo "=== AB flags='$FLAGS' ==="
  bash scripts/cluster/run_in_container.sh "PYTHONUNBUFFERED=1 timeout 600 ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py --task UW-FrankaLeap-GraspCube-JointAbs-v0 --num_envs 512 --diffusion_path logs/base/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast --cfg configs/rl/arm_rfs_joint_cfg.yaml $FLAGS --asymmetric_ac --profile_interval 20 --no_eval_at_start --no_eval_video --eval_interval 100000000 --headless"
done
