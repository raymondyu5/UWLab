#!/bin/bash
# Post-cleanup regression check: credit card seed 1 must reproduce 244/320 (284 ever); also a 2-minute training smoke run.
#SBATCH --job-name=validate_cleanup
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
echo "=== EVAL credit_card default ==="
bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/credit_card_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml --seed 1 --output_dir logs/eval/cleanup_credit_card --headless --num_envs 64 record_video=false"
echo "=== TRAIN smoke ==="
PYTHONUNBUFFERED=1 bash scripts/cluster/run_in_container.sh "timeout 300 ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py --task UW-FrankaLeap-GraspCube-JointAbs-v0 --num_envs 512 --diffusion_path logs/base/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast --cfg configs/rl/arm_rfs_joint_cfg.yaml --asymmetric_ac --profile_interval 20 --no_eval_at_start --no_eval_video --eval_interval 100000000 --headless"
