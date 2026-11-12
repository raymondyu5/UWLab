#!/bin/bash
# Short RFS PPO training run with step-timing profile (cube task). Submit: scripts/cluster/submit.sh scripts/cluster/rfs_profile_bf16.sbatch
#SBATCH --job-name=rfs_profile_bf16
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out

cd /mmfs1/gscratch/weirdlab/will/UWLab
bash scripts/cluster/run_in_container.sh "timeout 900 ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py --task UW-FrankaLeap-GraspCube-JointAbs-v0 --num_envs 512 --diffusion_path logs/base/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast --cfg configs/rl/arm_rfs_joint_cfg.yaml --cfm_autocast --asymmetric_ac --profile_interval 20 --no_eval_video --eval_interval 100000000 --headless"
