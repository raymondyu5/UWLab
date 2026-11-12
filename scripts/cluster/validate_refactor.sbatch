#!/bin/bash
# Smoke-validate the RFSWrapper.from_cfg refactor + speed options: (1) cube eval with fast cfg (expect 225/320),
# (2) short PPO training with fast+bf16 and step profile.
# Submit: scripts/cluster/submit.sh scripts/cluster/validate_refactor.sbatch
#SBATCH --job-name=validate_refactor
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
echo "=== EVAL cube fast (from_cfg) ==="
bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/cube_grasp_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml --fast_substeps --output_dir logs/eval/cube_fast_fromcfg --headless --num_envs 64 record_video=false"
echo "=== TRAIN fast+bf16 profile ==="
bash scripts/cluster/run_in_container.sh "timeout 420 ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py --task UW-FrankaLeap-GraspCube-JointAbs-v0 --num_envs 512 --diffusion_path logs/base/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast --cfg configs/rl/arm_rfs_joint_cfg.yaml --fast_substeps --cfm_autocast --asymmetric_ac --profile_interval 20 --no_eval_video --eval_interval 100000000 --headless"
