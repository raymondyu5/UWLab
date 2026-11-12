#!/bin/bash
# Speed/quality of a coarser physics step at the same control rate: 30 Hz x decimation 3 (default 60 Hz x 6). Experiment only.
# Eval compares with equiv_<task>_fast (default dynamics). Profile compares PPO-step time with the default A/B numbers.
#SBATCH --job-name=physx_dt
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
export_env="UWLAB_GRASP_PHYSICS_HZ=30 UWLAB_GRASP_DECIMATION=3"
for NAME in credit_card bottle_random_resets plate_in_dishrack; do
  echo "=== EVAL $NAME hz30dec3 ==="
  timeout 20m bash scripts/cluster/run_in_container.sh "$export_env ./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/${NAME}_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml --seed 1 --output_dir logs/eval/hz30dec3_${NAME} --headless --num_envs 64 record_video=false"
done
echo "=== PROFILE hz30dec3 ==="
PYTHONUNBUFFERED=1 bash scripts/cluster/run_in_container.sh "$export_env timeout 500 ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py --task UW-FrankaLeap-GraspCube-JointAbs-v0 --num_envs 512 --diffusion_path logs/base/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast --cfg configs/rl/arm_rfs_joint_cfg.yaml --asymmetric_ac --profile_interval 20 --no_eval_at_start --no_eval_video --eval_interval 100000000 --headless"
