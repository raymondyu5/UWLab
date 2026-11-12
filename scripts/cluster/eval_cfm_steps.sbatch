#!/bin/bash
# Quality of fewer CFM RK2 steps (trained with 5): 3 and 2 steps, fp32, vs equiv_<task>_fast (5 steps). seed 1, 64 envs x 5 trials.
#SBATCH --job-name=eval_cfm_steps
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
for NAME in credit_card bottle_random_resets plate_in_dishrack soccer_push; do
  for K in 3 2; do
    echo "=== EVAL $NAME steps=$K ==="
    timeout 20m bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/${NAME}_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml --cfm_steps $K --seed 1 --output_dir logs/eval/steps${K}_${NAME} --headless --num_envs 64 record_video=false"
  done
done
