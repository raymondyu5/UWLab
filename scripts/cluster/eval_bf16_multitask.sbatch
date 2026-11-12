#!/bin/bash
# bf16 CFM quality vs fp32 (compare with equiv_<task>_fast results of eval_equiv_multitask), seed 1, 64 envs x random_5_trials.
# Submit: scripts/cluster/submit.sh scripts/cluster/eval_bf16_multi_multitask.sbatch
#SBATCH --job-name=eval_bf16_multi
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
for NAME in credit_card bottle_random_resets plate_in_dishrack soccer_push; do
  for V in bf16; do
    FLAG="--cfm_autocast"
    echo "=== EVAL $NAME $V ==="
    timeout 20m bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/${NAME}_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml $FLAG --seed 1 --output_dir logs/eval/bf16_${NAME}_$V --headless --num_envs 64 record_video=false"
  done
done
