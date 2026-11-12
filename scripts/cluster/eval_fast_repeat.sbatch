#!/bin/bash
# Settle fast_substeps equivalence on credit card: paired base/fast evals over seeds 1-3 plus a repeated baseline (seed 1)
# to measure baseline run-to-run nondeterminism. Each eval = 64 envs x random_5_trials = 320 episodes.
# Submit: scripts/cluster/submit.sh scripts/cluster/eval_fast_repeat.sbatch
#SBATCH --job-name=eval_fast_repeat
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
run() {  # name seed flag
  echo "=== EVAL $1 seed=$2 flag='$3' ==="
  bash scripts/cluster/run_in_container.sh "./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/credit_card_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml $3 --seed $2 --output_dir logs/eval/repeat_$1_s$2 --headless --num_envs 64 record_video=false"
}
for S in 1 2 3; do run base $S ""; run fast $S "--fast_substeps"; run skipobs $S "--skip_intermediate_obs"; done
run base_rerun 1 ""
