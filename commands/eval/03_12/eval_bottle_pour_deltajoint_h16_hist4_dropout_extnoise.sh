#!/bin/bash
#SBATCH --job-name=uwlab_eval_pour
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/uwlab_eval_pour_dropout_extnoise_%j.out
#SBATCH --error=logs/uwlab_eval_pour_dropout_extnoise_%j.err

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"


/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/eval/play_bc.py \
    --eval_cfg configs/eval/bottle_pour_bc_deltajoint_random.yaml \
    --sim_type distill \
    checkpoint=logs/bc_cfm_pcd_bourbon_0312_deltajoint_h16_hist4_dropout_extnoise \
    action_horizon=1 \
    num_envs=4 \
    record_video=true \
    --enable_cameras --headless
'

echo "End: $(date)"
