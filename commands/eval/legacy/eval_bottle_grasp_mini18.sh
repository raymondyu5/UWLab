#!/bin/bash
#SBATCH --job-name=uwlab_eval_bottle_grasp
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/uwlab_eval_bottle_grasp_%j.out
#SBATCH --error=logs/uwlab_eval_bottle_grasp_%j.err

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"


/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/eval/play_bc_legacy.py \
    --checkpoint logs/real/mini_18/cfm/pcd_cfm/horizon_4_nobs_1 \
    --task UW-FrankaLeap-GraspBottle-IkRel-v0 \
    --obs_keys joint_pos \
    --num_envs 4 \
    --num_episodes 20 \
    --action_horizon 1 \
    --record_video --enable_cameras --headless
'

echo "End: $(date)"
