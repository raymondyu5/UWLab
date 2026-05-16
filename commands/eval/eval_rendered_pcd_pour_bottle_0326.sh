#!/bin/bash
#SBATCH --job-name=eval_rendered_pcd_0326
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_rendered_pcd_0326_%j.out
#SBATCH --error=logs/eval_rendered_pcd_0326_%j.err

# Eval the 0326 PPO checkpoint with rendered PCD (sim-to-real gap proxy).
# Mirrors collect_distill_pour_bottle_0327.sh conditions exactly, except:
#   --actor_pcd_key rendered  (both diffusion base and actor see camera-rendered seg_pc)
#   --num_episodes 100        (eval only, not full data collection)

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab
export EXTRA_APPTAINER_BINDS="--bind /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab:/tmp/isaaclab \
  --bind /gscratch/weirdlab/will:/gscratch/weirdlab/will"

/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/collect_distill_trajectories.py \
    --task UW-FrankaLeap-PourBottle-JointAbs-v0 \
    --diffusion_checkpoint /gscratch/weirdlab/will/UWLab_Docker/logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise \
    --ppo_checkpoint /gscratch/weirdlab/will/UWLab_Docker/logs/rfs/PourBottle-JointAbs_0326_2312_28e64e/model_000600.zip \
    --asymmetric_ac \
    --n_residual 23 \
    --finger_smooth_alpha 1.0 \
    --num_warmup_steps 5 \
    --horizon 112 \
    --num_episodes 100 \
    --actor_pcd_key rendered \
    --output_dir /tmp/eval_rendered_pcd_0326 \
    --headless
'

echo "End: $(date)"
