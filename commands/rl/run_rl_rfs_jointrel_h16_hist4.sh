#!/bin/bash
#SBATCH --job-name=dsrl_jointrel_h16
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rfs_jointrel_h16_%j.out
#SBATCH --error=logs/rfs_jointrel_h16_%j.err

# RFS fine-tuning on the h16_hist4 deltajoint BC policy.
# Base policy: horizon=16, n_obs_steps=4, 23D JointRel actions (7D delta arm + 16D abs hand).
# PPO noise space: 23 * 16 = 368 dims. Residual on hand dims [7:23].

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab
export EXTRA_APPTAINER_BINDS="--bind /gscratch/weirdlab/raymond/.tmp/${SLURM_JOB_ID}/isaaclab:/tmp/isaaclab"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$SCRIPT_DIR/../run_in_container.sh" '
  ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py \
    --task UW-FrankaLeap-PourBottle-JointRel-v0 \
    --num_envs 1024 \
    --diffusion_path logs/bc_cfm_pcd_bourbon_0312_deltajoint_h16_hist4 \
    --cfg configs/rl/dsrl_cfg_jointrel_h16.yaml \
    --eval_spawn bottle_pour_narrow \
    --wandb_project rfs_uwlab \
    --enable_cameras \
    --headless
'

echo "End: $(date)"
