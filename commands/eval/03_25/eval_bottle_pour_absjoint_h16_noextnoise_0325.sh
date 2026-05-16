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
#SBATCH --output=/mmfs1/gscratch/weirdlab/raymond/logs/uwlab_eval_pour_absjoint_h16_noextnoise_0325_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/raymond/logs/uwlab_eval_pour_absjoint_h16_noextnoise_0325_%j.err

mkdir -p /mmfs1/gscratch/weirdlab/raymond/logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /tmp/$SLURM_JOB_ID
export EXTRA_APPTAINER_BINDS="--bind /tmp/$SLURM_JOB_ID:/tmp"

/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/eval/play_bc.py \
    --eval_cfg configs/eval/bottle_pour_bc_jointabs.yaml \
    --sim_type distill \
    checkpoint=logs/bc_cfm_pcd_bourbon_0325_absjoint_h16_noextnoise \
    checkpoint_name=best.ckpt \
    action_horizon=8 \
    num_envs=4 \
    record_video=true \
    --enable_cameras --headless
'

echo "End: $(date)"
