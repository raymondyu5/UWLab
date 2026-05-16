#!/bin/bash
#SBATCH --job-name=uwlab_pc_overlay_screw
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/raymond/logs/pc_overlay_screw_ep28_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/raymond/logs/pc_overlay_screw_ep28_%j.err

mkdir -p /mmfs1/gscratch/weirdlab/raymond/logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /tmp/$SLURM_JOB_ID
export EXTRA_APPTAINER_BINDS="--bind /tmp/$SLURM_JOB_ID:/tmp"

/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/perception/open_loop_pc_overlay.py \
    --task UW-FrankaLeap-ScrewLightbulb-JointAbs-v0 \
    --trajectory_file /gscratch/weirdlab/raymond/04_21/screw_lightbulb/episode_28 \
    --output_dir logs/sysid/open_loop_pc_overlay/screw_ep28 \
    --mode direct_joints \
    --action_type joint \
    --enable_cameras --headless
'

echo "End: $(date)"
