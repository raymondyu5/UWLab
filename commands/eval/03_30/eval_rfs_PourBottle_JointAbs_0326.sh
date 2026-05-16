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
#SBATCH --output=/mmfs1/gscratch/weirdlab/raymond/logs/uwlab_eval_rfs_PourBottle_JointAbs_0326_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/raymond/logs/uwlab_eval_rfs_PourBottle_JointAbs_0326_%j.err

mkdir -p /mmfs1/gscratch/weirdlab/raymond/logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /tmp/$SLURM_JOB_ID
export EXTRA_APPTAINER_BINDS="--bind /tmp/$SLURM_JOB_ID:/tmp \
  --bind /gscratch/weirdlab/will:/gscratch/weirdlab/will"

/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/play_rfs.py \
    --task UW-FrankaLeap-PourBottle-JointAbs-v0 \
    --diffusion_path /gscratch/weirdlab/will/UWLab_Docker/logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise \
    --checkpoint /gscratch/weirdlab/will/UWLab_Docker/logs/rfs/PourBottle-JointAbs_0326_2312_28e64e/model_000600.zip \
    --cfg configs/rl/arm_rfs_joint_cfg.yaml \
    --asymmetric_ac \
    --num_envs 1024 \
    --num_episodes 1 \
    --output_dir logs/eval/bottle_pour_bc_jointabs/rfs_PourBottle_JointAbs_0326 \
    --headless
'

echo "End: $(date)"
