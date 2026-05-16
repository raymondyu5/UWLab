#!/bin/bash
#SBATCH --job-name=uwlab_cotrain_0329
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/raymond/logs/cotrain_bourbon_absjoint_h16_hist4_extnoise_0329_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/raymond/logs/cotrain_bourbon_absjoint_h16_hist4_extnoise_0329_%j.err

# Cotrain distill (sim, pour_bottle_0327) + real (03_24_bourbon_pour). With extrinsic noise.
# Sim action_key=actions (23D combined), real action_key=[arm_joint_pos_target,hand_action] (7+16=23D).

mkdir -p /mmfs1/gscratch/weirdlab/raymond/logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

export EXTRA_APPTAINER_BINDS="--bind /gscratch/weirdlab/will:/gscratch/weirdlab/will"

/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
    --config-name train_cfm_pcd_absjoint_h16_hist4 \
    dataset.data_path=/workspace/uwlab/logs/distill_collection/pour_bottle_0327 \
    dataset.action_key=actions \
    dataset.real_data_path=/gscratch/weirdlab/will/UWLab_Docker/data_storage/datasets/03_24_bourbon_pour \
    dataset.real_action_key=[arm_joint_pos_target,hand_action] \
    dataset.sim_ratio=0.95 \
    +dataset.noise_extrinsic=true \
    training.use_wandb=true \
    training.wandb_run_name=cotrain_bourbon_0329_absjoint_h16_hist4_extnoise \
    hydra.run.dir=/workspace/uwlab/logs/cotrain_bourbon_0329_absjoint_h16_hist4_extnoise
'

echo "End: $(date)"
