#!/bin/bash
#SBATCH --job-name=uwlab_train_cfm
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/raymond/logs/train_cfm_absjoint_h16_extnoise_dropout_0325_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/raymond/logs/train_cfm_absjoint_h16_extnoise_dropout_0325_%j.err

mkdir -p /mmfs1/gscratch/weirdlab/raymond/logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

export EXTRA_APPTAINER_BINDS="--bind /gscratch/weirdlab/will:/gscratch/weirdlab/will"

/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
    --config-name train_cfm_pcd_absjoint_h16 \
    dataset.data_path=/gscratch/weirdlab/will/UWLab_Docker/data_storage/datasets/03_24_bourbon_pour \
    +dataset.noise_extrinsic=true \
    +dataset.hand_dropout_prob=0.1 \
    training.use_wandb=true \
    training.wandb_run_name=cfm_pcd_bourbon_0325_absjoint_h16_extnoise_dropout \
    hydra.run.dir=/workspace/uwlab/logs/bc_cfm_pcd_bourbon_0325_absjoint_h16_extnoise_dropout
'

echo "End: $(date)"
