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
#SBATCH --output=logs/train_cfm_%j.out
#SBATCH --error=logs/train_cfm_%j.err

mkdir -p logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

export EXTRA_APPTAINER_BINDS="--bind /gscratch/weirdlab/will:/gscratch/weirdlab/will"

/mmfs1/gscratch/weirdlab/raymond/commands/uwlab/run_in_container.sh '
  ./uwlab.sh -p scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \
    --config-name train_cfm_pcd_deltajoint_h16_hist4 \
    dataset.data_path=/gscratch/weirdlab/will/UWLab_Docker/data_storage/datasets/03_24_bourbon_pour \
    dataset.hand_dropout_prob=0.1 \
    dataset.noise_extrinsic=true \
    training.use_wandb=true \
    training.wandb_run_name=cfm_pcd_bourbon_0324_deltajoint_h16_hist4_dropout_extnoise \
    hydra.run.dir=/workspace/uwlab/logs/bc_cfm_pcd_bourbon_0324_deltajoint_h16_hist4_dropout_extnoise
'

echo "End: $(date)"
