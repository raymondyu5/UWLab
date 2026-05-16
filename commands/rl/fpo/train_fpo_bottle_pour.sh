#!/bin/bash
# Usage: ./train_fpo_bottle_pour.sh [gpu_type] [version]
# BC: BC_COEF=0 to disable, BC_COEF=<val> to override default (0.1)
export TASK_SLUG="bottle_pour"
export TASK_ID="UW-FrankaLeap-PourBottleRandomResets-JointAbs-v0"
export BC_CKPT="logs/final_bc_checkpoints/bc_cfm_pcd_bottle_pour_all_0414_absjoint_h16_hist4_noextnoise_fast"
export BC_DATASETS="/gscratch/weirdlab/raymond/04_06/bottle_pour /gscratch/weirdlab/will/UWLab_Docker/data_storage/datasets/03_24_bourbon_pour /gscratch/weirdlab/raymond/04_12/bottle_pour_retry /gscratch/weirdlab/raymond/04_14/bottle_pour_retry2"
export NOISE_EXTRINSIC=false
source /gscratch/weirdlab/will/UWLab/commands/rl/fpo/_fpo_runner.sh "$@"
