#!/bin/bash
# Usage: ./train_fpo_bottle_grasp.sh [gpu_type] [version]
# BC: BC_COEF=0 to disable, BC_COEF=<val> to override default (0.1)
export TASK_SLUG="bottle_grasp"
export TASK_ID="UW-FrankaLeap-GraspBottleRandomResets-JointAbs-v0"

# pre 5/13
#export BC_CKPT="logs/final_bc_checkpoints/bc_cfm_pcd_bottle_grasp_mixed_0412_absjoint_h16_hist4_noextnoise_fast"
# export BC_DATASETS="/gscratch/weirdlab/raymond/04_03/bourbon_random_resets /gscratch/weirdlab/raymond/04_01/bourbon_grasp"


export BC_CKPT="/gscratch/weirdlab/raymond/UWLab/logs/bc_cfm_pcd_bourbon_0512_absjoint_h16_hist4_extnoise_fast"
export BC_DATASETS="/gscratch/weirdlab/raymond/05_12/bourbon"
export NOISE_EXTRINSIC=false
source /gscratch/weirdlab/will/UWLab/commands/rl/fpo/_fpo_runner.sh "$@"
