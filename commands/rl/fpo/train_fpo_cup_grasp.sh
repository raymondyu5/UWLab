#!/bin/bash
# Usage: ./train_fpo_cup_grasp.sh [gpu_type] [version]
# BC: BC_COEF=0 to disable, BC_COEF=<val> to override default (0.1)
export TASK_SLUG="cup_grasp"
export TASK_ID="UW-FrankaLeap-GraspPinkCupRandomResets-JointAbs-v0"

# pre 5/13
#export BC_CKPT="logs/final_bc_checkpoints/bc_cfm_pcd_cup_grasp_canonical_0414_absjoint_h16_hist4_noextnoise_fast"

export BC_CKPT="/gscratch/weirdlab/raymond/UWLab/logs/bc_cfm_pcd_cup_pick_0512_absjoint_h16_hist4_extnoise_fast"

# pre 5/13
# export BC_DATASETS="/gscratch/weirdlab/raymond/cup_pick /gscratch/weirdlab/raymond/04_14/cup_grasp_retry"
export BC_DATASETS="/gscratch/weirdlab/raymond/05_12/cup_pick"

export NOISE_EXTRINSIC=false
source /gscratch/weirdlab/will/UWLab/commands/rl/fpo/_fpo_runner.sh "$@"
