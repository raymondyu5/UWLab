#!/bin/bash
# Usage: ./train_fpo_cube_pinch.sh [gpu_type] [version]
# BC: BC_COEF=0 to disable, BC_COEF=<val> to override default (0.1)

# ./train_fpo_cube_pinch.sh --bc l40s v9

export TASK_SLUG="cube_pinch"
export TASK_ID="UW-FrankaLeap-GraspCube-JointAbs-v0"

# pre 5/13
#export BC_CKPT="logs/final_bc_checkpoints/bc_cfm_pcd_cube_grasp_normal_only_0412_absjoint_h16_hist4_noextnoise_fast"
# export BC_DATASETS="/gscratch/weirdlab/raymond/04_01/cube_pick"

export BC_CKPT="/gscratch/weirdlab/raymond/UWLab/logs/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast"
export BC_DATASETS="/gscratch/weirdlab/raymond/05_08/base_cube_70 /gscratch/weirdlab/raymond/05_08/base_cube_right /gscratch/weirdlab/raymond/05_08/retry_cube_40"

export NOISE_EXTRINSIC=false
source /gscratch/weirdlab/will/UWLab/commands/rl/fpo/_fpo_runner.sh "$@"
