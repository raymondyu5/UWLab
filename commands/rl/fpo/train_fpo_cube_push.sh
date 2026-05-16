#!/bin/bash
# Usage: ./train_fpo_cube_push.sh [gpu_type] [version]
# BC: BC_COEF=0 to disable, BC_COEF=<val> to override default (0.1)
export TASK_SLUG="cube_push"
export TASK_ID="UW-FrankaLeap-PushCubeToPoptartRandomResets-JointAbs-v0"
export BC_CKPT="logs/final_bc_checkpoints/bc_cfm_pcd_cube_push_all_0416_absjoint_h16_hist4_noextnoise_fast"
export BC_DATASETS="/gscratch/weirdlab/raymond/04_08/cube_push_random_resets /gscratch/weirdlab/raymond/04_03/cube_push /gscratch/weirdlab/raymond/04_14/cube_push_retry /gscratch/weirdlab/raymond/04_16/push_cube"
export NOISE_EXTRINSIC=false
source /gscratch/weirdlab/will/UWLab/commands/rl/fpo/_fpo_runner.sh "$@"
