#!/bin/bash
# Usage: ./train_fpo_dishrack_place.sh [gpu_type] [version]
# BC: BC_COEF=0 to disable, BC_COEF=<val> to override default (0.1)
export TASK_SLUG="dishrack_place"
export TASK_ID="UW-FrankaLeap-PlateInDishRack-JointAbs-v0"
export BC_CKPT="logs/final_bc_checkpoints/bc_cfm_pcd_dishrack_plate_0501_absjoint_h16_hist4_extnoise_fast"
export BC_DATASETS="/gscratch/weirdlab/raymond/05_01/dishrack_plate"
export NOISE_EXTRINSIC=true
source /gscratch/weirdlab/will/UWLab/commands/rl/fpo/_fpo_runner.sh "$@"
