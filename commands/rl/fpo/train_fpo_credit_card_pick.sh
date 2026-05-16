#!/bin/bash
# Usage: ./train_fpo_credit_card_pick.sh [gpu_type] [version]
# BC: BC_COEF=0 to disable, BC_COEF=<val> to override default (0.1)
export TASK_SLUG="credit_card_pick"
export TASK_ID="UW-FrankaLeap-GraspCreditCard-JointAbs-v0"
export BC_CKPT="logs/final_bc_checkpoints/bc_cfm_pcd_credit_card_0502_absjoint_h16_hist4_extnoise_fast"
export BC_DATASETS="/gscratch/weirdlab/raymond/05_02/credit_card /gscratch/weirdlab/raymond/05_02/credit_card_2"
export NOISE_EXTRINSIC=true
source /gscratch/weirdlab/will/UWLab/commands/rl/fpo/_fpo_runner.sh "$@"
