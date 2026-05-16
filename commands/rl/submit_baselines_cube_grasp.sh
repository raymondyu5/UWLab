#!/bin/bash
cd /gscratch/weirdlab/raymond/commands/uwlab/rl
sbatch train_dsrl_cube_grasp_fast_l40s.sh
sbatch train_residual_rl_cube_grasp_fast_l40s.sh
