#!/bin/bash
# Run inside the container from /workspace/uwlab

./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/play_rfs.py \
    --task UW-FrankaLeap-PourBottle-IkRel-v0 \
    --diffusion_path logs/bc_cfm_pcd_bourbon_0312 \
    --checkpoint logs/rfs/PourBottle_0314_1609/model_000300.zip \
    --cfg configs/rl/dsrl_cfg.yaml \
    --asymmetric_ac \
    --eval_spawn bottle_pour_narrow \
    --num_envs 1024 \
    --record_video \
    --enable_cameras \
    --headless
