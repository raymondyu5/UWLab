# Local RFS cube eval (RTX 5090), 2026-10-30

Sanity check that the local install reproduces klone results.

- Pulled from klone (`scripts/cluster/pull.sh` equivalent, rsync): `logs/rfs/GraspCube-JointAbs_0509_0950_e7765f` (no `eval/`, 131M) and `logs/base/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast` (1.1G).
- Run: `PYTHONPATH=$PWD ./uwlab.sh -p scripts/eval/play_rfs_wrapper.py --eval_cfg configs/eval/cube_grasp_rfs_joint_abs.yaml --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml --rfs_checkpoint logs/rfs/GraspCube-JointAbs_0509_0950_e7765f/model_000150.zip --output_dir logs/eval/local_cube_rfs --headless --enable_cameras --num_envs 16 record_video=false` (81 s).
- Result: success_rate 0.775 (62/80), success_ever 0.80, `random_5_trials` x 16 envs. Klone (64 envs, 320 episodes): 0.703 / 0.744. Difference is within sampling noise (SE ~0.05 at n=80).
- Outputs: `logs/eval/local_cube_rfs/eval/step_0000000000/results.json`, log `logs/eval/local_cube_rfs.log`.
