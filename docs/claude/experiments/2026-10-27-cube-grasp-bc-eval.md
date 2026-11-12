# Cube grasp BC eval with video (2026-10-27)

- Checkpoint: `logs/final_bc_checkpoints/bc_cfm_pcd_cube_grasp_normal_only_0412_absjoint_h16_hist4_noextnoise_fast` (best.ckpt, epoch 410, val_mse 0.0114)
- Task `UW-FrankaLeap-GraspCube-JointAbs-v0`, eval mode, 1 env, 5 random-reset episodes, action_horizon 8.
- Run: `scripts/cluster/submit.sh scripts/cluster/eval_cube_video.sbatch` (job 42505697, gpu-l40). Config: `configs/eval/cube_grasp_bc_joint_abs.yaml`.
- Result: 2/5 success (episodes 1 and 3); 2/5 grasped and lifted. Videos + results.json: `logs/eval/cube_grasp_video/` (`*_fixed.mp4` = fixed camera, others = viewer).
- Needed `play_bc.py` patch (pour-only metrics/cup optional) and the overlay (torchcfm lives in overlay.img).
