# ICL cube grasp student, RFS expert (2026-10-29)

Job 42574528 (gpu-l40, 1h02m), `logs/icl/icl_cube_rfs/` (policy_final.pt, metrics.jsonl, results.json, icl_curve.png). Implementation: docs/claude/implementations/icl-cube.md.

Setup: `train_icl_cube.py --rfs_checkpoint logs/rfs/GraspCube-JointAbs_0509_0950_e7765f/model_000150.zip --k 2 --feedback init_cube --num_envs 128 --num_iterations 30`. Expert = RFS (PPO noise+residual on base CFM `bc_cfm_pcd_cube_base70_retry40_...`) queried every step. Hidden task = cube xy (+-0.2 m). Student sees only joint pos, prev action, t/T.

Expert references (128 cubes, success_once): true cube 0.77, told default cube 0.27.

Final eval (iteration 29, success_once by context size h=0/1/2): 0.078 / 0.656 / 0.633. Stable over iterations 5-29 at ~0.60-0.70 for h>=1 (one dip to 0.40 at it 23); h=0 stays ~0.08.

Controls at the end (h=0/1/2): shuf_y 0.086/0.055/0.055, no_y 0.086/0.000/0.000, no_ctx 0.078/0.070/0.055.

Reading: the student learns in context. Blind at h=0 (~8%, below the expert-told-default 27%), reaches ~65% (near the 77% true-cube expert) after one episode with feedback, h=2 gives no further gain. Shuffled/no feedback destroys it, so the gain comes from the feedback token (initial cube - final gripper), not from leakage via trajectories. Only 1 seed, 128 eval tasks (+-4% noise); iteration-23 dip not investigated.

Next: other feedback types (final_cube, success), k>2, more seeds.
