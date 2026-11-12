# ICL cube grasp student with a noisy point cloud, RFS expert (2026-10-30)

Training job 42582767 (gpu-l40, 1h43m), `logs/icl/icl_cube_rfs_vision/` (policy_final.pt, metrics.jsonl, results.json, icl_alpha_curves.png). Plots/videos: `logs/icl/icl_cube_rfs_vision_sweep/{alpha_sweep.json,alpha_sweep.png}`, `logs/icl/icl_cube_rfs_vision_viz/videos/` (9 mp4), `logs/icl/pcd_dump/pcd_alpha_grid.png`.

Setup: `train_icl_cube.py --rfs_checkpoint logs/rfs/GraspCube-JointAbs_0509_0950_e7765f/model_000150.zip --k 2 --feedback init_cube --vision --num_envs 128 --num_iterations 30` (same expert/DAgger/feedback as [2026-10-29-icl-cube-rfs.md](2026-10-29-icl-cube-rfs.md)). New: the student also gets the episode's initial point cloud (2048 pts, env frame, standardised with fixed global stats) as `(1-alpha)*pcd + alpha*N(0,I)`, alpha ~ U(0,1) drawn independently per episode and not told to the student. A small PointNet (shared MLP + max-pool) maps it to one vector added to every decoder query. Feedback is still the ground-truth `init_cube` vector (cube0 - ee_final, ee_final). prior_expert_frac kept at 0.25 (in those slots the expert is told the default cube while the student's cloud shows the real one: slightly inconsistent, not ablated).

Expert references (128 cubes): true cube 0.758, told default cube 0.227.

Final eval (iteration 29, success_once for h=0/1/2, alpha from the fixed U(0,1) draws): 0.281 / 0.656 / 0.633. Per-iteration curves plateau by ~iteration 12.

Controls at the end (128 tasks, h=0/1/2):
- alpha=0 (clean cloud): 0.656 / 0.695 / 0.703
- alpha=1 (pure noise): 0.094 / 0.508 / 0.516
- shuf_y: 0.328 / 0.078 / 0.133; no_y: 0.297 / 0.000 / 0.000; no_ctx: 0.320 / 0.250 / 0.359

Alpha sweep (policy_final, 256 tasks, +-3% s.e., success_once h=0 / h=1 / h=2):
alpha 0.0: 0.684 / 0.695 / 0.680; 0.1: 0.633 / 0.723 / 0.688; 0.2: 0.555 / 0.688 / 0.707; 0.3: 0.531 / 0.664 / 0.660; 0.4: 0.422 / 0.672 / 0.652; 0.5: 0.305 / 0.652 / 0.613; 0.6: 0.160 / 0.648 / 0.613; 0.7: 0.086 / 0.648 / 0.617; 0.8: 0.070 / 0.578 / 0.539; 0.9: 0.117 / 0.586 / 0.523; 1.0: 0.090 / 0.523 / 0.562.

Reading:
- Vision works: with a clean cloud and no context the student reaches 0.68, close to the true-cube expert (0.76).
- h=0 falls off with noise, reaching the blind floor (~0.07-0.12, same as the no-vision student at 0.08) from alpha ~0.7; the cube is unreadable around alpha 0.5-0.7.
- Context + feedback compensates: h=1 stays 0.65-0.72 up to alpha 0.7 and 0.52 at alpha 1, so feedback recovers most but not all of the blind case (clean h=1 is 0.70). h=2 tracks h=1.
- The student relies on the feedback token when present (shuf_y / no_y push h>=1 below h=0); with no context it falls back to the cloud (no_ctx 0.25-0.36).
- Note on iteration 0 in the curves: the point is after one round of expert-driven data + 2000 gradient steps, not an untrained policy (an untrained baseline was not run).
- Single seed. The per-iteration alpha<0.5 / >=0.5 curves are buckets of the U(0,1) draws, not exact alpha=0/1 (those are only the final controls and the sweep).

Videos (9, alpha cycled 0/0.5/1, same alpha for both episodes; top row h=0, bottom h=1; left = policy's cloud, right = rollout): h0/h1 success by round = S/S, S/S, F/S, S/F, S/F, F/F, F/F, S/S, S/S. Clearest improvement: round 2 (alpha=1, h=0 fails, h=1 succeeds). Single episodes, so noisy.

Next: top-down or cube-ringed cloud views, prior_expert_frac=0, more seeds, untrained baseline.
