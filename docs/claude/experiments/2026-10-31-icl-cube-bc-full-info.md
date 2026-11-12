# Full-information BC baseline vs ICL (cube grasp), same expert budget (2026-10-31)

Run locally (RTX 5090, ~38 s/iter, 20 min; `docs/claude/info/local-install.md`). Log `logs/local/bc_full_info.out`, results `logs/icl/icl_cube_bc_full_info/` (policy_final.pt, metrics.jsonl, results.json), plot `logs/icl/icl_cube_bc_full_info/bc_vs_icl.png` (script in the scratchpad, `plot_bc.py`).

Setup: `train_icl_cube.py --bc_full_info --rfs_checkpoint logs/rfs/GraspCube-JointAbs_0509_0950_e7765f/model_000150.zip --num_envs 128 --num_iterations 30`. Each iteration: 128 new expert (RFS, true cube) episodes, no student rollouts, plain BC (same ICLPolicy network with an empty context, 2000 grad steps/iter, same lr schedule, seed 1). The student state is joint pos, previous action, t/T **plus the current true cube xyz** (full state; no point cloud, no feedback). Eval: one student episode on the same 128 fixed eval tasks as the ICL runs.
Comparison point: the ICL vision run `logs/icl/icl_cube_rfs_vision/` (also 128 tasks/iter = 128 expert-relabelled episodes/iter, but it additionally rolls out 256 student episodes/iter whose states the expert relabels).

Result (success_once, mean of last 5 iterations; final iteration in brackets; expert with true cube 0.76-0.77):
- BC, full info: 0.758 (0.781)
- ICL h=0 (noisy point cloud, no context): 0.320 (0.281)
- ICL h=1: 0.623 (0.656)
- ICL h=2: 0.642 (0.633)

Reading:
- BC with privileged cube position matches the expert (~0.76) and is the upper bound: after one iteration (128 episodes) it is already at 0.47, 0.70 by ~500 episodes.
- ICL with a noisy point cloud plus one feedback episode reaches ~0.62-0.64, i.e. ~82-85% of the full-information BC; the remaining gap is the cost of a noisy (alpha ~ U(0,1)) cloud, not data. ICL h=0 (0.32) is far below BC because the noisy cloud often hides the cube.
- ICL learns more slowly per expert episode (BC reaches 0.7 at ~500 episodes, ICL h=1 ~0.6 at ~1500).
- Caveats: single seed, 128 eval tasks (+-3-4% s.e.); BC sees the live cube position, not the initial one; compute differs (ICL also rolls out and relabels student episodes). Evaluation is on the local simulator; expert reference came out 0.766 here vs 0.758 on klone (different GPU, noise).

Incident: the first (klone) attempt had loss NaN from iteration 0 and 0.0 success: `set_norm` got a single dummy feedback row, std of one sample is NaN, and the NaN leaked through the masked attention keys. Fixed by passing 2 rows (`torch.zeros(2, Y)`).

Next: BC with the noisy point cloud instead of the cube position (isolates the vision gap), more seeds.
