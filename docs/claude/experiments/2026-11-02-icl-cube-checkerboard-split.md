# ICL vs BC on a checkerboard train/test split of cube positions (2026-11-02)

Question: with the same number of training cube positions (all from the "white" squares of a checkerboard over the workspace), does ICL (uses feedback from earlier episodes) generalise to the held-out "black" squares better than BC? All runs local (RTX 5090), seed 1, `--rfs_checkpoint logs/rfs/GraspCube-JointAbs_0509_0950_e7765f/model_000150.zip`. Logs `logs/local/checker*.out`, results `logs/icl/icl_cube_checker*/` (metrics.jsonl has `eval_id` / `eval_ood` per iteration, results.json has final controls).

## Code
`scripts/imitation_learning/icl/train_icl_cube.py`
- `--checker --checker_cells C --checker_margin m`: the cube_range square (+-0.2 m around (0.5, 0)) is a CxC board, cell (i, j) is white iff (i+j) even. Training cube xy is sampled from white cells only; points are kept `m` metres from every cell edge (train and eval).
- `--eval_tasks 256`: fixed eval tasks per region (white = `id`, black = `ood`), evaluated in chunks of `num_envs`. Metrics, refs, controls and `icl_curve.png` are reported per region; `eval` in metrics.jsonl stays the id result.
- `--bc_pcd`: new baseline, expert-only BC (no context, no feedback) whose student sees the episode's CLEAN initial point cloud (alpha=0) through the same PointNet as the vision student, and no cube position. `--bc_full_info` (older baseline) instead appends the true cube xyz to the state and has no point cloud.
- DAgger "prior expert" slots (told the default cube position) are unchanged.

## Setups
Common: 128 tasks / iteration (ICL: k=2 student episodes + 1 expert relabel episode per task; BC: 128 expert episodes), 2000 grad steps / iteration, ICL 30 iterations, BC 20 iterations. ICL vision = `--vision --k 2 --feedback init_cube` (noisy cloud, alpha ~ U(0,1)). Expert (true cube) reference on the eval sets: 2x2/0.04: 0.77-0.81 id, 0.85 ood; 3x3/0.06: 0.73 id, 0.86 ood.

## Results (success_once, mean of last 5 iterations, 256 tasks per region, 1 seed, s.e. ~ +-3%)
| Split | Method | white (id) | black (ood) | drop |
|---|---|---|---|---|
| 4x4, m=0 (stopped at it 14, mean it 10-14) | BC full-info | 0.76 | 0.75 | 0.01 |
| 3x3, m=0.04 | BC full-info | 0.728 | 0.668 | 0.06 |
| 3x3, m=0.06 | BC full-info | 0.745 | 0.593 | 0.15 |
| | BC clean cloud | 0.693 | 0.338 | 0.36 |
| | ICL vision h=0 / h=1 / h=2 | 0.440 / 0.684 / 0.675 | 0.227 / 0.384 / 0.402 | 0.21 / 0.30 / 0.27 |
| 2x2, m=0 | BC full-info | 0.73 | 0.59 | 0.14 |
| 2x2, m=0.04 | BC full-info | 0.795 | 0.477 | 0.32 |
| | BC clean cloud | 0.739 | 0.151 | 0.59 |
| | ICL vision h=0 / h=1 / h=2 | 0.466 / 0.722 / 0.727 | 0.185 / 0.380 / 0.386 | 0.28 / 0.34 / 0.34 |
| | ICL no vision h=0 / h=1 / h=2 | 0.018 / 0.748 / 0.741 | 0.082 / 0.332 / 0.320 | 0.42 (h=1) |

Final controls (ICL vision, h=0/1/2, ood): 2x2/0.04: no_y 0.20/0.00/0.00, shuf_y 0.19/0.16/0.16, no_ctx 0.22/0.15/0.18, alpha0 (clean cloud) 0.29/0.43/0.43, alpha1 (pure noise) 0.07/0.33/0.35. 3x3/0.06: no_y 0.23/0.00/0.00, shuf_y 0.24/0.11/0.14, alpha0 0.24/0.49/0.50, alpha1 0.06/0.25/0.27.

## Reading
- The 4x4 board (10 cm cells) is no test for BC full-info: it has no held-out drop (0.01) because it interpolates between neighbouring white cells from the cube xyz. Coarser cells and a margin make the drop appear: 0.06 (3x3/0.04), 0.14-0.15 (2x2/0, 3x3/0.06), 0.32 (2x2/0.04).
- BC full-info is a privileged baseline (true cube xyz at every step), not the right comparison for ICL. With a clean point cloud instead (`--bc_pcd`) BC collapses on black squares: 0.15 on 2x2/0.04 (white 0.74), 0.34 on 3x3/0.06 (white 0.69).
- ICL vision with feedback beats clean-cloud BC on black on 2x2/0.04: h=1/h=2 0.38-0.39 vs 0.15 at equal white performance (0.72-0.73 vs 0.74). Clear (>7 s.e.), even though ICL's cloud is noisy and BC's is clean. ICL did 3840 tasks vs BC 2560; at ICL iteration 19 (2560 tasks) black h=1 was ~0.33, so the gap holds at matched data.
- On 3x3/0.06 the ICL advantage over clean-cloud BC is small and within noise (0.38-0.40 vs 0.34).
- The feedback is what carries the gain: no_y gives 0.00 at h>=1 in both regions; shuf_y falls well below h=1.
- ICL's black success saturates at ~0.33-0.40 independent of the split (2x2 and 3x3, vision and no-vision), well below the expert (0.85) and below BC full-info on the same boards. Black h=1 reaches ~0.4 by iteration 2-3 and stays flat; a clean cloud at eval (alpha0) raises it to 0.43-0.50. Likely causes (not tested): the feedback token `[cube0 - ee_f, ee_f]` is out of its training range on black squares, and the decoder state path never saw black positions.
- No-vision ICL (feedback only) matches vision ICL on black (0.32-0.33 vs 0.38-0.40), so the noisy cloud is not what limits ICL on held-out squares; it mostly matters at h=0.

## Caveats
Single seed, +-3% per region; the 4x4 BC run was killed at iteration 14; BC and ICL data counts differ (2560 vs 3840 tasks); 3x3/m=0.06 means training is on five 1.3 cm squares; the 2x2 margin-0 run's expert reference differs between id and ood (0.73 vs 0.78).

Next: seeds 2 and 3 on 2x2/0.04 for ICL vision, BC clean cloud and BC full-info (see the seeds section below once filled in).

## Seeds 1-3 on 2x2, margin 0.04 (added 2026-11-03)
Seeds 2 and 3 run with the same flags (`--exp_name icl_cube_checker2_m0.04_{icl,bc_pcd,bc_full}_s{2,3}`, logs in `logs/local/`, plot `logs/icl/checker2_m0.04_seeds.png`, script `plot_seeds.py` in the scratchpad). Mean of the last 5 iterations per seed; 256 tasks per region.

| Method | white (id), seeds 1/2/3 | black (ood), seeds 1/2/3 | ood mean +- sd |
|---|---|---|---|
| BC full-info (cube xyz) | 0.795 / 0.816 / 0.810 | 0.477 / 0.244 / 0.448 | 0.390 +- 0.127 |
| BC clean cloud | 0.739 / 0.755 / 0.743 | 0.151 / 0.200 / 0.219 | 0.190 +- 0.035 |
| ICL vision h=0 | 0.466 / 0.482 / 0.476 | 0.185 / 0.174 / 0.142 | 0.167 +- 0.022 |
| ICL vision h=1 | 0.722 / 0.710 / 0.704 | 0.380 / 0.346 / 0.212 | 0.313 +- 0.089 |
| ICL vision h=2 | 0.727 / 0.727 / 0.741 | 0.386 / 0.351 / 0.227 | 0.321 +- 0.084 |

Reading (this supersedes the single-seed claims above where they differ):
- White performance is the same across seeds (ICL h>=1 0.70-0.74, BC clean cloud 0.74-0.76, BC full-info 0.80-0.82), so the comparison is at similar in-distribution performance (ICL slightly lower).
- ICL h=1/2 beats clean-cloud BC on black in seeds 1 and 2 (0.35-0.39 vs 0.15-0.20) but not in seed 3 (0.21-0.23 vs 0.22). Three-seed means: ICL 0.31-0.32 vs BC 0.19, a +0.12-0.13 difference, but the ICL seed sd (0.09) is large, so with 3 seeds this is suggestive, not established. The earlier "more than double" claim came from the two best seeds.
- The in-context gain (h=1 minus h=0) on black is +0.15 +- 0.07 across seeds; ICL h=0 (0.17) is about the same as clean-cloud BC (0.19), as expected.
- BC full-info on black varies a lot with the seed (0.24-0.48), so the single seed-1 value (0.477) was a favourable draw; ICL (0.31) is within its range.
- Not resolved: why ICL seed 3 stays low on black (0.21 at the end; h=1 hovered 0.2-0.3 from iteration ~5). No seed-to-seed control for the eval tasks (fixed across seeds), so this is training variance.
