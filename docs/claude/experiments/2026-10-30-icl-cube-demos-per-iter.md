# ICL cube (point cloud + noise interpolation): ablate demos per iteration (2026-10-30)

Jobs 42611427 (16), 42611512 (32), 42611534 (64), 42611624 (256), gpu-l40, 1.2-2.4 h each. Logs: `logs/icl/icl_cube_rfs_vision_demos{16,32,64,256}/` (policy_final.pt, metrics.jsonl, results.json), plot `logs/icl/icl_cube_rfs_vision_demos_ablation/demos_ablation.png`. The 128 point is the earlier run `logs/icl/icl_cube_rfs_vision/` ([2026-10-30-icl-cube-rfs-vision.md](2026-10-30-icl-cube-rfs-vision.md)), same config/seed, not rerun (it predates the flag).

Setup: identical to the vision run (`--rfs_checkpoint .../model_000150.zip --k 2 --feedback init_cube --vision --num_envs 128 --num_iterations 30`, 2000 grad steps/iter, seed 1) plus new `--demos_per_iter n`: each iteration adds n tasks (k=2 student episodes + 1 expert relabel episode each) to the dataset. Rollouts always run on 128 envs (n<128: extras dropped; n=256: two passes), so the 128-task eval is identical across runs. Dataset after 30 iterations: 480 / 960 / 1920 / 3840 / 7680 tasks. Grad steps fixed, so small-n runs also see each sample more often.

Final eval (iteration 29, success_once, h=0/1/2; expert with true cube ~0.75):
| demos/iter | h=0 | h=1 | h=2 | alpha=0 (h=0/1/2) | alpha=1 (h=0/1/2) |
|---|---|---|---|---|---|
| 16 | 0.281 | 0.320 | 0.289 | 0.43/0.36/0.36 | 0.07/0.14/0.12 |
| 32 | 0.297 | 0.391 | 0.445 | 0.55/0.44/0.50 | 0.09/0.26/0.23 |
| 64 | 0.305 | 0.461 | 0.484 | 0.54/0.56/0.54 | 0.06/0.36/0.35 |
| 128 | 0.281 | 0.656 | 0.633 | 0.66/0.70/0.70 | 0.09/0.51/0.52 |
| 256 | 0.320 | 0.680 | 0.703 | 0.67/0.69/0.75 | 0.09/0.59/0.59 |

Mean of last 5 iterations, h=1: 0.256 / 0.328 / 0.500 / 0.623 / 0.683 (less noisy than the single final point; +-3-4% s.e. per eval).

Controls at the end, all runs: no_y gives h>=1 = 0.000; shuf_y h>=1 = 0.06-0.13 (below h=0); no_ctx ~0.23-0.36. So every run's gain over h=0 comes from the feedback token, as before.

Reading:
- In-context gain (h=1 minus h=0) grows monotonically with demos/iter: ~0 at 16 (0.32 vs 0.28), +0.09 at 32, +0.16 at 64, +0.37 at 128, +0.36 at 256. It saturates around 128; 256 is within noise of 128 (h=1 0.68 vs 0.66) at ~1.4x the wall-clock.
- h=0 is roughly flat (0.28-0.32): the vision-only path is not very data-limited in this range; what scales is use of the feedback/context. With a clean cloud (alpha=0) h=0 also rises 0.43 -> 0.67, so the cloud encoder is data-limited too.
- With little data the student leans on the cloud: at 16 demos the h=1 result barely beats h=0 even though the context token is informative.
- Caveats: single seed; confounded with total dataset size and with grad steps per sample (fixed 2000/iter); 128 point from a pre-flag run (same code path otherwise); n<128 wastes rollouts so wall-clock does not shrink proportionally (137 s/it at n=16 vs ~190 at 128, mean per-iteration seconds depend on cluster load too).

Next: more seeds, fix total dataset size and vary iterations (separates demos/iter from total data), n=512, scale grad steps with data.
