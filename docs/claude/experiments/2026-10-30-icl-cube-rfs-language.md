# ICL cube grasp student with T5 language feedback, RFS expert (2026-10-30)

Job 42578495 (gpu-l40, 2h00m), `logs/icl/icl_cube_rfs_lang/` (policy_final.pt, metrics.jsonl, results.json).

Setup: `train_icl_cube.py --rfs_checkpoint ...model_000150.zip --k 2 --feedback language_tokens --num_envs 128 --num_iterations 40`. Same as the vector-feedback run (2026-10-29-icl-cube-rfs.md) except the feedback: the init_cube displacement (cube0 - ee_final) is written as a templated command ("move the hand 12.3 cm forward and 4.5 cm left and 0.7 cm down", random opener / axis order), encoded by frozen `google/flan-t5-small` (offline from /gscratch/weirdlab/will/hf_cache), one 512-d vector per word piece (40 max), passed through LayerNorm + linear + word-position embedding into context tokens. No point cloud (student sees joint pos, prev action, t/T).

Expert references: true cube 0.766, told default 0.242.

Final eval (iteration 39, success_once h=0/1/2): 0.070 / 0.336 / 0.297 (last 5 iterations: h=1 0.28-0.44, h=2 0.30-0.44).
Controls (h=0/1/2): shuf_y 0.086/0.062/0.094, no_y 0.062/0.086/0.078, no_ctx 0.078/0.078/0.070.

Reading: the student learns in context from language (h>=1 about 4x h=0, and the gain disappears with removed/shuffled feedback), but it is much weaker than the vector feedback (0.66 at h=1) and noisy; still slowly improving at the end (loss drifts up slightly as the DAgger dataset grows, as in the vector run). Consistent with ManiSkill, where language learned more slowly than the final-error vector. Single seed, 128 eval tasks (+-4%).

Next: longer training / larger tok count, h=2+ with k>2, combine language feedback with the point-cloud student.
