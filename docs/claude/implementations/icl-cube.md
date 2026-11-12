# In-context learner on the cube grasp task (port of ManiSkill `icl_episodic.py`)

Files
- `scripts/imitation_learning/icl/train_icl_cube.py`: rollouts, DAgger loop, eval + controls (shuf_y / no_y / no_ctx). Run through `scripts/cluster/icl_cube.sbatch` (args via `ICL_ARGS`).
- `source/uwlab/uwlab/policy/icl_policy.py`: transformer student `pi(a|s, c_h)` (context = subsampled (s,a) steps + one feedback token per past episode, learned null token).
- `source/uwlab/uwlab/policy/cfm_loading.py`: loads the CFM point-cloud expert from a checkpoint dir (same logic as play_bc.py's loader).

Setup
- Task `UW-FrankaLeap-GraspCube-JointAbs-v0`, eval mode (synthetic PCD, no cameras). Hidden variable: cube xy, uniform +-0.2 m around (0.5, 0), fixed across a task's episodes
  (cube teleported after `env.reset()`, z/orientation kept; obs recomputed with `observation_manager.compute()`).
- Student obs (47): joint positions (arm 7 + hand 16), previous executed action (23), t/T. No cube, no point cloud. Actions: absolute joint targets (23), one per step.
- Expert: CFM-PCD BC checkpoint, queried every step (action horizon 1: the formatter history is then consecutive frames as in training) for the first action of its chunk; labels every student state.
- Episode = 128 controlled steps (`max_episode_length - num_warmup_steps`). Only time_out terminates the env, so "terminate at first success" is emulated by masking:
  steps after the first success (cube z >= 0.20 above the table) are excluded from loss/context and the feedback is frozen at that step.
- Feedback (`--feedback`): `init_cube` = [initial cube pos - final gripper (ee) pos, final ee pos] (default; ManiSkill showed the start position solves the task whereas the final cube
  position is a noisy proxy when the cube is pushed); `final_cube` = same with the final cube pos; `success` = 0/1 flag.
- DAgger: iteration 0 and 25% of later student slots are driven by an expert told the default cube position (cube points within 6 cm of the cube centre are shifted to the default
  position in its point cloud) so contexts contain realistic failures. k student episodes (h = 0..k-1) + 1 expert episode per task; expert episode is paired with every context size.
- Eval: fixed fresh cube positions (separate RNG from env resets), student only, k+1 episodes (h = 0..k). References: expert with the true cube, expert told the default position.

Checkerboard train/test split (2026-11-02, see experiments/2026-11-02-icl-cube-checkerboard-split.md)
- `--checker --checker_cells C --checker_margin m` trains on white squares of a CxC board over the cube_range square and evaluates id (white) and ood (black) on `--eval_tasks` fixed tasks each; metrics.jsonl rows have `eval_id` / `eval_ood`.
- `--bc_pcd`: expert-only BC with the clean initial point cloud (no cube xyz, no feedback); `--bc_full_info`: expert-only BC with the cube xyz appended to the state.
