# Eval Scripts

## Which script to use?

| Checkpoint format | Script |
|---|---|
| New (Hydra-trained, has `.hydra/config.yaml` next to `checkpoints/`) | `play_bc.py` |
| Legacy (IsaacLab-trained, has `ckpt["state_dicts"]["ema_model"]`) | `play_bc_legacy.py` |

New checkpoints come from `scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py`.
Legacy checkpoints came from the old IsaacLab training pipeline.

---

## play_bc.py

Evaluates a new-format checkpoint using a YAML eval config.

### Usage

Via SLURM (recommended):
```bash
sbatch commands/uwlab/eval/<eval_script>.sh
```

Inside the container, basic run:
```bash
./uwlab.sh -p scripts/eval/play_bc.py \
    --eval_cfg configs/eval/bottle_grasp_bc.yaml \
    --headless
```

With checkpoint override, video recording, and multiple envs:
```bash
./uwlab.sh -p scripts/eval/play_bc.py \
    --eval_cfg configs/eval/bottle_grasp_bc.yaml \
    checkpoint=logs/bc_cfm_pcd_bourbon_0312_random_resets \
    record_video=true \
    num_envs=4 \
    --enable_cameras \
    --headless
```

Key=value overrides (e.g. `checkpoint=...`, `record_video=true`, `num_envs=4`) are applied
on top of the eval config yaml. `--enable_cameras` and `--headless` are Isaac Sim launcher
args and must use the `--flag` form.

### Eval config

Eval configs live in `configs/eval/*.yaml`. Key fields:

```yaml
task_id: UW-FrankaLeap-GraspBottle-IkRel-v0
checkpoint: logs/bc_cfm_pcd_grasp   # path to hydra run dir
obs_keys: [ee_pose, hand_joint_pos] # override training obs_keys if needed
num_envs: 1
action_horizon: 1
num_warmup_steps: 5
spawn: cardinal_3x3                 # spawn config from configs/eval/spawns/
record_video: true
```

`obs_keys`, `image_keys`, `downsample_points`, and all policy arch params are loaded
automatically from the checkpoint's `.hydra/config.yaml`. Only override `obs_keys` if
you need to evaluate with a different observation set than training.

### Output

Saved to `<checkpoint_dir>/../eval/<eval_config_stem>/<checkpoint_basename>/`:
- `results.json` — success rate, per-episode outcomes
- `trajectories.png` — EE xyz trajectories colored by success
- `videos/episode_NNN_{success,fail}.mp4` — if `record_video: true`

---

## play_bc_legacy.py

Deploys a legacy IsaacLab-format checkpoint. All args are passed on the CLI
(no eval config yaml). Architecture params are fixed to the standard arch
used for those checkpoints and are not read from the checkpoint cfg.

### Checkpoint format

- `ckpt["state_dicts"]["ema_model"]` — model weights
- `ckpt["cfg"]` — OmegaConf DictConfig embedded at save time

### Usage

Via SLURM (recommended):
```bash
sbatch commands/uwlab/eval/eval_bottle_grasp_mini18.sh
```

Or inside the container:
```bash
./uwlab.sh -p scripts/eval/play_bc_legacy.py \
    --checkpoint logs/real/mini_18/cfm/pcd_cfm/horizon_4_nobs_1 \
    --task UW-FrankaLeap-GraspBottle-IkRel-v0 \
    --obs_keys joint_pos \
    --num_envs 1 \
    --num_episodes 10 \
    --action_horizon 1 \
    --headless
```

Add `--record_video --enable_cameras` to save per-episode MP4s.

### Obs key mapping

Legacy policies used `joint_positions(7) + gripper_position(16) = 23D agent_pos`,
exposed as a single `joint_pos` key in UWLab. Pass `--obs_keys joint_pos`.

Newer legacy policies (pour) used `ee_pose(7) + hand_joint_pos(16)`:
pass `--obs_keys ee_pose hand_joint_pos`.

### Output

Saved to `<checkpoint>/eval_legacy/` (override with `--output_dir`):
- `results.json` — success rate, per-episode outcomes
- `trajectories.png` — EE xyz trajectories colored by success
- `videos/episode_NNN_{success,fail}.mp4` — written per-episode as they complete

---

## Running inside the container manually

All SLURM scripts use `commands/uwlab/run_in_container.sh` which handles all bind
mounts and env setup. To run interactively, see the canonical apptainer command in
`~/.claude/projects/.../memory/MEMORY.md` or any of the SLURM scripts.
