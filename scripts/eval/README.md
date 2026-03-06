# Eval Scripts

## play_bc_legacy.py

Deploys an IsaacLab-format BC checkpoint (CFMPCDPolicy) inside a UWLab environment.

### Checkpoint format

IsaacLab checkpoints differ from UWLab's flat format:
- `ckpt["state_dicts"]["ema_model"]` — model weights
- `ckpt["cfg"]` — OmegaConf DictConfig embedded at save time (contains `shape_meta`, `horizon`, `n_obs_steps`, etc.)

### Dependencies inside the container

The Isaac Sim container does not include `dill`, `imageio`, or `diffusion_policy`. Install them once on the host:

```bash
pip install -r third_party/requirements.txt --target third_party/pip_packages
```

`third_party/pip_packages/` is gitignored. The script adds it to `sys.path` automatically.
`diffusion_policy` lives at `third_party/diffusion_policy/` (tracked in git).

### Running inside the container

**Step 1**: Enter the Apptainer container (run from the host, outside the container):
```bash
export UWLAB_BASE=/gscratch/weirdlab/raymond/uwlab_docker
apptainer exec --nv \
  --bind /gscratch/weirdlab/raymond/UWLab/source:/workspace/uwlab/source \
  --bind /gscratch/weirdlab/raymond/UWLab/scripts:/workspace/uwlab/scripts \
  --bind /gscratch/weirdlab/raymond/UWLab/assets:/workspace/uwlab/assets \
  --bind /gscratch/weirdlab/raymond/UWLab/third_party:/workspace/uwlab/third_party \
  --bind /gscratch/weirdlab/raymond/UWLab/logs:/workspace/uwlab/logs \
  --bind $UWLAB_BASE/isaac-cache-kit:/isaac-sim/kit/cache \
  --bind $UWLAB_BASE/isaac-sim-data:/isaac-sim/kit/data \
  --bind $UWLAB_BASE/isaac-cache-ov:/root/.cache/ov \
  --bind $UWLAB_BASE/isaac-cache-pip:/root/.cache/pip \
  --bind $UWLAB_BASE/isaac-cache-gl:/root/.cache/nvidia/GLCache \
  --bind $UWLAB_BASE/isaac-cache-compute:/root/.nv/ComputeCache \
  --bind $UWLAB_BASE/outputs:/workspace/uwlab/outputs \
  --bind $UWLAB_BASE/data_storage:/workspace/uwlab/data_storage \
  --pwd /workspace/uwlab \
  $UWLAB_BASE/uw-lab_latest.sif bash
```

**Step 2**: Inside the container, unset conda env vars so `uwlab.sh` uses Isaac Sim python:
```bash
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
```

**Step 3**: Run the eval script:
```bash
./uwlab.sh -p scripts/eval/play_bc_legacy.py \
    --checkpoint /workspace/uwlab/logs/cup_pick_h4 \
    --task UW-FrankaLeap-GraspPinkCup-IkRel-v0 \
    --obs_keys joint_pos \
    --num_envs 1 \
    --num_episodes 10 \
    --action_horizon 1 \
    --headless
```

Add `--record_video --enable_cameras` to save per-episode MP4s from `fixed_camera`.

### Why `IkRel` and not `JointAbs`?

The `cup_pick` policy was trained with IK-delta actions:
- 6D EE delta (arm) + 16D hand joint absolute = 22D total
- Use `UW-FrankaLeap-GraspPinkCup-IkRel-v0`, NOT `JointAbs`

### Obs key mapping

The old IsaacLab policy used `joint_positions(7) + gripper_position(16) = 23D agent_pos`.
In UWLab the env returns this as a single `joint_pos` key (23D). Pass `--obs_keys joint_pos`.

### Output

Results are saved to `<checkpoint>/eval_legacy/` (override with `--output_dir`):
- `results.json` — success rate, per-episode outcomes
- `trajectories.png` — EE xyz trajectories colored by success
- `videos/episode_NNN_{success,fail}.mp4` — written per-episode as they complete (if `--record_video`)
