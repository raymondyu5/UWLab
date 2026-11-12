# Local install (RTX 5090, Ubuntu 24.04)

Verified 2026-10-30: `UW-FrankaLeap-GraspCube-JointAbs-v0` builds, resets and steps locally with `--enable_cameras`.

- Env: `source ~/.venvs/uwlab/bin/activate` (uv, Python 3.11; kept outside the repo).
- Isaac Sim 5.1.0 (pip) + torch 2.7.0+cu128. Needs Python 3.11 exactly.
- `./uwlab.sh --install` clones upstream IsaacLab `main` (needs py3.12). `_isaaclab/IsaacLab` is pinned to tag `v2.3.2` instead (gitignored; the script reuses an existing clone).
- Build workarounds: `uv pip install "setuptools<81" wheel && uv pip install --no-build-isolation flatdict==4.0.1`; `torch-cluster` from `-f https://data.pyg.org/whl/torch-2.7.0+cu128.html`.
- Task configs hardcode `/workspace/uwlab/...` (container path): `sudo ln -sfn ~/research/UWLab /workspace/uwlab`.
- First launch asks to accept the Omniverse EULA (interactive).
- NVIDIA driver must be 580.x (`nvidia-driver-580-open`). Driver 595.x segfaults in `librtx.scenedb.plugin.so` as soon as the RTX renderer starts (anything with `--enable_cameras`); non-camera runs are unaffected.
- Heavy training/eval still goes to klone (see remote-workflow.md).
- `play_rfs_wrapper.py` imports `scripts.reinforcement_learning...`: run with `PYTHONPATH=$PWD` (the container has the repo root on the path).
- Extra pip packages the cluster overlay has: `torchcfm zarr hydra-core imageio imageio-ffmpeg dill` (installed; torch stays 2.7.0+cu128).
- Local RFS cube eval reproduced klone: see experiments/2026-10-30-local-rfs-eval-cube.md.
