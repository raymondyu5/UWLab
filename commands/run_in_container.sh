#!/bin/bash
# Shared apptainer wrapper for UWLab SLURM scripts.
#
# Usage: run_in_container.sh INNER_COMMAND
#   INNER_COMMAND is a bash-evaluable command string to run inside the container.
#
# Optional env var: EXTRA_APPTAINER_BINDS
#   Space-separated extra --bind flags, e.g.:
#     export EXTRA_APPTAINER_BINDS="--bind /some/path:/container/path"

_UWLAB_BASE=/gscratch/weirdlab/raymond/uwlab_docker
_UWLAB_REPO=/gscratch/weirdlab/raymond/UWLab

apptainer exec --nv \
  --bind ${_UWLAB_REPO}/source:/workspace/uwlab/source \
  --bind ${_UWLAB_REPO}/scripts:/workspace/uwlab/scripts \
  --bind ${_UWLAB_REPO}/assets:/workspace/uwlab/assets \
  --bind ${_UWLAB_REPO}/third_party:/workspace/uwlab/third_party \
  --bind ${_UWLAB_REPO}/logs:/workspace/uwlab/logs \
  --bind ${_UWLAB_REPO}/configs:/workspace/uwlab/configs \
  --bind /gscratch/weirdlab/raymond:/gscratch/weirdlab/raymond \
  --bind ${_UWLAB_BASE}/isaac-cache-kit:/isaac-sim/kit/cache \
  --bind ${_UWLAB_BASE}/isaac-sim-data:/isaac-sim/kit/data \
  --bind ${_UWLAB_BASE}/isaac-cache-ov:/root/.cache/ov \
  --bind ${_UWLAB_BASE}/isaac-cache-pip:/root/.cache/pip \
  --bind ${_UWLAB_BASE}/isaac-cache-gl:/root/.cache/nvidia/GLCache \
  --bind ${_UWLAB_BASE}/isaac-cache-compute:/root/.nv/ComputeCache \
  --bind ${_UWLAB_BASE}/isaac-cache-warp:/root/.cache/warp \
  --bind ${_UWLAB_BASE}/outputs:/workspace/uwlab/outputs \
  --bind ${_UWLAB_BASE}/data_storage:/workspace/uwlab/data_storage \
  ${EXTRA_APPTAINER_BINDS:-} \
  --pwd /workspace/uwlab \
  ${_UWLAB_BASE}/uw-lab_latest.sif \
  /bin/bash -c '
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    export WANDB_INSECURE_DISABLE_SSL=true
    cd /workspace/uwlab
    eval "$1"
  ' -- "$1"
