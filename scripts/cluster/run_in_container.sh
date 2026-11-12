#!/bin/bash
# Run a command inside the UWLab apptainer on klone (call from sbatch scripts or an allocated node).
# Usage: run_in_container.sh "INNER_COMMAND"   (bash string evaluated in /workspace/uwlab)
# overlay.img (has torchcfm etc., not in the bare image) is mounted read-only by default; USE_OVERLAY=0 disables.
# Optional: EXTRA_APPTAINER_BINDS="--bind /a:/b".
B=/gscratch/weirdlab/will/UWLab_Docker
R=/gscratch/weirdlab/will/UWLab
JOB=${SLURM_JOB_ID:-interactive}
mkdir -p $B/tmp/$JOB/isaaclab $R/logs
OVERLAY=(); [ "${USE_OVERLAY:-1}" != 0 ] && OVERLAY=(--overlay $B/overlay.img:ro)

apptainer exec --nv --fakeroot "${OVERLAY[@]}" \
  --bind $R/source:/workspace/uwlab/source \
  --bind $R/scripts:/workspace/uwlab/scripts \
  --bind $R/assets:/workspace/uwlab/assets \
  --bind $R/third_party:/workspace/uwlab/third_party \
  --bind $R/configs:/workspace/uwlab/configs \
  --bind $R/logs:/workspace/uwlab/logs \
  --bind $B/isaac-cache-kit:/isaac-sim/kit/cache \
  --bind $B/isaac-sim-data:/isaac-sim/kit/data \
  --bind $B/isaac-cache-ov:/root/.cache/ov \
  --bind $B/isaac-cache-pip:/root/.cache/pip \
  --bind $B/isaac-cache-gl:/root/.cache/nvidia/GLCache \
  --bind $B/isaac-cache-compute:/root/.nv/ComputeCache \
  --bind $B/outputs:/workspace/uwlab/outputs \
  --bind $B/data_storage:/workspace/uwlab/data_storage \
  --bind $B/tmp/$JOB/isaaclab:/tmp/isaaclab \
  --bind /gscratch/weirdlab/will/hf_cache:/hf_cache \
  ${EXTRA_APPTAINER_BINDS:-} \
  --pwd /workspace/uwlab \
  $B/uw-lab_latest.sif \
  /bin/bash -c '
    unset CONDA_PREFIX CONDA_DEFAULT_ENV
    export WANDB_INSECURE_DISABLE_SSL=true HF_HOME=/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /workspace/uwlab
    eval "$1"
  ' -- "$1"
