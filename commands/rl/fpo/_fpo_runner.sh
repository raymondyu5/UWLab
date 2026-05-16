#!/bin/bash
# Shared Slurm+apptainer runner for FPO tasks.
# Do not call directly — source from a per-task script that sets:
#   TASK_ID, BC_CKPT, TASK_SLUG
# Optional BC regularization (datasets/noise defined in per-task script):
#   BC_DATASETS      — space-separated host paths to real zarr datasets
#   NOISE_EXTRINSIC  — "true" for tasks trained with extrinsic noise aug
# Usage (from per-task script): source _fpo_runner.sh [--bc] [gpu_type] [version]
#   --bc:     enable BC regularization (BC_COEF=0.02; override with BC_COEF=<val>)
#   gpu_type: Slurm GPU label, e.g. l40, l40s  (default: l40s)
#   version:  config version, e.g. v9, v12  (default: v9)

# Parse --bc and --n_cfm_samples flags, leaving positional args.
USE_BC=false
N_CFM_SAMPLES=""
POSITIONAL_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --bc) USE_BC=true ;;
        --n_cfm_samples) N_CFM_SAMPLES="$2"; shift ;;
        *) POSITIONAL_ARGS+=("$1") ;;
    esac
    shift
done
set -- "${POSITIONAL_ARGS[@]}"

GPU=${1:-l40s}
VERSION=${2:-v9}

if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p logs
    sbatch \
      --job-name=uwlab_fpo_${TASK_SLUG}_${GPU}_${VERSION} \
      --account=weirdlab \
      --partition=gpu-${GPU} \
      --nodes=1 \
      --ntasks-per-node=1 \
      --cpus-per-task=8 \
      --gres=gpu:${GPU}:1 \
      --mem=128G \
      --time=48:00:00 \
      --output=logs/uwlab_fpo_${TASK_SLUG}_${GPU}_${VERSION}_%j.out \
      --error=logs/uwlab_fpo_${TASK_SLUG}_${GPU}_${VERSION}_%j.err \
      "$0" "$@" $([ "$USE_BC" = "true" ] && echo "--bc") \
      $([ -n "$N_CFM_SAMPLES" ] && echo "--n_cfm_samples $N_CFM_SAMPLES")
    exit 0
fi

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

mkdir -p /gscratch/weirdlab/will/.tmp/${SLURM_JOB_ID}/isaaclab

export UWLAB_BASE=/gscratch/weirdlab/will/UWLab_Docker

# Auto-scale n_steps inversely with n_cfm_samples to keep memory constant.
CFM_ARGS=""
if [ -n "$N_CFM_SAMPLES" ] && [ "$N_CFM_SAMPLES" -gt 1 ]; then
    N_STEPS=$(( 200 / N_CFM_SAMPLES ))
    CFM_ARGS="--n_cfm_samples ${N_CFM_SAMPLES} --n_steps ${N_STEPS}"
    echo "[FPO] n_cfm_samples=${N_CFM_SAMPLES}, auto n_steps=${N_STEPS}"
fi

# BC regularization: enabled only when --bc flag is passed.
BC_TRAIN_ARGS=""
EXTRA_BINDS=()
if [ "$USE_BC" = "true" ] && [ -n "${BC_DATASETS:-}" ]; then
    for ds in ${BC_DATASETS}; do
        BC_TRAIN_ARGS="${BC_TRAIN_ARGS} --bc_dataset_path ${ds}"
    done
    BC_TRAIN_ARGS="${BC_TRAIN_ARGS} --bc_coef ${BC_COEF:-0.02}"
    [ "${NOISE_EXTRINSIC:-false}" = "true" ] && BC_TRAIN_ARGS="${BC_TRAIN_ARGS} --noise_extrinsic"
    # Bind will's data_storage at its host path so absolute paths work inside the container.
    EXTRA_BINDS+=(--bind /gscratch/weirdlab/will/UWLab_Docker/data_storage:/gscratch/weirdlab/will/UWLab_Docker/data_storage)
    echo "[BC] datasets: ${BC_DATASETS}"
    echo "[BC] coef: ${BC_COEF:-0.02}, noise_extrinsic: ${NOISE_EXTRINSIC:-false}"
fi

apptainer exec --nv \
  --fakeroot \
  --overlay /gscratch/weirdlab/will/UWLab_Docker/overlay.img:ro \
  --bind /gscratch/weirdlab/raymond:/gscratch/weirdlab/raymond \
  --bind /gscratch/weirdlab/will/UWLab/source:/workspace/uwlab/source \
  --bind /gscratch/weirdlab/will/UWLab/scripts:/workspace/uwlab/scripts \
  --bind /gscratch/weirdlab/will/UWLab/assets:/workspace/uwlab/assets \
  --bind /gscratch/weirdlab/will/UWLab/third_party:/workspace/uwlab/third_party \
  --bind $UWLAB_BASE/isaac-cache-kit:/isaac-sim/kit/cache \
  --bind $UWLAB_BASE/isaac-sim-data:/isaac-sim/kit/data \
  --bind $UWLAB_BASE/isaac-cache-ov:/root/.cache/ov \
  --bind $UWLAB_BASE/isaac-cache-pip:/root/.cache/pip \
  --bind $UWLAB_BASE/isaac-cache-gl:/root/.cache/nvidia/GLCache \
  --bind $UWLAB_BASE/isaac-cache-compute:/root/.nv/ComputeCache \
  --bind $UWLAB_BASE/logs:/workspace/uwlab/logs \
  --bind $UWLAB_BASE/outputs:/workspace/uwlab/outputs \
  --bind $UWLAB_BASE/data_storage:/workspace/uwlab/data_storage \
  --bind $UWLAB_BASE/tmp/isaaclab:/tmp/isaaclab \
  --bind $UWLAB_BASE/tmp:/tmp \
  --bind /gscratch/weirdlab/will/UWLab/configs:/workspace/uwlab/configs \
  "${EXTRA_BINDS[@]}" \
  --pwd /workspace/uwlab \
  $UWLAB_BASE/uw-lab_latest.sif \
  /bin/bash -c "
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    cd /workspace/uwlab

    export WANDB_INSECURE_DISABLE_SSL=true
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1

    ./uwlab.sh -p \
      scripts/reinforcement_learning/fpo/train.py \
      --task ${TASK_ID} \
      --num_envs 4096 \
      --diffusion_path ${BC_CKPT} \
      --cfg configs/rl/arm_fpo_cfg_${VERSION}.yaml \
      --wandb_project fpo_uwlab \
      --enable_cameras \
      --headless \
      ${BC_TRAIN_ARGS} \
      ${CFM_ARGS}
  "

echo "End: $(date)"
