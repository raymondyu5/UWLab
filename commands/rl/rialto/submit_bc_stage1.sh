#!/bin/bash
# Submit BC stage-1 pretraining jobs for all active tasks.
# Each job runs train_stage1_bc.py (standalone, no Isaac Sim) for 500 epochs.
#
# Usage: bash submit_bc_stage1.sh [gpu_type]
#   gpu_type: l40s (default) or l40

GPU=${1:-a40}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

UWLAB_BASE=/gscratch/weirdlab/will/UWLab_Docker

# --partition=ckpt \
# --nodes=1 \
# --ntasks-per-node=1 \
# --cpus-per-task=8 \
# --gpus-per-node=a40:1 \

# --partition=gpu-${GPU} \
# --nodes=1 \
# --ntasks-per-node=1 \
# --cpus-per-task=8 \
# --gres=gpu:${GPU}:1 \


submit() {
  local slug=$1 data_subdir=$2; shift 2
  local extra_args="$@"

  sbatch \
    --job-name="bc_stage1_${slug}" \
    --account=weirdlab \
    --partition=gpu-${GPU} \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --gres=gpu:${GPU}:1 \
    --mem=32G \
    --time=2:00:00 \
    --output="${LOG_DIR}/bc_stage1_${slug}_%j.out" \
    --error="${LOG_DIR}/bc_stage1_${slug}_%j.err" \
    << SBATCH_SCRIPT
#!/bin/bash
echo "Job ID: \$SLURM_JOB_ID | Node: \$SLURM_NODELIST | Start: \$(date)"
mkdir -p /gscratch/weirdlab/will/.tmp/\${SLURM_JOB_ID}/isaaclab

apptainer exec --nv \\
  --fakeroot \\
  --overlay ${UWLAB_BASE}/overlay.img:ro \\
  --bind /gscratch/weirdlab/will/UWLab/source:/workspace/uwlab/source \\
  --bind /gscratch/weirdlab/will/UWLab/scripts:/workspace/uwlab/scripts \\
  --bind /gscratch/weirdlab/will/UWLab/assets:/workspace/uwlab/assets \\
  --bind /gscratch/weirdlab/will/UWLab/third_party:/workspace/uwlab/third_party \\
  --bind ${UWLAB_BASE}/isaac-cache-kit:/isaac-sim/kit/cache \\
  --bind ${UWLAB_BASE}/isaac-sim-data:/isaac-sim/kit/data \\
  --bind ${UWLAB_BASE}/isaac-cache-ov:/root/.cache/ov \\
  --bind ${UWLAB_BASE}/isaac-cache-pip:/root/.cache/pip \\
  --bind ${UWLAB_BASE}/isaac-cache-gl:/root/.cache/nvidia/GLCache \\
  --bind ${UWLAB_BASE}/isaac-cache-compute:/root/.nv/ComputeCache \\
  --bind ${UWLAB_BASE}/logs:/workspace/uwlab/logs \\
  --bind ${UWLAB_BASE}/outputs:/workspace/uwlab/outputs \\
  --bind ${UWLAB_BASE}/data_storage:/workspace/uwlab/data_storage \\
  --bind ${UWLAB_BASE}/tmp/isaaclab:/tmp/isaaclab \\
  --bind ${UWLAB_BASE}/tmp:/tmp \\
  --bind /gscratch/weirdlab/will/UWLab/configs:/workspace/uwlab/configs \\
  --pwd /workspace/uwlab \\
  ${UWLAB_BASE}/uw-lab_latest.sif \\
  /bin/bash -c "
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    cd /workspace/uwlab
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1
    ./uwlab.sh -p \\
      scripts/reinforcement_learning/rialto/train_stage1_bc.py \\
      --data_path data_storage/rialto/${data_subdir} \\
      --log_dir logs/rialto/bc/${slug} \\
      --bc_epochs 1000 \\
      ${extra_args}
  "

echo "End: \$(date)"
SBATCH_SCRIPT
}

# ── Active tasks ───────────────────────────────────────────────────────────────
GRASP_KEYS="--obs_keys arm_joint_pos hand_joint_pos manipulated_object_pose target_object_pose contact_obs object_in_tip"
SCREW_KEYS="--obs_keys arm_joint_pos hand_joint_pos object_pose rotate_angle contact_obs object_in_tip"

# submit bottle_grasp       bottle_grasp_privileged  $GRASP_KEYS
# submit cup_grasp          cup_grasp_privileged          $GRASP_KEYS
# submit cube_grasp         cube_grasp_privileged     $GRASP_KEYS
# submit credit_card_grasp  credit_card_grasp_privileged  $GRASP_KEYS
# submit plate_dishrack     plate_dishrack_privileged     $GRASP_KEYS
submit screw_lightbulb    screw_lightbulb_privileged    $SCREW_KEYS
