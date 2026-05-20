#!/bin/bash
# Evaluate trained CFM PCD distillation policies in simulation.
# Produces scatter/success plots, metrics JSON, and optional video via PPOEvalCallback.
#
# Usage: bash submit_eval_cfm.sh [gpu_type|ckpt]
#   (default) a40 — gpu-a40 partition with --gres=gpu:a40:1
#   ckpt       — ckpt partition with --gpus-per-node=a40:1
#
# Edit the CHECKPOINT variables below to point to each task's trained CFM policy dir.

if [ "${1:-}" = "ckpt" ]; then
    GPU=a40
    PARTITION=ckpt
    GPU_ARGS="--gpus-per-node=a40:1"
else
    GPU=${1:-a40}
    PARTITION=gpu-${GPU}
    GPU_ARGS="--gres=gpu:${GPU}:1"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

UWLAB_BASE=/gscratch/weirdlab/will/UWLab_Docker

# ── Set checkpoints here ───────────────────────────────────────────────────
CUBE_CKPT=logs/rialto/distilled_policies/cfm_pcd_cube_grasp_rl_0518_absjoint_h16_hist4_extnoise_fast
CUP_CKPT=logs/rialto/distilled_policies/cfm_pcd_cup_grasp_rl_0518_absjoint_h16_hist4_extnoise_fast
BOTTLE_CKPT=logs/rialto/distilled_policies/cfm_pcd_bottle_grasp_rl_0518_absjoint_h16_hist4_extnoise_fast
# ──────────────────────────────────────────────────────────────────────────

submit() {
  local slug=$1 task=$2 checkpoint=$3; shift 3
  local extra_args="$@"

  sbatch \
    --job-name="eval_cfm_${slug}" \
    --account=weirdlab \
    --partition=${PARTITION} \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    ${GPU_ARGS} \
    --mem=64G \
    --time=4:00:00 \
    --output="${LOG_DIR}/eval_cfm_${slug}_%j.out" \
    --error="${LOG_DIR}/eval_cfm_${slug}_%j.err" \
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
  --bind /gscratch/weirdlab/will/UWLab/configs:/workspace/uwlab/configs \\
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
  --pwd /workspace/uwlab \\
  ${UWLAB_BASE}/uw-lab_latest.sif \\
  /bin/bash -c "
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    cd /workspace/uwlab
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1
    ./uwlab.sh -p \\
      scripts/eval/eval_cfm_rialto.py \\
      --task ${task} \\
      --checkpoint /workspace/uwlab/${checkpoint} \\
      --num_envs 256 \\
      --num_trials 20 \\
      --headless \\
      ${extra_args}
  "

echo "End: \$(date)"
SBATCH_SCRIPT
}

# ── Active evaluations ─────────────────────────────────────────────────────
submit cube_grasp \
  UW-FrankaLeap-GraspCube-JointAbs-PPO-Collect-v0 \
  ${CUBE_CKPT} \
  "--record_video --enable_cameras --num_envs 8"

submit cup_grasp \
  UW-FrankaLeap-GraspPinkCupRandomResets-JointAbs-PPO-Collect-v0 \
  ${CUP_CKPT} \
  "--record_video --enable_cameras --num_envs 8"

submit bottle_grasp \
  UW-FrankaLeap-GraspBottleRandomResets-JointAbs-PPO-Collect-v0 \
  ${BOTTLE_CKPT} \
  "--record_video --enable_cameras --num_envs 8"

# submit credit_card_grasp \
#   UW-FrankaLeap-GraspCreditCard-JointAbs-PPO-Collect-v0 \
#   ${CREDIT_CARD_CKPT}

# submit plate_dishrack \
#   UW-FrankaLeap-PlateInDishRack-JointAbs-PPO-Collect-v0 \
#   ${PLATE_CKPT}

# submit screw_lightbulb \
#   UW-FrankaLeap-ScrewLightbulb-JointAbs-PPO-Collect-v0 \
#   ${SCREW_CKPT}
