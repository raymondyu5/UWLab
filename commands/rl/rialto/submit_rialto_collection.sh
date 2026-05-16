#!/bin/bash
# Submit rialto rollout collection jobs for all tasks.
# Each job collects 1000 successful episodes (4h limit, l40s GPU).
#
# Usage: bash submit_rialto_collection.sh [gpu_type]
#   gpu_type: l40s (default) or l40

GPU=${1:-l40s}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

UWLAB_BASE=/gscratch/weirdlab/will/UWLab_Docker
RAYMOND_LOGS=/gscratch/weirdlab/raymond/UWLab/logs

submit() {
  local slug=$1 task=$2 ckpt=$3 out_subdir=$4 success_key=$5; shift 5
  local extra_args="$@"

  sbatch \
    --job-name="rialto_collect_${slug}" \
    --account=weirdlab \
    --partition=ckpt \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --gpus-per-node=a40:1 \
    --mem=128G \
    --time=8:00:00 \
    --output="${LOG_DIR}/rialto_${slug}_%j.out" \
    --error="${LOG_DIR}/rialto_${slug}_%j.err" \
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
  --bind /gscratch/weirdlab/raymond:/gscratch/weirdlab/raymond \\
  --pwd /workspace/uwlab \\
  ${UWLAB_BASE}/uw-lab_latest.sif \\
  /bin/bash -c "
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    cd /workspace/uwlab
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1
    ./uwlab.sh -p \\
      scripts/reinforcement_learning/rialto/collect_rialto_rollouts.py \\
      --task ${task} \\
      --checkpoint_dir ${ckpt} \\
      --output_dir data_storage/rialto/${out_subdir} \\
      --target_episodes 1000 \\
      --num_envs 1024 \\
      --success_key ${success_key} \\
      --headless \\
      ${extra_args}
  "

echo "End: \$(date)"
SBATCH_SCRIPT
}

GRASP_PRIV="--privileged_keys manipulated_object_pose target_object_pose contact_obs object_in_tip"
SCREW_PRIV="--privileged_keys ee_pose object_pose rotate_angle contact_obs object_in_tip"

# ── Grasp tasks (success = is_lifted) ─────────────────────────────────────────
submit bottle_grasp \
  UW-FrankaLeap-GraspBottleRandomResets-JointAbs-v0 \
  ${RAYMOND_LOGS}/bc_cfm_pcd_bourbon_0512_absjoint_h16_hist4_extnoise_fast \
  bottle_grasp_privileged \
  is_lifted $GRASP_PRIV

submit cup_grasp \
  UW-FrankaLeap-GraspPinkCupRandomResets-JointAbs-v0 \
  ${RAYMOND_LOGS}/bc_cfm_pcd_cup_pick_0512_absjoint_h16_hist4_extnoise_fast \
  cup_grasp_privileged \
  is_lifted $GRASP_PRIV

submit cube_grasp \
  UW-FrankaLeap-GraspCubeRandomResets-JointAbs-v0 \
  ${RAYMOND_LOGS}/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast \
  cube_grasp_privileged \
  is_lifted $GRASP_PRIV

submit credit_card_grasp \
  UW-FrankaLeap-GraspCreditCard-JointAbs-v0 \
  ${RAYMOND_LOGS}/bc_cfm_pcd_credit_card_0502_absjoint_h16_hist4_extnoise_fast \
  credit_card_grasp_privileged \
  is_lifted $GRASP_PRIV

# # ── Plate in dishrack (success = is_success, ever-true) ───────────────────────
submit plate_dishrack \
  UW-FrankaLeap-PlateInDishRack-JointAbs-v0 \
  ${RAYMOND_LOGS}/bc_cfm_pcd_dishrack_plate_0501_absjoint_h16_hist4_extnoise_fast \
  plate_dishrack_privileged \
  is_success $GRASP_PRIV


# # ── Screw lightbulb (success = cumulative_rotation > 0, .bool() in script) ────
submit screw_lightbulb \
  UW-FrankaLeap-ScrewLightbulb-JointAbs-v0 \
  ${RAYMOND_LOGS}/bc_cfm_pcd_screw_lightbulb_0503_absjoint_h16_hist4_extnoise_fast \
  screw_lightbulb_privileged \
  cumulative_rotation $SCREW_PRIV





############## DEPRECATED TASKS ##################
# # ── Push cube (success = is_success, ever-true) ───────────────────────────────
# submit cube_push \
#   UW-FrankaLeap-PushCubeToPoptartRandomResets-JointAbs-v0 \
#   ${RAYMOND_LOGS}/bc_cfm_pcd_cube_push_all_0416_absjoint_h16_hist4_noextnoise_fast \
#   cube_push \
#   is_success

# # ── Bottle pour (success = is_near_miss) ──────────────────────────────────────
# submit bottle_pour \
#   UW-FrankaLeap-PourBottleRandomResets-JointAbs-v0 \
#   ${RAYMOND_LOGS}/bc_cfm_pcd_bottle_pour_all_0414_absjoint_h16_hist4_noextnoise_fast \
#   bottle_pour \
#   is_near_miss
