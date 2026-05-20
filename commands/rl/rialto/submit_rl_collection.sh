#!/bin/bash
# Collect CFM training data (seg_pc + proprio) by rolling out trained RL policies.
#
# Uses collect_rl_rollouts.py: non-PPO task in RL_MODE (mesh-based seg_pc, no cameras),
# concatenates obs_keys the RL policy was trained on for inference, saves
# seg_pc + proprioception (no privileged state) to data_storage/rialto_rl/.
#
# Run submit_distill_sim.sh after this to train a CFM policy on the collected data.
#
# Usage: bash submit_rl_collection.sh [gpu_type|ckpt]
#   (default) a40 — gpu-a40 partition with --gres=gpu:a40:1
#   ckpt       — ckpt partition with --gpus-per-node=a40:1
#
# Edit the CHECKPOINT variables below to point at each task's best model.

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

submit() {
  local slug=$1 task=$2 checkpoint=$3 out_subdir=$4 success_key=$5; shift 5
  local extra_args="$@"

  sbatch \
    --job-name="rl_collect_${slug}" \
    --account=weirdlab \
    --partition=${PARTITION} \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    ${GPU_ARGS} \
    --mem=64G \
    --time=4:00:00 \
    --output="${LOG_DIR}/rl_collect_${slug}_%j.out" \
    --error="${LOG_DIR}/rl_collect_${slug}_%j.err" \
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
      scripts/reinforcement_learning/rialto/collect_rl_rollouts.py \\
      --task ${task} \\
      --checkpoint ${checkpoint} \\
      --output_dir data_storage/rialto_rl/rl/${out_subdir} \\
      --target_episodes 2000 \\
      --num_envs 1024 \\
      --success_key ${success_key} \\
      --headless \\
      ${extra_args}
  "

echo "End: \$(date)"
SBATCH_SCRIPT
}

# ── Obs key groups (must match what each policy was trained on) ───────────────
GRASP_OBS="--obs_keys arm_joint_pos hand_joint_pos manipulated_object_pose target_object_pose contact_obs object_in_tip"
# Credit card checkpoint (stage1_0514) was trained before the 57D obs update; use 30D keys.
CARD_OBS="--obs_keys arm_joint_pos hand_joint_pos manipulated_object_pose"
SCREW_OBS="--obs_keys arm_joint_pos hand_joint_pos object_pose rotate_angle contact_obs object_in_tip"
SOCCER_OBS="--obs_keys arm_joint_pos hand_joint_pos manipulated_object_pose goal_pose ball_velocity"
CUP_POUR_OBS="--obs_keys arm_joint_pos hand_joint_pos big_cup_pose ball_pose manipulated_object_pose ball_velocity"

# ── Checkpoint paths — edit these to point at the best model per task ─────────
# Pattern: logs/rialto/stage2/<slug>/stage1_MMDD_HHMM/model_XXXXXX.zip
CUBE_CKPT=logs/rialto/stage2/cube_grasp_vf01_cw30/stage1_0516_1537/model_000250.zip
SCREW_CKPT=logs/rialto/stage2/screw_lightbulb_vf01_cw30/stage1_0516_2144/model_000150.zip
BOTTLE_CKPT=logs/rialto/stage2/bottle_grasp_vf01_cw30/stage1_0516_1537/model_000900.zip
CUP_CKPT=logs/rialto/stage2/cup_grasp_vf01_cw30/stage1_0516_1537/model_000150.zip
CARD_CKPT=logs/rialto/stage2/credit_card_grasp_vf01_cw30/stage1_0514_1020/model_000850.zip

## not certain abt this one
PLATE_CKPT=logs/rialto/stage2/plate_dishrack_vf01_cw30/stage1_0516_1537/model_000100.zip
 
# todo
SOCCER_CKPT=logs/rialto/stage2/soccer_push_vf01_cw30/stage1_MMDD_HHMM/model_XXXXXX.zip
POUR_CKPT=logs/rialto/stage2/cup_pour_vf01_cw30/stage1_MMDD_HHMM/model_XXXXXX.zip

# ── Submit jobs (uncomment tasks whose checkpoints are ready) ─────────────────
#
# For a video preview + scatter plot, append: --record_video --enable_cameras --num_envs 8 --target_episodes 5
# Outputs go to <output_dir>/eval/collection_preview.mp4 and scatter_success.png

# submit cube_grasp UW-FrankaLeap-GraspCube-JointAbs-PPO-Collect-v0 "$CUBE_CKPT" cube_grasp_rl_privileged is_lifted $GRASP_OBS --record_video --enable_cameras --num_envs 32 --target_episodes 30

# submit bottle_grasp \
#   UW-FrankaLeap-GraspBottleRandomResets-JointAbs-PPO-Collect-v0 \
#   "$BOTTLE_CKPT" bottle_grasp_rl_privileged is_lifted $GRASP_OBS \
#   --record_video --enable_cameras --num_envs 8 --target_episodes 5

submit cup_grasp \
  UW-FrankaLeap-GraspPinkCupRandomResets-JointAbs-PPO-Collect-v0 \
  "$CUP_CKPT" cup_grasp_rl_privileged is_lifted $GRASP_OBS

# submit credit_card_grasp \
#   UW-FrankaLeap-GraspCreditCard-JointAbs-PPO-Collect-v0 \
#   "$CARD_CKPT" credit_card_grasp_rl_privileged is_lifted $CARD_OBS

# submit screw_lightbulb \
#   UW-FrankaLeap-ScrewLightbulb-JointAbs-PPO-Collect-v0 \
#   "$SCREW_CKPT" screw_lightbulb_rl_privileged cumulative_rotation $SCREW_OBS


# submit plate_dishrack \
#   UW-FrankaLeap-PlateInDishRack-JointAbs-PPO-Collect-v0 \
#   "$PLATE_CKPT" plate_dishrack_rl_privileged is_success $GRASP_OBS

# Soccer and cup_pour have no PPO variant — use the regular task (already has dict obs + seg_pc).
# submit soccer_push \
#   UW-FrankaLeap-SoccerPush-JointAbs-v0 \
#   "$SOCCER_CKPT" soccer_push_rl_privileged is_success $SOCCER_OBS

# submit cup_pour \
#   UW-FrankaLeap-CupPour-JointAbs-v0 \
#   "$POUR_CKPT" cup_pour_rl_privileged is_success $CUP_POUR_OBS
