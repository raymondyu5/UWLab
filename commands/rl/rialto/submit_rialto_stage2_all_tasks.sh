#!/bin/bash
# Submit rialto stage-2 PPO training for all tasks using the two best HP configs.
#
# Best configs from bottle_grasp tuning sweep:
#   cw60:      vf_coef=0.5,  critic_warmup_rollouts=60
#   vf01_cw30: vf_coef=0.1,  critic_warmup_rollouts=30
#
# Submits 12 jobs: 6 tasks × 2 configs.
#
# Usage: bash submit_rialto_stage2_all_tasks.sh [gpu_type]
#   gpu_type: a40 (default)

GPU=${1:-a40}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

UWLAB_BASE=/gscratch/weirdlab/will/UWLab_Docker

submit() {
  local slug=$1 task=$2 bc_ckpt=$3 data_path=$4; shift 4
  local extra_args="$@"

  sbatch \
    --job-name="rialto_${slug}" \
    --account=weirdlab \
    --partition=gpu-${GPU} \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --gres=gpu:${GPU}:1 \
    --mem=128G \
    --time=30:00:00 \
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
  --pwd /workspace/uwlab \\
  ${UWLAB_BASE}/uw-lab_latest.sif \\
  /bin/bash -c "
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    cd /workspace/uwlab
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1
    ./uwlab.sh -p \\
      scripts/reinforcement_learning/rialto/train_stage2_ppo.py \\
      --task ${task} \\
      --bc_checkpoint ${bc_ckpt} \\
      --data_path ${data_path} \\
      --num_envs 4096 \\
      --n_timesteps 1000000000 \\
      --warmup_rollouts 0 \\
      --ppo_lr 1e-4 \\
      --headless \\
      --no_eval_video \\
      --wandb_run_name rialto_${slug} \\
      --log_dir logs/rialto/stage2/${slug} \\
      ${extra_args}
  "

echo "End: \$(date)"
SBATCH_SCRIPT
}

# ── Task definitions ──────────────────────────────────────────────────────────
#
# Each entry: task_env  bc_checkpoint  data_path  [extra_args]
# screw_lightbulb uses ee_pose instead of manipulated_object_pose as BC obs.
# ─────────────────────────────────────────────────────────────────────────────

BOTTLE_TASK=UW-FrankaLeap-GraspBottleRandomResets-JointAbs-PPO-v0
BOTTLE_BC=logs/rialto/bc/bottle_grasp/bc_0515_0933/bc_best.pt
BOTTLE_DATA=data_storage/rialto/bottle_grasp_privileged

CUP_TASK=UW-FrankaLeap-GraspPinkCupRandomResets-JointAbs-PPO-v0
CUP_BC=logs/rialto/bc/cup_grasp/bc_0515_0933/bc_best.pt
CUP_DATA=data_storage/rialto/cup_grasp_privileged

CUBE_TASK=UW-FrankaLeap-GraspCube-JointAbs-PPO-v0
CUBE_BC=logs/rialto/bc/cube_grasp/bc_0515_0933/bc_best.pt
CUBE_DATA=data_storage/rialto/cube_grasp_privileged

CARD_TASK=UW-FrankaLeap-GraspCreditCard-JointAbs-PPO-v0
CARD_BC=logs/rialto/bc/credit_card_grasp/bc_0515_0933/bc_best.pt
CARD_DATA=data_storage/rialto/credit_card_privileged

PLATE_TASK=UW-FrankaLeap-PlateInDishRack-JointAbs-PPO-v0
PLATE_BC=logs/rialto/bc/plate_dishrack/bc_0515_0933/bc_best.pt
PLATE_DATA=data_storage/rialto/plate_dishrack_privileged

SCREW_TASK=UW-FrankaLeap-ScrewLightbulb-JointAbs-PPO-v0
SCREW_BC=logs/rialto/bc/screw_lightbulb/bc_0513_1759/bc_best.pt
SCREW_DATA=data_storage/rialto/screw_lightbulb_privileged

GRASP_OBS="--obs_keys arm_joint_pos hand_joint_pos manipulated_object_pose target_object_pose contact_obs object_in_tip"
SCREW_OBS="--obs_keys arm_joint_pos hand_joint_pos ee_pose object_pose rotate_angle contact_obs object_in_tip"

# ── Config 2: vf01_cw30 (vf_coef=0.1, critic_warmup=30) ─────────────────────
VF01_CW30_ARGS="--n_epochs 1 --bc_coef 1000 --log_std_init -3.0 --vf_coef 0.1 --critic_warmup_rollouts 30"

submit bottle_grasp_vf01_cw30      "$BOTTLE_TASK" "$BOTTLE_BC" "$BOTTLE_DATA" $VF01_CW30_ARGS $GRASP_OBS
submit cup_grasp_vf01_cw30         "$CUP_TASK"    "$CUP_BC"    "$CUP_DATA"    $VF01_CW30_ARGS $GRASP_OBS

submit cube_grasp_vf01_cw30        "$CUBE_TASK"   "$CUBE_BC"   "$CUBE_DATA"   $VF01_CW30_ARGS $GRASP_OBS
submit credit_card_grasp_vf01_cw30 "$CARD_TASK"   "$CARD_BC"   "$CARD_DATA"   $VF01_CW30_ARGS $GRASP_OBS
submit plate_dishrack_vf01_cw30    "$PLATE_TASK"  "$PLATE_BC"  "$PLATE_DATA"  $VF01_CW30_ARGS $GRASP_OBS
submit screw_lightbulb_vf01_cw30   "$SCREW_TASK"  "$SCREW_BC"  "$SCREW_DATA"  $VF01_CW30_ARGS $SCREW_OBS