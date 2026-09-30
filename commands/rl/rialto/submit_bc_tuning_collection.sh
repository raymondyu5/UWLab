#!/bin/bash
# Collect CFM training data for bc_coef tuning cube checkpoints.
#
# Usage: bash submit_bc_tuning_collection.sh [gpu_type|ckpt]
#   (default) a40 — gpu-a40 partition with --gres=gpu:a40:1
#   ckpt       — ckpt partition with --gpus-per-node=a40:1

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
  --bind /gscratch/scrubbed/wph52/rialto_rl:/workspace/uwlab/data_storage/rialto_rl \\
  --bind ${UWLAB_BASE}/tmp/isaaclab:/tmp/isaaclab \\
  --bind ${UWLAB_BASE}/tmp:/tmp \\
  --bind /gscratch/weirdlab/will/UWLab/configs:/workspace/uwlab/configs \\
  --pwd /workspace/uwlab \\
  ${UWLAB_BASE}/uw-lab_latest.sif \\
  /bin/bash -c "
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    unset VIRTUAL_ENV
    cd /workspace/uwlab
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1
    ./uwlab.sh -p \\
        scripts/reinforcement_learning/rialto/collect_rl_rollouts.py \\
        --task ${task} \\
        --checkpoint ${checkpoint} \\
        --output_dir data_storage/rialto_rl/${out_subdir} \\
        --target_episodes 2000 \\
        --num_envs 1024 \\
        --success_key ${success_key} \\
        --headless \\
        ${extra_args}
  "

echo "End: \$(date)"
SBATCH_SCRIPT
}

GRASP_OBS="--obs_keys arm_joint_pos hand_joint_pos manipulated_object_pose target_object_pose contact_obs object_in_tip"

CUBE_BC100000_CKPT=logs/rialto/stage2_tuning/bc100000/stage1_0523_1227/model_000500.zip
CUBE_BC10000_CKPT=logs/rialto/stage2_tuning/bc10000/stage1_0523_1157/model_000500.zip
CUBE_BC100_CKPT=logs/rialto/stage2_tuning/bc100/stage1_0523_1157/model_000500.zip

submit cube_bc100000 \
  UW-FrankaLeap-GraspCube-JointAbs-PPO-Collect-v0 \
  "$CUBE_BC100000_CKPT" cube_grasp_bc_tuning/bc100000 is_lifted $GRASP_OBS

submit cube_bc10000 \
  UW-FrankaLeap-GraspCube-JointAbs-PPO-Collect-v0 \
  "$CUBE_BC10000_CKPT" cube_grasp_bc_tuning/bc10000 is_lifted $GRASP_OBS

submit cube_bc100 \
  UW-FrankaLeap-GraspCube-JointAbs-PPO-Collect-v0 \
  "$CUBE_BC100_CKPT" cube_grasp_bc_tuning/bc100 is_lifted $GRASP_OBS
