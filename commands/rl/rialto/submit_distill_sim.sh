#!/bin/bash
# Submit CFM-PCD distillation training jobs using RL-collected sim rollouts.
# Trains a cfm_pcd_absjoint_h16_hist4_fast policy on data from submit_rl_collection.sh.
#
# Usage: bash submit_distill_sim.sh [gpu_type|ckpt]
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
DATE=$(date +%m%d)

submit() {
  local slug=$1 data_subdir=$2; shift 2
  local extra_args="$@"
  local run_name="cfm_pcd_${slug}_rl_${DATE}_absjoint_h16_hist4_extnoise_fast"

  sbatch \
    --job-name="distill_sim_${slug}" \
    --account=weirdlab \
    --partition=${PARTITION} \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    ${GPU_ARGS} \
    --mem=32G \
    --time=24:00:00 \
    --output="${LOG_DIR}/distill_sim_${slug}_%j.out" \
    --error="${LOG_DIR}/distill_sim_${slug}_%j.err" \
    << SBATCH_SCRIPT
#!/bin/bash
echo "Job ID: \$SLURM_JOB_ID | Node: \$SLURM_NODELIST | Start: \$(date)"

apptainer exec --nv \\
  --fakeroot \\
  --overlay ${UWLAB_BASE}/overlay.img:ro \\
  --bind /gscratch/weirdlab/will/UWLab/source:/workspace/uwlab/source \\
  --bind /gscratch/weirdlab/will/UWLab/scripts:/workspace/uwlab/scripts \\
  --bind /gscratch/weirdlab/will/UWLab/third_party:/workspace/uwlab/third_party \\
  --bind /gscratch/weirdlab/will/UWLab/configs:/workspace/uwlab/configs \\
  --bind ${UWLAB_BASE}/logs:/workspace/uwlab/logs \\
  --bind ${UWLAB_BASE}/data_storage:/workspace/uwlab/data_storage \\
  --bind ${UWLAB_BASE}/tmp:/tmp \\
  --pwd /workspace/uwlab \\
  ${UWLAB_BASE}/uw-lab_latest.sif \\
  /bin/bash -c "
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    cd /workspace/uwlab
    export PYTHONUNBUFFERED=1
    ./uwlab.sh -p \\
      scripts/imitation_learning/cfm_pcd/train_cfm_pcd.py \\
      --config-name train_cfm_pcd_absjoint_h16_hist4_fast \\
      dataset.data_path=/workspace/uwlab/data_storage/rialto_rl/rl/${data_subdir} \\
      ++dataset.noise_extrinsic=true \\
      training.use_wandb=true \\
      training.wandb_run_name=${run_name} \\
      hydra.run.dir=/workspace/uwlab/logs/rialto/distilled_policies/${run_name} \\
      ${extra_args}
  "

echo "End: \$(date)"
SBATCH_SCRIPT
}

# ── Active tasks ───────────────────────────────────────────────────────────────
#submit cube_grasp cube_grasp_rl_privileged
#submit cup_grasp cup_grasp_rl_privileged
submit bottle_grasp      bottle_grasp_rl_privileged
#submit credit_card_grasp credit_card_grasp_rl_privileged

# submit plate_dishrack    plate_dishrack_rl_privileged
# submit screw_lightbulb   screw_lightbulb_rl_privileged
# submit soccer_push       soccer_push_rl_privileged
# submit cup_pour          cup_pour_rl_privileged
