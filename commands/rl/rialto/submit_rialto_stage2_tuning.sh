#!/bin/bash
# Submit rialto stage-2 PPO hyperparameter tuning jobs.
# Each run trains for 100M steps on the bottle grasp task.
#
# Usage: bash submit_rialto_stage2_tuning.sh [gpu_type]
#   gpu_type: l40s (default) or l40

GPU=${1:-a40}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

UWLAB_BASE=/gscratch/weirdlab/will/UWLab_Docker
BC_CKPT=logs/rialto/bc/bc_0508_1320/bc_pretrained.pt
DATA_PATH=data_storage/rialto/bottle_grasp
TASK=UW-FrankaLeap-GraspBottleRandomResets-JointAbs-PPO-v0

#--gres=gpu:${GPU}:1 \
# gpu-${GPU} \
submit() {
  local slug=$1; shift
  local extra_args="$@"

  sbatch \
    --job-name="rialto_tune_${slug}" \
    --account=weirdlab \
    --partition=ckpt \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --gpus-per-node=a40:1 \
    --mem=128G \
    --time=20:00:00 \
    --output="${LOG_DIR}/rialto_tune_${slug}_%j.out" \
    --error="${LOG_DIR}/rialto_tune_${slug}_%j.err" \
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
      --task ${TASK} \\
      --bc_checkpoint ${BC_CKPT} \\
      --data_path ${DATA_PATH} \\
      --num_envs 4096 \\
      --n_timesteps 1000000000 \\
      --warmup_rollouts 0 \\
      --critic_warmup_rollouts 10 \\
      --vf_coef 0.5 \\
      --ppo_lr 1e-4 \\
      --headless \\
      --no_eval_video \\
      --wandb_run_name rialto_tune_${slug} \\
      --log_dir logs/rialto/stage2_tuning/${slug} \\
      ${extra_args}
  "

echo "End: \$(date)"
SBATCH_SCRIPT
}

# ── Sweep design ──────────────────────────────────────────────────────────────
#
# Finding: vf_coef and critic_warmup_rollouts are the key axes. bc_coef, log_std,
# and ppo_lr variations did not matter. All best runs: bc_coef=1000, log_std=-3.
#
# Best runs from previous sweep:
#   e1_bc1000_s3_cw30   — critic_warmup_rollouts=30
#   e1_bc1000_s3_cw60   — critic_warmup_rollouts=60
#   e1_bc1000_s3_vf001  — vf_coef=0.01
#   e1_bc1000_s3_vf01   — vf_coef=0.1 (~38% success at 63M steps)
#
# This sweep: cross the two best axes (vf_coef × cw), push boundaries, add lr.
# ──────────────────────────────────────────────────────────────────────────────

# ── 2×2 cross of the two key axes ────────────────────────────────────────────
submit e1_bc1000_s3_vf001_cw60   --n_epochs 1 --bc_coef 1000 --log_std_init -3.0 --vf_coef 0.01  --critic_warmup_rollouts 60
submit e1_bc1000_s3_vf01_cw60    --n_epochs 1 --bc_coef 1000 --log_std_init -3.0 --vf_coef 0.1   --critic_warmup_rollouts 60
submit e1_bc1000_s3_vf001_cw30   --n_epochs 1 --bc_coef 1000 --log_std_init -3.0 --vf_coef 0.01  --critic_warmup_rollouts 30
submit e1_bc1000_s3_vf01_cw30    --n_epochs 1 --bc_coef 1000 --log_std_init -3.0 --vf_coef 0.1   --critic_warmup_rollouts 30

# ── Boundary probes on each axis ──────────────────────────────────────────────
submit e1_bc1000_s3_vf0001       --n_epochs 1 --bc_coef 1000 --log_std_init -3.0 --vf_coef 0.001
submit e1_bc1000_s3_cw120        --n_epochs 1 --bc_coef 1000 --log_std_init -3.0 --critic_warmup_rollouts 120

# ── Best cross + slower lr ────────────────────────────────────────────────────
submit e1_bc1000_s3_vf001_cw60_lr3e5  --n_epochs 1 --bc_coef 1000 --log_std_init -3.0 --vf_coef 0.01 --critic_warmup_rollouts 60 --ppo_lr 3e-5
