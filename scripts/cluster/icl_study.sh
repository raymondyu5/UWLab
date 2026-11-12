#!/bin/bash
# Checkerboard step-budget study (docs/claude/experiments/2026-11-03-icl-checker-steps-study.md). Usage: icl_study.sh [smoke]
# Submits the dependency chain: collect (D_0, white pool, black pool) -> bc -> {bc_ext white/black, icl}, seeds 1..3.
# SKIP_EXT=1: skip the bc_ext jobs. SKIP_COLLECT=1: the caches already exist, submit only the training jobs. SEEDS="1 3": override the seed list.
source "$(dirname "$0")/config.sh"
SMOKE=${1:-}
CK=logs/rfs/GraspCube-JointAbs_0509_0950_e7765f/model_000150.zip
S=logs/icl/study
if [ "$SMOKE" = smoke ]; then X="--num_envs 16 --eval_tasks 16 --num_iterations 2 --grad_steps 50"; NT=32; POOL=64; SEEDS="1"; P=smoke_
else X="--num_iterations 15"; NT=2000; POOL=3840; SEEDS=${SEEDS:-"1 2 3"}; P=study_; fi
C="--checker --checker_cells 2 --checker_margin 0.04 --rfs_checkpoint $CK --k 2 $X"
[ "$SMOKE" = smoke ] && S=logs/icl/smoke_study
sub() { # dep exp_name args... -> job id
  local dep=$1 name=$2; shift 2
  local d=""; [ -n "$dep" ] && d="--dependency=afterok:$dep"
  ICL_ARGS="--exp_name $P$name $*" "$(dirname "$0")/submit.sh" scripts/cluster/icl_cube.sbatch --parsable $d | tail -1
}
if [ -n "$SKIP_COLLECT" ]; then D0=""; PW=""; PB=""; else
D0=$(sub "" collect_d0 $C --study collect --bc_pcd --n_trajs $NT --ext_region white --seed 100 --cache_out $S/d0_white.pt)
PW=$(sub "" collect_poolw $C --study collect --bc_pcd --n_trajs $POOL --ext_region white --seed 101 --cache_out $S/pool_white.pt)
PB=$(sub "" collect_poolb $C --study collect --bc_pcd --n_trajs $POOL --ext_region black --seed 102 --cache_out $S/pool_black.pt)
echo "collect: d0=$D0 pool_white=$PW pool_black=$PB"
fi
for s in $SEEDS; do
  BC=$(sub "$D0" bc_s$s $C --study bc --bc_pcd --d0_cache $S/d0_white.pt --seed $s)
  [ -n "$SKIP_EXT" ] || EW=$(sub "${D0:+$D0:$PW}" bcextw_s$s $C --study bc_ext --bc_pcd --d0_cache $S/d0_white.pt --ext_pool $S/pool_white.pt --seed $s)
  [ -n "$SKIP_EXT" ] || EB=$(sub "${D0:+$D0:$PB}" bcextb_s$s $C --study bc_ext --bc_pcd --d0_cache $S/d0_white.pt --ext_pool $S/pool_black.pt --seed $s)
  IC=$(sub "$BC" icl_s$s $C --study icl --vision --feedback init_cube --d0_cache $S/d0_white.pt --relabeler logs/icl/$P"bc_s$s"/policy_final.pt --seed $s)
  echo "seed $s: bc=$BC bc_ext_white=$EW bc_ext_black=$EB icl=$IC"
done
