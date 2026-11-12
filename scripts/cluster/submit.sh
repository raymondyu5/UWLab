#!/bin/bash
# Push, then sbatch on klone. Usage: submit.sh <sbatch file (repo-relative)> [sbatch args...]
source "$(dirname "$0")/config.sh"
[ $# -ge 1 ] || { echo "usage: $0 <sbatch file> [sbatch args]" >&2; exit 1; }
script=$1; shift
"$(dirname "$0")/push.sh" >/dev/null || exit $?
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && mkdir -p logs/slurm && ICL_ARGS=\"$ICL_ARGS\" sbatch --export=ALL $* $script"
