#!/bin/bash
# Usage: status.sh              queue + last day of jobs
#        status.sh --tail JOBID [N]   last N (default 40) lines of that job's slurm output
source "$(dirname "$0")/config.sh"
check_ssh
if [ "$1" = "--tail" ]; then
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR/logs/slurm && f=\$(ls *-$2.out 2>/dev/null | head -1); [ -n \"\$f\" ] && tail -n ${3:-40} \"\$f\" || echo \"no output yet for job $2\""
else
  ssh "$REMOTE_HOST" 'squeue -u $USER -o "%.10i %.9P %.20j %.8T %.10M %.12r"; echo; sacct -X -S now-1days -o JobID,JobName%24,Partition,State,Elapsed,ExitCode'
fi
