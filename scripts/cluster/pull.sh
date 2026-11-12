#!/bin/bash
# Pull results from klone into local logs/. Usage: pull.sh [--no-ckpt] [run-name ...]
# Run names are directories under logs/. With no names, pulls all of logs/ (can be large: use --no-ckpt).
source "$(dirname "$0")/config.sh"
check_ssh
args=(-az --itemize-changes); runs=()
for a in "$@"; do
  case $a in
    --no-ckpt) args+=(--exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt') ;;
    *) runs+=("$a") ;;
  esac
done
mkdir -p "$LOCAL_DIR/logs/slurm"
if [ ${#runs[@]} -eq 0 ]; then
  rsync "${args[@]}" "$REMOTE_HOST:$REMOTE_DIR/logs/" "$LOCAL_DIR/logs/"
else
  for r in "${runs[@]}"; do r=${r%/}; d=$(dirname "$r"); mkdir -p "$LOCAL_DIR/logs/$d"; rsync "${args[@]}" "$REMOTE_HOST:$REMOTE_DIR/logs/$r" "$LOCAL_DIR/logs/$d/"; done
  rsync "${args[@]}" "$REMOTE_HOST:$REMOTE_DIR/logs/slurm/" "$LOCAL_DIR/logs/slurm/"
fi
