#!/bin/bash
# Push the local working tree to klone. Usage: push.sh [--dry-run] [--delete]
# --delete removes remote files absent locally (logs/ is protected by the excludes).
source "$(dirname "$0")/config.sh"
check_ssh
args=(-rlzcp --chmod=Dgo+rx,Fgo+r --itemize-changes)
for a in "$@"; do
  case $a in
    --dry-run) args+=(--dry-run) ;;
    --delete) args+=(--delete) ;;
    *) echo "unknown arg $a" >&2; exit 1 ;;
  esac
done
rsync "${args[@]}" "${PUSH_EXCLUDES[@]}" "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"
