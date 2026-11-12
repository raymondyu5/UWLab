#!/usr/bin/env bash
# Report whether the klone ssh ControlMaster is alive and actually passing traffic.
# A master can be "running" yet stale (dead TCP); a timeout'd command catches that.
# Usage: ssh_check.sh [--kill]   (--kill drops a stale master so the next `ssh klone` re-auths)
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

if ! ssh -O check "$REMOTE_HOST" 2>/dev/null; then
  echo "no master running: next 'ssh $REMOTE_HOST' will prompt for Duo"
  exit 1
fi
if timeout 15 ssh -o BatchMode=yes "$REMOTE_HOST" true 2>/dev/null; then
  echo "master alive and responsive"
  exit 0
fi
echo "master STALE (process up, connection dead)"
[[ "${1:-}" == "--kill" ]] && ssh -O exit "$REMOTE_HOST" 2>&1
exit 3
