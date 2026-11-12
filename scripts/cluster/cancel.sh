#!/bin/bash
# Usage: cancel.sh JOBID...
source "$(dirname "$0")/config.sh"
check_ssh
ssh "$REMOTE_HOST" "scancel $*"
