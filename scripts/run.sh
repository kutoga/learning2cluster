#!/bin/bash
SCRIPT=$1
LOG=$2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH=$(pwd):$PYTHONPATH
$(which python3) -u $SCRIPT 2>&1 | bash $SCRIPT_DIR/date_log.sh | tee $LOG
