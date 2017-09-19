#!/bin/bash
SERVER=srv-lab-t-697
if [ ! -z "$1" ]; then
    SERVER="$1"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/.."
cd "$SOURCE_DIR"

rsync -avz "$(pwd)" meierbe8@$SERVER:~/code/tmp
ssh meierbe8@$SERVER "chmod -R 777 ~/code/tmp/ClusterNN" 2>/dev/null
