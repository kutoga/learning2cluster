#!/bin/bash
SERVER=srv-lab-t-697
if [ ! -z "$1" ]; then
    SERVER="$1"
fi
TARGET_DIR="/cygdrive/e/tmp/test/$SERVER"
if [ ! -z "$2" ]; then
    TARGET_DIR="$2"
fi
echo $TARGET_DIR

rsync -avz --delete meierbe8@$SERVER:~/data/MT "$TARGET_DIR"

