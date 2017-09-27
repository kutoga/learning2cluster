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

# TODO:
# Implement a download loop for rsync. Always use a timeout and the
# restart rsync (do this to avoid zhaw network problems, because
# sometimes just the connections hangs (forever)). When rsync no
# longer synchronized something, the process is done and the script
# can be stopped.
# https://unix.stackexchange.com/questions/198563/how-can-i-check-if-rsync-made-any-changes-in-bash
