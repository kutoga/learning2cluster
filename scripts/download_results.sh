#!/bin/bash

# Use the following command to only download the small files:
# RSYNC_MAX_SIZE=1m ./download_results.sh

SERVER=srv-lab-t-697
if [ ! -z "$1" ]; then
    SERVER="$1"
fi
TARGET_DIR="/cygdrive/e/tmp/test/$SERVER"
if [ ! -z "$2" ]; then
    TARGET_DIR="$2"
fi
echo $TARGET_DIR

exclude="TIMIT cache"
srv_base_dir="~/data/MT"

exclude_args=""
for exclude_path in $exclude; do
    exclude_args="$exclude_args --exclude MT/$exclude_path"
done

additional_args=""
if [ ! -z "$RSYNC_MAX_SIZE" ]; then
    additional_args="$additional_args --max-size=$RSYNC_MAX_SIZE"
fi

# See: https://unix.stackexchange.com/a/165417/246665
rsync -avz $additional_args $exclude_args --delete meierbe8@$SERVER:~/data/MT --progress --partial --append-verify "$TARGET_DIR"

# TODO:
# Implement a download loop for rsync. Always use a timeout and the
# restart rsync (do this to avoid zhaw network problems, because
# sometimes just the connections hangs (forever)). When rsync no
# longer synchronized something, the process is done and the script
# can be stopped.
# https://unix.stackexchange.com/questions/198563/how-can-i-check-if-rsync-made-any-changes-in-bash
