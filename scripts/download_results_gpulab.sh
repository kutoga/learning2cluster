#!/bin/bash

# Use the following command to only download the small files:
# RSYNC_MAX_SIZE=1m ./download_results.sh

SERVER=gpulogin.cloudlab.zhaw.ch
if [ ! -z "$1" ]; then
    SERVER="$1"
fi
TARGET_DIR="/cygdrive/g/tmp/test/$SERVER"
if [ ! -z "$2" ]; then
    TARGET_DIR="$2"
fi
echo $TARGET_DIR

exclude="TIMIT cache test/cache"
if [ -z "$MT_DIR_NAME" ]; then
    MT_DIR_NAME="MT_gpulab"
fi
srv_base_dir="/cluster/home/meierbe8/data/$MT_DIR_NAME"

exclude_args=""
for exclude_path in $exclude; do
    exclude_args="$exclude_args --exclude $MT_DIR_NAME/$exclude_path"
done

additional_args=""
if [ ! -z "$RSYNC_MAX_SIZE" ]; then
    additional_args="$additional_args --max-size=$RSYNC_MAX_SIZE"
fi

# See: https://unix.stackexchange.com/a/165417/246665
rsync -avz --checksum $additional_args $exclude_args --delete --progress --partial --append-verify meierbe8@$SERVER:~/data/$MT_DIR_NAME "$TARGET_DIR"

# TODO:
# Implement a download loop for rsync. Always use a timeout and the
# restart rsync (do this to avoid zhaw network problems, because
# sometimes just the connections hangs (forever)). When rsync no
# longer synchronized something, the process is done and the script
# can be stopped.
# https://unix.stackexchange.com/questions/198563/how-can-i-check-if-rsync-made-any-changes-in-bash
