#!/bin/bash

# TODO: Use flock for locking
# (But... locking is only for p*ssies, isn't it?)

cmd=$1
data_dir=data
mkdir -p "$data_dir"
echo "Add this job to the queue: $cmd"

while [ 1 ]; do
    fname="$data_dir/$(date +%s)"
    if [ -f $fname ]; then
        sleep 0.5
    else
        echo "$cmd" > $fname
        break
    fi
done

