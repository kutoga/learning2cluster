#!/bin/bash

while [ 1 ]; do
    job="$(./get.sh)"
    if [ -z "$job" ]; then
        echo "All jobs are done..."
        break
    fi
    echo "Job command: $job"
    eval $job
done

