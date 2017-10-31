#!/bin/bash

# TODO: Use flock for locking
# (But... locking is only for p*ssies, isn't it?)

data_dir=data
mkdir -p "$data_dir"

fname="$(ls $data_dir/|sort|head -1)"
if [ ! -z "$fname" ]; then
    fname="$data_dir/$fname"
    cat "$fname"
    rm "$fname"
fi

