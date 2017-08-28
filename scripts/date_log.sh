#!/bin/bash

# If required: Add some formatting things etc.
DATE_CMD=date

start=$($DATE_CMD)
while read line
do
    echo "$start / $($DATE_CMD) $line"
done
