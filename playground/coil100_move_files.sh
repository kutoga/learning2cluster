#!/usr/bin/env bash
for i in $(seq 1 100); do l=$(printf "%03d" $i); mkdir $l; mv obj${i}_* $l; done;

