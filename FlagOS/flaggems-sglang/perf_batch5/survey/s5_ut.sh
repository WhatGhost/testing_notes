#!/usr/bin/env bash
# unit tests of the 39 new ops on one gfx1100 card, one log per op
S=$(cat /tmp/fgs-s5.path)
GPU=$1
mkdir -p /tmp/s5_ut
cd $S
for op in $(cat /tmp/s5_ops.txt); do
  HIP_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_s5_$GPU timeout 1200 taskset -c 0-31,64-95 \
    /opt/venv/bin/python -m pytest -q -p no:cacheprovider tests/test_$op.py > /tmp/s5_ut/$op.log 2>&1
  echo "$op rc=$? $(grep -E '^[0-9]+ (passed|failed|skipped|error)|passed|failed|error' /tmp/s5_ut/$op.log | tail -1)"
done
