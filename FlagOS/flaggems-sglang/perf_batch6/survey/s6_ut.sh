#!/usr/bin/env bash
# unit tests of the 15 new ops on one gfx1100 card, one log per op
S=$(cat /tmp/fgs-s6.path)
GPU=$1
mkdir -p /tmp/s6_ut
cd $S
for op in $(cat /tmp/s6_ops.txt); do
  HIP_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_s6_$GPU timeout 1200 taskset -c 22-43 \
    /opt/venv/bin/python -m pytest -q -p no:cacheprovider tests/test_$op.py > /tmp/s6_ut/$op.log 2>&1
  echo "$op rc=$? $(grep -E '^[0-9]+ (passed|failed|skipped|error)|passed|failed|error' /tmp/s6_ut/$op.log | tail -1)"
done
