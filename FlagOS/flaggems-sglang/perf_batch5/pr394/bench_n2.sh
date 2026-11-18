#!/usr/bin/env bash
# Alternate master and the perf tree, 3 rounds, official benchmark harness.
# usage: bench_perf.sh GPU op [op...]
set -u
GPU=$1
shift
BASE=$(cat /tmp/fgs-master.path)
NEW=$(cat /tmp/fgs-n2.path)
OUT=/tmp/bench_n2
mkdir -p "$OUT"
run() {  # dir tag op
  local dir=$1 tag=$2 op=$3
  (cd "$dir" && HIP_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_bench_${tag}_$GPU \
    timeout 3600 taskset -c 0-31,64-95 /opt/venv/bin/python -m pytest -q -s -p no:cacheprovider \
    "benchmark/test_${op}.py" --level comprehensive --mode kernel --warmup 100 --iter 100 \
    --metrics latency > "$OUT/${op}_${tag}_${round}.log" 2>&1)
}
for round in 1 2 3; do
  for op in "$@"; do
    run "$BASE" master "$op"
    run "$NEW" patched "$op"
    echo "round $round $op done"
  done
done
