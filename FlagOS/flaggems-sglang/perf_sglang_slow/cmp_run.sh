#!/usr/bin/env bash
# usage: cmp_run.sh GPU group   (group: extend | silu | moe | gemm | lmhead)
# Alternates master and the perf tree, 3 rounds, with the user's SGLang bench scripts.
set -u
GPU=$1
GROUP=$2
OUT=/tmp/cmp
mkdir -p "$OUT"
source /target/flag-plugin-test/workdir/venv/bin/activate
cd /target/flag-plugin-test/logs/flaggems_sglang
declare -A SRC=([master]=/tmp/fgs-base-check.35O2N6/src [patched]=$(cat /tmp/fgs-perf.path)/src)
for round in 1 2 3; do
  for tree in master patched; do
    out="$OUT/${GROUP}_${tree}_${round}.json"
    env HIP_VISIBLE_DEVICES=$GPU PYTHONPATH=/tmp/pylibs FGS_SRC=${SRC[$tree]} OUT=$out \
      MOE_MS=1,2,4,8,16,32,64,128,256,1024,4096 GEMM_MS=1,2,4,8,16,32,64 ONLY=lmhead \
      TRITON_CACHE_DIR=/tmp/triton_cache_cmp_${tree}_$GPU \
      taskset -c 0-31,64-95 timeout 3600 \
      python $( [ "$GROUP" = lmhead ] && echo bench_more.py || echo "bench_kernels.py $GROUP" ) \
      > "$OUT/${GROUP}_${tree}_${round}.log" 2>&1
    echo "round $round $tree $GROUP exit $?"
  done
done
