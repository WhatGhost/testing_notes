#!/usr/bin/env bash
# grouped_gemm after the rule change: checks + unit tests (GPU 2), final
# timings (GPU 3), official kernel-mode harness ARCH=none vs rdna3 for 3
# rounds (GPU 0). Each job starts only if its GPU is idle.
set -u
cd /tmp/fgs-b.UUl8 || exit 1
PY=/opt/venv/bin/python3
state() { /tmp/idle_gpus.sh 2>&1 | grep "^HIP $1 "; }
idle() { state "$1" | grep -q " idle$"; }
OUT=/tmp/b14_harness_gg2
mkdir -p "$OUT"
: > "$OUT/progress.txt"
if idle 2; then
  ( export HIP_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton_cache_f5_2
    taskset -c 16-23,80-87 $PY -u /tmp/gg_check.py
    echo "== rdna3"; taskset -c 16-23,80-87 $PY -m pytest -q -p no:cacheprovider tests/test_grouped_gemm.py 2>&1 | tail -n 2
    echo "== none"; ARCH=none taskset -c 16-23,80-87 $PY -m pytest -q -p no:cacheprovider tests/test_grouped_gemm.py 2>&1 | tail -n 2
  ) > /tmp/gg_check5_out.txt 2>&1 &
else echo "GPU 2 busy" > /tmp/gg_check5_out.txt; fi
if idle 3; then
  HIP_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton_cache_f5_3 taskset -c 24-31,88-95 $PY -u /tmp/final_b14.py gg \
    > /tmp/final_gg5_out.txt 2>&1 &
else echo "GPU 3 busy" > /tmp/final_gg5_out.txt; fi
if idle 0; then
  ( for r in 1 2 3; do
      for side in master branch; do
        env $([ "$side" = master ] && echo ARCH=none) HIP_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton_cache_h5_${side} \
          taskset -c 0-7,64-71 timeout 3600 $PY -m pytest -q -s -p no:cacheprovider benchmark/test_grouped_gemm.py \
          --level comprehensive --mode kernel --warmup 1000 --iter 100 > "$OUT/grouped_gemm_${side}_r${r}.log" 2>&1
        echo "$(date +%H:%M:%S) round $r $side rc=$?" >> "$OUT/progress.txt"
      done
    done
    $PY /tmp/parse_bench.py "$OUT" grouped_gemm > "$OUT/summary.txt" 2>&1 ) &
else echo "GPU 0 busy" > "$OUT/summary.txt"; fi
if idle 1; then
  HIP_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton_cache_p32_1 taskset -c 8-15,72-79 $PY -u /tmp/gg_per32.py 5 \
    > /tmp/gg_per32_c.txt 2>&1 &
else echo "GPU 1 busy" > /tmp/gg_per32_c.txt; fi
wait
echo done
