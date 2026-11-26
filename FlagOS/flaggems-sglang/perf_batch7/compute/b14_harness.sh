#!/usr/bin/env bash
# Official kernel-mode benchmark for chunk_kda_scaled_dot_kkt and grouped_gemm:
# ARCH=none (the generic ops, which are master's on gfx1100) against the rdna3
# ops, alternated for 3 rounds, one op per idle GPU among 0-3.
set -u
cd /tmp/fgs-b.UUl8 || exit 1
OUT=/tmp/b14_harness
mkdir -p "$OUT"
state() { /tmp/idle_gpus.sh 2>&1 | grep "^HIP $1 "; }
bench() {
  local g=$1 cores=$2 op=$3
  if ! state "$g" | grep -q " idle$"; then
    echo "GPU $g busy, $op skipped" >> "$OUT/progress.txt"
    return
  fi
  for r in 1 2 3; do
    for side in master branch; do
      arch=$([ "$side" = master ] && echo none || echo "")
      env ${arch:+ARCH=$arch} HIP_VISIBLE_DEVICES=$g TRITON_CACHE_DIR=/tmp/triton_cache_h_${side}_$g \
        taskset -c "$cores" timeout 3600 /opt/venv/bin/python -m pytest -q -s -p no:cacheprovider \
        "benchmark/test_${op}.py" --level comprehensive --mode kernel --warmup 1000 --iter 100 \
        > "$OUT/${op}_${side}_r${r}.log" 2>&1
      echo "$(date +%H:%M:%S) round $r $op $side rc=$? GPU $g" >> "$OUT/progress.txt"
    done
  done
}
: > "$OUT/progress.txt"
bench 0 0-7,64-71 chunk_kda_scaled_dot_kkt &
bench 1 8-15,72-79 grouped_gemm &
wait
/opt/venv/bin/python /tmp/parse_bench.py "$OUT" chunk_kda_scaled_dot_kkt grouped_gemm
