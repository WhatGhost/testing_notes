"""Final timings for the rdna3 chunk_kda_scaled_dot_kkt, grouped_gemm and
absorbed_bmm_concat_cast_q_fp8 against the generic ops: official benchmark
shapes and inputs plus extra shapes, 3 rounds alternated, do_bench (cold)
medians, and warm back-to-back time."""

import importlib
import importlib.util
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
R = f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def warm(fn, n=30):
    for _ in range(3):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


def compare(label, gen, new):
    tg, tn, wg, wn = [], [], [], []
    for _ in range(3):
        tg.append(do_bench(gen, warmup=25, rep=200, return_mode="median") * 1e3)
        tn.append(do_bench(new, warmup=25, rep=200, return_mode="median") * 1e3)
    for _ in range(5):
        wg.append(warm(gen, 50))
        wn.append(warm(new, 50))
    tg, tn = sorted(tg)[1], sorted(tn)[1]
    wg, wn = sorted(wg)[2], sorted(wn)[2]
    print(f"{label:34s} cold generic {tg:8.1f} rdna3 {tn:8.1f} ({tg / tn:4.2f}x) | warm generic {wg:8.1f} rdna3 {wn:8.1f} ({wg / wn:4.2f}x)",
          flush=True)


ops = sys.argv[1].split(",") if len(sys.argv) > 1 else ["kkt", "gg", "ab"]

if "kkt" in ops:
    gen = importlib.import_module("flaggems_sglang.ops.chunk_kda_scaled_dot_kkt").chunk_kda_scaled_dot_kkt
    new = load("kkt_f", f"{R}/chunk_kda_scaled_dot_kkt.py").chunk_kda_scaled_dot_kkt
    bm = importlib.import_module("benchmark.test_chunk_kda_scaled_dot_kkt")
    print("== chunk_kda_scaled_dot_kkt (batch, seq, heads, head_dim)")
    for i, s in enumerate(list(bm.SHAPES) + list(bm.MORE_SHAPES) + [(1, 512, 8, 128), (2, 256, 8, 128), (1, 2048, 4, 64),
                                                                     (4, 1024, 16, 128), (1, 8192, 32, 128)]):
        args = next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))
        compare(f"{'official' if i < 4 else 'extra'} {s}", lambda: gen(*args), lambda: new(*args))

if "gg" in ops:
    gen = importlib.import_module("flaggems_sglang.ops.grouped_gemm").grouped_gemm
    new = load("gg_f", f"{R}/grouped_gemm.py").grouped_gemm
    bm = importlib.import_module("benchmark.test_grouped_gemm")
    print("== grouped_gemm (tokens, N, K, experts)")
    for i, s in enumerate(list(bm.SHAPES) + list(bm.MORE_SHAPES) + [(512, 512, 2048, 8), (2048, 512, 2048, 32), (8192, 512, 2048, 8),
                                                                     (1024, 1536, 2048, 64), (4096, 1536, 2048, 128), (4096, 2048, 768, 128),
                                                                     (4096, 512, 2048, 128), (512, 1536, 2048, 128), (768, 512, 2048, 32)]):
        args = next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))
        compare(f"{'official' if i < 6 else 'extra'} {s}", lambda: gen(*args), lambda: new(*args))

if "ab" in ops:
    gen = importlib.import_module("flaggems_sglang.ops.absorbed_bmm_concat_cast_q_fp8").absorbed_bmm_concat_cast_q_fp8
    new = load("ab_f", f"{R}/absorbed_bmm_concat_cast_q_fp8.py").absorbed_bmm_concat_cast_q_fp8
    bm = importlib.import_module("benchmark.test_absorbed_bmm_concat_cast_q_fp8")
    print("== absorbed_bmm_concat_cast_q_fp8 (tokens, heads)")
    for i, s in enumerate(list(bm.SHAPES) + list(bm.MORE_SHAPES) + [(768, 16), (1536, 16), (2048, 8), (512, 32)]):
        kw = next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))
        compare(f"{'official' if i < 7 else 'extra'} {s}", lambda: gen(**kw), lambda: new(**kw))
