"""rdna3 kkt vs generic around the size threshold and on more shapes:
3 rounds alternated, cold (do_bench) medians, plus warm back-to-back."""

import importlib.util
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
from flaggems_sglang.ops.chunk_kda_scaled_dot_kkt import chunk_kda_scaled_dot_kkt as gen  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "kkt_t", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/chunk_kda_scaled_dot_kkt.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
bm = __import__("benchmark.test_chunk_kda_scaled_dot_kkt", fromlist=["x"])


def warm(fn, n=50):
    for _ in range(5):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


shapes = list(bm.SHAPES) + list(bm.MORE_SHAPES) + [(1, 512, 8, 128), (2, 256, 8, 128), (1, 2048, 4, 64),
                                                     (4, 1024, 16, 128), (1, 8192, 32, 128), (1, 384, 16, 128)]
for s in shapes:
    args = next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))
    tg, tn = [], []
    for _ in range(3):
        tg.append(do_bench(lambda: gen(*args), warmup=25, rep=200, return_mode="median") * 1e3)
        tn.append(do_bench(lambda: new.chunk_kda_scaled_dot_kkt(*args), warmup=25, rep=200, return_mode="median") * 1e3)
    tg, tn = sorted(tg)[1], sorted(tn)[1]
    wg, wn = warm(lambda: gen(*args)), warm(lambda: new.chunk_kda_scaled_dot_kkt(*args))
    path = "generic" if s[0] * s[1] * s[2] < 4096 else "rdna3"
    print(f"{str(s):20s} [{path:7s}] cold generic {tg:7.1f} new {tn:7.1f} ({tg / tn:4.2f}x) | warm generic {wg:7.1f} new {wn:7.1f} ({wg / wn:4.2f}x)", flush=True)
