"""grouped_gemm tiles around 32 rows per expert with N = 512: the generic op
and rdna3 candidates, cold (do_bench, 3 medians) and warm (back to back,
median of 5 x 50 calls)."""

import importlib
import importlib.util
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
gg = importlib.import_module("flaggems_sglang.ops.grouped_gemm")
spec = importlib.util.spec_from_file_location("gg_p", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/grouped_gemm.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
meta = importlib.import_module("benchmark.test_grouped_gemm")._metadata

CFGS = [(16, 32, 128, 2, 3), (32, 64, 256, 8, 1), (16, 64, 128, 4, 2), (16, 64, 128, 2, 3), (16, 32, 256, 2, 2),
        (32, 32, 128, 4, 2), (32, 32, 256, 4, 1), (16, 128, 128, 4, 1), (32, 64, 128, 4, 2), (32, 64, 128, 8, 2),
        (64, 64, 128, 8, 3), (16, 64, 256, 4, 1), (32, 128, 128, 8, 1)]


def warm(fn, n=50):
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


def both(fn):
    c = sorted(do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3 for _ in range(3))[1]
    w = sorted(warm(fn) for _ in range(5))[2]
    return c, w


shapes = [(1024, 512, 2048, 32), (4096, 512, 2048, 128), (2048, 512, 2048, 64), (768, 512, 2048, 32),
          (1536, 512, 2048, 32), (512, 512, 2048, 32)]
if len(sys.argv) > 1:
    shapes = [shapes[int(i)] for i in sys.argv[1].split(",")]
for m, n, k, e in shapes:
    args = next(iter(importlib.import_module("benchmark.test_grouped_gemm")._input_fn((m, n, k, e), torch.bfloat16, "cuda")))
    gc, gw = both(lambda: gg.grouped_gemm(*args))
    print(f"== ({m},{n},{k},{e}) rows/expert {m / e:.0f}: generic cold {gc:7.1f} warm {gw:7.1f}", flush=True)
    res = []
    for cfg in CFGS:
        new._config = lambda mm, nn, kk, ee, c=cfg: c
        try:
            c, w = both(lambda: new.grouped_gemm(*args))
        except Exception as ex:  # noqa: BLE001
            print(f"   {cfg} failed: {type(ex).__name__}", flush=True)
            continue
        res.append((cfg, c, w))
        print(f"   {str(cfg):22s} cold {c:7.1f} ({gc / c:4.2f}x) warm {w:7.1f} ({gw / w:4.2f}x) worse-of {min(gc / c, gw / w):4.2f}x",
              flush=True)
    best = max(res, key=lambda r: min(gc / r[1], gw / r[2]))
    print(f"   best worse-of: {best[0]} cold {gc / best[1]:.2f}x warm {gw / best[2]:.2f}x", flush=True)
