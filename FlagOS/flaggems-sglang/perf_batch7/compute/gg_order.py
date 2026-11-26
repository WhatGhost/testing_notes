"""rdna3 grouped_gemm: row-block-fastest vs N-block-fastest launch order for
the sweep's best configs, 3 rounds alternated with the generic op, cold
medians, plus warm back-to-back."""

import importlib
import importlib.util
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
gg = importlib.import_module("flaggems_sglang.ops.grouped_gemm")
spec = importlib.util.spec_from_file_location("gg_o", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/grouped_gemm.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
bm_mod = importlib.import_module("benchmark.test_grouped_gemm")


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


CFGS = [(32, 64, 256, 8, 1), (64, 32, 64, 2, 2), (64, 64, 64, 8, 2), (16, 128, 128, 4, 1), (64, 64, 128, 8, 3),
        (32, 256, 64, 8, 1), (16, 32, 128, 2, 3), (64, 256, 64, 8, 1), (128, 128, 64, 8, 2), (64, 128, 64, 4, 2)]
for shape in [(1024, 512, 2048, 8), (1024, 512, 2048, 32), (4096, 512, 2048, 8), (4096, 512, 2048, 32),
              (8192, 512, 2048, 8), (4096, 2048, 768, 128), (512, 512, 2048, 8)]:
    a, b, e, counts, offs, boffs, sched = next(iter(bm_mod._input_fn(shape, torch.bfloat16, "cuda")))
    run_g = lambda: gg.grouped_gemm(a, b, e, counts, offs, boffs, sched)  # noqa: E731
    res = []
    for cfg in CFGS:
        for nf in (False, True):
            new._config = lambda m, n, k, ex, c=cfg + (nf,): c
            try:
                res.append((do_bench(lambda: new.grouped_gemm(a, b, e, counts, offs, boffs, sched), warmup=10, rep=50,
                                     return_mode="median") * 1e3, cfg + (nf,)))
            except Exception:  # noqa: BLE001
                continue
    res.sort()
    best = res[0][1]
    new._config = lambda m, n, k, ex, c=best: c
    run_n = lambda: new.grouped_gemm(a, b, e, counts, offs, boffs, sched)  # noqa: E731
    tg, tn = [], []
    for _ in range(3):
        tg.append(do_bench(run_g, warmup=10, rep=100, return_mode="median") * 1e3)
        tn.append(do_bench(run_n, warmup=10, rep=100, return_mode="median") * 1e3)
    tg, tn = sorted(tg)[1], sorted(tn)[1]
    wg, wn = warm(run_g), warm(run_n)
    rows_best = min((t, c) for t, c in res if not c[5])
    nfast_best = min((t, c) for t, c in res if c[5])
    f = 2 * shape[0] * shape[1] * shape[2]
    print(f"{str(shape):24s} generic {tg:7.1f}/{wg:7.1f} | best {best} {tn:7.1f}/{wn:7.1f} ({tg / tn:4.2f}x cold, {wg / wn:4.2f}x warm, "
          f"{f / tn / 1e6:4.1f} TFLOPs) | rows-fast best {rows_best[1][:5]} {rows_best[0]:.1f}, n-fast best {nfast_best[1][:5]} {nfast_best[0]:.1f}",
          flush=True)
