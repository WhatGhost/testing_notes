"""Pick the small-M swap config for w8a8 robustly: a few candidates per (M, N),
each timed 3 times (do_bench medians), against the generic op."""

import importlib.util
import itertools
import os
import statistics

import torch
from triton.testing import do_bench

T = open("/tmp/fgs-f8.path").read().strip()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_w8", os.path.join(T, "src/flaggems_sglang/ops/w8a8_block_fp8_matmul.py"))
new = load("new_w8", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/w8a8_block_fp8_matmul.py"))
FP8 = torch.float8_e4m3fn


def bench(fn):
    return statistics.median(do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3 for _ in range(3))


K = 4096
for n in (1024, 2048, 4096, 7168):
    for m in (1, 2, 4, 8, 16):
        g = torch.Generator(device="cuda").manual_seed(0)
        fmax = torch.finfo(FP8).max
        A = torch.randn(m, K, device="cuda", generator=g).clamp(-fmax, fmax).to(FP8)
        B = torch.randn(n, K, device="cuda", generator=g).clamp(-fmax, fmax).to(FP8)
        As = 1e-2 * torch.rand(m, K // 128, device="cuda", generator=g)
        Bs = 1e-2 * torch.rand(n // 128, K // 128, device="cuda", generator=g)
        tg = bench(lambda: gen.w8a8_block_fp8_matmul(A, B, As, Bs, [128, 128], torch.bfloat16))
        res = []
        for bn, nw, sp in itertools.product((32, 64), (4, 8), (1, 2, 4, 8)):
            new._small_config = lambda m_, n_, bn=bn, nw=nw, sp=sp: (16, bn, 128, nw, sp)
            t = bench(lambda: new._small(A, B, As, Bs, torch.bfloat16))
            res.append((t, (bn, nw, sp)))
        res.sort()
        print(f"N={n} M={m:2d}: generic {tg:7.1f} us | best {res[0][0]:7.1f} us {res[0][1]} ({tg / res[0][0]:4.2f}x) | "
              f"{' '.join(f'{c}:{t:.1f}' for t, c in res[1:5])}", flush=True)
