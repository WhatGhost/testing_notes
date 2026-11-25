"""Router candidates interleaved, 5 rounds, medians: shipped, master's scalar
kernel with 2 or 8 warps, and the single-kernel GEMV."""

import importlib.util
import os
import sys

import torch
import triton

sys.path.insert(0, "/tmp")
from opt_router_gpu_lib import gemv, scalar, new, bench  # noqa: E402

g = torch.Generator(device="cuda").manual_seed(0)
print(" M    E | shipped | scalar w2 | scalar w8 |  gemv  | best")
for e in (64, 128, 256, 384, 512):
    for m in (1, 2, 3, 4, 6, 8, 12, 16):
        h = torch.randn(m, 7168, device="cuda", generator=g).bfloat16()
        w = torch.randn(e, 7168, device="cuda", generator=g).bfloat16()
        mb = triton.next_power_of_2(m)
        fns = {
            "shipped": lambda: new.dsv3_router_gemm(h, w, torch.float32),
            "scalar w2": lambda: scalar(h, w, 2),
            "scalar w8": lambda: scalar(h, w, 8),
            "gemv": lambda: gemv(h, w, mb, 2, 1024, 4 if mb <= 4 else 8),
        }
        t = {k: [] for k in fns}
        for _ in range(5):
            for k, fn in fns.items():
                t[k].append(bench(fn))
        med = {k: sorted(v)[2] for k, v in t.items()}
        best = min(med, key=med.get)
        print(f"{m:2d} {e:4d} | {med['shipped']:7.1f} | {med['scalar w2']:9.1f} | {med['scalar w8']:9.1f} | {med['gemv']:6.1f} | {best} {med['shipped'] / med[best]:.2f}x", flush=True)
