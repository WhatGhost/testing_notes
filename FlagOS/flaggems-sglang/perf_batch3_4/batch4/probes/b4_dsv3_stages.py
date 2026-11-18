"""dsv3_fused_a_gemm: interleaved split-K, num_stages 1 vs 2 over N / K / M."""

import importlib
import os
import sys

import torch
import triton
from triton.testing import do_bench

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.dsv3_fused_a_gemm")
g = importlib.import_module("flaggems_sglang.ops.dsv3_fused_a_gemm")


def run(a, b, st):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    part = torch.empty((4, m, n), dtype=torch.float32, device=a.device)
    r._split_k[(triton.cdiv(n, 64) * 4, triton.cdiv(m, 16))](
        a, b, part, m, n, k, *a.stride(), *b.stride(), SPLITS=4, BM=16, BN=64, BK=256, EVEN_K=k % 256 == 0,
        num_warps=4, num_stages=st)
    r._reduce[(triton.cdiv(m * n, 256),)](part, out, m * n, SPLITS=4, BLOCK=256, num_warps=1)
    return out


for k in (5120, 7168, 8192):
    for n in (576, 1536, 2112, 2560, 3072, 4096):
        for m in (4, 16):
            a = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            w = torch.randn(n, k, device="cuda").to(torch.bfloat16)
            bm = w.t()
            tg = min(do_bench(lambda: g.dsv3_fused_a_gemm(a, bm), warmup=25, rep=200, return_mode="median")
                     for _ in range(2)) * 1e3
            t1 = min(do_bench(lambda: run(a, bm, 1), warmup=25, rep=200, return_mode="median")
                     for _ in range(2)) * 1e3
            t2 = min(do_bench(lambda: run(a, bm, 2), warmup=25, rep=200, return_mode="median")
                     for _ in range(2)) * 1e3
            print(f"K={k} N={n:4d} M={m:2d} | master {tg:6.1f} | stages1 {t1:6.1f} ({tg / t1:.2f}x) | "
                  f"stages2 {t2:6.1f} ({tg / t2:.2f}x)", flush=True)
