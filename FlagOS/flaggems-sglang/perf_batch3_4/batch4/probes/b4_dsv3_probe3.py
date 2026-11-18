"""dsv3_fused_a_gemm: K blocks per split vs K (power-of-two spacing check)."""

import importlib
import os
import sys

import torch
import triton
from triton.testing import do_bench

sys.path.insert(0, "/tmp")
B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401
import b4_dsv3_lib as L  # noqa: E402

gen = importlib.import_module("flaggems_sglang.ops.dsv3_fused_a_gemm")


def run(a, b, per_split, bn=64, bk=256, nw=4):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    k_split = per_split * bk
    splits = triton.cdiv(k, k_split)
    part = torch.empty((splits, m, n), dtype=torch.float32, device=a.device)
    L._split_k[(triton.cdiv(n, bn) * splits, triton.cdiv(m, 16))](
        a, b, part, m, n, k, *a.stride(), *b.stride(), k_split, splits, 16, bn, bk, k % bk == 0,
        num_warps=nw, num_stages=1)
    L._reduce[(triton.cdiv(m * n, 256),)](part, out, m * n, splits, 256, num_warps=1)
    return out


for k in (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192):
    for n in (1536, 2112):
        for m in (4, 16):
            g = torch.Generator(device="cuda").manual_seed(0)
            a = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
            w = torch.randn(n, k, device="cuda", generator=g).to(torch.bfloat16)
            bm = w.t()
            t_gen = do_bench(lambda: gen.dsv3_fused_a_gemm(a, bm), warmup=25, rep=200, return_mode="median") * 1e3
            cells = []
            for per in range(1, 9):
                if per * 256 >= k:
                    break
                t = do_bench(lambda: run(a, bm, per), warmup=25, rep=200, return_mode="median") * 1e3
                cells.append(f"{per}:{t:6.1f}")
            print(f"K={k:5d} N={n} M={m:2d} | generic {t_gen:6.1f} | blocks per split -> " + " ".join(cells),
                  flush=True)
