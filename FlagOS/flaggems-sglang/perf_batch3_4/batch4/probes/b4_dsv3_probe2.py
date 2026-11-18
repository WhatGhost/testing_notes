"""dsv3_fused_a_gemm: split count / tile sweep over several (K, N)."""

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


def run(a, b, splits, bn, bk, nw):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    k_split = triton.cdiv(triton.cdiv(k, bk), splits) * bk
    splits = triton.cdiv(k, k_split)
    part = torch.empty((splits, m, n), dtype=torch.float32, device=a.device)
    L._split_k[(triton.cdiv(n, bn) * splits, triton.cdiv(m, 16))](
        a, b, part, m, n, k, *a.stride(), *b.stride(), k_split, splits, 16, bn, bk, k % bk == 0,
        num_warps=nw, num_stages=1)
    L._reduce[(triton.cdiv(m * n, 256),)](part, out, m * n, splits, 256, num_warps=1)
    return out


for k, n in ((7168, 2112), (4096, 1536), (7168, 1536), (2048, 2112), (7168, 576), (7168, 3072)):
    for m in (3, 4, 8, 16):
        g = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
        w = torch.randn(n, k, device="cuda", generator=g).to(torch.bfloat16)
        bm = w.t()
        ref = a.float() @ w.float().T
        o = gen.dsv3_fused_a_gemm(a, bm)
        e_gen = ((o.float() - ref).abs() / (1 + ref.abs())).max().item()
        t_gen = do_bench(lambda: gen.dsv3_fused_a_gemm(a, bm), warmup=25, rep=200, return_mode="median") * 1e3
        res = {}
        for splits in (2, 3, 4, 5, 6, 8):
            for bn, bk, nw in ((32, 256, 4), (64, 256, 4), (64, 128, 4), (128, 256, 8)):
                o = run(a, bm, splits, bn, bk, nw)
                e = ((o.float() - ref).abs() / (1 + ref.abs())).max().item()
                if e > max(2e-2, 2 * e_gen):
                    print("BAD", k, n, m, splits, bn, bk, nw, e)
                    continue
                res[(splits, bn, bk, nw)] = do_bench(lambda: run(a, bm, splits, bn, bk, nw), warmup=25, rep=200,
                                                     return_mode="median") * 1e3
        best = sorted(res, key=res.get)[:3]
        s4 = res.get((4, 64, 256, 4), 0)
        print(f"K={k} N={n} M={m:2d} | generic {t_gen:6.1f} | (4,64,256,4) {s4:6.1f} | best: " +
              ", ".join(f"{c} {res[c]:.1f}" for c in best), flush=True)
