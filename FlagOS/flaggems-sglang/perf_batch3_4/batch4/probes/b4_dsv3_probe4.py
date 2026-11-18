"""dsv3_fused_a_gemm: interleaved split-K vs contiguous split-K vs generic."""

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


def run_il(a, b, splits, bn=64, bk=256, nw=4):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    splits = min(splits, triton.cdiv(k, bk))
    part = torch.empty((splits, m, n), dtype=torch.float32, device=a.device)
    L._split_k_il[(triton.cdiv(n, bn) * splits, triton.cdiv(m, 16))](
        a, b, part, m, n, k, *a.stride(), *b.stride(), splits, 16, bn, bk, k % bk == 0, num_warps=nw, num_stages=1)
    L._reduce[(triton.cdiv(m * n, 256),)](part, out, m * n, splits, 256, num_warps=1)
    return out


for k in (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192):
    for n in (576, 1536, 2112, 3072):
        for m in (3, 4, 8, 16):
            if n in (576, 3072) and m in (3, 8):
                continue
            g = torch.Generator(device="cuda").manual_seed(0)
            a = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
            w = torch.randn(n, k, device="cuda", generator=g).to(torch.bfloat16)
            bm = w.t()
            ref = a.float() @ w.float().T
            t_gen = do_bench(lambda: gen.dsv3_fused_a_gemm(a, bm), warmup=25, rep=200, return_mode="median") * 1e3
            cells = []
            for s in (2, 3, 4, 5, 6, 8):
                for bn, nw in ((64, 4), (32, 4)):
                    o = run_il(a, bm, s, bn=bn, nw=nw)
                    e = ((o.float() - ref).abs() / (1 + ref.abs())).max().item()
                    t = do_bench(lambda: run_il(a, bm, s, bn=bn, nw=nw), warmup=25, rep=200,
                                 return_mode="median") * 1e3
                    cells.append((t, f"S{s}/{bn}", e))
            d = {c[1]: c[0] for c in cells}
            best = sorted(cells)[:3]
            bad = [c[1] for c in cells if c[2] > 2e-2]
            print(f"K={k:5d} N={n} M={m:2d} | generic {t_gen:6.1f} | S4/64 {d['S4/64']:6.1f} S8/64 {d['S8/64']:6.1f} "
                  f"S4/32 {d['S4/32']:6.1f} | best: " + ", ".join(f"{c[1]} {c[0]:.1f}" for c in best) +
                  (f" BAD {bad}" if bad else ""), flush=True)
