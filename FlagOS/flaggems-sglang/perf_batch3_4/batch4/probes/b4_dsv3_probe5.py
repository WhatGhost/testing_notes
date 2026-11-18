"""dsv3_fused_a_gemm K=7168 N=2112: interleaved split-K tile sweep."""

import os
import sys

import torch
import triton
from triton.testing import do_bench

sys.path.insert(0, "/tmp")
B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import b4_dsv3_lib as L  # noqa: E402


def run_il(a, b, splits, bn, bk, nw, st):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    part = torch.empty((splits, m, n), dtype=torch.float32, device=a.device)
    L._split_k_il[(triton.cdiv(n, bn) * splits, triton.cdiv(m, 16))](
        a, b, part, m, n, k, *a.stride(), *b.stride(), splits, 16, bn, bk, k % bk == 0, num_warps=nw, num_stages=st)
    L._reduce[(triton.cdiv(m * n, 256),)](part, out, m * n, splits, 256, num_warps=1)
    return out


res = {}
for m in (4, 16):
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(m, 7168, device="cuda", generator=g).to(torch.bfloat16)
    w = torch.randn(2112, 7168, device="cuda", generator=g).to(torch.bfloat16)
    bm = w.t()
    for s in (3, 4, 5, 6):
        for bn, bk, nw in ((32, 256, 4), (64, 256, 4), (64, 128, 4), (64, 512, 8), (128, 256, 8), (64, 256, 8),
                           (32, 512, 4), (128, 128, 4)):
            for st in (1, 2):
                try:
                    t = do_bench(lambda: run_il(a, bm, s, bn, bk, nw, st), warmup=25, rep=200,
                                 return_mode="median") * 1e3
                except Exception:  # noqa: BLE001
                    continue
                res.setdefault((s, bn, bk, nw, st), []).append(t)
res = {c: v for c, v in res.items() if len(v) == 2}
ranked = sorted(res, key=lambda c: sum(res[c]))
for c in ranked[:10]:
    print(c, " ".join(f"{t:.1f}" for t in res[c]))
