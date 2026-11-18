"""fused_softmax_pool on gfx1100: current dispatch vs the generic file's own
kernels with other grids, per official shape (do_bench median)."""

import importlib
import os
import sys

import torch
import triton
from triton.testing import do_bench

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

m = importlib.import_module("flaggems_sglang.ops.fused_softmax_pool")

for bs in (1, 64, 1024, 4096):
    for k in (4, 16, 64):
        g = torch.Generator(device="cuda").manual_seed(9)
        x = torch.randn(bs, k, 256, device="cuda", generator=g).to(torch.bfloat16)
        ref = m.fused_softmax_pool(x, 128)
        t_cur = do_bench(lambda: m.fused_softmax_pool(x, 128), warmup=25, rep=200, return_mode="median") * 1e3
        res = []

        def try_(name, fn):
            out = torch.empty_like(ref)
            try:
                fn(out)
                err = (out.float() - ref.float()).abs().max().item()
                if err > 2e-2:
                    return
                t = do_bench(lambda: fn(out), warmup=25, rep=200, return_mode="median") * 1e3
                res.append((t, name))
            except Exception as e:  # noqa: BLE001
                print("ERR", name, str(e)[:100])

        for bd in (8, 16, 32, 64, 128):
            for nw in (1, 2, 4, 8):
                for group in (1, 2, 4):
                    if group > bs:
                        continue
                    try_(f"vector bd{bd} w{nw} g{group}",
                         lambda o, bd=bd, nw=nw, group=group: m._vector[(triton.cdiv(bs, group), 128 // bd)](
                             x, o, bs, k, 128, triton.next_power_of_2(k), bd, False, group,
                             num_warps=nw, enable_fp_fusion=False))
        for bd in (32, 64, 128):
            for q in (1, 2, 4, 8):
                if q > k:
                    continue
                for nw in (1, 2, 4):
                    try_(f"online bd{bd} q{q} w{nw}",
                         lambda o, bd=bd, q=q, nw=nw: m._online[(min(bs, 4096), 128 // bd)](
                             x, o, bs, k, 128, bd, q, 1, num_warps=nw, enable_fp_fusion=False))
        res.sort()
        print(f"bs={bs:5d} k={k:3d} | current {t_cur:6.1f} us | best: " + ", ".join(f"{n} {t:.1f}" for t, n in res[:3]),
              flush=True)
