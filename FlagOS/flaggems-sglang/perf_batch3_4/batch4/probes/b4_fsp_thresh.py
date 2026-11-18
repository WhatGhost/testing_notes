"""fused_softmax_pool: where the _online kernel beats the current HIP dispatch."""

import importlib
import os
import sys

import torch
from triton.testing import do_bench

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

m = importlib.import_module("flaggems_sglang.ops.fused_softmax_pool")
CFG = {64: [(32, 4, 4), (64, 4, 4), (32, 2, 4), (128, 2, 2)],
       16: [(128, 2, 4), (64, 2, 4), (32, 4, 4), (64, 4, 4)],
       4: [(64, 1, 4), (128, 1, 4), (32, 1, 4), (64, 2, 4)]}

for k in (4, 16, 64):
    for bs in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        g = torch.Generator(device="cuda").manual_seed(9)
        x = torch.randn(bs, k, 256, device="cuda", generator=g).to(torch.bfloat16)
        ref = m.fused_softmax_pool(x, 128)
        exact = (x[..., :128].float() * torch.softmax(x[..., 128:].float(), 1)).sum(1)
        e_cur = (ref.float() - exact).abs().max().item()
        t_cur = do_bench(lambda: m.fused_softmax_pool(x, 128), warmup=25, rep=200, return_mode="median") * 1e3
        cells = []
        for bd, q, nw in CFG[k]:
            out = torch.empty_like(ref)

            def run(out=out, bd=bd, q=q, nw=nw):
                m._online[(min(bs, 4096), 128 // bd)](x, out, bs, k, 128, bd, q, 1, num_warps=nw,
                                                      enable_fp_fusion=False)

            run()
            e = (out.float() - exact).abs().max().item()
            t = do_bench(run, warmup=25, rep=200, return_mode="median") * 1e3
            cells.append(f"{(bd, q, nw)} {t:6.1f}{'!' if e > max(2 * e_cur, 1e-2) else ''}")
        print(f"k={k:2d} bs={bs:5d} | current {t_cur:6.1f} (err {e_cur:.1e}) | " + " | ".join(cells), flush=True)
