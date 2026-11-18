"""fused_moe grouped path end to end over candidate configs (GROUP_M=8)."""

import importlib
import itertools
import json
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401

r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
model = sys.argv[1]
H, I, E, topk = {"official": (2048, 1408, 64, 8), "35B": (2048, 128, 256, 8)}[model]
UP = [(64, 64, 4, 2), (32, 128, 4, 2), (64, 128, 4, 2), (128, 128, 8, 1), (32, 64, 4, 2)]
DOWN = [(256, 32, 8, 2), (128, 64, 8, 2), (256, 64, 8, 2), (128, 32, 8, 2), (128, 128, 8, 1), (64, 64, 4, 2)]
Ts = [int(t) for t in sys.argv[2].split(",")]


def bench(fn):
    return do_bench(fn, warmup=3, rep=30, return_mode="median") * 1e3


gen = torch.Generator(device="cuda").manual_seed(0)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
rows = []
for T in Ts:
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
    ids = ti.to(torch.int32).reshape(-1).contiguous()
    wts = tw.float().reshape(-1).contiguous()
    out = torch.empty_like(x)
    res = {}
    for bm, u, d in itertools.product((16, 32), UP, DOWN):
        cfg = (bm, u + (8,), d + (8,))
        try:
            res[str(cfg)] = bench(lambda: r._fused_moe_grouped(x, w1, w2, ids, wts, topk, 0, 1.0, out, cfg=cfg))
        except Exception:  # noqa: BLE001
            pass
    best = sorted(res, key=res.get)[:3]
    rows.append(dict(T=T, res=res))
    print(f"{model} T={T:5d} | " + " | ".join(f"{c} {res[c]:.1f}" for c in best), flush=True)
json.dump(rows, open(f"/tmp/n3_grp_full_{model}_{sys.argv[2].replace(',', '_')}.json", "w"))
