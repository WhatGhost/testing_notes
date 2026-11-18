"""fused_moe: generic (master) vs rdna3 pair path vs rdna3 grouped path."""

import importlib
import json
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401

r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
g = importlib.import_module("flaggems_sglang.ops.fused_moe")
only = sys.argv[1] if len(sys.argv) > 1 else "all"


def bench(fn):
    return do_bench(fn, warmup=5, rep=50, return_mode="median") * 1e3


rows = []
models = {"official": (2048, 1408, 64, 8), "35B": (2048, 128, 256, 8)}
for model, (H, I, E, topk) in models.items():
    if only != "all" and only != model:
        continue
    gen = torch.Generator(device="cuda").manual_seed(0)
    w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    for T in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
        tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
        to = (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None)
        ids = to[1].reshape(-1).contiguous()
        wts = to[0].reshape(-1).contiguous()
        out = torch.empty_like(x)
        t_gen = bench(lambda: g.fused_moe(x, w1, w2, to))
        t_new = bench(lambda: r.fused_moe(x, w1, w2, to))
        t_pair = bench(lambda: r._fused_moe_pairs(x, w1, w2, ids, wts, topk, 0, 1.0, out)) if T * topk <= 4096 else None
        t_grp = bench(lambda: r._fused_moe_grouped(x, w1, w2, ids, wts, topk, 0, 1.0, out))
        rows.append(dict(model=model, T=T, generic=t_gen, new=t_new, pairs=t_pair, grouped=t_grp))
        tp = f"{t_pair:9.1f}" if t_pair is not None else "        -"
        print(f"{model:8s} T={T:5d} | generic {t_gen:9.1f} | new {t_new:9.1f} ({t_gen / t_new:5.2f}x) | pairs {tp} | "
              f"grouped {t_grp:9.1f}", flush=True)
json.dump(rows, open(f"/tmp/n3_moe_perf_{only}.json", "w"), indent=1)
