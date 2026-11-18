"""Large-T grouped fused_moe: block configs with and without the hi/lo split,
end to end, plus max error vs the fp32 reference (T <= 1024 only)."""

import importlib.util
import itertools
import json
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
from flaggems_sglang.reference import get_reference  # noqa: E402

spec = importlib.util.spec_from_file_location("old_moe", "/tmp/n3_fused_moe.before_cleanup.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
ref = get_reference("fused_moe")
model = sys.argv[1]
H, I, E, topk = {"official": (2048, 1408, 64, 8), "35B": (2048, 128, 256, 8)}[model]


def bench(fn):
    return do_bench(fn, warmup=5, rep=40, return_mode="median") * 1e3


ups = [(64, 64, 4, 2, 8), (32, 128, 4, 2, 8), (128, 64, 8, 2, 8), (64, 128, 8, 2, 8), (128, 128, 8, 2, 8)]
downs = [(256, 64, 8, 2, 8), (128, 64, 4, 2, 8), (128, 128, 8, 2, 8), (64, 64, 4, 2, 8), (256, 32, 8, 2, 8)]
rows = []
gen = torch.Generator(device="cuda").manual_seed(0)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
for T in (1024, 2048, 4096):
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
    ids = ti.to(torch.int32).reshape(-1).contiguous()
    wts = tw.float().reshape(-1).contiguous()
    out = torch.empty_like(x)
    want = ref(x, w1, w2, (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None), "silu", 1.0).float() \
        if T <= 1024 else None
    best = {}
    for split_a in (True, False):
        for bm, up, down in itertools.product((32, 64, 128), ups, downs):
            if bm == 128 and up[0] == 128 and up[1] == 128:
                continue
            cfg = (bm, up, down)
            try:
                fn = lambda: old._fused_moe_grouped(x, w1, w2, ids, wts, topk, 0, 1.0, out, cfg=cfg, split_a=split_a)
                fn()
                t = bench(fn)
            except Exception as e:  # out of resources etc.
                continue
            err = (out.float() - want).abs().max().item() if want is not None else None
            rows.append(dict(model=model, T=T, split_a=split_a, bm=bm, up=up, down=down, t=t, err=err))
            if split_a not in best or t < best[split_a][0]:
                best[split_a] = (t, cfg, err)
    for split_a, (t, cfg, err) in best.items():
        print(f"{model} T={T} split_a={split_a}: best {t:.1f} us cfg={cfg} err={err}", flush=True)
json.dump(rows, open(f"/tmp/n3_large_sweep_{model}.json", "w"), indent=1)
