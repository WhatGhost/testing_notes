"""fused_moe grouped path: per BLOCK_M, best gate/up and down tiles."""

import importlib
import json
import os
import sys

import torch
import triton
from triton.testing import do_bench

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401

r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
UP = [(32, 64, 4, 2), (64, 64, 4, 2), (64, 128, 4, 2), (128, 64, 8, 2), (32, 128, 4, 2), (64, 32, 4, 2),
      (128, 32, 8, 2), (64, 64, 8, 2), (128, 128, 8, 1), (32, 256, 4, 1), (16, 256, 2, 1)]
DOWN = [(64, 64, 4, 2), (128, 64, 8, 2), (64, 128, 4, 2), (256, 64, 8, 2), (128, 32, 8, 2), (64, 32, 4, 2),
        (32, 128, 4, 2), (128, 128, 8, 1), (256, 32, 8, 2)]
model = sys.argv[1]
H, I, E, topk = {"official": (2048, 1408, 64, 8), "35B": (2048, 128, 256, 8)}[model]


def bench(fn):
    return do_bench(fn, warmup=3, rep=30, return_mode="median") * 1e3


gen = torch.Generator(device="cuda").manual_seed(0)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
rows = []
for T in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
    ids = ti.to(torch.int32).reshape(-1).contiguous()
    wts = tw.float().reshape(-1).contiguous()
    P = T * topk
    res = {}
    for bm in (16, 32, 64):
        t_align = bench(lambda: r._moe_align(ids, E, bm))
        sorted_ids, block_expert, nb = r._moe_align(ids, E, bm)
        c1 = torch.empty((nb * bm, I), dtype=torch.float32, device="cuda")
        c3 = torch.empty((P, H), dtype=torch.float32, device="cuda")
        tu = {}
        for bn, bk, nw, st in UP:
            def f(bn=bn, bk=bk, nw=nw, st=st):
                r._grouped_gate_up_kernel[(nb, triton.cdiv(I, bn))](
                    x, w1, sorted_ids, block_expert, c1, P, E, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1),
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, ACT=0, SPLIT_STORE=False, num_warps=nw, num_stages=st)
            try:
                tu[str((bn, bk, nw, st))] = bench(f)
            except Exception:  # noqa: BLE001
                pass
        td = {}
        for bn, bk, nw, st in DOWN:
            def f(bn=bn, bk=bk, nw=nw, st=st):
                r._grouped_down_kernel[(nb, triton.cdiv(H, bn))](
                    c1, w2, sorted_ids, block_expert, wts, c3, P, E, H, I, w2.stride(0), w2.stride(1), BLOCK_M=bm,
                    BLOCK_N=bn, BLOCK_K=bk, SPLIT_A=True, SPLIT_STORE=False, num_warps=nw, num_stages=st)
            try:
                td[str((bn, bk, nw, st))] = bench(f)
            except Exception:  # noqa: BLE001
                pass
        res[bm] = dict(align=t_align, up=tu, down=td)
    rows.append(dict(T=T, res=res))
    summary = []
    for bm, d in res.items():
        bu = min(d["up"], key=d["up"].get)
        bd = min(d["down"], key=d["down"].get)
        summary.append((d["align"] + d["up"][bu] + d["down"][bd], bm, bu, d["up"][bu], bd, d["down"][bd], d["align"]))
    summary.sort()
    print(f"{model} T={T:5d} | " + " | ".join(
        f"BM={s[1]}: {s[0]:.1f} (align {s[6]:.1f}, up {s[2]} {s[3]:.1f}, down {s[4]} {s[5]:.1f})" for s in summary[:2]),
        flush=True)
json.dump(rows, open(f"/tmp/n3_grp_sweep_{model}.json", "w"))
