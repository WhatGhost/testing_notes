"""fused_moe pair path with split K: gate/up and down swept separately per split
count, best total per shape."""

import importlib
import itertools
import os
import sys

import torch
import triton
from triton.testing import do_bench

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401

r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
UP = [(16, 256, 8), (16, 128, 8), (32, 128, 8), (16, 256, 4), (32, 256, 8), (64, 128, 8), (16, 64, 4), (32, 64, 4)]
DOWN = [(32, 128, 8), (16, 128, 4), (64, 128, 4), (16, 256, 8), (64, 128, 8), (128, 128, 8), (16, 64, 2), (32, 64, 4)]


def bench(fn):
    return do_bench(fn, warmup=5, rep=40, return_mode="median") * 1e3


for model, (H, I, E, topk), Ts in (("official", (2048, 1408, 64, 8), (1, 2, 4, 8)),
                                     ("35B", (2048, 128, 256, 8), (1, 2, 4, 8, 16, 32))):
    gen = torch.Generator(device="cuda").manual_seed(0)
    w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    for T in Ts:
        x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
        tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
        ids = ti.to(torch.int32).reshape(-1).contiguous()
        wts = tw.float().reshape(-1).contiguous()
        P = T * topk
        out = torch.empty_like(x)
        best = []
        for S in (1, 2, 4, 8):
            part = torch.empty((S, P, 2, I), dtype=torch.float32, device="cuda")
            tu = []
            for bn, bk, nw in UP:
                if S * bk > H:
                    continue

                def f(bn=bn, bk=bk, nw=nw):
                    r._pair_gate_up_kernel[(P, triton.cdiv(I, bn), S)](
                        x, w1, ids, part, P, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1), SPLITS=S,
                        BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
                tu.append((bench(f), (bn, bk, nw)))
            td = []
            for bn, bk, nw in DOWN:
                def f(bn=bn, bk=bk, nw=nw):
                    r._token_down_kernel[(T, triton.cdiv(H, bn))](
                        part, w2, ids, wts, out, P, H, I, topk, 1.0, w2.stride(0), w2.stride(1), out.stride(0),
                        SPLITS=S, ACT=0, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
                td.append((bench(f), (bn, bk, nw)))
            if not tu:
                continue
            u, d = min(tu), min(td)
            best.append((u[0] + d[0], S, u, d))
        best.sort()
        cur = [b for b in best if b[1] == 1][0]
        print(f"{model:8s} T={T:3d} | S=1 best total {cur[0]:7.1f} | best: " +
              ", ".join(f"S={b[1]} up{b[2][1]} {b[2][0]:.1f} + down{b[3][1]} {b[3][0]:.1f} = {b[0]:.1f}" for b in best[:2]),
              flush=True)
