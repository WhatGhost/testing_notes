"""fused_moe grouped path, large T: GROUP_M swizzle with the best tiles."""

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
model = sys.argv[1]
H, I, E, topk = {"official": (2048, 1408, 64, 8), "35B": (2048, 128, 256, 8)}[model]
UP = [(64, 64, 4, 2), (32, 128, 4, 2), (64, 128, 4, 2), (128, 64, 8, 2), (32, 64, 4, 2), (128, 128, 8, 1)]
DOWN = [(256, 32, 8, 2), (128, 64, 8, 2), (128, 32, 8, 2), (256, 64, 8, 2), (64, 32, 4, 2)]


def bench(fn):
    return do_bench(fn, warmup=3, rep=30, return_mode="median") * 1e3


gen = torch.Generator(device="cuda").manual_seed(0)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
for T in (256, 512, 1024, 2048, 4096):
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
    ids = ti.to(torch.int32).reshape(-1).contiguous()
    wts = tw.float().reshape(-1).contiguous()
    P = T * topk
    for bm in (16, 32):
        sorted_ids, block_expert, nb = r._moe_align(ids, E, bm)
        c1 = torch.empty((2, nb * bm, I), dtype=torch.bfloat16, device="cuda")
        c3 = torch.empty((P, H), dtype=torch.float32, device="cuda")
        tu = []
        for (bn, bk, nw, st), gm in itertools.product(UP, (1, 4, 8, 16)):
            n = triton.cdiv(I, bn)

            def f(bn=bn, bk=bk, nw=nw, st=st, gm=gm, n=n):
                r._grouped_gate_up_kernel[(nb * n,)](
                    x, w1, sorted_ids, block_expert, c1, P, E, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1),
                    n, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, ACT=0, GROUP_M=gm, SPLIT_STORE=False, num_warps=nw,
                    num_stages=st)
            tu.append((bench(f), (bn, bk, nw, st, gm)))
        td = []
        for (bn, bk, nw, st), gm in itertools.product(DOWN, (1, 4, 8, 16)):
            n = triton.cdiv(H, bn)

            def f(bn=bn, bk=bk, nw=nw, st=st, gm=gm, n=n):
                r._grouped_down_kernel[(nb * n,)](
                    c1, w2, sorted_ids, block_expert, wts, c3, P, E, H, I, w2.stride(0), w2.stride(1), n, BLOCK_M=bm,
                    BLOCK_N=bn, BLOCK_K=bk, SPLIT_A=True, GROUP_M=gm, SPLIT_STORE=False, num_warps=nw, num_stages=st)
            td.append((bench(f), (bn, bk, nw, st, gm)))
        tu.sort()
        td.sort()
        print(f"{model} T={T:5d} BM={bm} | up best " + ", ".join(f"{c} {t:.1f}" for t, c in tu[:3]) +
              " | down best " + ", ".join(f"{c} {t:.1f}" for t, c in td[:3]), flush=True)
