"""fused_moe pair path: tile sweep of the gate/up and token-down kernels."""

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


def bench(fn):
    return do_bench(fn, warmup=5, rep=50, return_mode="median") * 1e3


for model, (H, I, E, topk), Ts in (("official", (2048, 1408, 64, 8), (1, 2, 4, 8)),
                                     ("35B", (2048, 128, 256, 8), (1, 2, 4, 8, 16))):
    gen = torch.Generator(device="cuda").manual_seed(0)
    w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    for T in Ts:
        x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
        tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
        ids = ti.to(torch.int32).reshape(-1).contiguous()
        wts = tw.float().reshape(-1).contiguous()
        P = T * topk
        c1 = torch.empty((P, I), dtype=torch.float32, device="cuda")
        out = torch.empty_like(x)
        up = []
        for bn, bk, nw in itertools.product((16, 32, 64, 128), (16, 32, 64, 128, 256), (1, 2, 4, 8)):
            if bn * bk // (32 * nw) > 256:
                continue

            def f(bn=bn, bk=bk, nw=nw):
                r._pair_gate_up_kernel[(P, triton.cdiv(I, bn))](
                    x, w1, ids, c1, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1), BLOCK_N=bn, BLOCK_K=bk,
                    ACT=0, num_warps=nw)
            try:
                up.append((bench(f), (bn, bk, nw)))
            except Exception:  # noqa: BLE001
                pass
        up.sort()
        dn = []
        for bn, bk, nw in itertools.product((16, 32, 64, 128), (16, 32, 64, 128, 256), (1, 2, 4, 8)):
            if bn * bk // (32 * nw) > 256:
                continue

            def f(bn=bn, bk=bk, nw=nw):
                r._token_down_kernel[(T, triton.cdiv(H, bn))](
                    c1, w2, ids, wts, out, H, I, topk, 1.0, w2.stride(0), w2.stride(1), out.stride(0), BLOCK_N=bn,
                    BLOCK_K=bk, num_warps=nw)
            try:
                dn.append((bench(f), (bn, bk, nw)))
            except Exception:  # noqa: BLE001
                pass
        dn.sort()
        wbytes = P * (2 * I * H + H * I) * 2
        print(f"{model:8s} T={T:3d} | gate/up best " + ", ".join(f"{c} {t:.1f}" for t, c in up[:3]) +
              f" | down best " + ", ".join(f"{c} {t:.1f}" for t, c in dn[:3]) +
              f" | weight bytes {wbytes / 1e6:.1f} MB", flush=True)
