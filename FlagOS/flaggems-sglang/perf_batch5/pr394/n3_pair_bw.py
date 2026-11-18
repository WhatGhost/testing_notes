"""Pair path at small T: time and effective weight bandwidth of each kernel."""

import importlib
import os
import sys

import torch
import triton
from triton.testing import do_bench

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")


def bench(fn):
    return do_bench(fn, warmup=10, rep=100, return_mode="median") * 1e3


for name, (H, I, E, topk) in {"official": (2048, 1408, 64, 8), "35B": (2048, 128, 256, 8)}.items():
    gen = torch.Generator(device="cuda").manual_seed(0)
    w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    for T in (1, 2, 4, 8):
        x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
        tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
        ids = ti.to(torch.int32).reshape(-1).contiguous()
        wts = tw.float().reshape(-1).contiguous()
        out = torch.empty_like(x)
        P = T * topk
        n_exp = ids.unique().numel()
        splits, up, down = r._pair_config(T, P, H, I)
        part = torch.empty((splits, P, 2, I), dtype=torch.float32, device="cuda")

        def gate_up():
            r._pair_gate_up_kernel[(P, triton.cdiv(I, up[0]), splits)](
                x, w1, ids, part, P, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1),
                SPLITS=splits, BLOCK_N=up[0], BLOCK_K=up[1], num_warps=up[2])

        def down_k():
            r._token_down_kernel[(T, triton.cdiv(H, down[0]))](
                part, w2, ids, wts, out, P, H, I, topk, 1.0, w2.stride(0), w2.stride(1), out.stride(0),
                SPLITS=splits, ACT=0, BLOCK_N=down[0], BLOCK_K=down[1], num_warps=down[2])

        t_up, t_down = bench(gate_up), bench(down_k)
        t_all = bench(lambda: r._fused_moe_pairs(x, w1, w2, ids, wts, topk, 0, 1.0, out))
        # bytes actually needed: every distinct expert once (L2 may serve repeats)
        b_up = n_exp * 2 * I * H * 2
        b_down = n_exp * H * I * 2
        print(f"{name:8s} T={T} experts={n_exp:3d} | gate/up {t_up:7.1f} us {b_up / t_up / 1e3:6.0f} GB/s "
              f"(per pair {P * 2 * I * H * 2 / t_up / 1e3:6.0f}) | down {t_down:7.1f} us {b_down / t_down / 1e3:6.0f} GB/s "
              f"(per pair {P * H * I * 2 / t_down / 1e3:6.0f}) | total {t_all:7.1f}", flush=True)
