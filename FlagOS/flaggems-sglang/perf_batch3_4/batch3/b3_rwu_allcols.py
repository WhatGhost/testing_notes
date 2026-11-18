"""recompute_w_u small T on gfx1100: one program per (chunk-head, row block)
walking every column tile with A loaded once, vs the current rdna3 config."""

import importlib
import itertools
import os
import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

B = open("/tmp/fgs-b3.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

gen = importlib.import_module("flaggems_sglang.ops.recompute_w_u")
new = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.recompute_w_u")


@triton.jit
def _allcols(Kp, Vp, Bp, Gp, Ap, Wp, Up, H: tl.constexpr, HG: tl.constexpr, DK: tl.constexpr,
             DV: tl.constexpr, BT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr):
    q = tl.program_id(0)
    h = q % H
    t = q // H * BT
    mr = tl.program_id(1) * BM + tl.arange(0, BM)
    rr = tl.arange(0, BR)
    a = tl.load(Ap + ((t + mr[:, None]) * H + h) * BT + rr[None, :], (mr[:, None] < BT) & (rr[None, :] < BT), 0)
    beta = tl.load(Bp + (t + rr) * H + h, rr < BT, 0).to(tl.float32)
    gate = tl.exp(tl.load(Gp + (t + rr) * H + h, rr < BT, 0).to(tl.float32))
    for j in tl.static_range((DK + BN - 1) // BN):
        nn = j * BN + tl.arange(0, BN)
        k = tl.load(Kp + ((t + rr[:, None]) * HG + h // (H // HG)) * DK + nn[None, :],
                    (rr[:, None] < BT) & (nn[None, :] < DK), 0)
        x = ((k.to(tl.float32) * beta[:, None]) * gate[:, None]).to(k.dtype)
        tl.store(Wp + ((t + mr[:, None]) * H + h) * DK + nn[None, :], tl.dot(a, x),
                 (mr[:, None] < BT) & (nn[None, :] < DK))
    for j in tl.static_range((DV + BN - 1) // BN):
        nn = j * BN + tl.arange(0, BN)
        v = tl.load(Vp + ((t + rr[:, None]) * H + h) * DV + nn[None, :], (rr[:, None] < BT) & (nn[None, :] < DV), 0)
        x = (v.to(tl.float32) * beta[:, None]).to(v.dtype)
        tl.store(Up + ((t + mr[:, None]) * H + h) * DV + nn[None, :], tl.dot(a, x),
                 (mr[:, None] < BT) & (nn[None, :] < DV))


def allcols(k, v, beta, g, A, bm, bn, nw):
    Bsz, T, Hg, K = k.shape
    H, V = v.shape[-2:]
    BT = A.shape[-1]
    w = torch.empty((Bsz, T, H, K), dtype=k.dtype, device=k.device)
    u = torch.empty_like(v)
    q = Bsz * (T // BT) * H
    _allcols[(q, triton.cdiv(BT, bm))](k, v, beta, g, A, w, u, H, Hg, K, V, BT, bm, bn,
                                       max(16, triton.next_power_of_2(BT)), num_warps=nw, num_stages=1,
                                       enable_fp_fusion=False)
    return w, u


for t in (64, 256, 1024):
    g = torch.Generator(device="cuda").manual_seed(9)
    k = torch.randn(1, t, 4, 128, device="cuda", generator=g).to(torch.bfloat16)
    v = torch.randn(1, t, 16, 128, device="cuda", generator=g).to(torch.bfloat16)
    beta = torch.rand(1, t, 16, device="cuda", generator=g).to(torch.bfloat16)
    gc = -torch.rand(1, t, 16, device="cuda", generator=g).cumsum(1).float()
    A = (torch.randn(1, t, 16, 64, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    rw, ru = gen.recompute_w_u(k, v, beta, gc, A, None)
    t_cur = do_bench(lambda: new.recompute_w_u(k, v, beta, gc, A, None), warmup=50, rep=400, return_mode="median") * 1e3
    res = []
    for bm, bn, nw in itertools.product((16, 32, 64), (32, 64, 128), (2, 4, 8)):
        w, u = allcols(k, v, beta, gc, A, bm, bn, nw)
        same = torch.equal(w, rw) and torch.equal(u, ru)
        tm = do_bench(lambda: allcols(k, v, beta, gc, A, bm, bn, nw), warmup=50, rep=400, return_mode="median") * 1e3
        res.append((tm, (bm, bn, nw), same))
    res.sort()
    print(f"T={t:5d} | current rdna3 {t_cur:.1f} us | all-columns best: " +
          ", ".join(f"{c} {tm:.1f}{'' if s else ' (differs)'}" for tm, c, s in res[:4]), flush=True)
