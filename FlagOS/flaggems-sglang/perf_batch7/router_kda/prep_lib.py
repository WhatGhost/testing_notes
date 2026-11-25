"""Shared pieces for the KDA _prepare experiments."""

import importlib.util
import os
import sys

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


new = load("new_kda_lib", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_recurrent_kda.py")
bm = __import__("benchmark.test_fused_recurrent_kda", fromlist=["x"])
_sum_rows = new._sum_rows


def bench(fn):
    return do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3


@triton.jit
def _prep(Q, K, G, QP, KP, D, N, H: tl.constexpr, T: tl.constexpr, KD: tl.constexpr, QS: tl.constexpr,
          KS: tl.constexpr, GS: tl.constexpr, SCALE: tl.constexpr, NORM: tl.constexpr, BK: tl.constexpr,
          R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    hi = rows % H
    ti = rows // H % T
    bi = rows // (T * H)
    kk = tl.arange(0, BK)
    m = (rows[:, None] < N) & (kk[None, :] < KD)
    off = rows[:, None] * KD + kk[None, :]
    q = tl.load(Q + (bi * QS[0] + ti * QS[1] + hi * QS[2])[:, None] + kk[None, :] * QS[3], m, 0).to(tl.float32)
    k = tl.load(K + (bi * KS[0] + ti * KS[1] + hi * KS[2])[:, None] + kk[None, :] * KS[3], m, 0).to(tl.float32)
    g = tl.load(G + (bi * GS[0] + ti * GS[1] + hi * GS[2])[:, None] + kk[None, :] * GS[3], m, 0).to(tl.float32)
    if NORM:
        qd = libdevice.sqrt((_sum_rows(tl.fma(q, q, -0.0), R, BK) + 1.0e-6).to(tl.float64)).to(tl.float32)
        kd = libdevice.sqrt((_sum_rows(tl.fma(k, k, -0.0), R, BK) + 1.0e-6).to(tl.float64)).to(tl.float32)
        q = (q.to(tl.float64) / qd.to(tl.float64)[:, None]).to(tl.float32)
        k = (k.to(tl.float64) / kd.to(tl.float64)[:, None]).to(tl.float32)
    q = tl.fma(q, SCALE, -0.0)
    d = libdevice.exp(g.to(tl.float64)).to(tl.float32)
    tl.store(QP + off, q, m)
    tl.store(KP + off, k, m)
    tl.store(D + off, d, m)


def setup(kw):
    q, k, g = kw["q"], kw["k"], kw["g"]
    b, t, h, kd = k.shape
    bufs = [torch.empty((b, t, h, kd), dtype=torch.float32, device="cuda") for _ in range(3)]

    def shipped():
        new._prepare[(b * t * h,)](q, k, g, *bufs, h, t, kd, q.stride(), k.stride(), g.stride(), kw["scale"],
                                   kw["use_qk_l2norm_in_kernel"], 128, num_warps=4, enable_fp_fusion=False)

    def variant(r, nw):
        n = b * t * h
        return lambda: _prep[(triton.cdiv(n, r),)](q, k, g, *bufs, n, h, t, kd, q.stride(), k.stride(), g.stride(),
                                                  kw["scale"], kw["use_qk_l2norm_in_kernel"], 128, r,
                                                  num_warps=nw, enable_fp_fusion=False)
    return bufs, shipped, variant


def decode_case(b, h=8, kd=128, vd=128, norm=True):
    gen = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *d: torch.randn(*d, device="cuda", generator=gen)  # noqa: E731
    return dict(q=r(b, 1, h, kd).bfloat16(), k=r(b, 1, h, kd).bfloat16(), v=r(b, 1, h, vd).bfloat16(),
                g=-0.05 * torch.rand(b, 1, h, kd, device="cuda", generator=gen),
                beta=torch.rand(b, 1, h, device="cuda", generator=gen).bfloat16(), scale=kd ** -0.5,
                initial_state=0.1 * r(b, h, vd, kd), use_qk_l2norm_in_kernel=norm)
