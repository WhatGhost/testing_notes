"""KDA _prepare variants: rows per program, fp32 correctly rounded division in
place of the fp64 one (bitwise equal by the double-rounding theorem), and
what the fp64 exp and the q/k round trip cost."""

import importlib.util
import itertools
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


new = load("new", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_recurrent_kda.py")
bm = __import__("benchmark.test_fused_recurrent_kda", fromlist=["x"])
_sum_rows = new._sum_rows


def bench(fn):
    return do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3


@triton.jit
def _prep(Q, K, G, QP, KP, D, N, H: tl.constexpr, T: tl.constexpr, KD: tl.constexpr, QS: tl.constexpr,
          KS: tl.constexpr, GS: tl.constexpr, SCALE: tl.constexpr, NORM: tl.constexpr, BK: tl.constexpr,
          R: tl.constexpr, DIV32: tl.constexpr, EXP: tl.constexpr, LIGHT: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    hi = rows % H
    ti = rows // H % T
    bi = rows // (T * H)
    kk = tl.arange(0, BK)
    m = (rows[:, None] < N) & (kk[None, :] < KD)
    off = rows[:, None] * KD + kk[None, :]
    g = tl.load(G + (bi * GS[0] + ti * GS[1] + hi * GS[2])[:, None] + kk[None, :] * GS[3], m, 0).to(tl.float32)
    if not LIGHT:
        q = tl.load(Q + (bi * QS[0] + ti * QS[1] + hi * QS[2])[:, None] + kk[None, :] * QS[3], m, 0).to(tl.float32)
        k = tl.load(K + (bi * KS[0] + ti * KS[1] + hi * KS[2])[:, None] + kk[None, :] * KS[3], m, 0).to(tl.float32)
        if NORM:
            qd = libdevice.sqrt((_sum_rows(tl.fma(q, q, -0.0), R, BK) + 1.0e-6).to(tl.float64)).to(tl.float32)
            kd = libdevice.sqrt((_sum_rows(tl.fma(k, k, -0.0), R, BK) + 1.0e-6).to(tl.float64)).to(tl.float32)
            if DIV32:
                q = tl.div_rn(q, qd[:, None])
                k = tl.div_rn(k, kd[:, None])
            else:
                q = (q.to(tl.float64) / qd.to(tl.float64)[:, None]).to(tl.float32)
                k = (k.to(tl.float64) / kd.to(tl.float64)[:, None]).to(tl.float32)
        q = tl.fma(q, SCALE, -0.0)
        tl.store(QP + off, q, m)
        tl.store(KP + off, k, m)
    if EXP == 0:
        d = libdevice.exp(g.to(tl.float64)).to(tl.float32)
    elif EXP == 1:
        d = libdevice.exp(g)
    else:
        d = g
    tl.store(D + off, d, m)


def setup(kw):
    q, k, g = kw["q"], kw["k"], kw["g"]
    b, t, h, kd = k.shape
    bufs = [torch.empty((b, t, h, kd), dtype=torch.float32, device="cuda") for _ in range(3)]

    def shipped():
        new._prepare[(b * t * h,)](q, k, g, *bufs, h, t, kd, q.stride(), k.stride(), g.stride(), kw["scale"],
                                   kw["use_qk_l2norm_in_kernel"], 128, num_warps=4, enable_fp_fusion=False)

    def variant(r, nw, div32=False, exp=0, light=False, out=None):
        o = out or bufs
        n = b * t * h
        return lambda: _prep[(triton.cdiv(n, r),)](q, k, g, *o, n, h, t, kd, q.stride(), k.stride(), g.stride(),
                                                  kw["scale"], kw["use_qk_l2norm_in_kernel"], 128, r, div32, exp,
                                                  light, num_warps=nw, enable_fp_fusion=False)
    return bufs, shipped, variant


def decode_case(b, h=8, kd=128, vd=128):
    gen = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *d: torch.randn(*d, device="cuda", generator=gen)  # noqa: E731
    return dict(q=r(b, 1, h, kd).bfloat16(), k=r(b, 1, h, kd).bfloat16(), v=r(b, 1, h, vd).bfloat16(),
                g=-0.05 * torch.rand(b, 1, h, kd, device="cuda", generator=gen),
                beta=torch.rand(b, 1, h, device="cuda", generator=gen).bfloat16(), scale=kd ** -0.5,
                initial_state=0.1 * r(b, h, vd, kd), use_qk_l2norm_in_kernel=True)


cases = [(f"bench {s}", next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))) for s in [(1, 64), (4, 64), (16, 64), (64, 64)]]
cases += [(f"decode b={b} (L2 norm)", decode_case(b)) for b in (64, 256)]
cases += [("bench (64,64) + L2 norm", dict(next(iter(bm._input_fn((64, 64), torch.bfloat16, "cuda"))), use_qk_l2norm_in_kernel=True))]
grid = [(r, nw) for r, nw in itertools.product((1, 2, 4, 8, 16), (1, 2, 4, 8)) if 32 * nw <= r * 128]

for name, kw in cases:
    bufs, shipped, variant = setup(kw)
    shipped()
    ref = [x.clone() for x in bufs]
    norm = kw["use_qk_l2norm_in_kernel"]
    t0 = bench(shipped)
    rows = min((bench(variant(r, nw)), (r, nw)) for r, nw in grid)
    r, nw = rows[1]
    variant(r, nw)()
    same = all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(ref, bufs))
    line = f"{name:24s} shipped {t0:6.1f} | rows/program best {rows[0]:6.1f} @(R,warps)={rows[1]} bitwise {'ok' if same else 'DIFF'}"
    if norm:
        d32 = min((bench(variant(r2, nw2, div32=True)), (r2, nw2)) for r2, nw2 in grid)
        variant(*d32[1], div32=True)()
        same = all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(ref, bufs))
        line += f" | + fp32 div_rn {d32[0]:6.1f} @{d32[1]} bitwise {'ok' if same else 'DIFF'}"
    line += f" | fp32 exp {bench(variant(r, nw, div32=norm, exp=1)):6.1f} | no exp {bench(variant(r, nw, div32=norm, exp=2)):6.1f}"
    if not norm:
        line += f" | decay only {min(bench(variant(r2, nw2, light=True)) for r2, nw2 in grid):6.1f}"
    print(line, flush=True)

print("\n== fp32 div_rn vs fp64 division over 4M random L2-normalized rows ==")
n = 1 << 22
gen = torch.Generator(device="cuda").manual_seed(3)
kw = dict(q=(torch.randn(n, 1, 1, 128, device="cuda", generator=gen) * torch.exp(4 * torch.randn(n, 1, 1, 1, device="cuda", generator=gen))).bfloat16(),
          k=torch.randn(n, 1, 1, 128, device="cuda", generator=gen).bfloat16(), g=-torch.rand(n, 1, 1, 128, device="cuda", generator=gen) * 10,
          scale=128 ** -0.5, use_qk_l2norm_in_kernel=True)
bufs, shipped, variant = setup(kw)
shipped()
ref = [x.clone() for x in bufs]
variant(4, 4, div32=True)()
print("q, k, decay bitwise:", [torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(ref, bufs)],
      "mismatching words:", [(a.view(torch.int32) != b.view(torch.int32)).sum().item() for a, b in zip(ref, bufs)])
