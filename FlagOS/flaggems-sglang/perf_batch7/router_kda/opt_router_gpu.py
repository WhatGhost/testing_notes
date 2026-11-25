"""Router headroom on GPU at the benchmark's token counts: launch / read
floors, the split-K pieces, master's scalar kernel with more warps and a
single-kernel GEMV, each against the shipped op."""

import importlib.util
import itertools
import os

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


amd = load("amd_router", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/ops/dsv3_router_gemm.py"))
new = load("new_router", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/dsv3_router_gemm.py"))
g = torch.Generator(device="cuda").manual_seed(0)


def bench(fn):
    return do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3


def inputs(m, e, hidden=7168):
    return (torch.randn(m, hidden, device="cuda", generator=g).bfloat16(),
            torch.randn(e, hidden, device="cuda", generator=g).bfloat16())


@triton.jit
def _empty(OUT):
    tl.store(OUT, 1.0)


@triton.jit
def _read(X, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(pid * BLOCK, N, tl.num_programs(0) * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(X + offs, offs < N, 0).to(tl.float32)
    tl.store(OUT + pid, tl.sum(acc, 0))


@triton.jit
def _gemv(X, WT, OUT, M, E: tl.constexpr, K: tl.constexpr, SX0, SX1, SW0, SW1,
          MB: tl.constexpr, EB: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, MB)
    re = tl.program_id(0) * EB + tl.arange(0, EB)
    rk = tl.arange(0, BK)
    acc = tl.zeros((MB, EB, BK), tl.float32)
    for k in range(0, K, BK):
        kk = k + rk
        x = tl.load(X + rm[:, None] * SX0 + kk[None, :] * SX1, (rm[:, None] < M) & (kk[None, :] < K), 0).to(tl.float32)
        w = tl.load(WT + re[:, None] * SW0 + kk[None, :] * SW1, (re[:, None] < E) & (kk[None, :] < K), 0).to(tl.float32)
        acc += x[:, None, :] * w[None, :, :]
    tl.store(OUT + rm[:, None] * E + re[None, :], tl.sum(acc, 2), (rm[:, None] < M) & (re[None, :] < E))


def gemv(h, w, mb, eb, bk, nw):
    m, k = h.shape
    e = w.shape[0]
    out = torch.empty((m, e), dtype=torch.float32, device="cuda")
    _gemv[(triton.cdiv(e, eb),)](h, w, out, m, e, k, *h.stride(), *w.stride(), mb, eb, bk, num_warps=nw)
    return out


def scalar(h, w, nw):
    m, k = h.shape
    e = w.shape[0]
    out = torch.empty((m, e), dtype=torch.float32, device="cuda")
    new._dsv3_router_gemm_small_batch_kernel[(m, e)](h, w, out, e, k, *h.stride(), *w.stride(), 8192,
                                                    num_warps=nw, enable_fp_fusion=False)
    return out


def dot_only(h, w):
    m, k = h.shape
    e = w.shape[0]
    bm_, be, bk, warps, stages, split = new._config(m, e) or (16, 16, 128, 1, 1, 8)
    part = torch.empty((split, m, e), dtype=torch.float32, device="cuda")
    return lambda: new._router_dot_kernel[(triton.cdiv(m, bm_), triton.cdiv(e, be), split)](
        h, w, part, m, split, e, k, *h.stride(), *w.stride(), bm_, be, bk, num_warps=warps, num_stages=stages)


print("== 1. floors ==")
o = torch.empty(4096, device="cuda")
print(f"one empty kernel {bench(lambda: _empty[(1,)](o)):.1f} us, two {bench(lambda: (_empty[(1,)](o), _empty[(1,)](o))):.1f} us")
for e in (128, 256, 384, 512):
    wt = torch.randn(e, 7168, device="cuda").bfloat16()
    n = wt.numel()
    best = min((bench(lambda: _read[(p,)](wt, o, n, 2048, num_warps=nw)), p, nw) for p in (96, 192, 384, 768) for nw in (4, 8))
    print(f"read the {e}-expert weight once ({n * 2 / 1e6:.2f} MB): {best[0]:.1f} us (programs {best[1]}, warps {best[2]})")

print("\n== 2. shipped op vs its pieces, master scalar kernel with more warps, single-kernel GEMV ==")
shapes = [(m, e) for m in (1, 2, 4, 8, 16) for e in (256, 384)] + [(1, 128), (16, 512), (8, 64)]
for m, e in shapes:
    h, w = inputs(m, e)
    ref = h.float() @ w.float().t()
    t_master = bench(lambda: amd.dsv3_router_gemm(h, w, torch.float32))
    t_new = bench(lambda: new.dsv3_router_gemm(h, w, torch.float32))
    t_dot = bench(dot_only(h, w)) if new._config(m, e) else float("nan")
    sc = min((bench(lambda: scalar(h, w, nw)), nw) for nw in (1, 2, 4, 8))
    mb = triton.next_power_of_2(m)
    res = []
    for eb, bk, nw in itertools.product((1, 2, 4, 8, 16), (128, 256, 512, 1024), (2, 4, 8)):
        if mb * eb * bk > 128 * 32 * nw or mb * eb * bk < 32 * nw * 4:
            continue
        try:
            res.append((bench(lambda: gemv(h, w, mb, eb, bk, nw)), (eb, bk, nw)))
        except Exception:  # noqa: BLE001
            continue
    res.sort()
    tg, cg = res[0]
    err = (gemv(h, w, mb, *cg) - ref).abs().max().item()
    err_m = (amd.dsv3_router_gemm(h, w, torch.float32) - ref).abs().max().item()
    print(f"({m:2d},{e:3d}) master {t_master:5.1f} | shipped {t_new:5.1f} (dot only {t_dot:5.1f}) | scalar best {sc[0]:5.1f} @warps {sc[1]} | "
          f"gemv best {tg:5.1f} @(EB,BK,warps)={cg} err {err:.1e} (master {err_m:.1e}) | next {res[1][0]:.1f}@{res[1][1]}")
