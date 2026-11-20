"""triton_scaled_mm, small M on gfx1100: A converted from fp8 inside the kernel
(no widen launch), B read as packed fp8 (4 per uint32) with wider column
blocks, interleaved split-K plus a reduce kernel. N, K constexpr."""

import importlib.util
import itertools
import os
import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

T = open("/tmp/fgs-f8.path").read().strip()
spec = importlib.util.spec_from_file_location("gen_tsm", os.path.join(T, "src/flaggems_sglang/ops/triton_scaled_mm.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
_unpack4 = g._unpack4
_e4m3_to_fp16 = g._e4m3_to_fp16


@triton.jit
def _small(A8, B, C, P, sa_ptr, sb_ptr, M, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    split = tl.program_id(1)
    pid_m = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn4 = pid_n * (BN // 4) + tl.arange(0, BN // 4)
    rk = tl.arange(0, BK)
    m_mask = (rm < M)[:, None]
    ap = A8 + rm[:, None].to(tl.int64) * K + rk[None, :]
    bp = B + rk[:, None].to(tl.int64) * (N // 4) + rn4[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(split * BK, K, SPLIT * BK):
        a = _e4m3_to_fp16(tl.load(ap + k, mask=m_mask, other=0).to(tl.uint32))
        b = _unpack4(tl.load(bp + k * (N // 4)))
        acc = tl.dot(a, b, acc)
    rn = pid_n * BN + tl.arange(0, BN)
    if SPLIT == 1:
        sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
        sb = tl.load(sb_ptr + rn)
        acc = acc * sa[:, None] * sb[None, :]
        tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty), mask=m_mask)
    else:
        tl.store(P + (split * M + rm[:, None]).to(tl.int64) * N + rn[None, :], acc, mask=m_mask)


@triton.jit
def _reduce(P, C, sa_ptr, sb_ptr, M, N: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(P + s * M * N + offs, mask=mask, other=0.0)
    acc = acc * tl.load(sa_ptr + offs // N, mask=mask, other=0.0) * tl.load(sb_ptr + offs % N, mask=mask, other=0.0)
    tl.store(C + offs, acc.to(C.dtype.element_ty), mask=mask)


def bench(fn):
    return do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3


K = 4096
SHAPES = [tuple(int(v) for v in x.split("x")) for x in sys.argv[1].split(",")]
for M, N in SHAPES:
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = torch.clamp(0.1 * torch.randn(M, K, dtype=torch.float16, device="cuda", generator=gen), -0.3, 0.3).to(
        torch.float8_e4m3fn)
    b = torch.clamp(0.1 * torch.randn(K, N, dtype=torch.float16, device="cuda", generator=gen), -0.3, 0.3).to(
        torch.float8_e4m3fn)
    sa = 0.1 + 0.05 * torch.rand(M, 1, device="cuda", generator=gen)
    sb = 0.1 + 0.05 * torch.rand(N, 1, device="cuda", generator=gen)
    ref = g.triton_scaled_mm(a, b, sa, sb, torch.float16)
    t_gen = bench(lambda: g.triton_scaled_mm(a, b, sa, sb, torch.float16))
    a8 = a.view(torch.uint8)
    b32 = b.view(torch.uint8).view(torch.uint32)
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    bm = 16 if M <= 16 else 32 if M <= 32 else 64
    res = []
    for bn, bk, nw, ns, sp in itertools.product((64, 128, 256), (64, 128, 256), (4, 8), (1, 2), (1, 2, 4, 8)):
        programs = triton.cdiv(N, bn) * sp * triton.cdiv(M, bm)
        if programs < 64 or programs > 4096 or K % (sp * bk) != 0 or bm * bn * bk > 64 * 128 * 256:
            continue
        part = torch.empty((sp, M, N), device="cuda", dtype=torch.float32) if sp > 1 else out

        def fn(bn=bn, bk=bk, nw=nw, ns=ns, sp=sp, part=part):
            _small[(triton.cdiv(N, bn), sp, triton.cdiv(M, bm))](
                a8, b32, out, part, sa, sb, M, N, K, BM=bm, BN=bn, BK=bk, SPLIT=sp, num_warps=nw, num_stages=ns)
            if sp > 1:
                _reduce[(triton.cdiv(M * N, 1024),)](part, out, sa, sb, M, N, SPLIT=sp, BLOCK=1024, num_warps=4)
        try:
            fn()
            t = bench(fn)
        except Exception:  # noqa: BLE001
            continue
        res.append((t, (bn, bk, nw, ns, sp), fn))
    res.sort(key=lambda r: r[0])
    t, cfg, fn = res[0]
    fn()
    err = (out.float() - ref.float()).abs().max().item()
    top = " ".join(f"{r[1]}:{r[0]:.1f}" for r in res[1:4])
    print(f"M={M:3d} N={N}: generic {t_gen:7.1f} us | small best {t:7.1f} us (no widen) weight {N * K / t / 1e3:4.0f} GB/s "
          f"cfg (BN,BK,warps,stages,split)={cfg} speedup {t_gen / t:4.2f}x | diff {err:.1e} | next {top}", flush=True)
