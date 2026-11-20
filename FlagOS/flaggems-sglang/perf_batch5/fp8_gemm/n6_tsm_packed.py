"""triton_scaled_mm, small and medium M on gfx1100: A widened to fp16 once, B
read as packed fp8 (4 per uint32) and converted in registers inside the GEMM.
N, K, leading dims constexpr; optional interleaved split-K with a reduce
kernel for small grids. Swept per (M, N); generic as the reference."""

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


@triton.jit
def _packed(A, B, C, P, sa_ptr, sb_ptr, M, N: tl.constexpr, K: tl.constexpr, LDA: tl.constexpr, LDB4: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, SPLIT: tl.constexpr):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n: tl.constexpr = (N + BN - 1) // BN
    in_group = GROUP_M * num_pid_n
    first_m = pid // in_group * GROUP_M
    group_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + pid % in_group % group_m
    pid_n = pid % in_group // group_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn4 = pid_n * (BN // 4) + tl.arange(0, BN // 4)
    rk = tl.arange(0, BK)
    m_mask = (rm < M)[:, None]
    ap = A + rm[:, None].to(tl.int64) * LDA + rk[None, :]
    bp = B + rk[:, None].to(tl.int64) * LDB4 + rn4[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(split * BK, K, SPLIT * BK):
        a = tl.load(ap + k, mask=m_mask, other=0.0)
        b = _unpack4(tl.load(bp + k * LDB4))
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
    m = offs // N
    n = offs % N
    acc = acc * tl.load(sa_ptr + m, mask=mask, other=0.0) * tl.load(sb_ptr + n, mask=mask, other=0.0)
    tl.store(C + offs, acc.to(C.dtype.element_ty), mask=mask)


def bench(fn):
    return do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3


K = 4096
SHAPES = [tuple(int(v) for v in x.split('x')) for x in sys.argv[1].split(',')]
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
    a16 = a.to(torch.float16)
    b32 = b.view(torch.uint8).view(torch.uint32)
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    if M <= 16:
        space = itertools.product((16,), (32, 64), (128, 256), (4, 8), (1, 2), (1, 4))
    elif M <= 64:
        space = itertools.product((64,), (32, 64, 128), (64, 128), (4, 8), (1, 2), (1, 2))
    else:
        space = itertools.product((64, 128), (64, 128), (64, 128), (4, 8), (1, 2), (1,))
    best = None
    for bm, bn, bk, nw, ns, sp in space:
        programs = triton.cdiv(M, bm) * triton.cdiv(N, bn) * sp
        if (sp > 1 and programs > 4096) or bm * bn * bk > 128 * 128 * 128 or K // (sp * bk) < 1:
            continue
        part = torch.empty((sp, M, N), device="cuda", dtype=torch.float32) if sp > 1 else out

        def fn(bm=bm, bn=bn, bk=bk, nw=nw, ns=ns, sp=sp, part=part):
            _packed[(triton.cdiv(M, bm) * triton.cdiv(N, bn), sp)](
                a16, b32, out, part, sa, sb, M, N, K, K, N // 4, BM=bm, BN=bn, BK=bk, GROUP_M=8, SPLIT=sp,
                num_warps=nw, num_stages=ns)
            if sp > 1:
                _reduce[(triton.cdiv(M * N, 1024),)](part, out, sa, sb, M, N, SPLIT=sp, BLOCK=1024, num_warps=4)
        try:
            fn()
            t = bench(fn)
        except Exception:  # noqa: BLE001
            continue
        if best is None or t < best[0]:
            best = (t, (bm, bn, bk, nw, ns, sp), fn)
    best[2]()
    err = (out.float() - ref.float()).abs().max().item()
    t_a = bench(lambda: a.to(torch.float16))
    fl = 2 * M * N * K
    print(f"M={M:4d} N={N}: generic {t_gen:8.1f} us | packed best {best[0]:8.1f} us (+widen A {t_a:5.1f}) "
          f"{fl / best[0] / 1e6:5.1f} TF, weight {N * K / best[0] / 1e3:4.0f} GB/s, cfg (BM,BN,BK,warps,stages,split)="
          f"{best[1]} | speedup {t_gen / (best[0] + t_a):4.2f}x | max diff vs generic {err:.2e}", flush=True)
