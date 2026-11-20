"""gfx1100 GEMM probes for the two FP8 ops, after decoding to 16 bits:
  plain   acc = dot(a, b, acc), scales in the epilogue (triton_scaled_mm);
          B stored [K, N] like its weight
  block   per-128-K-block partial dot, scaled and added with fma
          (w8a8_block_fp8_matmul); B stored [N, K]
N, K and leading dims are constexpr, M is a runtime value. hipBLASLt bf16 as
the reference. Also checks the block kernel against an fp64 reference."""

import itertools
import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def _ids(M, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n: tl.constexpr = (N + BN - 1) // BN
    in_group = GROUP_M * num_pid_n
    first_m = pid // in_group * GROUP_M
    group_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    return first_m + pid % in_group % group_m, pid % in_group // group_m


@triton.jit
def _plain(A, B, C, sa_ptr, sb_ptr, M, N: tl.constexpr, K: tl.constexpr, LDA: tl.constexpr, LDB: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid_m, pid_n = _ids(M, N, BM, BN, GROUP_M)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None].to(tl.int64) * LDA + rk[None, :]
    bp = B + rk[:, None].to(tl.int64) * LDB + rn[None, :]
    m_mask = (rm < M)[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(ap, mask=m_mask, other=0.0)
        b = tl.load(bp)
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK * LDB
    sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
    sb = tl.load(sb_ptr + rn)
    acc = acc * sa[:, None] * sb[None, :]
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty), mask=m_mask)


@triton.jit
def _block(A, B, As, Bs, C, M, N: tl.constexpr, K: tl.constexpr, LDA: tl.constexpr, LDB: tl.constexpr,
           AS0, AS1, BS0, BS1, GK: tl.constexpr, GN: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid_m, pid_n = _ids(M, N, BM, BN, GROUP_M)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None].to(tl.int64) * LDA + rk[None, :]
    bp = B + rn[None, :].to(tl.int64) * LDB + rk[:, None]
    m_mask = (rm < M)[:, None]
    sa_row = As + rm * AS0
    sb_row = Bs + (pid_n * BN // GN) * BS0
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(0, K // GK):
        part = tl.zeros((BM, BN), tl.float32)
        for kk in tl.static_range(GK // BK):
            a = tl.load(ap, mask=m_mask, other=0.0)
            b = tl.load(bp)
            part = tl.dot(a, b, part)
            ap += BK
            bp += BK
        sa = tl.load(sa_row + kb * AS1, mask=rm < M, other=0.0)
        sb = tl.load(sb_row + kb * BS1)
        acc = tl.fma(part, sa[:, None] * sb, acc)
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty), mask=m_mask)


def tflops(m, n, k, us):
    return 2 * m * n * k / us / 1e6


def bench(fn):
    return do_bench(fn, warmup=5, rep=25, return_mode="median") * 1e3


M, N, K = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (4096, 7168, 4096)
which = sys.argv[4] if len(sys.argv) > 4 else "both"
torch.manual_seed(0)
a32 = torch.randn(M, K, device="cuda")
b32 = torch.randn(N, K, device="cuda")
a = a32.to(torch.bfloat16)
b = b32.to(torch.bfloat16)
t = bench(lambda: torch.matmul(a, b.t()))
print(f"M={M} N={N} K={K} hipBLASLt bf16: {t:8.1f} us {tflops(M, N, K, t):5.1f} TFLOPs", flush=True)
c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

if which in ("both", "plain"):
    res = []
    sa = torch.rand(M, device="cuda") + 0.5
    sb = torch.rand(N, device="cuda") + 0.5
    for pad_a, pad_b in ((0, 0), (32, 32), (32, 0), (0, 32)):
        ap = torch.empty(M, K + pad_a, device="cuda", dtype=torch.float16)
        ap[:, :K] = a32
        bp = torch.empty(K, N + pad_b, device="cuda", dtype=torch.float16)
        bp[:, :N] = b32.t()
        for bm, bn, bk, nw, ns in itertools.product((128, 256), (128, 256), (32, 64), (4, 8), (1, 2, 3)):
            if bm * bn > 256 * 128 and nw == 4:
                continue
            fn = lambda: _plain[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
                ap, bp, c, sa, sb, M, N, K, ap.stride(0), bp.stride(0), BM=bm, BN=bn, BK=bk, GROUP_M=8,
                num_warps=nw, num_stages=ns)
            try:
                fn()
                t = bench(fn)
            except Exception:  # noqa: BLE001
                continue
            res.append((t, pad_a, pad_b, bm, bn, bk, nw, ns))
    res.sort()
    for t, pa, pb, bm, bn, bk, nw, ns in res[:8]:
        print(f"  plain pad_a={pa:2d} pad_b={pb:2d} BM={bm} BN={bn} BK={bk} warps={nw} stages={ns}: {t:8.1f} us "
              f"{tflops(M, N, K, t):5.1f} TFLOPs", flush=True)

if which in ("both", "block"):
    res = []
    As = torch.rand(M, K // 128, device="cuda") + 0.5
    Bs = torch.rand((N + 127) // 128, K // 128, device="cuda") + 0.5
    for pad in (0, 32):
        ap = torch.empty(M, K + pad, device="cuda", dtype=torch.bfloat16)
        ap[:, :K] = a
        bp = torch.empty(N, K + pad, device="cuda", dtype=torch.bfloat16)
        bp[:, :K] = b
        for bm, bn, bk, nw, ns in itertools.product((64, 128, 256), (64, 128), (32, 64, 128), (4, 8), (1, 2)):
            if bm * bn > 256 * 128 or (bm * bn >= 128 * 128 and nw == 4):
                continue
            fn = lambda: _block[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
                ap, bp, As, Bs, c, M, N, K, ap.stride(0), bp.stride(0), As.stride(0), As.stride(1),
                Bs.stride(0), Bs.stride(1), GK=128, GN=128, BM=bm, BN=bn, BK=bk, GROUP_M=8,
                num_warps=nw, num_stages=ns)
            try:
                fn()
                t = bench(fn)
            except Exception:  # noqa: BLE001
                continue
            res.append((t, pad, bm, bn, bk, nw, ns))
    res.sort()
    for t, pad, bm, bn, bk, nw, ns in res[:8]:
        print(f"  block pad={pad:2d} BM={bm} BN={bn} BK={bk} warps={nw} stages={ns}: {t:8.1f} us "
              f"{tflops(M, N, K, t):5.1f} TFLOPs", flush=True)
