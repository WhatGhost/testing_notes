"""w8a8 block FP8 GEMM core on gfx1100 with the accumulator split in two
column halves: per 128-K block, two partial dots (BM x BN/2 each) share the A
tile, each scaled and fma-added into its own half. One partial instead of a
full-size second accumulator. B stored [N, K] as given."""

import itertools
import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def _halves(A, B, As, Bs, C, M, N: tl.constexpr, K: tl.constexpr, LDA: tl.constexpr, LDB: tl.constexpr,
            AS0, AS1, BS0, BS1, GK: tl.constexpr, GN: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    HALF: tl.constexpr = BN // 2
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n: tl.constexpr = (N + BN - 1) // BN
    in_group = GROUP_M * num_pid_n
    first_m = pid // in_group * GROUP_M
    group_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + pid % in_group % group_m
    pid_n = pid % in_group // group_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn0 = pid_n * BN + tl.arange(0, HALF)
    rn1 = rn0 + HALF
    rk = tl.arange(0, BK)
    m_mask = (rm < M)[:, None]
    ap = A + rm[:, None].to(tl.int64) * LDA + rk[None, :]
    bp0 = B + rn0[None, :].to(tl.int64) * LDB + rk[:, None]
    bp1 = B + rn1[None, :].to(tl.int64) * LDB + rk[:, None]
    sa_row = As + rm * AS0
    sb_row0 = Bs + (pid_n * BN // GN) * BS0
    sb_row1 = Bs + ((pid_n * BN + HALF) // GN) * BS0
    acc0 = tl.zeros((BM, HALF), tl.float32)
    acc1 = tl.zeros((BM, HALF), tl.float32)
    for kb in range(0, K // GK):
        sa = tl.load(sa_row + kb * AS1, mask=rm < M, other=0.0)
        part = tl.zeros((BM, HALF), tl.float32)
        for kk in tl.static_range(GK // BK):
            a = tl.load(ap + kk * BK, mask=m_mask, other=0.0)
            part = tl.dot(a, tl.load(bp0 + kk * BK), part)
        acc0 = tl.fma(part, sa[:, None] * tl.load(sb_row0 + kb * BS1), acc0)
        part = tl.zeros((BM, HALF), tl.float32)
        for kk in tl.static_range(GK // BK):
            a = tl.load(ap + kk * BK, mask=m_mask, other=0.0)
            part = tl.dot(a, tl.load(bp1 + kk * BK), part)
        acc1 = tl.fma(part, sa[:, None] * tl.load(sb_row1 + kb * BS1), acc1)
        ap += GK
        bp0 += GK
        bp1 += GK
    cp = C + rm[:, None].to(tl.int64) * N
    tl.store(cp + rn0[None, :], acc0.to(C.dtype.element_ty), mask=m_mask)
    tl.store(cp + rn1[None, :], acc1.to(C.dtype.element_ty), mask=m_mask)


def tflops(m, n, k, us):
    return 2 * m * n * k / us / 1e6


def bench(fn):
    return do_bench(fn, warmup=5, rep=25, return_mode="median") * 1e3


M, N, K = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (4096, 7168, 4096)
torch.manual_seed(0)
a = torch.randn(M, K, device="cuda").to(torch.bfloat16)
b = torch.randn(N, K, device="cuda").to(torch.bfloat16)
As = torch.rand(M, K // 128, device="cuda") + 0.5
Bs = torch.rand((N + 127) // 128, K // 128, device="cuda") + 0.5
c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
want = None
res = []
for pad in (0, 32):
    ap = torch.empty(M, K + pad, device="cuda", dtype=torch.bfloat16)
    ap[:, :K] = a
    bp = torch.empty(N, K + pad, device="cuda", dtype=torch.bfloat16)
    bp[:, :K] = b
    for bm, bn, bk, nw, ns in itertools.product((64, 128, 256), (128, 256), (32, 64, 128), (4, 8), (1, 2)):
        if bm * bn > 256 * 128 or bm * bn // 2 * bk > 128 * 64 * 128:
            continue
        fn = lambda: _halves[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
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
for t, pad, bm, bn, bk, nw, ns in res[:10]:
    print(f"  halves pad={pad:2d} BM={bm} BN={bn} BK={bk} warps={nw} stages={ns}: {t:8.1f} us "
          f"{tflops(M, N, K, t):5.1f} TFLOPs", flush=True)
# correctness of the best config against an fp32 reference (sampled rows)
t, pad, bm, bn, bk, nw, ns = res[0]
ap = torch.zeros(M, K + pad, device="cuda", dtype=torch.bfloat16)
ap[:, :K] = a
bp = torch.zeros(N, K + pad, device="cuda", dtype=torch.bfloat16)
bp[:, :K] = b
c32 = torch.empty(M, N, device="cuda", dtype=torch.float32)
_halves[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
    ap, bp, As, Bs, c32, M, N, K, ap.stride(0), bp.stride(0), As.stride(0), As.stride(1),
    Bs.stride(0), Bs.stride(1), GK=128, GN=128, BM=bm, BN=bn, BK=bk, GROUP_M=8, num_warps=nw, num_stages=ns)
rows = torch.arange(0, M, max(1, M // 64), device="cuda")
af = a[rows].double().view(len(rows), K // 128, 128)
bf = b.double().view(N, K // 128, 128)
part = torch.einsum("mbk,nbk->mnb", af, bf)
sc = As[rows].double()[:, None, :] * Bs.double().repeat_interleave(128, 0)[:N][None, :, :]
ref = (part * sc).sum(-1)
err = ((c32[rows].double() - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
print(f"best config fp32 output: max relative error vs fp64 reference {err:.2e}")
