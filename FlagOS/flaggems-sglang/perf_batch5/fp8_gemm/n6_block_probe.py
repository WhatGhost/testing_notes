"""w8a8 block FP8 GEMM core on gfx1100, after decoding to bf16: per-128-K-block
partial dot, scaled by As[m, kb] * Bs[nb, kb] and added with fma. Compares B
stored [N, K] (as given) with B transposed to [K, N] by the decode pass, over
tiles / warps / stages / padding. N, K and leading dims constexpr."""

import itertools
import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def _block(A, B, As, Bs, C, M, N: tl.constexpr, K: tl.constexpr, LDA: tl.constexpr, LDB: tl.constexpr,
           AS0, AS1, BS0, BS1, B_KN: tl.constexpr, GK: tl.constexpr, GN: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n: tl.constexpr = (N + BN - 1) // BN
    in_group = GROUP_M * num_pid_n
    first_m = pid // in_group * GROUP_M
    group_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + pid % in_group % group_m
    pid_n = pid % in_group // group_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_mask = (rm < M)[:, None]
    ap = A + rm[:, None].to(tl.int64) * LDA + rk[None, :]
    if B_KN:
        bp = B + rk[:, None].to(tl.int64) * LDB + rn[None, :]
        b_step: tl.constexpr = BK * LDB
    else:
        bp = B + rn[None, :].to(tl.int64) * LDB + rk[:, None]
        b_step: tl.constexpr = BK
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
            bp += b_step
        sa = tl.load(sa_row + kb * AS1, mask=rm < M, other=0.0)
        sb = tl.load(sb_row + kb * BS1)
        acc = tl.fma(part, sa[:, None] * sb, acc)
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty), mask=m_mask)


def tflops(m, n, k, us):
    return 2 * m * n * k / us / 1e6


def bench(fn):
    return do_bench(fn, warmup=5, rep=25, return_mode="median") * 1e3


M, N, K = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (4096, 7168, 4096)
torch.manual_seed(0)
a = torch.randn(M, K, device="cuda").to(torch.bfloat16)
b = torch.randn(N, K, device="cuda").to(torch.bfloat16)
t = bench(lambda: torch.matmul(a, b.t()))
print(f"M={M} N={N} K={K} hipBLASLt bf16: {t:8.1f} us {tflops(M, N, K, t):5.1f} TFLOPs", flush=True)
As = torch.rand(M, K // 128, device="cuda") + 0.5
Bs = torch.rand((N + 127) // 128, K // 128, device="cuda") + 0.5
c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
res = []
for b_kn, pad in ((1, 0), (1, 32), (0, 0), (0, 32)):
    ap = torch.empty(M, K + pad, device="cuda", dtype=torch.bfloat16)
    ap[:, :K] = a
    if b_kn:
        bp = torch.empty(K, N + pad, device="cuda", dtype=torch.bfloat16)
        bp[:, :N] = b.t()
    else:
        bp = torch.empty(N, K + pad, device="cuda", dtype=torch.bfloat16)
        bp[:, :K] = b
    for bm, bn, bk, nw, ns in itertools.product((64, 128, 256), (64, 128), (32, 64, 128), (4, 8), (1, 2)):
        if bm * bn > 256 * 128 or bm * bn * bk > 128 * 128 * 128:
            continue
        fn = lambda: _block[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
            ap, bp, As, Bs, c, M, N, K, ap.stride(0), bp.stride(0), As.stride(0), As.stride(1),
            Bs.stride(0), Bs.stride(1), B_KN=b_kn, GK=128, GN=128, BM=bm, BN=bn, BK=bk, GROUP_M=8,
            num_warps=nw, num_stages=ns)
        try:
            fn()
            t = bench(fn)
        except Exception:  # noqa: BLE001
            continue
        res.append((t, b_kn, pad, bm, bn, bk, nw, ns))
res.sort()
for t, b_kn, pad, bm, bn, bk, nw, ns in res[:10]:
    print(f"  B {'[K,N]' if b_kn else '[N,K]'} pad={pad:2d} BM={bm} BN={bn} BK={bk} warps={nw} stages={ns}: "
          f"{t:8.1f} us {tflops(M, N, K, t):5.1f} TFLOPs", flush=True)
for b_kn in (1, 0):
    best = min(r for r in res if r[1] == b_kn)
    print(f"best with B {'[K,N]' if b_kn else '[N,K]'}: {best[0]:.1f} us {tflops(M, N, K, best[0]):.1f} TFLOPs {best[2:]}")
