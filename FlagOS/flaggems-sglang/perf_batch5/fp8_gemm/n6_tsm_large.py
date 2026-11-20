"""triton_scaled_mm, large M on gfx1100: where the time goes in the generic
path (widen A, widen B, GEMM) and what the GEMM gains from 2 stages and from
constexpr N / K / leading dims."""

import importlib.util
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


@triton.jit
def _plain(A, B, C, sa_ptr, sb_ptr, M, N: tl.constexpr, K: tl.constexpr, LDA: tl.constexpr, LDB: tl.constexpr,
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


def bench(fn):
    return do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3


for M, K, N in ((4096, 4096, 7168), (4096, 4096, 4096), (512, 4096, 7168), (512, 4096, 4096)):
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = torch.clamp(0.1 * torch.randn(M, K, dtype=torch.float16, device="cuda", generator=gen), -0.3, 0.3).to(
        torch.float8_e4m3fn)
    b = torch.clamp(0.1 * torch.randn(K, N, dtype=torch.float16, device="cuda", generator=gen), -0.3, 0.3).to(
        torch.float8_e4m3fn)
    sa = 0.1 + 0.05 * torch.rand(M, 1, device="cuda", generator=gen)
    sb = 0.1 + 0.05 * torch.rand(N, 1, device="cuda", generator=gen)
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    t_total = bench(lambda: g.triton_scaled_mm(a, b, sa, sb, torch.float16))
    a16 = torch.empty((M, K), device="cuda", dtype=torch.float16)
    b16 = torch.empty((K, N), device="cuda", dtype=torch.float16)
    na4, nb4 = M * K // 4, K * N // 4
    wa = lambda: g._widen_fp8_vec_kernel[(triton.cdiv(na4, 1024),)](a.view(torch.uint8).view(torch.uint32), a16, na4,
                                                                     BLOCK=1024, num_warps=4)
    wb = lambda: g._widen_fp8_vec_kernel[(triton.cdiv(nb4, 1024),)](b.view(torch.uint8).view(torch.uint32), b16, nb4,
                                                                     BLOCK=1024, num_warps=4)
    wa()
    wb()
    t_wa, t_wb = bench(wa), bench(wb)
    BM, BN, BK, GM, nw = (128, 128, 64, 8, 4) if M > 128 else (64, 64, 64, 16, 8)

    def gemm(ns):
        g._gemm_kernel[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](
            a16, b16, out, out, sa, sb, M, N, K, a16.stride(0), a16.stride(1), b16.stride(0), b16.stride(1),
            out.stride(0), out.stride(1), 0, 0, OUT_DTYPE=tl.float16, IS_INT8=False, HAS_BIAS=False, SA_BCAST=0,
            SB_BCAST=0, EVEN_K=(K % BK == 0), EVEN_M=(M % BM == 0), EVEN_N=(N % BN == 0), BLOCK_M=BM, BLOCK_N=BN,
            BLOCK_K=BK, GROUP_M=GM, num_warps=nw, num_stages=ns)

    t_g1, t_g2 = bench(lambda: gemm(1)), bench(lambda: gemm(2))
    best = None
    for bm, bn, bk, nw2, ns in ((128, 128, 64, 4, 2), (128, 128, 32, 4, 1), (128, 128, 32, 4, 2), (128, 128, 64, 4, 1),
                                (256, 128, 32, 8, 1), (128, 256, 32, 8, 1)):
        fn = lambda: _plain[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
            a16, b16, out, sa, sb, M, N, K, K, N, BM=bm, BN=bn, BK=bk, GROUP_M=8, num_warps=nw2, num_stages=ns)
        t = bench(fn)
        if best is None or t < best[0]:
            best = (t, (bm, bn, bk, nw2, ns))
    fl = 2 * M * N * K
    print(f"M={M} K={K} N={N}: generic total {t_total:8.1f} us | widen A {t_wa:6.1f} B {t_wb:6.1f} | "
          f"generic GEMM 1 stage {t_g1:8.1f} ({fl / t_g1 / 1e6:5.1f} TF) 2 stages {t_g2:8.1f} ({fl / t_g2 / 1e6:5.1f} TF) | "
          f"constexpr GEMM {best[0]:8.1f} ({fl / best[0] / 1e6:5.1f} TF) {best[1]}", flush=True)
