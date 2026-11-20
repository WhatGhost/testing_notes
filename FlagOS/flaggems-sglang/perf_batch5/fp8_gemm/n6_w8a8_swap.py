"""w8a8 block FP8 GEMM for small M on gfx1100, swap-AB: out^T tile [BN, BM] =
B_tile [BN, BK] (fp8 read as uint32 along K, converted in registers) @
A_tile^T [BK, BM]. Per 128-K block partial dot scaled by Bs (scalar per tile)
x As (per row) with fma; K blocks split across programs plus a reduce when
the grid is small. A either pre-decoded to fp16 or converted in the kernel."""

import importlib.util
import itertools
import os
import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

T = open("/tmp/fgs-f8.path").read().strip()
spec = importlib.util.spec_from_file_location("gen_w8", os.path.join(T, "src/flaggems_sglang/ops/w8a8_block_fp8_matmul.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
spec = importlib.util.spec_from_file_location("gen_tsm", os.path.join(T, "src/flaggems_sglang/ops/triton_scaled_mm.py"))
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)
_e4m3_to_fp16 = gt._e4m3_to_fp16


@triton.jit
def _unpack4_k(w):
    b0 = _e4m3_to_fp16(w & 0xFF)
    b1 = _e4m3_to_fp16((w >> 8) & 0xFF)
    b2 = _e4m3_to_fp16((w >> 16) & 0xFF)
    b3 = _e4m3_to_fp16((w >> 24) & 0xFF)
    return tl.interleave(tl.interleave(b0, b2), tl.interleave(b1, b3))


@triton.jit
def _swap(A, B32, As, Bs, C, P, M, N: tl.constexpr, K: tl.constexpr, AS0, AS1, BS0, BS1,
          GK: tl.constexpr, GN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          SPLIT: tl.constexpr, A_FP8: tl.constexpr):
    pid_n = tl.program_id(0)
    split = tl.program_id(1)
    pid_m = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN)
    rm = pid_m * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    rk4 = tl.arange(0, BK // 4)
    m_ok = rm < M
    bp = B32 + rn[:, None].to(tl.int64) * (K // 4) + rk4[None, :]
    ap = A + rm[None, :].to(tl.int64) * K + rk[:, None]
    sb_row = Bs + (pid_n * BN // GN) * BS0
    acc = tl.zeros((BN, BM), tl.float32)
    for kb in range(split, K // GK, SPLIT):
        part = tl.zeros((BN, BM), tl.float32)
        for kk in tl.static_range(GK // BK):
            k0 = kb * GK + kk * BK
            b = _unpack4_k(tl.load(bp + k0 // 4))
            if A_FP8:
                a = _e4m3_to_fp16(tl.load(ap + k0, mask=m_ok[None, :], other=0).to(tl.uint32))
            else:
                a = tl.load(ap + k0, mask=m_ok[None, :], other=0.0)
            part = tl.dot(b, a, part)
        sa = tl.load(As + rm * AS0 + kb * AS1, mask=m_ok, other=0.0)
        acc = tl.fma(part, tl.load(sb_row + kb * BS1) * sa[None, :], acc)
    if SPLIT == 1:
        tl.store(C + rm[None, :].to(tl.int64) * N + rn[:, None], acc.to(C.dtype.element_ty), mask=m_ok[None, :])
    else:
        tl.store(P + (split * M + rm[None, :]).to(tl.int64) * N + rn[:, None], acc, mask=m_ok[None, :])


@triton.jit
def _reduce(P, C, M, N: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(P + s * M * N + offs, mask=mask, other=0.0)
    tl.store(C + offs, acc.to(C.dtype.element_ty), mask=mask)


def bench(fn):
    return do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3


K = 4096
SHAPES = [tuple(int(v) for v in x.split("x")) for x in sys.argv[1].split(",")]
for M, N in SHAPES:
    gen = torch.Generator(device="cuda").manual_seed(0)
    fmax = torch.finfo(torch.float8_e4m3fn).max
    A = torch.randn(M, K, device="cuda", generator=gen).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda", generator=gen).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
    As = 1e-2 * torch.rand(M, K // 128, device="cuda", generator=gen)
    Bs = 1e-2 * torch.rand(N // 128, K // 128, device="cuda", generator=gen)
    ref = g.w8a8_block_fp8_matmul(A, B, As, Bs, [128, 128], torch.bfloat16)
    t_gen = bench(lambda: g.w8a8_block_fp8_matmul(A, B, As, Bs, [128, 128], torch.bfloat16))
    a16 = A.to(torch.float16)
    t_a = bench(lambda: A.to(torch.float16))
    a8 = A.view(torch.uint8)
    b32 = B.view(torch.uint8).view(torch.uint32)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    bms = (16,) if M <= 16 else (32, 64) if M <= 64 else (64, 128)
    a_modes = (0,)
    res = []
    for a_fp8, bm, bn, bk, nw, ns, sp in itertools.product(a_modes, bms, (32, 64, 128), (64, 128), (4, 8), (2,),
                                                          (1, 2, 4, 8)):
        programs = triton.cdiv(N, bn) * sp * triton.cdiv(M, bm)
        if programs < 32 or programs > 4096 or (K // 128) % sp != 0:
            continue
        if a_fp8 and M > 64:
            continue
        part = torch.empty((sp, M, N), device="cuda", dtype=torch.float32) if sp > 1 else out
        src = a8 if a_fp8 else a16

        def fn(bm=bm, bn=bn, bk=bk, nw=nw, ns=ns, sp=sp, part=part, src=src, a_fp8=a_fp8):
            if not a_fp8:
                torch.Tensor.copy_(a16, A)  # stands in for the decode launch of A
            _swap[(triton.cdiv(N, bn), sp, triton.cdiv(M, bm))](
                src, b32, As, Bs, out, part, M, N, K, As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1),
                GK=128, GN=128, BM=bm, BN=bn, BK=bk, SPLIT=sp, A_FP8=a_fp8, num_warps=nw, num_stages=ns)
            if sp > 1:
                _reduce[(triton.cdiv(M * N, 1024),)](part, out, M, N, SPLIT=sp, BLOCK=1024, num_warps=4)
        try:
            fn()
            t = bench(fn)
        except Exception:  # noqa: BLE001
            continue
        res.append((t, (a_fp8, bm, bn, bk, nw, ns, sp), fn))
    res.sort(key=lambda r: r[0])
    t, cfg, fn = res[0]
    fn()
    err = ((out.float() - ref.float()).abs() / ref.float().abs().max()).max().item()
    print(f"M={M:4d} N={N}: generic {t_gen:8.1f} us | swap best {t:8.1f} us weight {N * K / t / 1e3:4.0f} GB/s "
          f"cfg (A_fp8,BM,BN,BK,warps,stages,split)={cfg} speedup {t_gen / t:4.2f}x | max diff/max|ref| {err:.1e} | "
          f"next {' '.join(f'{r[1]}:{r[0]:.1f}' for r in res[1:3])}", flush=True)
