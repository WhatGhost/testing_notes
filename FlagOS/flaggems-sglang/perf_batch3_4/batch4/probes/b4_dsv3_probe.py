"""dsv3_fused_a_gemm M>=3 on gfx1100: split-K + reduction vs the generic op."""

import importlib
import itertools
import os
import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

gen = importlib.import_module("flaggems_sglang.ops.dsv3_fused_a_gemm")


@triton.jit
def _splitk(a_ptr, b_ptr, part_ptr, M, N, K, split_k, k_steps, stride_am, stride_bk, stride_bn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_n = pid % num_n
    pid_k = (pid // num_n + pid_n) % split_k
    k_start = pid_k * k_steps * BLOCK_K
    k_end = tl.minimum(k_start + k_steps * BLOCK_K, K)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :], mask=m_mask, other=0.0)
        b = tl.load(b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=n_mask, other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(part_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :], acc, mask=m_mask & n_mask)


@triton.jit
def _reduce(part_ptr, out_ptr, total, split_k, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, split_k):
        acc += tl.load(part_ptr + s * total + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def splitk(a, b, bn, steps, nw, bk=256):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    split = triton.cdiv(triton.cdiv(k, bk), steps)
    part = torch.empty((split, m, n), dtype=torch.float32, device=a.device)
    _splitk[(triton.cdiv(n, bn) * split,)](a, b, part, m, n, k, split, steps, a.stride(0), b.stride(0), b.stride(1),
                                           BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=1)
    _reduce[(triton.cdiv(m * n, 256),)](part, out, m * n, split, BLOCK=256, num_warps=1, num_stages=1)
    return out


for m in (3, 4, 8, 16):
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(m, 7168, device="cuda", generator=g).to(torch.bfloat16)
    w = torch.randn(2112, 7168, device="cuda", generator=g).to(torch.bfloat16)
    bmat = w.t()
    ref = (a.float() @ w.float().T)
    o = gen.dsv3_fused_a_gemm(a, bmat)
    e_gen = ((o.float() - ref).abs() / (1 + ref.abs())).max().item()
    t_gen = do_bench(lambda: gen.dsv3_fused_a_gemm(a, bmat), warmup=25, rep=200, return_mode="median") * 1e3
    res = []
    for bn, steps, nw in itertools.product((32, 64, 128), (1, 2, 3, 4, 5, 7), (2, 4, 8)):
        o = splitk(a, bmat, bn, steps, nw)
        e = ((o.float() - ref).abs() / (1 + ref.abs())).max().item()
        if e > max(2e-2, 2 * e_gen):
            print("BAD", m, bn, steps, nw, e)
            continue
        t = do_bench(lambda: splitk(a, bmat, bn, steps, nw), warmup=25, rep=200, return_mode="median") * 1e3
        res.append((t, (bn, steps, nw)))
    res.sort()
    print(f"M={m:2d} | generic {t_gen:6.1f} us | split-K best (BN, steps, warps): " +
          ", ".join(f"{c} {t:.1f}" for t, c in res[:4]), flush=True)
