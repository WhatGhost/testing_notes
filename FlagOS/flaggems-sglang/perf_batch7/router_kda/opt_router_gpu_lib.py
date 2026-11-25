"""Shared pieces for the router experiments."""

import importlib.util
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


amd = load("amd_router_lib", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/ops/dsv3_router_gemm.py"))
new = load("new_router_lib", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/dsv3_router_gemm.py"))


def bench(fn):
    return do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3


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
