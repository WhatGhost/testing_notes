"""fused_moe pair path at small T (official layer): where the time goes and
what helps. Kernels timed separately with do_bench (cache flushed):
  gate/up  current grid (pair, n, split) vs n-major grid (n, pair, split)
  down     current (token, n) vs slot-split (token, n, slot) + partial sum
  read     a pure streaming read of the same expert rows (ceiling)"""

import importlib
import itertools
import os
import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")


@triton.jit
def _gate_up_nmajor(x_ptr, w1_ptr, ids_ptr, part_ptr, P, H, inter, top_k, stride_xm, stride_we, stride_wn,
                    SPLITS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pair = tl.program_id(1)
    split = tl.program_id(2)
    token = pair // top_k
    expert = tl.load(ids_ptr + pair).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < inter
    offs_k = tl.arange(0, BLOCK_K)
    x_row = x_ptr + token.to(tl.int64) * stride_xm
    w_gate = w1_ptr + expert * stride_we + offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None]
    w_up = w_gate + inter * stride_wn
    acc_gate = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for k in range(split * BLOCK_K, H, SPLITS * BLOCK_K):
        k_mask = k + offs_k < H
        a = tl.load(x_row + k + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        mask = k_mask[:, None] & n_mask[None, :]
        b_gate = tl.load(w_gate + k, mask=mask, other=0.0).to(tl.float32)
        b_up = tl.load(w_up + k, mask=mask, other=0.0).to(tl.float32)
        acc_gate += a[:, None] * b_gate
        acc_up += a[:, None] * b_up
    row = part_ptr + ((split * P + pair).to(tl.int64) * 2) * inter + offs_n
    tl.store(row, tl.sum(acc_gate, axis=0), mask=n_mask)
    tl.store(row + inter, tl.sum(acc_up, axis=0), mask=n_mask)


@triton.jit
def _down_slot_split(part_ptr, w2_ptr, ids_ptr, weights_ptr, dpart_ptr, T, P, H, inter, top_k, stride_we, stride_wn,
                     SPLITS: tl.constexpr, ACT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # one program per (n block, token, slot); writes float32 partial rows
    pid_n = tl.program_id(0)
    token = tl.program_id(1)
    slot = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < H
    offs_k = tl.arange(0, BLOCK_K)
    pair = token * top_k + slot
    expert = tl.load(ids_ptr + pair).to(tl.int64)
    weight = tl.load(weights_ptr + pair).to(tl.float32)
    w_row = w2_ptr + expert * stride_we + offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None]
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for k in range(0, inter, BLOCK_K):
        k_mask = k + offs_k < inter
        gate = tl.zeros((BLOCK_K,), dtype=tl.float32)
        up = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for split in tl.static_range(SPLITS):
            row = part_ptr + ((split * P + pair).to(tl.int64) * 2) * inter
            gate += tl.load(row + k + offs_k, mask=k_mask, other=0.0)
            up += tl.load(row + inter + k + offs_k, mask=k_mask, other=0.0)
        a = r._activation(gate, ACT) * up * weight
        b = tl.load(w_row + k, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        acc += a[:, None] * b
    tl.store(dpart_ptr + (slot * T + token).to(tl.int64) * H + offs_n, tl.sum(acc, axis=0), mask=n_mask)


@triton.jit
def _sum_slots(dpart_ptr, out_ptr, T, H, scale, stride_om, TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for slot in tl.static_range(TOP_K):
        acc += tl.load(dpart_ptr + (slot * T + token).to(tl.int64) * H + offs, mask=mask, other=0.0)
    tl.store(out_ptr + token.to(tl.int64) * stride_om + offs, (acc * scale).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _read_rows(w_ptr, ids_ptr, out_ptr, R, H, stride_we, stride_wn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # streaming read of R rows of each selected expert (ceiling for the access pattern)
    pid_n = tl.program_id(0)
    pair = tl.program_id(1)
    expert = tl.load(ids_ptr + pair).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    ptrs = w_ptr + expert * stride_we + offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None]
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for k in range(0, H, BLOCK_K):
        acc += tl.load(ptrs + k, mask=(offs_n < R)[None, :], other=0.0).to(tl.float32)
    tl.store(out_ptr + pair * R + offs_n, tl.sum(acc, axis=0), mask=offs_n < R)


def bench(fn):
    return do_bench(fn, warmup=10, rep=60, return_mode="median") * 1e3


H, I, E, topk = 2048, 1408, 64, 8
gen = torch.Generator(device="cuda").manual_seed(0)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
for T in (1, 2, 4, 8):
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
    ids = ti.to(torch.int32).reshape(-1).contiguous()
    wts = tw.float().reshape(-1).contiguous()
    P = T * topk
    n_exp = ids.unique().numel()
    out = torch.empty_like(x)
    splits, up_cfg, down_cfg = r._pair_config(T, P, H, I)
    part = torch.empty((splits, P, 2, I), dtype=torch.float32, device="cuda")
    b_up, b_down = n_exp * 2 * I * H * 2, n_exp * H * I * 2

    def cur_up():
        r._pair_gate_up_kernel[(P, triton.cdiv(I, up_cfg[0]), splits)](
            x, w1, ids, part, P, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1),
            SPLITS=splits, BLOCK_N=up_cfg[0], BLOCK_K=up_cfg[1], num_warps=up_cfg[2])

    def cur_down():
        r._token_down_kernel[(T, triton.cdiv(H, down_cfg[0]))](
            part, w2, ids, wts, out, P, H, I, topk, 1.0, w2.stride(0), w2.stride(1), out.stride(0),
            SPLITS=splits, ACT=0, BLOCK_N=down_cfg[0], BLOCK_K=down_cfg[1], num_warps=down_cfg[2])

    cur_up()
    ref_part = part.clone()
    cur_down()
    ref_out = out.clone()
    t_up, t_down = bench(cur_up), bench(cur_down)
    print(f"T={T} experts={n_exp} | current: gate/up {t_up:6.1f} us ({b_up / t_up / 1e3:4.0f} GB/s) "
          f"down {t_down:6.1f} us ({b_down / t_down / 1e3:4.0f} GB/s) cfg S={splits} {up_cfg} {down_cfg}", flush=True)

    # ceiling: pure read of the gate/up rows of the selected experts
    best = None
    rbuf = torch.empty(P * 2 * I, dtype=torch.float32, device="cuda")
    for bn, bk, nw in itertools.product((8, 16, 32), (256, 512, 1024), (4, 8)):
        fn = lambda: _read_rows[(triton.cdiv(2 * I, bn), P)](w1, ids, rbuf, 2 * I, H, w1.stride(0), w1.stride(1),
                                                           BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
        t = bench(fn)
        if best is None or t < best[0]:
            best = (t, (bn, bk, nw))
    print(f"      read-only gate/up rows: {best[0]:6.1f} us ({b_up / best[0] / 1e3:4.0f} GB/s) {best[1]}", flush=True)

    # gate/up, n-major grid
    best = None
    for bn, bk, nw in itertools.product((8, 16, 32), (256, 512, 1024), (4, 8)):
        if bk * splits > max(H, 256):
            continue
        fn = lambda: _gate_up_nmajor[(triton.cdiv(I, bn), P, splits)](
            x, w1, ids, part, P, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1),
            SPLITS=splits, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
        t = bench(fn)
        if best is None or t < best[0]:
            best = (t, (bn, bk, nw))
    fn = lambda: _gate_up_nmajor[(triton.cdiv(I, best[1][0]), P, splits)](
        x, w1, ids, part, P, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1),
        SPLITS=splits, BLOCK_N=best[1][0], BLOCK_K=best[1][1], num_warps=best[1][2])
    fn()
    same_part = torch.equal(part, ref_part) if best[1][1] == up_cfg[1] else "n/a (other BLOCK_K)"
    print(f"      gate/up n-major best: {best[0]:6.1f} us ({b_up / best[0] / 1e3:4.0f} GB/s) {best[1]} "
          f"same partials as current: {same_part}", flush=True)
    part.copy_(ref_part)

    # down, slot split + partial sum
    dpart = torch.empty((topk, T, H), dtype=torch.float32, device="cuda")
    best = None
    for bn, bk, nw in itertools.product((8, 16, 32, 64), (128, 256, 512), (2, 4, 8)):
        def fn(bn=bn, bk=bk, nw=nw):
            _down_slot_split[(triton.cdiv(H, bn), T, topk)](
                part, w2, ids, wts, dpart, T, P, H, I, topk, w2.stride(0), w2.stride(1),
                SPLITS=splits, ACT=0, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
            _sum_slots[(triton.cdiv(H, 1024), T)](dpart, out, T, H, 1.0, out.stride(0), TOP_K=topk, BLOCK=1024,
                                                  num_warps=4)
        t = bench(fn)
        if best is None or t < best[0]:
            best = (t, (bn, bk, nw), fn)
    best[2]()
    err = (out.float() - ref_out.float()).abs().max().item()
    print(f"      down slot-split + sum best: {best[0]:6.1f} us ({b_down / best[0] / 1e3:4.0f} GB/s) {best[1]} "
          f"max diff vs current {err:.2e}", flush=True)
