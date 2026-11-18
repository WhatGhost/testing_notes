"""Does the cache flush decide the small-T fused_moe numbers? Same kernels timed
with three flushes before every call:
  zero   do_bench's default: zero a 256 MB buffer (leaves dirty lines)
  read   sum over a 512 MB buffer (evicts the weights, nothing dirty)
  none   no flush (weights may stay in the 96 MB Infinity Cache)"""

import importlib
import os
import statistics
import sys

import torch
import triton

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
sys.path.insert(0, "/tmp")
exp = importlib.import_module("n4_pair_exp_kernels")

zero_buf = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
read_buf = torch.ones(512 * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
sink = torch.empty(1, dtype=torch.float32, device="cuda")


def flush(kind):
    if kind == "zero":
        zero_buf.zero_()
    elif kind == "read":
        torch.sum(read_buf, dtype=torch.float32)


def timed(fn, kind, reps=60, warmup=10):
    for _ in range(warmup):
        flush(kind)
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        flush(kind)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


H, I, E, topk = 2048, 1408, 64, 8
gen = torch.Generator(device="cuda").manual_seed(0)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
big = torch.empty(46 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")  # 92 MiB
big2 = torch.empty_like(big)
for T in (1, 4):
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
    to = (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None)
    ids = to[1].reshape(-1).contiguous()
    wts = to[0].reshape(-1).contiguous()
    P = T * topk
    n_exp = ids.unique().numel()
    out = torch.empty_like(x)
    splits, up_cfg, down_cfg = r._pair_config(T, P, H, I)
    part = torch.empty((splits, P, 2, I), dtype=torch.float32, device="cuda")
    rbuf = torch.empty(P * 2 * I, dtype=torch.float32, device="cuda")
    b_up, b_down = n_exp * 2 * I * H * 2, n_exp * H * I * 2
    fns = {
        "read-only gate/up rows": (lambda: exp._read_rows[(triton.cdiv(2 * I, 8), P)](
            w1, ids, rbuf, 2 * I, H, w1.stride(0), w1.stride(1), BLOCK_N=8, BLOCK_K=1024, num_warps=8), b_up),
        "gate/up (current)": (lambda: r._pair_gate_up_kernel[(P, triton.cdiv(I, up_cfg[0]), splits)](
            x, w1, ids, part, P, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1),
            SPLITS=splits, BLOCK_N=up_cfg[0], BLOCK_K=up_cfg[1], num_warps=up_cfg[2]), b_up),
        "down (current)": (lambda: r._token_down_kernel[(T, triton.cdiv(H, down_cfg[0]))](
            part, w2, ids, wts, out, P, H, I, topk, 1.0, w2.stride(0), w2.stride(1), out.stride(0),
            SPLITS=splits, ACT=0, BLOCK_N=down_cfg[0], BLOCK_K=down_cfg[1], num_warps=down_cfg[2]), b_down),
        "whole fused_moe": (lambda: r.fused_moe(x, w1, w2, to), b_up + b_down),
        "torch copy 92 MiB": (lambda: big2.copy_(big), 2 * big.numel() * 2),
        "torch sum 92 MiB (read)": (lambda: torch.sum(big, dtype=torch.float32), big.numel() * 2),
    }
    print(f"T={T} (experts {n_exp}, weights {(b_up + b_down) / 1e6:.0f} MB)")
    for name, (fn, nbytes) in fns.items():
        res = {k: timed(fn, k) for k in ("zero", "read", "none")}
        print(f"  {name:26s} " + " | ".join(f"{k} {v:7.1f} us {nbytes / v / 1e3:4.0f} GB/s" for k, v in res.items()),
              flush=True)
