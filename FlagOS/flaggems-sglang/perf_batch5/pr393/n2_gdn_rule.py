"""GDN packed decode: candidate launch configs over head counts (TP splits)
and batch sizes; bitwise check against the current config."""

import importlib
import json
import os
import sys

import torch
import triton
from triton.testing import do_bench

N = open("/tmp/fgs-n1.path").read().strip()
sys.path.insert(0, os.path.join(N, "src"))
import flaggems_sglang  # noqa: E402,F401

lib = importlib.import_module("flaggems_sglang.ops.fused_recurrent_gated_delta_rule_packed_decode")
DEV = "cuda"
CFGS = [(64, 8), (16, 8), (32, 8), (32, 16), (16, 4), (64, 16)]


def run_cfg(mixed, a, b, A_log, dt_bias, scale, state, out, idx, bv, nw):
    B = mixed.shape[0]
    HV, V, K = state.shape[-3:]
    H = (mixed.shape[1] - HV * V) // 2 // K
    lib._packed_decode_kernel[(triton.cdiv(V, bv), B * HV)](
        mixed, a, b, A_log, dt_bias, state, out, idx, float(scale), HV, H, mixed.stride(0), state.stride(0),
        state.stride(1), state.stride(2), state.stride(3), out.stride(0), out.stride(2), V=V, K=K, BV=bv,
        BK=triton.next_power_of_2(K), EVEN_V=V % bv == 0, EVEN_K=True, USE_L2NORM=True, num_warps=nw)


rows = []
for h, hv in ((2, 6), (4, 8), (4, 12), (8, 16), (8, 24), (16, 32), (16, 48)):
    k = v = 128
    for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        pool = max(64, 2 * bs + 1)
        g = torch.Generator(device=DEV).manual_seed(0)
        mixed = torch.randn(bs, 2 * h * k + hv * v, dtype=torch.bfloat16, device=DEV, generator=g)
        a = torch.randn(bs, hv, dtype=torch.bfloat16, device=DEV, generator=g)
        b = torch.randn(bs, hv, dtype=torch.bfloat16, device=DEV, generator=g)
        A_log = torch.randn(hv, dtype=torch.float32, device=DEV, generator=g)
        dt_bias = torch.randn(hv, dtype=torch.float32, device=DEV, generator=g)
        state0 = torch.randn(pool, hv, v, k, dtype=torch.float32, device=DEV, generator=g) * 0.1
        idx = torch.randperm(pool - 1, device=DEV)[:bs].to(torch.int32) + 1
        out = torch.empty(bs, 1, hv, v, dtype=torch.bfloat16, device=DEV)
        scale = k ** -0.5
        ref_s, ref_o = state0.clone(), torch.empty_like(out)
        run_cfg(mixed, a, b, A_log, dt_bias, scale, ref_s, ref_o, idx, 64, 8)
        times, bitwise = {}, {}
        for bv, nw in CFGS:
            s, o = state0.clone(), torch.empty_like(out)
            run_cfg(mixed, a, b, A_log, dt_bias, scale, s, o, idx, bv, nw)
            bitwise[str((bv, nw))] = bool(torch.equal(s, ref_s) and torch.equal(o, ref_o))
            st = state0.clone()
            times[str((bv, nw))] = do_bench(lambda: run_cfg(mixed, a, b, A_log, dt_bias, scale, st, out, idx, bv, nw),
                                            warmup=10, rep=100, return_mode="median") * 1e3
        rows.append(dict(h=h, hv=hv, bs=bs, times=times, bitwise=bitwise))
        cur = times[str((64, 8))]
        best = min(times, key=times.get)
        print(f"H={h:2d} HV={hv:2d} bs={bs:3d} B*HV={bs * hv:5d} | cur {cur:7.1f} | (16,8) {times['(16, 8)']:7.1f} "
              f"(32,8) {times['(32, 8)']:7.1f} (32,16) {times['(32, 16)']:7.1f} | best {best} {times[best]:.1f} | "
              f"bitwise {all(bitwise.values())}", flush=True)
json.dump(rows, open("/tmp/n2_gdn_rule.json", "w"))
