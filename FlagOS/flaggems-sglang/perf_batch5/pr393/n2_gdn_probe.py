"""GDN packed decode (fused_recurrent_gated_delta_rule_packed_decode) on gfx1100:
library generic op vs SGLang vs other tilings of the library kernel, against an
fp32 copy of the same state bytes."""

import importlib
import itertools
import os
import sys

import torch
import triton
from triton.testing import do_bench

N = open("/tmp/fgs-n1.path").read().strip()
sys.path.insert(0, os.path.join(N, "src"))
sys.path.insert(0, "/tmp/sglang_new/standalone")
import flaggems_sglang  # noqa: E402,F401
import sgl_gdn_decode as sgl  # noqa: E402

lib = importlib.import_module("flaggems_sglang.ops.fused_recurrent_gated_delta_rule_packed_decode")
DEV = "cuda"


def run_cfg(mixed, a, b, A_log, dt_bias, scale, state, out, idx, bv, nw):
    B = mixed.shape[0]
    HV, V, K = state.shape[-3:]
    H = (mixed.shape[1] - HV * V) // 2 // K
    lib._packed_decode_kernel[(triton.cdiv(V, bv), B * HV)](
        mixed, a, b, A_log, dt_bias, state, out, idx, float(scale), HV, H, mixed.stride(0), state.stride(0),
        state.stride(1), state.stride(2), state.stride(3), out.stride(0), out.stride(2), V=V, K=K, BV=bv,
        BK=triton.next_power_of_2(K), EVEN_V=V % bv == 0, EVEN_K=True, USE_L2NORM=True, num_warps=nw)


def bench(fn):
    return do_bench(fn, warmup=10, rep=100, return_mode="median") * 1e3


for model, h, hv in (("27B", 4, 12), ("35B-A3B", 4, 8)):
    k = v = 128
    for bs in (1, 4, 8, 16, 32, 64, 128):
        pool = max(360, 2 * bs + 1)
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
        # correctness of the library op vs SGLang on one step
        s_ref, s_lib = state0.clone(), state0.clone()
        o_ref, o_lib = torch.empty_like(out), torch.empty_like(out)
        sgl.fused_recurrent_gated_delta_rule_packed_decode(mixed, a, b, A_log, dt_bias, scale, s_ref, o_ref, idx,
                                                          use_qk_l2norm_in_kernel=True)
        lib.fused_recurrent_gated_delta_rule_packed_decode(mixed, a, b, A_log, dt_bias, scale, s_lib, o_lib, idx,
                                                          use_qk_l2norm_in_kernel=True)
        err_o = (o_ref.float() - o_lib.float()).abs().max().item()
        err_s = (s_ref - s_lib).abs().max().item()
        state = state0.clone()
        t_lib = bench(lambda: lib.fused_recurrent_gated_delta_rule_packed_decode(
            mixed, a, b, A_log, dt_bias, scale, state, out, idx, use_qk_l2norm_in_kernel=True))
        t_sgl = bench(lambda: sgl.fused_recurrent_gated_delta_rule_packed_decode(
            mixed, a, b, A_log, dt_bias, scale, state, out, idx, use_qk_l2norm_in_kernel=True))
        res = []
        for bv, nw in itertools.product((16, 32, 64, 128), (1, 2, 4, 8, 16)):
            try:
                run_cfg(mixed, a, b, A_log, dt_bias, scale, state, out, idx, bv, nw)
                res.append((bench(lambda: run_cfg(mixed, a, b, A_log, dt_bias, scale, state, out, idx, bv, nw)),
                            (bv, nw)))
            except Exception:  # noqa: BLE001
                pass
        res.sort()
        n_state = bs * hv * v * k
        src = torch.empty(n_state, dtype=torch.float32, device=DEV)
        dst = torch.empty_like(src)
        t_copy = bench(lambda: dst.copy_(src))
        gbs = 2 * n_state * 4 / (t_lib * 1e-6) / 1e9
        print(f"{model:8s} bs={bs:3d} | lib {t_lib:7.1f} us ({gbs:5.0f} GB/s) | sglang {t_sgl:7.1f} | copy {t_copy:7.1f} "
              f"| best (BV, warps): " + ", ".join(f"{c} {t:.1f}" for t, c in res[:3]) +
              f" | err out {err_o:.1e} state {err_s:.1e}", flush=True)
