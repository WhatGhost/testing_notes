"""rdna3 fused_recurrent_gated_delta_rule_packed_decode vs the generic op:
output and state bitwise."""

import importlib
import os
import sys

import torch

N = open("/tmp/fgs-n1.path").read().strip()
sys.path.insert(0, os.path.join(N, "src"))
import flaggems_sglang  # noqa: E402,F401

g = importlib.import_module("flaggems_sglang.ops.fused_recurrent_gated_delta_rule_packed_decode")
r = importlib.import_module(
    "flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_recurrent_gated_delta_rule_packed_decode")
fails = cases = 0
for h, hv in ((1, 1), (2, 6), (4, 8), (4, 12), (8, 24), (16, 48), (3, 9)):
    for kv in (64, 128):
        for bs in (1, 3, 8, 17, 64, 200):
            for l2 in (True, False):
                for dtype in (torch.bfloat16, torch.float16):
                    gen = torch.Generator(device="cuda").manual_seed(bs * 7 + hv)
                    pool = 2 * bs + 4
                    mixed = torch.randn(bs, 2 * h * kv + hv * kv, dtype=dtype, device="cuda", generator=gen)
                    a = torch.randn(bs, hv, dtype=dtype, device="cuda", generator=gen)
                    b = torch.randn(bs, hv, dtype=dtype, device="cuda", generator=gen)
                    A_log = torch.randn(hv, dtype=torch.float32, device="cuda", generator=gen)
                    dt_bias = torch.randn(hv, dtype=torch.float32, device="cuda", generator=gen)
                    state0 = torch.randn(pool, hv, kv, kv, dtype=torch.float32, device="cuda", generator=gen) * 0.1
                    idx = torch.randperm(pool - 1, device="cuda")[:bs].to(torch.int32) + 1
                    if bs > 2:
                        idx[1] = 0  # padded slot: zero output, state untouched
                        idx[2] = -1
                    outs = []
                    for mod in (g, r):
                        st = state0.clone()
                        out = torch.full((bs, 1, hv, kv), 3.0, dtype=dtype, device="cuda")
                        mod.fused_recurrent_gated_delta_rule_packed_decode(
                            mixed, a, b, A_log, dt_bias, kv ** -0.5, st, out, idx, use_qk_l2norm_in_kernel=l2)
                        outs.append((out, st))
                    cases += 1
                    if not (torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])):
                        fails += 1
                        print("FAIL", h, hv, kv, bs, l2, dtype, flush=True)
print("cases", cases, "fails", fails)
