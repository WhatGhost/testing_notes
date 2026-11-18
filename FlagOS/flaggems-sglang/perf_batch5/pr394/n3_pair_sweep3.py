"""Pair path, small T: gate/up (BLOCK_N, BLOCK_K, warps, splits) and down
(BLOCK_N, BLOCK_K, warps) swept separately, kernel times."""

import importlib
import itertools
import json
import os
import sys

import torch
import triton
from triton.testing import do_bench

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
model = sys.argv[1]
H, I, E, topk = {"official": (2048, 1408, 64, 8), "35B": (2048, 128, 256, 8)}[model]


def bench(fn):
    return do_bench(fn, warmup=10, rep=60, return_mode="median") * 1e3


rows = []
gen = torch.Generator(device="cuda").manual_seed(0)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
Ts = (1, 2, 4, 8, 16, 32) if model == "35B" else (1, 2, 4, 8, 16)
for T in Ts:
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
    ids = ti.to(torch.int32).reshape(-1).contiguous()
    wts = tw.float().reshape(-1).contiguous()
    out = torch.empty_like(x)
    P = T * topk
    cur_s, cur_up, cur_down = r._pair_config(T, P, H, I)
    best_up = {}
    for bn, bk, nw, s in itertools.product((8, 16, 32, 64), (64, 128, 256, 512), (2, 4, 8), (1, 2, 4, 8)):
        if bk * s > max(H, 256) or bn > max(16, triton.next_power_of_2(I)):
            continue
        part = torch.empty((s, P, 2, I), dtype=torch.float32, device="cuda")

        def gate_up():
            r._pair_gate_up_kernel[(P, triton.cdiv(I, bn), s)](
                x, w1, ids, part, P, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1),
                SPLITS=s, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)

        try:
            t = bench(gate_up)
        except Exception:
            continue
        rows.append(dict(model=model, T=T, kernel="up", bn=bn, bk=bk, nw=nw, s=s, t=t))
        if s not in best_up or t < best_up[s][0]:
            best_up[s] = (t, (bn, bk, nw))
    best_down = {}
    for s in (1, 2, 4, 8):
        part = torch.zeros((s, P, 2, I), dtype=torch.float32, device="cuda")
        for bn, bk, nw in itertools.product((4, 8, 16, 32, 64), (64, 128, 256, 512), (1, 2, 4, 8)):
            if bk > max(64, triton.next_power_of_2(I)):
                continue

            def down_k():
                r._token_down_kernel[(T, triton.cdiv(H, bn))](
                    part, w2, ids, wts, out, P, H, I, topk, 1.0, w2.stride(0), w2.stride(1), out.stride(0),
                    SPLITS=s, ACT=0, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)

            try:
                t = bench(down_k)
            except Exception:
                continue
            rows.append(dict(model=model, T=T, kernel="down", bn=bn, bk=bk, nw=nw, s=s, t=t))
            if s not in best_down or t < best_down[s][0]:
                best_down[s] = (t, (bn, bk, nw))
    combo = {s: best_up[s][0] + best_down[s][0] for s in best_up if s in best_down}
    s_best = min(combo, key=combo.get)
    cur_t = [x_["t"] for x_ in rows if x_["T"] == T and x_["kernel"] == "up" and x_["s"] == cur_s
             and (x_["bn"], x_["bk"], x_["nw"]) == cur_up]
    cur_d = [x_["t"] for x_ in rows if x_["T"] == T and x_["kernel"] == "down" and x_["s"] == cur_s
             and (x_["bn"], x_["bk"], x_["nw"]) == cur_down]
    print(f"{model} T={T}: best S={s_best} up {best_up[s_best]} down {best_down[s_best]} sum {combo[s_best]:.1f} | "
          f"current S={cur_s} up {cur_up} {cur_t[0] if cur_t else float('nan'):.1f} down {cur_down} "
          f"{cur_d[0] if cur_d else float('nan'):.1f} | per-S {{{', '.join(f'{k}: {v:.1f}' for k, v in sorted(combo.items()))}}}",
          flush=True)
json.dump(rows, open(f"/tmp/n3_pair_sweep3_{model}.json", "w"), indent=1)
