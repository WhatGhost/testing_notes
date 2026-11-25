"""Router: master's scalar kernel by num_warps vs the shipped op, 3 rounds,
medians, over token counts 1-16 and expert counts 64-512."""

import importlib.util
import os

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


new = load("new_router", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/dsv3_router_gemm.py"))
g = torch.Generator(device="cuda").manual_seed(0)


def bench(fn):
    return do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3


def scalar(h, w, nw):
    m, k = h.shape
    e = w.shape[0]
    out = torch.empty((m, e), dtype=torch.float32, device="cuda")
    new._dsv3_router_gemm_small_batch_kernel[(m, e)](h, w, out, e, k, *h.stride(), *w.stride(), 8192,
                                                    num_warps=nw, enable_fp_fusion=False)
    return out


print(" M    E | shipped | scalar warps=1     2     4     8 | best/shipped")
for e in (64, 128, 256, 384, 512):
    for m in (1, 2, 3, 4, 6, 8, 12, 16):
        h = torch.randn(m, 7168, device="cuda", generator=g).bfloat16()
        w = torch.randn(e, 7168, device="cuda", generator=g).bfloat16()
        ts, tw = [], {nw: [] for nw in (1, 2, 4, 8)}
        for _ in range(3):
            ts.append(bench(lambda: new.dsv3_router_gemm(h, w, torch.float32)))
            for nw in tw:
                tw[nw].append(bench(lambda: scalar(h, w, nw)))
        s = sorted(ts)[1]
        med = {nw: sorted(v)[1] for nw, v in tw.items()}
        best = min(med.values())
        print(f"{m:2d} {e:4d} | {s:7.1f} | " + " ".join(f"{med[nw]:5.1f}" for nw in (1, 2, 4, 8)) + f" | {s / best:4.2f}x", flush=True)
