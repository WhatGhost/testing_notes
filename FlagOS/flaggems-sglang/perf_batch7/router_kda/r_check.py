"""rdna3 dsv3_router_gemm vs the reference and master's AMD op: accuracy under
the unit-test tolerance, a config sweep and timings (do_bench, the benchmark's
dict inputs cannot run in the official harness)."""

import importlib.util
import itertools
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


amd = load("amd_router", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/ops/dsv3_router_gemm.py"))
new = load("new_router", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/dsv3_router_gemm.py"))
g = torch.Generator(device="cuda").manual_seed(0)
TOL = {torch.float32: 1e-4, torch.bfloat16: 1.5e-2}


def inputs(rows, experts, dt=torch.bfloat16, strided=False, hidden=7168):
    h = torch.randn(rows, hidden + (64 if strided else 0), device="cuda", generator=g).to(dt)[:, :hidden]
    w = torch.randn(experts, hidden, device="cuda", generator=g).to(dt)
    return h, w


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


BENCH = [(m, e) for m in (1, 2, 4, 8, 16) for e in (256, 384)] + [(1, 128), (16, 512), (8, 64)]
mode = sys.argv[1] if len(sys.argv) > 1 else "check"
if mode == "check":
    cases = [(f"bench {s}", inputs(*s), torch.float32) for s in BENCH]
    cases += [(f"rows={r}", inputs(r, 256), torch.float32) for r in (3, 5, 31, 64, 512, 4096)]
    cases += [("fp16", inputs(7, 256, torch.float16), torch.float32), ("bf16 out", inputs(4, 256), torch.bfloat16),
              ("strided hidden", inputs(9, 256, strided=True), torch.float32), ("experts=100", inputs(5, 100), torch.float32),
              ("hidden=1000", inputs(6, 64, hidden=1000), torch.float32), ("fp32 inputs", inputs(4, 64, torch.float32), torch.float32)]
    bad = 0
    for name, (h, w), od in cases:
        ref = (h.float() @ w.float().t()).to(od).float()
        a = amd.dsv3_router_gemm(h, w, od).float()
        b = new.dsv3_router_gemm(h, w, od).float()
        tol = TOL[od]
        ok = b.dtype == ref.dtype and torch.allclose(b, ref, atol=tol, rtol=tol) and b.shape == ref.shape
        ea, eb = (a - ref).abs().max().item(), (b - ref).abs().max().item()
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:18s} max err master {ea:.2e} new {eb:.2e} (|ref| max {ref.abs().max().item():.1f})", flush=True)
    print(f"{len(cases) - bad}/{len(cases)} within the unit-test tolerance")
elif mode == "sweep":
    for s in BENCH + [(64, 256), (512, 256), (4096, 256)]:
        h, w = inputs(*s)
        tm = bench(lambda: amd.dsv3_router_gemm(h, w, torch.float32))
        res = []
        bms = (16,) if s[0] <= 16 else (16, 32, 64, 128)
        for bm, be, bk, nw, ns, sp in itertools.product(bms, (16, 32, 64), (128, 256, 512), (1, 2, 4, 8), (1, 2), (1, 2, 4, 8, 16)):
            if nw * 32 * 4 > bm * be or bm * be * bk // (nw * 32) > 4096:
                continue
            new._config = lambda r, e, c=(bm, be, bk, nw, ns, sp): c
            try:
                res.append((bench(lambda: new.dsv3_router_gemm(h, w, torch.float32)), (bm, be, bk, nw, ns, sp)))
            except Exception:  # noqa: BLE001
                continue
        res.sort()
        print(f"{s}: master {tm:6.1f} | " + " ".join(f"{c}:{t:.1f}" for t, c in res[:4]), flush=True)
else:
    for s in BENCH + [(64, 256), (512, 256), (4096, 256)]:
        h, w = inputs(*s)
        tm, tn = [], []
        for _ in range(3):
            tm.append(bench(lambda: amd.dsv3_router_gemm(h, w, torch.float32)))
            tn.append(bench(lambda: new.dsv3_router_gemm(h, w, torch.float32)))
        tm, tn = sorted(tm)[1], sorted(tn)[1]
        print(f"{str(s):12s} master {tm:8.1f} us  new {tn:7.1f} us  {tm / tn:5.2f}x", flush=True)
