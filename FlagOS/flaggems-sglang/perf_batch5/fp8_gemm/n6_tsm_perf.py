"""rdna3 triton_scaled_mm vs the generic op on the official benchmark shapes
and inputs (same process, alternating, 3 rounds of do_bench medians), plus
the packed / widened crossover over M."""

import importlib.util
import json
import os
import statistics
import sys

import torch
from triton.testing import do_bench

T = open("/tmp/fgs-f8.path").read().strip()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_tsm", os.path.join(T, "src/flaggems_sglang/ops/triton_scaled_mm.py"))
new = load("new_tsm", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/triton_scaled_mm.py"))


def inputs(m, k, n):
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.clamp(0.1 * torch.randn(m, k, dtype=torch.float16, device="cuda", generator=g), -0.3, 0.3).to(
        torch.float8_e4m3fn)
    b = torch.clamp(0.1 * torch.randn(k, n, dtype=torch.float16, device="cuda", generator=g), -0.3, 0.3).to(
        torch.float8_e4m3fn)
    sa = 0.1 + 0.05 * torch.rand(m, 1, dtype=torch.float32, device="cuda", generator=g)
    sb = 0.1 + 0.05 * torch.rand(n, 1, dtype=torch.float32, device="cuda", generator=g)
    return dict(input=a, weight=b, scale_a=sa, scale_b=sb, out_dtype=torch.float16)


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


rows = []
shapes = [(m, k, n) for m in (1, 8, 64, 512, 4096) for k, n in ((4096, 4096), (4096, 7168))] + [(32, 32, 32), (17, 64, 96)]
for m, k, n in shapes:
    kw = inputs(m, k, n)
    tg, tn = [], []
    for _ in range(3):
        tg.append(bench(lambda: gen.triton_scaled_mm(**kw)))
        tn.append(bench(lambda: new.triton_scaled_mm(**kw)))
    g_, n_ = statistics.median(tg), statistics.median(tn)
    rows.append(dict(m=m, k=k, n=n, master=g_, new=n_))
    print(f"official M={m:5d} K={k:5d} N={n:5d}: master {g_:9.1f} us  new {n_:9.1f} us  {g_ / n_:5.2f}x", flush=True)

if len(sys.argv) > 1 and sys.argv[1] == "crossover":
    for n in (4096, 7168):
        for m in (256, 512, 768, 1024, 1536, 2048):
            kw = inputs(m, 4096, n)
            a, b = kw["input"], kw["weight"]
            out = torch.empty(m, n, device="cuda", dtype=torch.float16)
            args = (out, out, kw["scale_a"], kw["scale_b"], 0, 0, new._tl_dtype(torch.float16), False, 0, 0)
            tp = bench(lambda: new._packed(a, b, *args))
            tf = bench(lambda: new._fast(a, b, *args))
            tg = bench(lambda: gen.triton_scaled_mm(**kw))
            print(f"crossover N={n} M={m:5d}: packed {tp:8.1f} us  widened {tf:8.1f} us  master {tg:8.1f} us", flush=True)
json.dump(rows, open("/tmp/n6_tsm_perf.json", "w"), indent=1)
