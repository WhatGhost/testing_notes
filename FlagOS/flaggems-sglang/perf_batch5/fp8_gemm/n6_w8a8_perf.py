"""rdna3 w8a8_block_fp8_matmul vs the generic op on the official benchmark
shapes and inputs (same process, alternating, 3 rounds of do_bench medians);
with 'halves' also forces the decode + halves path at medium M."""

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


gen = load("gen_w8", os.path.join(T, "src/flaggems_sglang/ops/w8a8_block_fp8_matmul.py"))
new = load("new_w8", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/w8a8_block_fp8_matmul.py"))
FP8 = torch.float8_e4m3fn


def inputs(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(0)
    fmax = torch.finfo(FP8).max
    A = torch.randn(m, k, device="cuda", generator=g).clamp(-fmax, fmax).to(FP8)
    B = torch.randn(n, k, device="cuda", generator=g).clamp(-fmax, fmax).to(FP8)
    As = (1e-2 * torch.rand(m, k // 128, device="cuda", generator=g)).contiguous()
    Bs = (1e-2 * torch.rand(n // 128, k // 128, device="cuda", generator=g)).contiguous()
    return dict(A=A, B=B, As=As, Bs=Bs, block_size=[128, 128], output_dtype=torch.bfloat16)


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


mode = sys.argv[1] if len(sys.argv) > 1 else "official"
rows = []
if mode == "official":
    shapes = [(m, n, k) for m in (1, 8, 64, 512, 4096) for n, k in ((1024, 4096), (4096, 4096), (7168, 4096))]
    shapes += [(7, 256, 512), (64, 1024, 512), (256, 1024, 4096)]
    for m, n, k in shapes:
        kw = inputs(m, n, k)
        tg, tn = [], []
        for _ in range(3):
            tg.append(bench(lambda: gen.w8a8_block_fp8_matmul(**kw)))
            tn.append(bench(lambda: new.w8a8_block_fp8_matmul(**kw)))
        g_, n_ = statistics.median(tg), statistics.median(tn)
        rows.append(dict(m=m, n=n, k=k, master=g_, new=n_))
        print(f"official M={m:5d} N={n:5d} K={k:5d}: master {g_:9.1f} us  new {n_:9.1f} us  {g_ / n_:5.2f}x", flush=True)
    json.dump(rows, open("/tmp/n6_w8a8_perf.json", "w"), indent=1)
elif mode == "halves":
    for n in (1024, 4096, 7168):
        for m in (32, 64, 96, 128, 192, 256):
            kw = inputs(m, n, 4096)
            th = bench(lambda: new._halves(kw["A"], kw["B"], kw["As"], kw["Bs"], torch.bfloat16))
            tg = bench(lambda: gen.w8a8_block_fp8_matmul(**kw))
            print(f"halves N={n} M={m:4d}: {th:8.1f} us  master {tg:8.1f} us", flush=True)

if mode == "cross":
    for n in (1024, 4096, 7168):
        for m in (64, 96, 128, 192, 256, 384):
            kw = inputs(m, n, 4096)
            a, b, sa, sb = kw["A"], kw["B"], kw["As"], kw["Bs"]
            ts = bench(lambda: new._small(a, b, sa, sb, torch.bfloat16))
            th = bench(lambda: new._halves(a, b, sa, sb, torch.bfloat16))
            tg = bench(lambda: gen.w8a8_block_fp8_matmul(**kw))
            print(f"cross N={n} M={m:4d}: small {ts:8.1f} us  halves {th:8.1f} us  master {tg:8.1f} us", flush=True)
