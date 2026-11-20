"""Config sweep of the rdna3 awq_gemm kernel per (dtype, shape): quick timing of
every config, then the best 4 re-timed against the generic op over 3 rounds."""

import importlib.util
import itertools
import os
import sys

import torch
import triton
from triton.testing import do_bench

T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, T)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_awq", os.path.join(T, "src/flaggems_sglang/ops/awq_gemm.py"))
new = load("new_awq", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/awq_gemm.py"))
bm = importlib.import_module("benchmark.test_awq_gemm")
dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def quick(fn):
    return do_bench(fn, warmup=5, rep=25, return_mode="median") * 1e3


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


jobs = [a.split(":") for a in sys.argv[1:]]  # dtype:m,n
for name, shp in jobs:
    m, n = (int(x) for x in shp.split(","))
    kw = next(iter(bm._input_fn((m, n), torch.float32, "cuda")))
    kw["input"] = kw["input"].to(dtypes[name])
    kw["scales"] = kw["scales"].to(dtypes[name])
    hilo = name == "fp32"
    bms = [b for b in (16, 32, 64, 128) if b <= max(16, triton.next_power_of_2(m))]
    if m >= 256:
        bms = [64, 128]
    res = []
    for bmm, bn, bk, nw, ns in itertools.product(bms, (32, 64, 128, 256), (32, 64, 128), (1, 2, 4, 8), (1, 2)):
        if bn > max(32, n) or nw * 256 > bmm * bn or bmm * bn > (16384 if hilo else 32768):
            continue
        if m >= 256 and (bn < 64 or nw < 4):
            continue
        tiles = triton.cdiv(m, bmm) * triton.cdiv(n, bn)
        for sp in (1, 2, 4, 8, 16):
            if sp > 1 and tiles * sp > 1536:
                continue
            if sp < 16 and tiles * sp < 48:
                continue
            new._config = lambda m_, n_, g_, gr_, h_, c=(bmm, bn, bk, nw, ns, sp): c
            try:
                t = quick(lambda: new.awq_gemm(**kw))
            except Exception:  # noqa: BLE001
                continue
            res.append((t, (bmm, bn, bk, nw, ns, sp)))
    res.sort()
    top = []
    for _, cfg in res[:4]:
        new._config = lambda m_, n_, g_, gr_, h_, c=cfg: c
        ts = []
        for _ in range(3):
            ts.append(bench(lambda: new.awq_gemm(**kw)))
        top.append((sorted(ts)[1], cfg))
    top.sort()
    tg = sorted(bench(lambda: gen.awq_gemm(**kw)) for _ in range(3))[1]
    t, cfg = top[0]
    print(f"{name} ({m}, {n}): generic {tg:7.1f} us | best {t:7.1f} us {cfg} {tg / t:5.2f}x | "
          + " ".join(f"{c}:{tt:.1f}" for tt, c in top[1:]) + f" | {len(res)} configs", flush=True)
