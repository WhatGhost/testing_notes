"""rdna3 triton_sparse_mla_fwd (fused online softmax-values) vs the generic op
on the benchmark shapes: config sweep of the values kernel, accuracy vs the
generic op (max |diff| / max |out|)."""

import importlib.util
import itertools
import os
import sys

import torch
from triton.testing import do_bench

T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, T)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_smla", os.path.join(T, "src/flaggems_sglang/ops/triton_sparse_mla_fwd.py"))
new = load("new_smla", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/triton_sparse_mla_fwd.py"))
bm = importlib.import_module("benchmark.test_triton_sparse_mla_fwd")


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


for shape in list(bm.SHAPES) + list(bm.MORE_SHAPES):
    kw = next(iter(bm._input_fn(shape, torch.float8_e4m3fn, "cuda")))
    ref = gen.triton_sparse_mla_fwd(**kw).float()
    tg = bench(lambda: gen.triton_sparse_mla_fwd(**kw))
    res = []
    for bn, bv, nw, ns, sp in itertools.product((32, 64), (64, 128, 256), (4, 8), (1,), (1, 2, 4, 8, 16)):
        if sp > 1 and shape[1] // sp < bn:
            continue
        new._values_config = lambda s, h, t, c=(bn, bv, nw, ns, sp): c
        try:
            out = new.triton_sparse_mla_fwd(**kw).float()
            t = bench(lambda: new.triton_sparse_mla_fwd(**kw))
        except Exception as e:  # noqa: BLE001
            continue
        err = ((out - ref).abs().max() / ref.abs().max()).item()
        res.append((t, (bn, bv, nw, ns), err))
    res.sort()
    t, cfg, err = res[0]
    print(f"shape {shape}: generic {tg:7.1f} us | best {t:7.1f} us {cfg} ({tg / t:4.2f}x) diff {err:.1e} | "
          f"{' '.join(f'{c}:{tt:.1f}' for tt, c, _ in res[1:4])}", flush=True)
