"""rdna3 triton_sparse_mla_fwd (fp8 gathers) vs the generic op on the benchmark
shapes: max |diff| / max |generic| and do_bench median over 3 alternating
rounds."""

import importlib.util
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
new = load("new_smla", sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1].endswith(".py") else os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/triton_sparse_mla_fwd.py"))
bm = importlib.import_module("benchmark.test_triton_sparse_mla_fwd")


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


shapes = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1:]] or list(bm.SHAPES) + list(bm.MORE_SHAPES)
for shape in shapes:
    kw = next(iter(bm._input_fn(shape, torch.float8_e4m3fn, "cuda")))
    ref = gen.triton_sparse_mla_fwd(**kw).float()
    out = new.triton_sparse_mla_fwd(**kw).float()
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    tg, tn = [], []
    for _ in range(3):
        tg.append(bench(lambda: gen.triton_sparse_mla_fwd(**kw)))
        tn.append(bench(lambda: new.triton_sparse_mla_fwd(**kw)))
    tg, tn = sorted(tg)[1], sorted(tn)[1]
    print(f"shape {str(shape):12s} generic {tg:7.1f} us  new {tn:7.1f} us  {tg / tn:4.2f}x  diff {err:.1e}", flush=True)
