"""rdna3 awq_gemm vs the generic op: official benchmark inputs (fp32, k=4096,
group 128, split_k 8) and the same shapes with fp16 / bf16 activations and
scales. do_bench median, 3 alternating rounds, middle value."""

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


gen = load("gen_awq", os.path.join(T, "src/flaggems_sglang/ops/awq_gemm.py"))
new = load("new_awq", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/awq_gemm.py"))
bm = importlib.import_module("benchmark.test_awq_gemm")


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
sel = [a for a in sys.argv[1:] if a in dtypes] or ["fp32", "fp16"]
shapes = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1:] if "," in a] or list(bm.SHAPES) + list(bm.MORE_SHAPES)
for name in sel:
    for shape in shapes:
        kw = next(iter(bm._input_fn(shape, torch.float32, "cuda")))
        kw["input"] = kw["input"].to(dtypes[name])
        kw["scales"] = kw["scales"].to(dtypes[name])
        tg, tn = [], []
        for _ in range(3):
            tg.append(bench(lambda: gen.awq_gemm(**kw)))
            tn.append(bench(lambda: new.awq_gemm(**kw)))
        tg, tn = sorted(tg)[1], sorted(tn)[1]
        m, n = shape
        tflops = 2 * m * n * 4096 / tn * 1e-6
        print(f"{name} {str(shape):12s} generic {tg:7.1f} us  new {tn:7.1f} us  {tg / tn:5.2f}x  ({tflops:5.1f} TFLOPs)", flush=True)
