import importlib.util, os, sys, torch
from triton.testing import do_bench
T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, T)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
new = load("new_awq", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/awq_gemm.py"))
bm = importlib.import_module("benchmark.test_awq_gemm")
bench = lambda fn: do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3
cands = [(32, 32, 4, 1, 8), (32, 32, 4, 1, 4), (32, 32, 4, 1, 1), (32, 64, 1, 1, 16), (32, 64, 8, 2, 1), (32, 64, 4, 1, 4), (64, 16, 2, 2, 16), (32, 64, 2, 1, 8)]
dts = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
for name in dts:
    for n in (1024, 4096):
        kw = next(iter(bm._input_fn((1, n), torch.float32, "cuda")))
        kw["input"] = kw["input"].to(dts[name]); kw["scales"] = kw["scales"].to(dts[name])
        row = []
        for c in cands:
            new._gemv_config = lambda n_, g_, c=c: c
            row.append(sorted(bench(lambda: new.awq_gemm(**kw)) for _ in range(3))[1])
        print(f"{name} (1, {n}): " + "  ".join(f"{c}:{t:.1f}" for c, t in zip(cands, row)), flush=True)
