import importlib.util, os, sys, torch
from triton.testing import do_bench
T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, T)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
gen = load("gen_awq", os.path.join(T, "src/flaggems_sglang/ops/awq_gemm.py"))
new = load("new_awq", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/awq_gemm.py"))
bm = importlib.import_module("benchmark.test_awq_gemm")
bench = lambda fn: do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3
cfgs = [(16, 32, 32, 1, 1, 8), (16, 32, 32, 1, 2, 8), (16, 32, 32, 1, 1, 4), (16, 64, 32, 1, 1, 8), (16, 64, 32, 2, 1, 8),
        (32, 32, 32, 1, 1, 8), (32, 32, 32, 2, 1, 4), (32, 64, 32, 2, 1, 4), (32, 64, 32, 2, 2, 2), (16, 32, 64, 1, 1, 8),
        (32, 64, 64, 2, 1, 2), (64, 32, 32, 2, 1, 4), (16, 128, 32, 2, 1, 4), (32, 128, 32, 4, 1, 2)]
for shape in ((512, 4096), (512, 1024), (64, 4096), (64, 1024)):
    kw = next(iter(bm._input_fn(shape, torch.float32, "cuda")))
    tg = bench(lambda: gen.awq_gemm(**kw))
    res = []
    for c in cfgs:
        new._config = lambda m, n, g, gr, h, c=c: c
        res.append((bench(lambda: new.awq_gemm(**kw)), c))
    res.sort()
    print(f"fp32 {shape}: generic {tg:6.1f} | " + " ".join(f"{c}:{t:.1f}" for t, c in res[:5]), flush=True)
