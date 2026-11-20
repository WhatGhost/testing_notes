import importlib.util, os, sys, time, torch
T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, T)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
gen = load("gen_awq", os.path.join(T, "src/flaggems_sglang/ops/awq_gemm.py"))
new = load("new_awq", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/awq_gemm.py"))
bm = importlib.import_module("benchmark.test_awq_gemm")
for shape in ((1, 1024), (16, 32)):
    kw = next(iter(bm._input_fn(shape, torch.float32, "cuda")))
    for name, fn in (("gen", gen.awq_gemm), ("new", new.awq_gemm)):
        for split in ((1, 16) if name == "new" else (None,)):
            if split:
                new._config = lambda m, n, g, gr, h, s=split: (16, 32, 64, 2, 1, s)
            for _ in range(20): fn(**kw)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(2000): fn(**kw)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f"{shape} {name} split={split}: CPU {1e6 * (t1 - t0) / 2000:.1f} us/call, wall {1e6 * (t2 - t0) / 2000:.1f} us/call", flush=True)
