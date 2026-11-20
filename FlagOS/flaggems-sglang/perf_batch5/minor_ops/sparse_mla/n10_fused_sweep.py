import importlib.util, itertools, os, sys, torch
from triton.testing import do_bench
T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, T)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
gen = load("gen_smla", os.path.join(T, "src/flaggems_sglang/ops/triton_sparse_mla_fwd.py"))
new = load("fused", "/tmp/n10_smla_fused.py")
bm = importlib.import_module("benchmark.test_triton_sparse_mla_fwd")
bench = lambda fn: do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3
for shape in ((128, 2048), (16, 512), (128, 128)):
    kw = next(iter(bm._input_fn(shape, torch.float8_e4m3fn, "cuda")))
    tg = bench(lambda: gen.triton_sparse_mla_fwd(**kw))
    res = []
    for bn, nw, ns, sp in itertools.product((128,), (4, 8), (1,), (1, 2, 4, 8)):
        if shape[1] // sp < bn:
            continue
        new._attend_config = lambda s, h, t, c=(bn, nw, ns, sp): c
        try:
            t = bench(lambda: new.triton_sparse_mla_fwd(**kw))
        except Exception:
            continue
        res.append((t, (bn, nw, ns, sp)))
    res.sort()
    print(f"{shape}: generic {tg:6.1f} | best {res[0][0]:6.1f} {res[0][1]} {tg / res[0][0]:4.2f}x | " + " ".join(f"{c}:{t:.1f}" for t, c in res[1:4]), flush=True)
