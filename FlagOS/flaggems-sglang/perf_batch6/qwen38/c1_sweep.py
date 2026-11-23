import importlib, importlib.util, itertools, os, sys, torch
from triton.testing import do_bench
W = open("/tmp/fgs-a.path").read().strip()
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
gen = load("gen_ple", os.path.join(W, "src/flaggems_sglang/ops/ple_short_conv_.py"))
new = load("new_ple", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/ple_short_conv_.py"))
bm = importlib.import_module("benchmark.test_ple_short_conv_")
bench = lambda fn: do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3
for rows in (1, 7, 64, 256, 1024):
    *a, kw = next(iter(bm._input_fn((rows,), torch.bfloat16, "cuda")))
    tg = bench(lambda: gen.ple_short_conv_(*a, **kw))
    res = []
    for bc, nw in itertools.product((64, 128, 256, 512, 1024), (1, 2, 4, 8)):
        if nw * 32 > bc or bc * 16 // (nw * 32) > 64:
            continue
        new._decode_config = lambda t, c, cfg=(bc, nw): cfg
        res.append((bench(lambda: new.ple_short_conv_(*a, **kw)), (bc, nw)))
    res.sort()
    print(f"rows={rows:5d}: generic {tg:7.1f} | " + " ".join(f"{c}:{t:.1f}" for t, c in res[:5]), flush=True)
