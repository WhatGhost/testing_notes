import importlib, importlib.util, itertools, os, sys, torch
from triton.testing import do_bench
W = open("/tmp/fgs-a.path").read().strip()
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))
spec = importlib.util.spec_from_file_location("pre", os.path.join(W, "src/flaggems_sglang/ops/ple_prefill_short_conv_.py"))
pre = importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)
bp = importlib.import_module("benchmark.test_ple_prefill_short_conv_")
bench = lambda fn: do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3
for shape in ((1, 1024), (4, 1024)):
    *a, kw = next(iter(bp._input_fn(shape, torch.bfloat16, "cuda")))
    base = bench(lambda: pre.ple_prefill_short_conv_(*a, **kw))
    res = []
    for bt, bc in itertools.product((8, 16, 32, 64, 128), (64, 128, 256, 512)):
        if bt * bc > 16384:
            continue
        res.append((bench(lambda: pre.ple_prefill_short_conv_(*a, **kw, block_t=bt, block_c=bc)), (bt, bc)))
    res.sort()
    print(f"{shape}: default {base:.1f} | " + " ".join(f"{c}:{t:.1f}" for t, c in res[:6]), flush=True)
