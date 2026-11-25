import importlib, importlib.util, os, sys, torch
from torch.profiler import profile, ProfilerActivity
W = open("/tmp/fgs-b.path").read().strip()
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))
spec = importlib.util.spec_from_file_location("gen_kda", os.path.join(W, "src/flaggems_sglang/ops/fused_recurrent_kda.py"))
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
bm = importlib.import_module("benchmark.test_fused_recurrent_kda")
src = open(os.path.join(W, "benchmark/test_fused_recurrent_kda.py")).read()
import re
for shape in ((1, 64), (4, 64), (16, 64), (64, 64), (32, 128)):
    args = next(iter(bm._input_fn(shape, torch.bfloat16, "cuda")))
    a, kw = (args, {}) if not isinstance(args, dict) else ((), args)
    if isinstance(args, tuple) and isinstance(args[-1], dict):
        a, kw = args[:-1], args[-1]
    fn = lambda: gen.fused_recurrent_kda(*a, **kw)
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): fn()
        torch.cuda.synchronize()
    tot = {}
    for e in p.events():
        if e.device_type.name == "CUDA":
            tot[e.name] = tot.get(e.name, 0) + e.device_time_total / 10
    print(shape, ", ".join(f"{k[:20]} {v:.1f}us" for k, v in sorted(tot.items(), key=lambda x: -x[1])), flush=True)
