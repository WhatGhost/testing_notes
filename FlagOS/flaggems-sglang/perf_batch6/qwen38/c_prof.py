import os, sys, torch
from torch.profiler import profile, ProfilerActivity
W = open("/tmp/fgs-a.path").read().strip()
sys.path.insert(0, os.path.join(W, "src")); sys.path.insert(0, W)
import importlib
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
pre = load("pre", os.path.join(W, "src/flaggems_sglang/ops/ple_prefill_short_conv_.py"))
dec = load("dec", os.path.join(W, "src/flaggems_sglang/ops/ple_short_conv_.py"))
bp = importlib.import_module("benchmark.test_ple_prefill_short_conv_")
bd = importlib.import_module("benchmark.test_ple_short_conv_")
def prof(name, fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): fn()
        torch.cuda.synchronize()
    tot = {}
    for e in p.events():
        if e.device_type.name == "CUDA":
            tot[e.name] = tot.get(e.name, 0) + e.device_time_total / 10
    print(name, ", ".join(f"{k[:34]} {v:.1f}us" for k, v in sorted(tot.items(), key=lambda x: -x[1])), flush=True)
for shape in ((1, 1024), (4, 1024)):
    *a, kw = next(iter(bp._input_fn(shape, torch.bfloat16, "cuda")))
    prof(f"prefill {shape}:", lambda: pre.ple_prefill_short_conv_(*a, **kw))
for shape in ((64,), (1024,)):
    *a, kw = next(iter(bd._input_fn(shape, torch.bfloat16, "cuda")))
    prof(f"decode {shape}:", lambda: dec.ple_short_conv_(*a, **kw))
