import importlib, importlib.util, os, sys, torch
from torch.profiler import profile, ProfilerActivity
W = open("/tmp/fgs-a.path").read().strip()
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
gen = load("gen_pre", os.path.join(W, "src/flaggems_sglang/ops/ple_prefill_short_conv_.py"))
new = load("new_pre", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/ple_prefill_short_conv_.py"))
bp = importlib.import_module("benchmark.test_ple_prefill_short_conv_")
*a, kw = next(iter(bp._input_fn((1, 1024), torch.bfloat16, "cuda")))
state = a[2]; snap = state.clone()
for name, mod in (("generic", gen), ("rdna3", new)):
    for _ in range(5): state.copy_(snap); mod.ple_prefill_short_conv_(*a, **kw)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        for _ in range(10):
            state.copy_(snap); torch.cuda.synchronize(); mod.ple_prefill_short_conv_(*a, **kw); torch.cuda.synchronize()
    tot = {}
    for e in p.events():
        if e.device_type.name == "CUDA" and "ple" in e.name:
            tot[e.name] = tot.get(e.name, 0) + e.device_time_total / 10
    print(name, ", ".join(f"{k[:34]} {v:.1f}us" for k, v in tot.items()), flush=True)
