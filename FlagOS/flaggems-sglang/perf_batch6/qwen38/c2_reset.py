"""ple_prefill_short_conv_ timed as the official harness does for in-place
ops (restore the state, sync, events around one call, median of 25), for the
generic op and rdna3 configs."""

import importlib
import importlib.util
import itertools
import os
import statistics
import sys

import torch

W = open("/tmp/fgs-a.path").read().strip()
sys.path.insert(0, W)
sys.path.insert(0, os.path.join(W, "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_pre", os.path.join(W, "src/flaggems_sglang/ops/ple_prefill_short_conv_.py"))
new = load("new_pre", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/ple_prefill_short_conv_.py"))
bp = importlib.import_module("benchmark.test_ple_prefill_short_conv_")


def reset_time(fn, state):
    snap = state.clone()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        state.copy_(snap)
        fn()
    samples = []
    for _ in range(25):
        state.copy_(snap)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    state.copy_(snap)
    return statistics.median(samples)


for shape in ((1, 1024), (4, 1024)):
    *a, kw = next(iter(bp._input_fn(shape, torch.bfloat16, "cuda")))
    state = a[2]
    tg = statistics.median(reset_time(lambda: gen.ple_prefill_short_conv_(*a, **kw), state) for _ in range(3))
    res = []
    for bt, bc, nw in itertools.product((8, 16, 32), (128, 256, 512, 1024), (4, 8, 16)):
        per_lane = bt * bc // (nw * 32)
        if per_lane > 64 or per_lane < 2:
            continue
        new._prefill_config = lambda c, cfg=(bt, bc, nw): cfg
        t = statistics.median(reset_time(lambda: new.ple_prefill_short_conv_(*a, **kw), state) for _ in range(3))
        res.append((t, (bt, bc, nw)))
    res.sort()
    print(f"{shape}: generic {tg:.1f} | " + " ".join(f"{c}:{t:.1f}" for t, c in res[:6]), flush=True)
