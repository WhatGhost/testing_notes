"""Compute-type ops on gfx1100 with the official benchmarks' inputs: the
implementation that runs, kernels per call, cold (do_bench) and warm
(back-to-back) time, a copy of the same bytes, and FLOPs from the reference
math as TFLOPs and share of the 106 TFLOPs bf16 matrix peak."""

import importlib
import json
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
import flaggems_sglang  # noqa: E402

PEAK = 106e12


def flops(op, s):
    if op == "chunk_kda_scaled_dot_kkt":
        b, t, h, k = s
        return 4 * b * t * h * 64 * k
    if op == "grouped_gemm":
        m, n, k, _ = s
        return 2 * m * n * k
    if op.startswith("step_"):
        tok = sum(s[0]) * s[2]
        a, b = (s[3], s[5]) if op in ("step_a_q", "step_b_v") else (s[3], s[4])
        return 2 * tok * a * b
    if op == "absorbed_bmm_concat_cast_q_fp8":
        return 2 * s[0] * s[1] * 128 * 512
    return 0


def unpack(inp):
    args, kwargs = [], {}
    for item in inp if isinstance(inp, tuple) else (inp,):
        (kwargs.update(item) if isinstance(item, dict) else args.append(item))
    return args, kwargs


def tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from tensors(o)
    elif hasattr(obj, "__dataclass_fields__"):
        for f in obj.__dataclass_fields__:
            yield from tensors(getattr(obj, f))


def warm(fn, n=50):
    for _ in range(5):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


ops = sys.argv[1].split(",")
rows = []
print(f"{'op':32s} {'shape':40s} impl  k  cold_us  warm_us  copy_us ratio  TFLOPs cold/warm (% peak)   GB/s cold/warm")
for op in ops:
    tm = importlib.import_module(f"benchmark.test_{op}")
    fn_op = flaggems_sglang.get_op(op)
    impl = fn_op.__module__.split(".backend.")[-1].replace(".ops.", "/") if "backend" in fn_op.__module__ else "generic"
    for shape in dict.fromkeys((*tm.SHAPES, *getattr(tm, "MORE_SHAPES", []))):
        args, kwargs = unpack(next(iter(tm._input_fn(shape, torch.bfloat16, "cuda"))))
        fn = lambda: fn_op(*args, **kwargs)  # noqa: E731
        out = fn()
        torch.cuda.synchronize()
        nbytes = sum(t.numel() * t.element_size() for t in tensors((args, kwargs))) + sum(
            t.numel() * t.element_size() for t in tensors(out))
        t_cold = do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3
        t_warm = warm(fn)
        src = torch.empty(max(nbytes // 4, 1), dtype=torch.bfloat16, device="cuda")
        dst = torch.empty_like(src)
        t_copy = do_bench(lambda: dst.copy_(src), warmup=25, rep=200, return_mode="median") * 1e3
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        kern = sum(1 for e in prof.events() if e.device_type.name == "CUDA")
        f = flops(op, shape)
        tf_c, tf_w = f / t_cold / 1e6, f / t_warm / 1e6
        row = dict(op=op, shape=str(shape), impl=impl, kernels=kern, cold=t_cold, warm=t_warm, copy=t_copy,
                   ratio=t_cold / t_copy, flops=f, mbytes=nbytes / 1e6)
        rows.append(row)
        print(f"{op:32s} {str(shape)[:40]:40s} {impl[:5]:5s} {kern:2d} {t_cold:8.1f} {t_warm:8.1f} {t_copy:8.1f} {t_cold / t_copy:5.2f}"
              f"  {tf_c:6.1f}/{tf_w:6.1f} ({tf_c * 1e12 / PEAK * 100:4.1f}%/{tf_w * 1e12 / PEAK * 100:4.1f}%)"
              f"  {nbytes / t_cold / 1e3:6.0f}/{nbytes / t_warm / 1e3:6.0f}", flush=True)
        del args, kwargs, out
        torch.cuda.empty_cache()
json.dump(rows, open("/tmp/c_survey.json", "w"), indent=1)
