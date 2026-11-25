"""New-op survey on gfx1100 with the official benchmarks' own inputs (third-round
method): op time, a copy moving the same total bytes (read half, write half),
the host share in back-to-back calls, kernels per call, and the largest tensors
so the byte count can be checked op by op.

Handles both benchmark styles: a baseline workload (SHAPES / DTYPES /
_case_fn / _build_inputs) and the newer module-level SHAPES with _input_fn."""

import importlib
import inspect
import json
import os
import re
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile
from triton.testing import do_bench

S = open("/tmp/fgs-s7.path").read().strip()
sys.path.insert(0, S)
sys.path.insert(0, os.path.join(S, "src"))
import flaggems_sglang  # noqa: E402
from flaggems_sglang.reference import get_reference  # noqa: E402
from flaggems_sglang.reference_candidate import get_candidate  # noqa: E402

OPS = sys.argv[1].split(",")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/s7_survey.json"


def unpack(inp):
    args, kwargs = [], {}
    for item in inp if isinstance(inp, tuple) else (inp,):
        if isinstance(item, (torch.Tensor, int, float, list, tuple, torch.dtype)) or item is None:
            args.append(item)
        elif isinstance(item, dict):
            kwargs.update(item)
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


def named_tensors(args, kwargs):
    for i, a in enumerate(args):
        for t in tensors(a):
            yield f"arg{i}", t
    for k, val in kwargs.items():
        for t in tensors(val):
            yield k, t


def cases(op):
    tm = importlib.import_module(f"benchmark.test_{op}")
    if hasattr(tm, "workload"):
        wl = tm.workload
        cand = get_candidate(op, reference=get_reference(op))
        inplace = getattr(wl, "IS_INPLACE", False)
        for shape in dict.fromkeys((*wl.SHAPES, *getattr(wl, "MORE_SHAPES", []))):
            for dtype in wl.DTYPES:
                for plan in tm._case_fn(shape, dtype):
                    yield cand, inplace, shape, dtype, lambda p=plan, d=dtype: unpack(tm._build_inputs(p, d, "cuda"))
        return
    cand = flaggems_sglang.get_op(op) or get_candidate(op, reference=get_reference(op))
    src = inspect.getsource(tm)
    m = re.search(r"dtypes=\[([^\]]*)\]", src)
    dtypes = eval("[" + m.group(1) + "]", {"torch": torch}) if m else [torch.bfloat16]
    shapes = list(dict.fromkeys((*getattr(tm, "SHAPES", []), *getattr(tm, "MORE_SHAPES", []))))
    if not shapes:
        # newer benchmarks pass the shapes inline to OpBenchmark(...)
        for name in ("shapes", "more_shapes"):
            m = re.search(rf"^\s*{name}=(.+),\s*$", src, re.M)
            if m:
                shapes += list(eval(m.group(1), vars(tm)))
        shapes = list(dict.fromkeys(shapes))
    for shape in shapes:
        for dtype in dtypes:
            try:
                n = sum(1 for _ in tm._input_fn(shape, dtype, "cuda"))
            except Exception as e:  # noqa: BLE001
                print(f"{op} {str(shape)[:40]}: BAD SHAPE in the benchmark itself ({type(e).__name__}: {str(e)[:80]})",
                      flush=True)
                continue
            for k in range(n):
                def build(s=shape, d=dtype, k=k):
                    it = iter(tm._input_fn(s, d, "cuda"))
                    for _ in range(k):
                        next(it)
                    return unpack(next(it))
                yield cand, None, shape, dtype, build


rows = []
for op in OPS:
    try:
        cs = list(cases(op))
    except Exception as e:  # noqa: BLE001
        print(f"{op}: cannot load ({str(e)[:160]})", flush=True)
        continue
    for cand, inplace, shape, dtype, build in cs:
        key = str(shape)
        try:
            args, kwargs = build()
            fn = lambda: cand(*args, **kwargs)  # noqa: E731
            out = fn()
            torch.cuda.synchronize()
            ins = list(named_tensors(args, kwargs))
            in_bytes = sum(t.numel() * t.element_size() for _, t in ins)
            outs = list(tensors(out))
            out_bytes = sum(t.numel() * t.element_size() for t in outs)
            returns_none = out is None
            nbytes = max(in_bytes + out_bytes, 8)
            t_op = do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3
            src_t = torch.empty(max(nbytes // 4, 1), dtype=torch.bfloat16, device="cuda")
            dst_t = torch.empty_like(src_t)
            t_copy = do_bench(lambda: dst_t.copy_(src_t), warmup=25, rep=100, return_mode="median") * 1e3
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            n = 50
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            t_cont = (time.perf_counter() - t0) / n * 1e6
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            kernels = sum(1 for e in prof.events() if e.device_type.name == "CUDA")
            big = sorted(ins, key=lambda nt: -nt[1].numel() * nt[1].element_size())[:3]
            big_s = ", ".join(f"{nm}{list(t.shape)}:{str(t.dtype).replace('torch.', '')}" for nm, t in big)
            row = dict(op=op, dtype=str(dtype).split(".")[-1], shape=key, t_op=t_op, t_copy=t_copy,
                       ratio=t_op / t_copy, t_cont=t_cont, host=max(0.0, 1 - t_op / t_cont) if t_cont > 0 else 0.0,
                       kernels=kernels, mbytes=nbytes / 1e6, in_mb=in_bytes / 1e6, out_mb=out_bytes / 1e6,
                       returns_none=returns_none, inplace=inplace, big=big_s)
            rows.append(row)
            print(f"{op:34s} {row['dtype']:8s} {key[:34]:34s} {t_op:9.1f}us copy {t_copy:8.1f} ratio {row['ratio']:5.2f} "
                  f"cont {t_cont:8.1f} host {row['host']:4.0%} k{kernels} in {in_bytes / 1e6:8.2f}MB out "
                  f"{out_bytes / 1e6:7.2f}MB{' (None)' if returns_none else ''} | {big_s[:90]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{op} {key[:40]}: ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
        torch.cuda.empty_cache()
json.dump(rows, open(OUT, "w"), indent=1)
