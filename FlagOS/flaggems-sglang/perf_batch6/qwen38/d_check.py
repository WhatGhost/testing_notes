"""rdna3 qwen4_hc_inject_combine / qwen4_hc_gate_reduce vs the generic ops:
bitwise outputs (benchmark inputs, fp16, 3-D inputs), a block/warps sweep and
timings."""

import importlib
import importlib.util
import itertools
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-a.path").read().strip()
sys.path.insert(0, W)
sys.path.insert(0, os.path.join(W, "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ops = {}
for op in ("qwen4_hc_inject_combine", "qwen4_hc_gate_reduce"):
    ops[op] = (load(f"gen_{op}", os.path.join(W, f"src/flaggems_sglang/ops/{op}.py")),
               load(f"new_{op}", os.path.join(W, f"src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/{op}.py")),
               importlib.import_module(f"benchmark.test_{op}").make_inputs)


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


def to_dtype(args, dt):
    return tuple(a.to(dt) if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args)


mode = sys.argv[1] if len(sys.argv) > 1 else "check"
for op, (gen, new, make) in ops.items():
    fn_name = op
    if mode == "check":
        bad = 0
        cases = []
        for rows in (1, 7, 64, 1024, 16384):
            cases.append((f"bf16 rows={rows}", make(rows, "cuda")))
        cases.append(("fp16 rows=333", to_dtype(make(333, "cuda"), torch.float16)))
        a = make(64, "cuda")
        cases.append(("3-D [8, 8, ...]", tuple(t.reshape(8, 8, *t.shape[1:]) if isinstance(t, torch.Tensor) and t.is_contiguous()
                                                else (t.unflatten(0, (8, 8)) if isinstance(t, torch.Tensor) else t) for t in a)))
        for name, args in cases:
            x = getattr(gen, fn_name)(*args)
            y = getattr(new, fn_name)(*args)
            same = x.shape == y.shape and x.dtype == y.dtype and torch.equal(x.view(torch.int16), y.view(torch.int16))
            bad += not same
            print(f"{'ok  ' if same else 'FAIL'} {op:26s} {name}", flush=True)
        print(f"{op}: {len(cases) - bad}/{len(cases)} bitwise identical", flush=True)
    elif mode == "sweep":
        for rows in (1, 64, 1024, 16384):
            args = make(rows, "cuda")
            tg = bench(lambda: getattr(gen, fn_name)(*args))
            res = []
            for bh, nw in itertools.product((256, 512, 1024, 2048, 4096), (1, 2, 4, 8, 16)):
                if nw * 32 * 2 > bh:
                    continue
                new._config = lambda h, c=(bh, nw): c
                res.append((bench(lambda: getattr(new, fn_name)(*args)), (bh, nw)))
            res.sort()
            print(f"{op} rows={rows:5d}: generic {tg:7.1f} | " + " ".join(f"{c}:{t:.1f}" for t, c in res[:5]), flush=True)
    else:
        for rows in (1, 7, 64, 128, 256, 512, 1024, 16384):
            args = make(rows, "cuda")
            tg, tn = [], []
            for _ in range(3):
                tg.append(bench(lambda: getattr(gen, fn_name)(*args)))
                tn.append(bench(lambda: getattr(new, fn_name)(*args)))
            tg, tn = sorted(tg)[1], sorted(tn)[1]
            print(f"{op} rows={rows:5d}: generic {tg:8.1f} us  new {tn:8.1f} us  {tg / tn:5.2f}x", flush=True)
