"""rdna3 grouped_gemm: config sweep per shape (cold do_bench), correctness of
the best config against the reference, and the generic op's time."""

import importlib
import importlib.util
import itertools
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
gg = importlib.import_module("flaggems_sglang.ops.grouped_gemm")
from flaggems_sglang.reference.grouped_gemm import reference  # noqa: E402

spec = importlib.util.spec_from_file_location("gg_new", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/grouped_gemm.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
bm_mod = importlib.import_module("benchmark.test_grouped_gemm")


def inputs(shape):
    return next(iter(bm_mod._input_fn(shape, torch.bfloat16, "cuda")))


shapes = [(128, 512, 2048, 8), (128, 512, 2048, 32), (1024, 512, 2048, 8), (1024, 512, 2048, 32),
          (4096, 512, 2048, 8), (4096, 512, 2048, 32)]
if len(sys.argv) > 1 and sys.argv[1] == "more":
    shapes = [(4096, 1536, 2048, 128), (4096, 2048, 768, 128), (1024, 1536, 2048, 64), (512, 512, 2048, 8),
              (2048, 512, 2048, 32), (256, 1536, 2048, 128), (8192, 512, 2048, 8)]
import json  # noqa: E402

PRUNED = json.load(open("/tmp/prune_regs.json"))["grouped_gemm"]
for shape in shapes:
    cfgs = [tuple(c[2:7]) for c in PRUNED if (c[0], c[1]) == (shape[1], shape[2])]
    a, b, e, counts, offs, boffs, sched = inputs(shape)
    ref = reference(a, b, e, counts, offs, boffs, sched)
    tg = do_bench(lambda: gg.grouped_gemm(a, b, e, counts, offs, boffs, sched), warmup=10, rep=50, return_mode="median") * 1e3
    res = []
    for cfg in cfgs:
        new._config = lambda m, n, k, ex, c=cfg: c
        try:
            res.append((do_bench(lambda: new.grouped_gemm(a, b, e, counts, offs, boffs, sched), warmup=5, rep=25,
                                 return_mode="median") * 1e3, cfg))
        except Exception:  # noqa: BLE001
            continue
    res.sort()
    top = []
    for t, cfg in res[:6]:
        new._config = lambda m, n, k, ex, c=cfg: c
        t2 = do_bench(lambda: new.grouped_gemm(a, b, e, counts, offs, boffs, sched), warmup=10, rep=100, return_mode="median") * 1e3
        top.append((t2, cfg))
    top.sort()
    t, cfg = top[0]
    new._config = lambda m, n, k, ex, c=cfg: c
    out = new.grouped_gemm(a, b, e, counts, offs, boffs, sched)
    err = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), ref.float(), atol=1.5e-2, rtol=1.5e-2)
    f = 2 * shape[0] * shape[1] * shape[2]
    print(f"{str(shape):24s} generic {tg:7.1f} | best {cfg} {t:7.1f} ({tg / t:4.2f}x, {f / t / 1e6:5.1f} TFLOPs) ok={ok} err {err:.1e} | "
          + " ".join(f"{c}:{x:.1f}" for x, c in top[1:4]), flush=True)
