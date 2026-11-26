"""grouped_gemm rule search: the generic op and 20 rdna3 candidates (10 tiles
x 2 launch orders) on a grid of shapes, cold medians; one (N, K) group per
process. Writes json for the rule fit."""

import importlib
import importlib.util
import json
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
gg = importlib.import_module("flaggems_sglang.ops.grouped_gemm")
spec = importlib.util.spec_from_file_location("gg_g", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/grouped_gemm.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
meta = importlib.import_module("benchmark.test_grouped_gemm")._metadata

n, k = int(sys.argv[1]), int(sys.argv[2])
CFGS = [(32, 64, 256, 8, 1), (64, 32, 64, 2, 2), (64, 64, 64, 8, 2), (16, 128, 128, 4, 1), (64, 64, 128, 8, 3),
        (32, 256, 64, 8, 1), (16, 32, 128, 2, 3), (64, 256, 64, 8, 1), (128, 128, 64, 8, 2), (64, 128, 64, 4, 2)]
rows = []
for m in (512, 1024, 2048, 4096, 8192):
    for e in (8, 32, 128):
        g = torch.Generator(device="cuda").manual_seed(9)
        a = torch.randn(m, k, device="cuda", generator=g).bfloat16()
        b = (0.1 * torch.randn(e, n, k, device="cuda", generator=g)).bfloat16()
        ids = torch.sort(torch.randint(0, e, (m,), device="cuda", generator=g, dtype=torch.int32))[0].contiguous()
        counts, offs, boffs, sched = meta(ids, e)
        args = (a, b, e, counts, offs, boffs, sched)
        tg = sorted(do_bench(lambda: gg.grouped_gemm(*args), warmup=10, rep=50, return_mode="median") * 1e3 for _ in range(3))[1]
        cand = {}
        for cfg in CFGS:
            for nf in (False, True):
                new._config = lambda mm, nn, kk, ee, c=cfg + (nf,): c
                try:
                    cand[str(cfg + (nf,))] = do_bench(lambda: new.grouped_gemm(*args), warmup=10, rep=50, return_mode="median") * 1e3
                except Exception:  # noqa: BLE001
                    pass
        best = min(cand, key=cand.get)
        rows.append(dict(m=m, n=n, k=k, e=e, generic=tg, cand=cand))
        print(f"m={m:5d} E={e:3d} N={n} K={k}: generic {tg:7.1f} best {best} {cand[best]:7.1f} ({tg / cand[best]:4.2f}x)", flush=True)
        del a, b
        torch.cuda.empty_cache()
json.dump(rows, open(f"/tmp/gg_grid_{n}_{k}.json", "w"))
