"""per_group_transpose: current rdna3 launch vs row-split configs on extra shapes."""

import itertools
import json
import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, "/tmp")
from b4_pgt_lib import cur, split  # noqa: E402

CFGS = [(s, bm, bk, nw) for s in (1, 2, 4, 8, 16)
        for bm, bk, nw in ((32, 32, 4), (32, 32, 8), (32, 64, 8), (32, 128, 8), (64, 64, 8), (64, 128, 8))]


def offsets(sizes):
    cum = [0]
    for s in sizes:
        cum.append(cum[-1] + s)
    return torch.tensor(cum, dtype=torch.int32, device="cuda")


cases = []
for dtype in (torch.bfloat16, torch.float32):
    for k in (128, 256, 512, 1024, 2048, 4096, 7168):
        for rows in (16, 128, 256, 512, 1024, 2048):
            if dtype == torch.float32 and k not in (128, 512, 4096):
                continue
            cases.append((str(dtype)[6:], 8, k, [rows] * 8))
g = torch.Generator().manual_seed(0)
for k in (128, 512, 2048, 4096):
    for mean in (256, 1024):
        sizes = (torch.rand(8, generator=g) * 2 * mean).int().tolist()
        cases.append(("bfloat16", 8, k, sizes))
    cases.append(("bfloat16", 32, k, [256] * 32))
    cases.append(("bfloat16", 32, k, [64] * 32))

rows_out = []
for dt, e, k, sizes in cases:
    dtype = getattr(torch, dt)
    a = torch.randn(sum(sizes), k, device="cuda").to(dtype)
    offs = offsets(sizes)
    ref = cur.per_group_transpose(a, offs)
    t_cur = do_bench(lambda: cur.per_group_transpose(a, offs), warmup=25, rep=200, return_mode="median") * 1e3
    times = {}
    for c in CFGS:
        o = split(a, offs, *c)
        if not torch.equal(o, ref):
            print("MISMATCH", dt, e, k, sizes[:4], c)
            continue
        times[str(c)] = do_bench(lambda: split(a, offs, *c), warmup=25, rep=200, return_mode="median") * 1e3
    best = min(times, key=times.get)
    rows_out.append(dict(dtype=dt, e=e, k=k, sizes=sizes, cur=t_cur, times=times))
    print(f"{dt[:4]} E{e:2d} k{k:5d} rows~{sum(sizes) // e:5d} | cur {t_cur:7.1f} | best {best} {times[best]:7.1f}",
          flush=True)
json.dump(rows_out, open("/tmp/b4_pgt_sweep.json", "w"))
