"""Packed extend kernel at head_dim 256: candidate tiles over a wider grid."""

import itertools
import json
import sys

import torch

sys.path.insert(0, "/tmp")
from extend_sweep import bench, build, packed  # noqa: E402

CFGS = [(16, 32, 4, 1), (32, 32, 8, 1), (64, 64, 4, 1), (128, 64, 8, 1)]
out = {}
for (hq, hkv), bs, ext, prefix in itertools.product(((6, 1), (4, 1), (8, 1), (4, 2), (2, 1)), (1, 2, 4, 8),
                                                     (256, 512, 1024, 2048, 4096), (0, 4096)):
    if bs * (ext + prefix) > 32768:
        continue
    args = build(hq, hkv, bs, ext, prefix)
    ref = packed(args, 64, 64, 4, 1)
    row = {}
    for c in CFGS:
        if c[0] // (hq // hkv) < 1:
            continue
        o = packed(args, *c)
        assert (o - ref).abs().max().item() < 2e-2
        row[str(c)] = bench(lambda c=c: packed(args, *c), rep=15)
    key = f"g{hq // hkv} hkv{hkv} bs{bs} ext{ext} pre{prefix}"
    out[key] = row
    best = min(row, key=row.get)
    print(key, {k: round(v) for k, v in row.items()}, "best", best, flush=True)
    del args
    torch.cuda.empty_cache()
json.dump(out, open("/tmp/extend_eval.json", "w"), indent=1)
