"""Evaluate candidate (BLOCK_N, warps, stages) x splits on benchmark, unit-test
style and random-index inputs; prints, per shape, every candidate's time so a
rule can be chosen by worst-case regret."""

import itertools
import json
import sys

import torch

sys.argv, sel = sys.argv[:1], sys.argv[1:]
exec(open("/tmp/a_sweep.py").read().split("cases = [")[0])  # helpers: bench_inputs, test_inputs, rand_inputs, run, bench

cands = [(32, 8, 1), (32, 8, 2), (16, 8, 3), (16, 8, 1), (16, 4, 1), (32, 4, 2)]
cases = [("bench", r, s) for r, s in ((1, 512), (7, 2048), (64, 2048))]
cases += [("test", r, t) for r in (1, 7, 64) for t in (513, 2051)]
cases += [("rand", r, t) for r in (1, 7, 64, 300) for t in (513, 2051)]
out = {}
for kind, rows, topk in cases:
    if sel and f"{kind}:{rows},{topk}" not in sel:
        continue
    inp = {"bench": bench_inputs, "test": test_inputs, "rand": rand_inputs}[kind](rows, topk)
    res = {}
    for cfg, sp in itertools.product(cands, (1, 2, 4, 8, 16, 32)):
        fn = run(inp, 16, cfg, sp)
        fn()
        res[f"{cfg}|{sp}"] = bench(fn)
    best = min(res.values())
    out[f"{kind}:{rows},{topk}"] = res
    print(f"{kind}:{rows},{topk} best {best:.1f} " + json.dumps({k: round(v, 1) for k, v in res.items()}), flush=True)
