"""qsa_sparse_paged_attention on gfx1100: sweep BLOCK_M floor, BLOCK_N, warps,
stages and TopK splits through the generic module's own entry points, on the
benchmark inputs and on unit-test style inputs (page size 16, reversed table)."""

import importlib.util
import itertools
import math
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-a.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
spec = importlib.util.spec_from_file_location("gen_qsa", os.path.join(W, "src/flaggems_sglang/ops/qsa_sparse_paged_attention.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def bench_inputs(rows, selected):
    q = torch.randn(rows, 3, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(rows * selected, 1, 1, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    idx = torch.arange(selected, device="cuda", dtype=torch.int32).expand(rows, -1).contiguous()
    table = torch.arange(rows * selected, device="cuda", dtype=torch.int32).view(rows, selected)
    req = torch.arange(rows, device="cuda", dtype=torch.int32)
    return q, k, v, idx, table, req, 256 ** -0.5, torch.randn_like(q)


def test_inputs(rows, topk, page=16):
    pages = math.ceil(topk / page)
    q = torch.randn(rows, 3, 264, device="cuda", dtype=torch.bfloat16)[..., :256]
    k = torch.randn(rows * pages, page, 1, 264, device="cuda", dtype=torch.bfloat16)[..., :256]
    v = torch.randn(rows * pages, page, 1, 264, device="cuda", dtype=torch.bfloat16)[..., :256]
    table = torch.arange(rows * pages, dtype=torch.int32, device="cuda").reshape(rows, pages).flip(1)
    idx = torch.arange(topk, dtype=torch.int32, device="cuda").expand(rows, -1).clone()
    idx[:, -1] = -1
    idx[:, -2] = pages * page + 5
    req = torch.arange(rows, dtype=torch.int32, device="cuda")
    return q, k, v, idx, table, req, 256 ** -0.5, torch.randn_like(q)


def rand_inputs(rows, topk, page=16):
    pages = math.ceil(topk / page)
    q = torch.randn(rows, 3, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(rows * pages, page, 1, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.randperm(rows * pages, device="cuda").int().reshape(rows, pages)
    idx = torch.stack([torch.randperm(pages * page, device="cuda")[:topk] for _ in range(rows)]).int()
    req = torch.arange(rows, dtype=torch.int32, device="cuda")
    return q, k, v, idx, table, req, 256 ** -0.5, torch.randn_like(q)


def run(inp, block_m, cfg, splits):
    q, k, v, idx, table, req, scale, gate = inp
    ws = None
    if splits > 1:
        ws = (torch.empty(q.shape[0], q.shape[1], splits, q.shape[2], device="cuda"),
              torch.empty(q.shape[0], q.shape[1], splits, device="cuda"),
              torch.empty(q.shape[0], q.shape[1], splits, device="cuda"))
    return lambda: gen._qsa_sparse_paged_attention(q, k, v, idx, table, req, scale, None, gate, ws,
                                                   min_block_m=block_m, launch_config=cfg)


def bench(fn):
    return do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3


cases = [("bench", r, s) for r, s in ((1, 512), (7, 2048), (64, 2048))]
cases += [("test", r, t) for r in (1, 7, 64) for t in (513, 2051)]
cases += [("rand", r, t) for r in (1, 7, 64, 300) for t in (513, 2051)]
sel = sys.argv[1:]
for kind, rows, topk in cases:
    if sel and f"{kind}:{rows},{topk}" not in sel:
        continue
    inp = {"bench": bench_inputs, "test": test_inputs, "rand": rand_inputs}[kind](rows, topk)
    ref = run(inp, 8, (16, 4, 2), 1)().float()
    t_master = bench(run(inp, 8, (16, 4, 2), 1))
    res = []
    grid = ((16, 32, 64), (4, 8), (1, 2, 3), (1, 2, 4, 8, 16, 32)) if kind == "rand" else ((32, 64), (4, 8), (1, 2), (1, 2, 4, 8, 16, 32))
    for bn, nw, ns, sp in itertools.product(*grid):
        fn = run(inp, 16, (bn, nw, ns), sp)
        try:
            out = fn().float()
            t = bench(fn)
        except Exception as e:  # noqa: BLE001
            continue
        err = (out - ref).abs().max().item()
        res.append((t, (bn, nw, ns, sp), err))
    res.sort()
    print(f"{kind} rows={rows:3d} topk={topk:5d}: master {t_master:7.1f} us | best {res[0][0]:7.1f} {res[0][1]} "
          f"{t_master / res[0][0]:5.2f}x maxdiff {res[0][2]:.1e} | "
          + " ".join(f"{c}:{t:.1f}" for t, c, _ in res[1:5]), flush=True)
