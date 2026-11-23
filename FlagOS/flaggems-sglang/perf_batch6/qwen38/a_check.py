"""rdna3 qsa_sparse_paged_attention vs the repo reference (CPU, float) and the
generic op: page sizes, reversed tables, sentinel / out-of-table tokens, GQA
groups, head dims, gate on/off, caller workspace, invalid request rows; then
timings against the generic op."""

import importlib.util
import math
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-a.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_qsa", os.path.join(W, "src/flaggems_sglang/ops/qsa_sparse_paged_attention.py"))
new = load("new_qsa", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/qsa_sparse_paged_attention.py"))
ref = load("ref_qsa", os.path.join(W, "src/flaggems_sglang/reference/qsa_sparse_paged_attention.py")).reference
torch.manual_seed(0)


def make(rows, topk, page, head_dim=256, group=3, kv_heads=1, gate=True, bad_rows=False):
    pages = math.ceil(topk / page)
    hq = group * kv_heads
    q = torch.randn(rows, hq, head_dim + 8, device="cuda", dtype=torch.bfloat16)[..., :head_dim]
    k = torch.randn(rows * pages, page, kv_heads, head_dim + 8, device="cuda", dtype=torch.bfloat16)[..., :head_dim]
    v = torch.randn(rows * pages, page, kv_heads, head_dim + 8, device="cuda", dtype=torch.bfloat16)[..., :head_dim]
    table = torch.arange(rows * pages, dtype=torch.int32, device="cuda").reshape(rows, pages).flip(1)
    idx = torch.randperm(pages * page, device="cuda")[:topk].int().expand(rows, -1).clone()
    if topk > 2:
        idx[:, -1] = -1
        idx[:, -2] = pages * page + 5
    req = torch.arange(rows, dtype=torch.int32, device="cuda")
    if bad_rows and rows > 2:
        req[1] = -1
        req[2] = rows + 3
    g = torch.randn_like(q) if gate else None
    return q, k, v, idx, table, req, head_dim ** -0.5, g


def err(a, b):
    d = (a.float() - b.float()).abs().flatten()
    return d.max().item(), d.square().mean().sqrt().item(), torch.quantile(d, 0.99).item()


cases = [("bench", dict(rows=r, topk=t, page=1)) for r, t in ((1, 512), (7, 2048), (64, 2048))]
cases += [("page16", dict(rows=r, topk=t, page=16)) for r in (1, 7, 64) for t in (33, 513, 2051)]
cases += [("dims", dict(rows=5, topk=300, page=16, head_dim=d, group=gp, kv_heads=kh))
          for d in (64, 128) for gp, kh in ((1, 1), (4, 2), (8, 1))]
cases += [("nogate", dict(rows=9, topk=513, page=16, gate=False)),
          ("bad rows", dict(rows=8, topk=513, page=16, bad_rows=True)),
          ("rows300", dict(rows=300, topk=513, page=16))]
fails = 0
for name, kw in cases:
    q, k, v, idx, table, req, scale, g = make(**kw)
    args_cpu = [t.cpu() for t in (q, k, v, idx, table, req)]
    expected = ref(*args_cpu, gate=g.cpu() if g is not None else None).float()
    a = gen.qsa_sparse_paged_attention(q, k, v, idx, table, req, scale, None, g).cpu()
    b = new.qsa_sparse_paged_attention(q, k, v, idx, table, req, scale, None, g).cpu()
    out = torch.full_like(q, float("nan"))
    ws = (torch.empty(*q.shape[:2], 4, q.shape[2], device="cuda"), torch.empty(*q.shape[:2], 4, device="cuda"),
          torch.empty(*q.shape[:2], 4, device="cuda"))
    c = new.qsa_sparse_paged_attention(q, k, v, idx, table, req, scale, out, g, ws).cpu()
    eg, en, ec = err(a, expected)[0], err(b, expected)[0], err(c, expected)[0]
    mx, rmse, p99 = err(b, a)
    ok = (en <= max(1.5 * eg, 2e-2) and ec <= max(1.5 * eg, 2e-2) and mx <= 1e-2 and rmse <= 1e-3 and p99 <= 2e-3
          and torch.isfinite(b).all().item() and torch.isfinite(c).all().item())
    if kw.get("bad_rows"):
        ok &= bool((b[1] == 0).all() and (b[2] == 0).all())
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:8s} {str(kw)[:70]:70s} | ref err generic {eg:.1e} new {en:.1e} ws4 {ec:.1e} "
          f"| vs generic max {mx:.1e} rmse {rmse:.1e} p99 {p99:.1e}", flush=True)
print(f"{len(cases) - fails}/{len(cases)} ok")


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


for name, kw in [("bench", dict(rows=r, topk=t, page=1)) for r, t in ((1, 512), (7, 2048), (64, 2048))] + \
        [("page16", dict(rows=r, topk=t, page=16)) for r, t in ((1, 513), (7, 2051), (64, 2051), (300, 513))]:
    q, k, v, idx, table, req, scale, g = make(**kw)
    tg, tn = [], []
    for _ in range(3):
        tg.append(bench(lambda: gen.qsa_sparse_paged_attention(q, k, v, idx, table, req, scale, None, g)))
        tn.append(bench(lambda: new.qsa_sparse_paged_attention(q, k, v, idx, table, req, scale, None, g)))
    tg, tn = sorted(tg)[1], sorted(tn)[1]
    print(f"{name:6s} rows={kw['rows']:3d} topk={kw['topk']:5d}: generic {tg:8.1f} us  new {tn:7.1f} us  {tg / tn:5.2f}x",
          flush=True)
