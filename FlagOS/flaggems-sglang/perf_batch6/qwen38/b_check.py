"""rdna3 qwen4_qsa_topk vs the repo reference (exact) and the generic op:
ties, -inf / +-0 inside the window, empty / short / long windows, column and
topk variants, int64 bounds, strided logits; then timings."""

import importlib.util
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


gen = load("gen_topk", os.path.join(W, "src/flaggems_sglang/ops/qwen4_qsa_topk.py"))
new = load("new_topk", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/qwen4_qsa_topk.py"))
ref = load("ref_topk", os.path.join(W, "src/flaggems_sglang/reference/qwen4_qsa_topk.py")).reference
g = torch.Generator(device="cuda").manual_seed(0)


def bench_inputs(rows):
    logits = torch.arange(rows * 640, device="cuda", dtype=torch.float32).reshape(rows, 640)
    starts = torch.arange(rows, device="cuda", dtype=torch.int32) % 4
    lengths = torch.tensor([513, 0, 1, 511, 512, 600, 636], device="cuda", dtype=torch.int32)
    ends = starts + lengths[torch.arange(rows, device="cuda") % 7]
    return logits, starts, ends, 512


def windows(rows, cols, idx_dtype=torch.int32):
    lengths = torch.tensor([0, 1, 511, 512, 513, cols, cols // 2, 3], device="cuda")
    starts = torch.randint(0, max(1, cols // 4), (rows,), device="cuda", generator=g)
    ends = torch.minimum(starts + lengths[torch.arange(rows, device="cuda") % len(lengths)], torch.tensor(cols, device="cuda"))
    return starts.to(idx_dtype), ends.to(idx_dtype)


cases = []
for rows in (1, 7, 64):
    a = bench_inputs(rows)
    cases.append((f"bench rows={rows}", a))
    cases.append((f"bench neg rows={rows}", (-a[0],) + a[1:]))
for cols, topk in ((640, 512), (1000, 512), (2048, 512), (100, 512), (640, 100), (640, 1024)):
    s, e = windows(37, cols)
    cases.append((f"randn cols={cols} topk={topk}", (torch.randn(37, cols, device="cuda", generator=g), s, e, topk)))
    ties = torch.randint(-3, 4, (37, cols), device="cuda", generator=g).float()
    cases.append((f"ties cols={cols} topk={topk}", (ties, s, e, topk)))
s, e = windows(29, 640, torch.int64)
x = torch.randn(29, 640, device="cuda", generator=g)
x[:, ::7] = -float("inf")
x[:, 3::11] = 0.0
x[:, 5::13] = -0.0
cases.append(("-inf/+-0, int64 bounds", (x, s, e, 512)))
cases.append(("bf16 logits (ties)", (torch.randn(29, 640, device="cuda", generator=g).bfloat16(), s, e, 512)))
wide = torch.randn(23, 1280, device="cuda", generator=g)
s, e = windows(23, 640)
cases.append(("strided logits", (wide[:, ::2], s, e, 512)))

bad = 0
for name, args in cases:
    expected = ref(*[t.cpu() if isinstance(t, torch.Tensor) else t for t in args])
    got = new.qwen4_qsa_topk(*args).cpu()
    ok = got.shape == expected.shape and got.dtype == expected.dtype and torch.equal(got, expected)
    gen_ok = torch.equal(gen.qwen4_qsa_topk(*args).cpu(), expected)
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:32s} (generic {'matches' if gen_ok else 'differs'})", flush=True)
print(f"{len(cases) - bad}/{len(cases)} exact")


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


for rows in (1, 7, 64, 1024, 4096):
    a = bench_inputs(rows)
    tg, tn = [], []
    for _ in range(3):
        tg.append(bench(lambda: gen.qwen4_qsa_topk(*a)))
        tn.append(bench(lambda: new.qwen4_qsa_topk(*a)))
    tg, tn = sorted(tg)[1], sorted(tn)[1]
    print(f"rows={rows:5d}: generic {tg:8.1f} us  new {tn:7.1f} us  {tg / tn:6.2f}x", flush=True)
