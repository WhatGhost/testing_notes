"""rdna3 ple_short_conv_ vs the generic op: output and updated state bitwise,
decode (fused kernel) and the ragged / profiling paths; then timings."""

import importlib
import importlib.util
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


gen = load("gen_ple", os.path.join(W, "src/flaggems_sglang/ops/ple_short_conv_.py"))
new = load("new_ple", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/ple_short_conv_.py"))
g = torch.Generator(device="cuda").manual_seed(0)


def bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def decode_case(rows, C=10240, width=4, dilation=3, dt=torch.bfloat16, sdt=None, extra=0, initial=None,
                bad=False, perm=False, strided=False, zeros=False):
    sdt = sdt or dt
    S = (width - 1) * dilation + extra
    x = torch.randn(rows, C + (64 if strided else 0), device="cuda", generator=g).to(dt)[:, :C]
    state = torch.randn(rows + 3, C, S, device="cuda", generator=g).to(sdt)
    if zeros:
        state[..., ::2] = 0.0
        state[..., 1::4] = -0.0
        x[:, ::3] = -0.0
    w = (torch.randn(C, width, device="cuda", generator=g) * 0.1).to(dt)
    idx = torch.randperm(rows + 3, device="cuda", generator=g)[:rows].int() if perm else torch.arange(rows, device="cuda", dtype=torch.int32)
    if bad and rows > 2:
        idx[0] = -1
        idx[1] = rows + 100
    out_store = torch.empty(rows, C + (64 if strided else 0), device="cuda", dtype=dt)
    out = out_store[:, :C]
    kw = dict(dilation=dilation)
    if initial is not None:
        kw["has_initial_state"] = (torch.arange(rows, device="cuda") % 3 != 0) if initial else None
    return (x, out, state, w, idx), kw


def run_pair(args, kw):
    x, out, state, w, idx = args
    s1, s2 = state.clone(), state.clone()
    o1, o2 = torch.full_like(out, float("nan")), torch.full_like(out, float("nan"))
    gen.ple_short_conv_(x, o1, s1, w, idx, **kw)
    new.ple_short_conv_(x, o2, s2, w, idx, **kw)
    return torch.equal(bits(o1), bits(o2)) and torch.equal(bits(s1), bits(s2))


cases = []
for rows in (1, 7, 64, 1024):
    cases.append((f"bench rows={rows}", decode_case(rows)))
cases += [
    ("zeros/-0.0", decode_case(33, zeros=True)),
    ("fp16", decode_case(17, dt=torch.float16)),
    ("fp32", decode_case(17, dt=torch.float32)),
    ("bf16 x, fp32 state", decode_case(17, sdt=torch.float32)),
    ("has_initial mixed", decode_case(29, initial=True)),
    ("null / out-of-range rows", decode_case(29, bad=True)),
    ("permuted rows", decode_case(29, perm=True)),
    ("strided x / out", decode_case(29, strided=True)),
    ("extra capacity", decode_case(29, extra=5)),
    ("odd channels", decode_case(9, C=1000)),
]
for width, dilation in ((2, 1), (3, 2), (8, 1), (5, 3)):
    cases.append((f"width={width} dilation={dilation}", decode_case(13, C=2048, width=width, dilation=dilation)))
bad = 0
for name, (args, kw) in cases:
    ok = run_pair(args, kw)
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} decode {name}", flush=True)
# ragged (mode 2) and profiling (mode 0) go through the copied generic kernels
x = torch.randn(40, 512, device="cuda", generator=g).bfloat16()
state = torch.randn(6, 512, 9, device="cuda", generator=g).bfloat16()
w = (torch.randn(512, 4, device="cuda", generator=g) * 0.1).bfloat16()
starts = torch.tensor([0, 10, 10, 25, 40], device="cuda", dtype=torch.int32)
idx = torch.tensor([2, 0, 5, 1], device="cuda", dtype=torch.int32)
for name, kw, ix in (("ragged", dict(dilation=3, query_start_loc=starts), idx), ("profiling", dict(dilation=3), None)):
    s1, s2 = state.clone(), state.clone()
    o1, o2 = torch.empty_like(x), torch.empty_like(x)
    gen.ple_short_conv_(x, o1, s1, w, ix, **kw)
    new.ple_short_conv_(x, o2, s2, w, ix, **kw)
    ok = torch.equal(bits(o1), bits(o2)) and torch.equal(bits(s1), bits(s2))
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
print(f"{len(cases) + 2 - bad}/{len(cases) + 2} bitwise identical (output and state)")

bm = importlib.import_module("benchmark.test_ple_short_conv_")


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


for rows in (1, 7, 64, 1024):
    *a, kw = next(iter(bm._input_fn((rows,), torch.bfloat16, "cuda")))
    tg, tn = [], []
    for _ in range(3):
        tg.append(bench(lambda: gen.ple_short_conv_(*a, **kw)))
        tn.append(bench(lambda: new.ple_short_conv_(*a, **kw)))
    tg, tn = sorted(tg)[1], sorted(tn)[1]
    print(f"decode rows={rows:5d}: generic {tg:8.1f} us  new {tn:8.1f} us  {tg / tn:5.2f}x", flush=True)
