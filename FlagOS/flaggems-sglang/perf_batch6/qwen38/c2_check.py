"""rdna3 ple_prefill_short_conv_ vs the generic op: output and updated state
bitwise; then a block sweep and timings."""

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


gen = load("gen_pre", os.path.join(W, "src/flaggems_sglang/ops/ple_prefill_short_conv_.py"))
new = load("new_pre", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/ple_prefill_short_conv_.py"))
bp = importlib.import_module("benchmark.test_ple_prefill_short_conv_")
g = torch.Generator(device="cuda").manual_seed(0)


def bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def case(lengths, C=10240, width=4, dilation=3, dt=torch.bfloat16, sdt=None, initial=None, null=False, zeros=False):
    sdt = sdt or dt
    S = (width - 1) * dilation
    n = len(lengths)
    T = sum(lengths)
    x = torch.randn(max(T, 1), C, device="cuda", generator=g).to(dt)[:T]
    state = torch.randn(n + 2, C, S, device="cuda", generator=g).to(sdt)
    if zeros:
        x[:, ::3] = -0.0
        state[..., ::2] = -0.0
    w = (torch.randn(C, width, device="cuda", generator=g) * 0.1).to(dt)
    starts = torch.tensor([0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32)
    idx = torch.randperm(n + 2, device="cuda", generator=g)[:n].int()
    if null and n > 1:
        idx[1] = -1
    init = torch.ones(n, device="cuda", dtype=torch.bool) if initial is None else (torch.arange(n, device="cuda") % 2 == 0)
    kw = dict(num_prefills=n, max_len=max(lengths), state_len=S, kernel_width=width, dilation=dilation, null_block_id=-1)
    return (x, torch.empty_like(x), state, w, starts, idx, init), kw


def pair(args, kw):
    x, out, state, w, starts, idx, init = args
    s1, s2 = state.clone(), state.clone()
    o1, o2 = torch.full_like(x, float("nan")), torch.full_like(x, float("nan"))
    gen.ple_prefill_short_conv_(x, o1, s1, w, starts, idx, init, **kw)
    new.ple_prefill_short_conv_(x, o2, s2, w, starts, idx, init, **kw)
    return torch.equal(bits(o1), bits(o2)) and torch.equal(bits(s1), bits(s2))


mode = sys.argv[1] if len(sys.argv) > 1 else "check"
if mode == "check":
    cases = []
    for shape in ((1, 1024), (4, 1024)):
        *a, kw = next(iter(bp._input_fn(shape, torch.bfloat16, "cuda")))
        cases.append((f"bench {shape}", (tuple(a), kw)))
    cases += [
        ("ragged short/empty", case([5, 0, 1, 300, 17, 64], C=2048)),
        ("has_initial mixed + null", case([40, 33, 70, 9], C=2048, initial=False, null=True)),
        ("zeros/-0.0", case([20, 100], C=2048, zeros=True)),
        ("fp16", case([50, 200], C=1024, dt=torch.float16)),
        ("fp32", case([50, 200], C=1024, dt=torch.float32)),
        ("bf16 x, fp32 state", case([50, 200], C=1024, sdt=torch.float32)),
        ("odd channels", case([31, 77], C=1000)),
    ]
    for width, dilation in ((2, 1), (3, 2), (8, 1), (5, 3)):
        cases.append((f"width={width} dilation={dilation}", case([10, 90, 3], C=768, width=width, dilation=dilation)))
    bad = 0
    for name, (args, kw) in cases:
        ok = pair(args, kw)
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    print(f"{len(cases) - bad}/{len(cases)} bitwise identical (output and state)")
else:
    def bench(fn):
        return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3

    for shape in ((1, 1024), (4, 1024)):
        *a, kw = next(iter(bp._input_fn(shape, torch.bfloat16, "cuda")))
        if mode == "sweep":
            tg = bench(lambda: gen.ple_prefill_short_conv_(*a, **kw))
            res = []
            for bt, bc, nw in itertools.product((8, 16, 32), (256, 512, 1024, 2048), (4, 8, 16)):
                if bt * bc // (nw * 32) > 64 or bt * bc // (nw * 32) < 4:
                    continue
                new._prefill_config = lambda c, cfg=(bt, bc, nw): cfg
                res.append((bench(lambda: new.ple_prefill_short_conv_(*a, **kw)), (bt, bc, nw)))
            res.sort()
            print(f"{shape}: generic {tg:.1f} | " + " ".join(f"{c}:{t:.1f}" for t, c in res[:6]), flush=True)
        else:
            tg, tn = [], []
            for _ in range(3):
                tg.append(bench(lambda: gen.ple_prefill_short_conv_(*a, **kw)))
                tn.append(bench(lambda: new.ple_prefill_short_conv_(*a, **kw)))
            tg, tn = sorted(tg)[1], sorted(tn)[1]
            print(f"prefill {shape}: generic {tg:8.1f} us  new {tn:8.1f} us  {tg / tn:5.2f}x", flush=True)
