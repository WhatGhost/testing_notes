"""rdna3 fused_recurrent_kda vs the generic op: output and final state bitwise,
then a (rows per program, warps) sweep and timings with do_bench (the
benchmark's dict inputs cannot run in the official harness)."""

import importlib
import importlib.util
import itertools
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path.insert(0, W)
sys.path.insert(0, os.path.join(W, "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_kda", os.path.join(W, "src/flaggems_sglang/ops/fused_recurrent_kda.py"))
new = load("new_kda", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_recurrent_kda.py"))
bm = importlib.import_module("benchmark.test_fused_recurrent_kda")
g = torch.Generator(device="cuda").manual_seed(0)
BENCH = list(bm.SHAPES) + list(bm.MORE_SHAPES)


def case(b, t, h, kd, vd, l2norm=False, headwise=False, state=True, dt=torch.bfloat16):
    r = lambda *d: torch.randn(*d, device="cuda", generator=g)  # noqa: E731
    beta = torch.rand(b, t, h, *(() if headwise else (vd,)), device="cuda", generator=g).to(dt)
    s0 = r(b, h, vd, kd) * 0.1 if state else torch.zeros(b, h, vd, kd, device="cuda")
    return dict(q=r(b, t, h, kd).to(dt), k=r(b, t, h, kd).to(dt), v=r(b, t, h, vd).to(dt),
                g=-0.05 * torch.rand(b, t, h, kd, device="cuda", generator=g), beta=beta, scale=kd ** -0.5,
                initial_state=s0, use_qk_l2norm_in_kernel=l2norm)


def bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


mode = sys.argv[1] if len(sys.argv) > 1 else "check"
if mode == "check":
    cases = [(f"bench {s}", next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))) for s in BENCH]
    cases += [("test (1,16,2,64,64)", case(1, 16, 2, 64, 64)), ("test (2,64,4,128,128)", case(2, 64, 4, 128, 128)),
              ("l2norm", case(1, 32, 2, 64, 64, l2norm=True)), ("headwise beta", case(1, 32, 2, 64, 64, headwise=True)),
              ("decode t=1 b=64", case(64, 1, 8, 128, 128)), ("decode t=1 b=3", case(3, 1, 8, 128, 128)),
              ("fp16 b=5", case(5, 33, 4, 128, 128, dt=torch.float16)), ("vd=96 kd=128", case(4, 20, 8, 128, 96)),
              ("l2norm b=16", case(16, 64, 8, 128, 128, l2norm=True))]
    bad = 0
    for name, kw in cases:
        o1, s1 = gen.fused_recurrent_kda(**kw)
        o2, s2 = new.fused_recurrent_kda(**kw)
        ok = torch.equal(bits(o1), bits(o2)) and torch.equal(bits(s1), bits(s2))
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    print(f"{len(cases) - bad}/{len(cases)} bitwise identical (output and final state)")
elif mode == "sweep":
    for s in BENCH + [(1, 1), (64, 1)]:
        kw = next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))
        tg = bench(lambda: gen.fused_recurrent_kda(**kw))
        res = []
        for rv, nw in itertools.product((1, 2, 4, 8, 16, 32, 64, 128), (1, 2, 4, 8)):
            per_lane = rv * 128 // (nw * 32)
            if per_lane < 1 or per_lane > 128:
                continue
            new._config = lambda b, t, h, vd, c=(rv, nw): c
            try:
                res.append((bench(lambda: new.fused_recurrent_kda(**kw)), (rv, nw)))
            except Exception:  # noqa: BLE001
                continue
        res.sort()
        print(f"{s}: generic {tg:7.1f} | " + " ".join(f"{c}:{t:.1f}" for t, c in res[:5]), flush=True)
else:
    for s in BENCH + [(1, 1), (64, 1)]:
        kw = next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))
        tg, tn = [], []
        for _ in range(3):
            tg.append(bench(lambda: gen.fused_recurrent_kda(**kw)))
            tn.append(bench(lambda: new.fused_recurrent_kda(**kw)))
        tg, tn = sorted(tg)[1], sorted(tn)[1]
        print(f"{str(s):10s} generic {tg:8.1f} us  new {tn:8.1f} us  {tg / tn:5.2f}x", flush=True)
