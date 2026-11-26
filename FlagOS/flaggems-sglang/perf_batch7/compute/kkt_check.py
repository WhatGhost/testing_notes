"""rdna3 chunk_kda_scaled_dot_kkt vs the fp32 reference and the generic op:
max error and the unit-test tolerance on the test cases plus edge cases,
then cold / warm timings on the benchmark shapes."""

import importlib.util
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
from flaggems_sglang.ops.chunk_kda_scaled_dot_kkt import chunk_kda_scaled_dot_kkt as gen  # noqa: E402
from flaggems_sglang.reference.chunk_kda_scaled_dot_kkt import reference  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


new = load("kkt_new", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/chunk_kda_scaled_dot_kkt.py")
bm = __import__("benchmark.test_chunk_kda_scaled_dot_kkt", fromlist=["x"])


def case(b, t, h, d, gate=lambda s, g: -0.05 * torch.rand(*s, device="cuda", generator=g), gk_scale=1.0,
         strided=False, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    wide = 2 * d if strided else d
    q = torch.randn(b, t, h, wide, device="cuda", generator=g).bfloat16()[..., :d]
    k = torch.randn(b, t, h, wide, device="cuda", generator=g).bfloat16()[..., :d]
    tp = (t + 63) // 64 * 64
    gt = gate((b, tp, h, d), g)
    gk = gt.reshape(b, -1, 64, h, d).cumsum(2).reshape(b, tp, h, d)[:, :t].contiguous()
    beta = torch.rand(b, t, h, device="cuda", generator=g)
    return dict(q=q, k=k, gk=gk, beta=beta, scale=d ** -0.5, gk_scale=gk_scale, chunk_size=64)


cases = [("test (1,64,4,64)", case(1, 64, 4, 64)), ("test (2,256,4,128)", case(2, 256, 4, 128)),
         ("test gk_scale=log2e", case(1, 128, 8, 64, gk_scale=1.4426950408889634))]
cases += [(f"test (1,{t},8,128)", case(1, t, 8, 128)) for t in (64, 256, 1024, 4096)]
cases += [("t=1100 (tail chunk)", case(1, 1100, 4, 128)), ("b=2 t=1030 h=3 d=64", case(2, 1030, 3, 64)),
          ("strided q/k (generic)", case(2, 192, 4, 128, strided=True)), ("d=96", case(1, 1024, 4, 96)),
          ("d=256", case(1, 1024, 4, 256)), ("d=32 h=16", case(1, 512, 16, 32)),
          ("strong decay (-5/step)", case(1, 1024, 8, 128, gate=lambda s, g: -5 * torch.rand(*s, device="cuda", generator=g))),
          ("medium decay (-1/step)", case(1, 1024, 8, 128, gate=lambda s, g: -1 * torch.rand(*s, device="cuda", generator=g))),
          ("non-monotone small", case(1, 1024, 8, 128, gate=lambda s, g: 0.05 * torch.randn(*s, device="cuda", generator=g))),
          ("non-monotone large", case(1, 1024, 8, 128, gate=lambda s, g: 1.0 * torch.randn(*s, device="cuda", generator=g))),
          ("large k values (x100)", dict(case(1, 1024, 8, 128), k=case(1, 1024, 8, 128)["k"] * 100))]

bad = 0
for name, kw in cases:
    ref = reference(**kw)
    out_g = gen(**kw)
    out_n = new.chunk_kda_scaled_dot_kkt(**kw)
    errs = []
    ok = True
    for r, og, on in zip(ref, out_g, out_n):
        errs.append(((og - r).abs().max().item(), (on - r).abs().max().item()))
        ok &= bool(torch.allclose(on, r, atol=2e-2, rtol=2e-2)) and on.shape == r.shape
        ok &= not torch.isnan(on).any().item()
    bad += not ok
    gen_ok = all(bool(torch.allclose(og, r, atol=2e-2, rtol=2e-2)) for og, r in zip(out_g, ref))
    print(f"{'ok  ' if ok else 'FAIL'} {name:26s} (generic {'ok' if gen_ok else 'FAIL'}) A err generic {errs[0][0]:.2e} new {errs[0][1]:.2e} | "
          f"Aqk err generic {errs[1][0]:.2e} new {errs[1][1]:.2e} (|ref| max {ref[0].abs().max().item():.1f})", flush=True)
print(f"{len(cases) - bad}/{len(cases)} within the unit-test tolerance")

if len(sys.argv) > 1 and sys.argv[1] == "time":
    def warm(fn, n=50):
        for _ in range(5):
            fn()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        for _ in range(n):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / n * 1e3

    for s in list(bm.SHAPES) + list(bm.MORE_SHAPES):
        args = next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))
        tg, tn = [], []
        for _ in range(3):
            tg.append(do_bench(lambda: gen(*args), warmup=25, rep=200, return_mode="median") * 1e3)
            tn.append(do_bench(lambda: new.chunk_kda_scaled_dot_kkt(*args), warmup=25, rep=200, return_mode="median") * 1e3)
        tg, tn = sorted(tg)[1], sorted(tn)[1]
        wg, wn = warm(lambda: gen(*args)), warm(lambda: new.chunk_kda_scaled_dot_kkt(*args))
        print(f"{str(s):18s} cold generic {tg:7.1f} new {tn:7.1f} ({tg / tn:4.2f}x) | warm generic {wg:7.1f} new {wn:7.1f} ({wg / wn:4.2f}x)", flush=True)
