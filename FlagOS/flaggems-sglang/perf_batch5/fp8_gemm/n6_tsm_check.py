"""rdna3 triton_scaled_mm vs the reference (unit-test tolerances) and vs the
generic op (max abs difference), over the unit-test cases, the benchmark
shapes, M edges, output dtypes, scalar scales, bias, odd N / K and int8."""

import importlib.util
import os
import sys

import torch

T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, os.path.join(T, "src"))
from flaggems_sglang.reference import get_reference  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_tsm", os.path.join(T, "src/flaggems_sglang/ops/triton_scaled_mm.py"))
new = load("new_tsm", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/triton_scaled_mm.py"))
ref = get_reference("triton_scaled_mm")


def case(m, k, n, in_dtype, out_dtype=torch.float16, seed=0, scale_range=None, scalar=(False, False), bias=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if in_dtype == torch.int8:
        a = torch.randint(-8, 8, (m, k), dtype=in_dtype, device="cuda", generator=g)
        b = torch.randint(-8, 8, (k, n), dtype=in_dtype, device="cuda", generator=g)
        rtol, atol = 0.15, 0.1
    else:
        a = torch.clamp(0.1 * torch.randn(m, k, dtype=torch.float16, device="cuda", generator=g), -0.3, 0.3).to(in_dtype)
        b = torch.clamp(0.1 * torch.randn(k, n, dtype=torch.float16, device="cuda", generator=g), -0.3, 0.3).to(in_dtype)
        rtol, atol = 0.25, 0.15
    lo, width = (0.1, 0.05) if scale_range is None else (scale_range[0], scale_range[1] - scale_range[0])
    sa = lo + width * torch.rand(1 if scalar[0] else m, 1, dtype=torch.float32, device="cuda", generator=g)
    sb = lo + width * torch.rand(1 if scalar[1] else n, 1, dtype=torch.float32, device="cuda", generator=g)
    kw = dict(input=a, weight=b, scale_a=sa, scale_b=sb, out_dtype=out_dtype)
    if bias == "row":
        kw["bias"] = torch.randn(1, n, device="cuda", generator=g).to(out_dtype)
    elif bias == "full":
        kw["bias"] = torch.randn(m, n, device="cuda", generator=g).to(out_dtype)
    return kw, rtol, atol


cases = []
for m, k, n, dt in ((32, 32, 32, torch.int8), (17, 64, 96, torch.int8), (32, 32, 32, torch.float8_e4m3fn),
                    (17, 64, 96, torch.float8_e4m3fn)):
    cases.append(case(m, k, n, dt))
for i, (m, k, n) in enumerate(((1, 4096, 256), (17, 1024, 96), (64, 4096, 256))):
    cases.append(case(m, k, n, torch.float8_e4m3fn, seed=11 + i, scale_range=(0.5, 1.5)))
for m in (1, 8, 64, 512, 4096):
    for k, n in ((4096, 4096), (4096, 7168)):
        cases.append(case(m, k, n, torch.float8_e4m3fn))
for m in (2, 3, 7, 15, 16, 17, 31, 33, 63, 65, 100, 127, 128, 129, 255, 256, 257, 511, 1000, 1024, 2048):
    cases.append(case(m, 1024, 2048, torch.float8_e4m3fn, seed=m, scale_range=(0.5, 1.5)))
for out_dtype in (torch.bfloat16, torch.float32):
    for m in (1, 8, 64, 512, 2048):
        cases.append(case(m, 2048, 1024, torch.float8_e4m3fn, out_dtype, seed=m + 7, scale_range=(0.5, 1.5)))
for scalar in ((True, False), (False, True), (True, True)):
    for m in (1, 8, 300, 2048):
        cases.append(case(m, 1024, 1024, torch.float8_e4m3fn, seed=m + 3, scale_range=(0.5, 1.5), scalar=scalar))
for bias in ("row", "full"):
    for m in (8, 300, 2048):
        cases.append(case(m, 1024, 1024, torch.float8_e4m3fn, seed=m + 5, scale_range=(0.5, 1.5), bias=bias))
for m, k, n in ((8, 1000, 1024), (300, 1000, 1024), (8, 1024, 1000), (300, 1024, 1000), (2048, 1024, 1000),
                (2048, 1000, 1024), (5, 96, 80), (130, 4096, 7168)):
    cases.append(case(m, k, n, torch.float8_e4m3fn, seed=m + 9, scale_range=(0.5, 1.5)))
for m in (1, 8, 300, 2048):
    cases.append(case(m, 1024, 1024, torch.int8, seed=m + 13))

fails = 0
worst = 0.0
for kw, rtol, atol in cases:
    want = ref(**kw).float()
    out = new.triton_scaled_mm(**kw)
    ok = torch.allclose(out.float(), want, rtol=rtol, atol=atol)
    tight = ((out.float() - want).abs().max() / want.abs().max().clamp_min(1e-6)).item()
    ok = ok and tight < 4e-3
    # the generic op reads past a [1, N] bias, so it is not called for those
    if kw.get("bias") is not None and kw["bias"].shape[0] == 1 and kw["input"].shape[0] > 1:
        rel = 0.0
    else:
        base = gen.triton_scaled_mm(**kw)
        assert out.dtype == base.dtype and out.shape == base.shape
        d = (out.float() - base.float()).abs()
        rel = (d.max() / base.float().abs().max().clamp_min(1e-6)).item()
    worst = max(worst, rel)
    tag = "ok  " if ok and rel < 4e-3 else "FAIL"
    fails += tag == "FAIL"
    m, k = kw["input"].shape
    n = kw["weight"].shape[1]
    if tag == "FAIL":
        print(f"{tag} M={m} K={k} N={n} {str(kw['input'].dtype)[6:]}->{str(kw['out_dtype'])[6:]} "
              f"sa{tuple(kw['scale_a'].shape)} sb{tuple(kw['scale_b'].shape)} bias={'bias' in kw} "
              f"ref_ok={ok} tight={tight:.1e} vs_generic={rel:.1e}", flush=True)
print(f"cases {len(cases)} fails {fails} worst rel diff vs generic {worst:.2e}")
