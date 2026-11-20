"""rdna3 w8a8_block_fp8_matmul vs the reference (unit-test check and a tight
check: max |out - ref| / max |ref|) and vs the generic op, over the unit-test
cases, the benchmark shapes, M edges, output dtypes, odd shapes (fallback)
and non-contiguous inputs."""

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


gen = load("gen_w8", os.path.join(T, "src/flaggems_sglang/ops/w8a8_block_fp8_matmul.py"))
new = load("new_w8", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/w8a8_block_fp8_matmul.py"))
ref = get_reference("w8a8_block_fp8_matmul")
FP8 = torch.float8_e4m3fn
SMALL = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # only M > SMALL when checking one path


def case(m, n, k, dtype=torch.bfloat16, seed=0, scale_range=(0.0, 1e-2), block=(128, 128)):
    g = torch.Generator(device="cuda").manual_seed(seed)
    fmax = torch.finfo(FP8).max
    A = torch.randn(m, k, device="cuda", generator=g).clamp(-fmax, fmax).to(FP8)
    B = torch.randn(n, k, device="cuda", generator=g).clamp(-fmax, fmax).to(FP8)
    lo, hi = scale_range
    bn, bk = block
    As = (lo + (hi - lo) * torch.rand(m, (k + bk - 1) // bk, device="cuda", generator=g)).contiguous()
    Bs = (lo + (hi - lo) * torch.rand((n + bn - 1) // bn, (k + bk - 1) // bk, device="cuda", generator=g)).contiguous()
    return dict(A=A, B=B, As=As, Bs=Bs, block_size=list(block), output_dtype=dtype)


cases = [case(7, 256, 512), case(64, 1024, 512), case(256, 1024, 4096)]
cases += [case(7, 256, 512, seed=11, scale_range=(0.5, 1.5)), case(64, 1024, 512, seed=12, scale_range=(0.5, 1.5)),
          case(256, 1024, 4096, seed=13, scale_range=(0.5, 1.5))]
cases += [case(m, n, k) for m in (1, 8, 64, 512, 4096) for n, k in ((1024, 4096), (4096, 4096), (7168, 4096))]
cases += [case(m, 2048, 1024, seed=m, scale_range=(0.5, 1.5))
          for m in (2, 3, 15, 16, 17, 31, 33, 63, 65, 100, 127, 128, 129, 255, 256, 257, 300, 511, 1000, 2048)]
for dtype in (torch.float16, torch.float32):
    cases += [case(m, 1024, 2048, dtype, seed=m + 1, scale_range=(0.5, 1.5)) for m in (1, 8, 64, 300, 2048)]
cases += [case(300, 1000, 1024, seed=5), case(300, 1024, 1000, seed=6), case(70, 384, 640, seed=7, block=(64, 64)),
          case(200, 512, 1024, seed=8, block=(128, 64))]
# non-contiguous A (a column slice): falls back to the generic op
big = case(300, 1024, 1024, seed=9, scale_range=(0.5, 1.5))
wide = torch.zeros(300, 1152, device="cuda").to(FP8)
wide[:, :1024] = big["A"]
big["A"] = wide[:, :1024]
cases.append(big)

fails = 0
worst = 0.0
for kw in cases:
    m = kw["A"].shape[0]
    if m <= SMALL:
        continue
    want = ref(**kw).float()
    out = new.w8a8_block_fp8_matmul(**kw)
    base = gen.w8a8_block_fp8_matmul(**kw)
    assert out.dtype == base.dtype and out.shape == base.shape
    ok_test = torch.allclose(out.float(), want, atol=5e-2, rtol=1e-2)
    peak = want.abs().max().clamp_min(1e-12)
    tight = ((out.float() - want).abs().max() / peak).item()
    # one output ulp at the largest magnitude
    limit = {torch.float32: 1e-5, torch.float16: 1e-3}.get(kw["output_dtype"], 7.9e-3)
    gen_tight = ((base.float() - want).abs().max() / peak).item()
    worst = max(worst, tight)
    ok = ok_test and tight <= max(limit, 1.5 * gen_tight)
    fails += not ok
    if not ok:
        print(f"FAIL M={m} N={kw['B'].shape[0]} K={kw['A'].shape[1]} {str(kw['output_dtype'])[6:]} "
              f"block={kw['block_size']} test_ok={ok_test} tight={tight:.2e} generic_tight={gen_tight:.2e}", flush=True)
print(f"cases {sum(1 for kw in cases if kw['A'].shape[0] > SMALL)} fails {fails} worst tight error {worst:.2e}")
