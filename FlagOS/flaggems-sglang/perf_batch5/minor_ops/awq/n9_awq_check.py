"""rdna3 awq_gemm accuracy: max |out - exact| / max |exact| for the new op,
the generic op and the reference, exact = fp64 matmul of the reference
dequantized weight (rounded to the scales dtype as in the reference)."""

import importlib.util
import os
import sys

import torch

T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, os.path.join(T, "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_awq", os.path.join(T, "src/flaggems_sglang/ops/awq_gemm.py"))
new = load("new_awq", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/awq_gemm.py"))
deq = load("ref_deq", os.path.join(T, "src/flaggems_sglang/reference/awq_dequantize.py"))


def case(m, k, n, gs, dtype, seed=0, stride_pad=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gs = k if gs == -1 else gs
    x = torch.rand(m, k + stride_pad, dtype=torch.float32, device="cuda", generator=g)[:, :k].to(dtype)
    qweight = torch.randint(0, torch.iinfo(torch.int32).max, (k, n // 8), dtype=torch.int32, device="cuda", generator=g)
    qzeros = torch.randint(0, torch.iinfo(torch.int32).max, (k // gs, n // 8), dtype=torch.int32, device="cuda", generator=g)
    scales = torch.rand(k // gs, n, dtype=torch.float32, device="cuda", generator=g).to(dtype)
    return x, qweight, scales, qzeros


cases = []
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    cases += [(16, 128, 32, 128, dtype, 0), (24, 128, 64, 32, dtype, 0), (32, 128, 128, -1, dtype, 0)]
    cases += [(m, 4096, n, 128, dtype, 0) for m in (1, 8, 64, 512) for n in (1024, 4096)]
    cases += [(m, 4096, n, 128, dtype, 0) for m, n in ((16, 32), (24, 64), (32, 128))]
    cases += [(3, 512, 40, 64, dtype, 0), (17, 1024, 96, 32, dtype, 0), (100, 2048, 200, 128, dtype, 0),
              (130, 768, 264, 256, dtype, 0), (5, 4096, 1024, 128, dtype, 64), (64, 4096, 4096, 128, dtype, 32)]

worst = {}
fails = 0
for m, k, n, gs, dtype, pad in cases:
    x, qw, sc, qz = case(m, k, n, gs, dtype, stride_pad=pad)
    w = deq.reference(qw, sc, qz)
    exact = x.double() @ w.double()
    scale = exact.abs().max().item()
    out = new.awq_gemm(x, qw, sc, qz, 8)
    errs = {"new": ((out.double() - exact).abs().max().item()) / scale}
    try:
        errs["gen"] = ((gen.awq_gemm(x, qw, sc, qz, 8).double() - exact).abs().max().item()) / scale
    except Exception as e:  # noqa: BLE001
        errs["gen"] = float("nan")
    try:
        errs["ref"] = ((torch.matmul(x, w).double() - exact).abs().max().item()) / scale
    except Exception as e:  # noqa: BLE001
        errs["ref"] = float("nan")
    # one output ulp of the largest value, or 1e-5 for fp32
    limit = {torch.float32: 1e-5, torch.float16: 2 ** -10, torch.bfloat16: 2 ** -7}[dtype]
    ok = out.dtype == sc.dtype and out.shape == (m, n) and errs["new"] <= limit
    fails += not ok
    key = str(dtype).split(".")[-1]
    worst[key] = max(worst.get(key, 0), errs["new"])
    print(f"{'ok  ' if ok else 'FAIL'} {key:8s} m={m:4d} k={k:5d} n={n:5d} gs={gs:4d} pad={pad:2d} | "
          f"new {errs['new']:.1e} gen {errs['gen']:.1e} ref {errs['ref']:.1e}", flush=True)
print(f"{len(cases) - fails}/{len(cases)} ok; worst new error {worst}")
