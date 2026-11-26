"""rdna3 absorbed_bmm with the 6144-row rule: unit-test style checks on cases
that reach the kernel (and two that fall back), plus CPU cost per call."""

import importlib
import importlib.util
import os
import sys
import time

import torch

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
gen = importlib.import_module("flaggems_sglang.ops.absorbed_bmm_concat_cast_q_fp8").absorbed_bmm_concat_cast_q_fp8
from flaggems_sglang.reference.absorbed_bmm_concat_cast_q_fp8 import reference  # noqa: E402

spec = importlib.util.spec_from_file_location("ab_c2", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/absorbed_bmm_concat_cast_q_fp8.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)


def case(t, h, pad_t=None, pad_h=None, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *s: (0.25 * torch.randn(*s, device="cuda", generator=g)).to(dtype)  # noqa: E731
    pad = torch.full((pad_t or t, pad_h or h, 576), 3.0, device="cuda").to(torch.float8_e4m3fn)
    return dict(q_fp8_pad=pad, q_nope=r(t, h, 128).contiguous(), w_kc=r(h, 128, 512).contiguous(),
                q_rope=r(t, h, 64).contiguous(), num_heads=h)


def fresh(kw):
    return dict(kw, q_fp8_pad=kw["q_fp8_pad"].clone())


bad = 0
cases = [("(512,32)", 512, 32, case(512, 32)), ("(1024,16)", 1024, 16, case(1024, 16)),
         ("(1536,16)", 1536, 16, case(1536, 16)), ("(1000,32)", 1000, 32, case(1000, 32)),
         ("(2048,8) in a (4096,16) buffer", 2048, 8, case(2048, 8, 4096, 16)),
         ("fp16 (generic path, read as bf16)", 768, 16, case(768, 16, dtype=torch.float16)),
         ("(768,16) at the threshold", 768, 16, case(768, 16)),
         ("(767,16) below it (generic)", 767, 16, case(767, 16)),
         ("(2047,16) just below 2^24", 2047, 16, case(2047, 16)),
         ("(2048,16) at 2^24 (generic)", 2048, 16, case(2048, 16)),
         ("(4096,16) above (generic)", 4096, 16, case(4096, 16))]
for name, t, h, kw in cases:
    out = new.absorbed_bmm_concat_cast_q_fp8(**fresh(kw))
    ref = reference(**fresh(kw))
    rope_ok = torch.equal(out[:t, :h, 512:].view(torch.uint8), ref[:t, :h, 512:].view(torch.uint8))
    a, e = out[:t, :h, :512].float(), ref[:t, :h, :512].float()
    mism = ((a - e).abs() > (6e-2 + 6e-2 * e.abs())).float().mean().item()
    pad_ok = torch.equal(out[:, h:].view(torch.uint8), ref[:, h:].view(torch.uint8)) and torch.equal(
        out[t:].view(torch.uint8), ref[t:].view(torch.uint8))
    cd = (gen(**fresh(kw)).view(torch.uint8) != out.view(torch.uint8)).float().mean().item()
    ok = rope_ok and mism < 2e-2 and pad_ok
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:32s} rope exact {rope_ok}, nope mismatch {mism:.1e}, padding kept {pad_ok}, "
          f"codes differing from generic {cd:.1e}", flush=True)
print(f"{len(cases) - bad}/{len(cases)} pass the unit-test checks")


def per_call(fn, n=1000):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


kw = case(768, 16)
print(f"back-to-back per call at (768,16): generic {per_call(lambda: gen(**kw)):.1f} us, rdna3 {per_call(lambda: new.absorbed_bmm_concat_cast_q_fp8(**kw)):.1f} us")
