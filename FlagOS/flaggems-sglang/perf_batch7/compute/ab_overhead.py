"""CPU cost per call of the absorbed_bmm wrappers (small T, so the GPU is
idle-bound): generic, rdna3 wrapper, its checks alone, the int32 view alone,
and a direct kernel launch with prepared arguments."""

import importlib
import importlib.util
import os
import sys
import time

import torch

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
gen = importlib.import_module("flaggems_sglang.ops.absorbed_bmm_concat_cast_q_fp8").absorbed_bmm_concat_cast_q_fp8
spec = importlib.util.spec_from_file_location("ab_o", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/absorbed_bmm_concat_cast_q_fp8.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
new._config = lambda t, h: (64, 128, 64, 8, True)

t, h = 16, 16
g = torch.Generator(device="cuda").manual_seed(0)
r = lambda *s: (0.25 * torch.randn(*s, device="cuda", generator=g)).bfloat16()  # noqa: E731
kw = dict(q_fp8_pad=torch.zeros((t, h, 576), device="cuda").to(torch.float8_e4m3fn), q_nope=r(t, h, 128).contiguous(),
          w_kc=r(h, 128, 512).contiguous(), q_rope=r(t, h, 64).contiguous(), num_heads=h)
out = kw["q_fp8_pad"]
o32 = out.view(torch.int32)


def per_call(fn, n=2000):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def direct():
    new._absorbed_kernel[(1 * h * 4,)](kw["q_nope"], kw["w_kc"], kw["q_rope"], o32, t, out.stride(0), out.stride(1),
                                       h, 128, 512, 64, 64, 128, 64, 64, True, num_warps=8)


def checks():
    q, w, rr = kw["q_nope"], kw["w_kc"], kw["q_rope"]
    return (out.dtype != torch.float8_e4m3fn or not all(x.is_contiguous() for x in (q, w, rr))
            or max(out.numel(), q.numel(), w.numel()) >= 1 << 31 or any(new._overlaps(out, x) for x in (q, w, rr)))


print(f"generic wrapper   {per_call(lambda: gen(**kw)):6.1f} us/call")
print(f"rdna3 wrapper     {per_call(lambda: new.absorbed_bmm_concat_cast_q_fp8(**kw)):6.1f} us/call")
print(f"direct launch     {per_call(direct):6.1f} us/call")
print(f"checks only       {per_call(checks):6.1f} us/call")
print(f"int32 view only   {per_call(lambda: out.view(torch.int32)):6.1f} us/call")
