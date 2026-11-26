"""absorbed_bmm: generic vs the rdna3 kernel (best sweep config) across token
counts for 8, 16 and 32 heads, 3 rounds alternated, cold medians plus warm,
to place the window where the rdna3 kernel is used."""

import importlib
import importlib.util
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
gen = importlib.import_module("flaggems_sglang.ops.absorbed_bmm_concat_cast_q_fp8").absorbed_bmm_concat_cast_q_fp8
spec = importlib.util.spec_from_file_location("ab_w", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/absorbed_bmm_concat_cast_q_fp8.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
new._config = lambda t, h: (64, 128, 64, 8, True)


def case(t, h):
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *s: (0.25 * torch.randn(*s, device="cuda", generator=g)).bfloat16()  # noqa: E731
    pad = torch.full((t, h, 576), 3.0, device="cuda").to(torch.float8_e4m3fn)
    return dict(q_fp8_pad=pad, q_nope=r(t, h, 128).contiguous(), w_kc=r(h, 128, 512).contiguous(),
                q_rope=r(t, h, 64).contiguous(), num_heads=h)


def warm(fn, n=30):
    for _ in range(3):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


for h in (8, 16, 32):
    for t in (128, 256, 384, 512, 768, 1024, 1536, 2048, 4096):
        kw = case(t, h)
        tg, tn = [], []
        for _ in range(3):
            tg.append(do_bench(lambda: gen(**kw), warmup=10, rep=100, return_mode="median") * 1e3)
            tn.append(do_bench(lambda: new.absorbed_bmm_concat_cast_q_fp8(**kw), warmup=10, rep=100, return_mode="median") * 1e3)
        tg, tn = sorted(tg)[1], sorted(tn)[1]
        wg, wn = warm(lambda: gen(**kw)), warm(lambda: new.absorbed_bmm_concat_cast_q_fp8(**kw))
        print(f"H={h:2d} T={t:5d} rows={t * h:6d}: cold generic {tg:7.1f} new {tn:7.1f} ({tg / tn:4.2f}x) | warm generic {wg:7.1f} new {wn:7.1f} ({wg / wn:4.2f}x)", flush=True)
