"""rdna3 absorbed_bmm_concat_cast_q_fp8: unit-test style checks against the
reference, code agreement with the generic op, and a config sweep (cold
do_bench) per benchmark shape plus warm timing of the best config."""

import importlib
import importlib.util
import itertools
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
gen = importlib.import_module("flaggems_sglang.ops.absorbed_bmm_concat_cast_q_fp8").absorbed_bmm_concat_cast_q_fp8
from flaggems_sglang.reference.absorbed_bmm_concat_cast_q_fp8 import reference  # noqa: E402

spec = importlib.util.spec_from_file_location("ab_new", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/absorbed_bmm_concat_cast_q_fp8.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)


def case(t, h, pad_t=None, pad_h=None, k=128, n=512, d=64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: (0.25 * torch.randn(*s, device="cuda", generator=g)).bfloat16()  # noqa: E731
    pad = torch.full((pad_t or t, pad_h or h, n + d), 3.0, device="cuda").to(torch.float8_e4m3fn)
    return dict(q_fp8_pad=pad, q_nope=r(t, h, k).contiguous(), w_kc=r(h, k, n).contiguous(),
                q_rope=r(t, h, d).contiguous(), num_heads=h)


def fresh(kw):
    return dict(kw, q_fp8_pad=kw["q_fp8_pad"].clone())


def check(kw, out, t, h):
    ref = reference(**fresh(kw))
    rope_ok = torch.equal(out[:t, :h, 512:].view(torch.uint8), ref[:t, :h, 512:].view(torch.uint8))
    a = out[:t, :h, :512].float()
    e = ref[:t, :h, :512].float()
    mism = ((a - e).abs() > (6e-2 + 6e-2 * e.abs())).float().mean().item()
    pad_ok = torch.equal(out[:, h:].view(torch.uint8), ref[:, h:].view(torch.uint8)) and torch.equal(
        out[t:].view(torch.uint8), ref[t:].view(torch.uint8))
    g = gen(**fresh(kw))
    code_diff = (g.view(torch.uint8) != out.view(torch.uint8)).float().mean().item()
    return rope_ok and mism < 2e-2 and pad_ok, mism, code_diff


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


print("== checks (default config) ==")
for name, kw, t, h in [("(16,8)", case(16, 8), 16, 8), ("(512,16)", case(512, 16), 512, 16),
                       ("(64,8) padded", case(64, 8, pad_t=128, pad_h=16), 64, 8),
                       ("(1024,16)", case(1024, 16), 1024, 16), ("(4096,16)", case(4096, 16), 4096, 16),
                       ("(100,16)", case(100, 16), 100, 16), ("(1000,32)", case(1000, 32), 1000, 32)]:
    out = new.absorbed_bmm_concat_cast_q_fp8(**fresh(kw))
    ok, mism, cd = check(kw, out, t, h)
    print(f"{'ok  ' if ok else 'FAIL'} {name:16s} nope mismatch {mism:.2e}, codes differing from generic {cd:.2e}", flush=True)

if len(sys.argv) > 1 and sys.argv[1] == "sweep":
    import json

    cfgs = [tuple(c[:5]) for c in json.load(open("/tmp/prune_regs.json"))["absorbed"]]
    for t, h in [(16, 16), (128, 16), (512, 16), (1024, 16), (4096, 16), (16, 8), (64, 8), (2048, 16), (256, 16)]:
        kw = case(t, h)
        tg = do_bench(lambda: gen(**kw), warmup=10, rep=50, return_mode="median") * 1e3
        res = []
        for cfg in cfgs:
            new._config = lambda tt, hh, c=cfg: c
            try:
                res.append((do_bench(lambda: new.absorbed_bmm_concat_cast_q_fp8(**kw), warmup=5, rep=25, return_mode="median") * 1e3, cfg))
            except Exception:  # noqa: BLE001
                continue
        res.sort()
        top = []
        for _, cfg in res[:5]:
            new._config = lambda tt, hh, c=cfg: c
            top.append((do_bench(lambda: new.absorbed_bmm_concat_cast_q_fp8(**kw), warmup=10, rep=100, return_mode="median") * 1e3, cfg))
        top.sort()
        best_t, best = top[0]
        new._config = lambda tt, hh, c=best: c
        wg, wn = warm(lambda: gen(**kw)), warm(lambda: new.absorbed_bmm_concat_cast_q_fp8(**kw))
        print(f"({t:4d},{h:2d}) generic cold {tg:6.1f} warm {wg:6.1f} | best {best} cold {best_t:6.1f} ({tg / best_t:4.2f}x) warm {wn:6.1f} ({wg / wn:4.2f}x) | "
              + " ".join(f"{c}:{x:.1f}" for x, c in top[1:4]), flush=True)
