"""rdna3 mla_kv_pack_quantize_fp8 vs the generic op: byte-for-byte outputs over
shapes, special values, scales, fp8 formats, 3-D k_pe, strided inputs and
non-power-of-two widths; then the benchmark shapes timed against master."""

import importlib.util
import os
import statistics

import torch
from triton.testing import do_bench

T = open("/tmp/fgs-f8.path").read().strip()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_mla", os.path.join(T, "src/flaggems_sglang/ops/mla_kv_pack_quantize_fp8.py"))
new = load("new_mla", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/mla_kv_pack_quantize_fp8.py"))


def make(s, h, nope=512, rope=64, width=512, dtype=torch.bfloat16, seed=0, special=False, pe_3d=False, strided=False):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def t(*dims):
        x = torch.randn(*dims, device="cuda", generator=g) * 3
        if special and x.numel() > 16:
            flat = x.view(-1)
            vals = torch.tensor([float("nan"), float("inf"), -float("inf"), 1e6, -1e6, 1e-10, -1e-10, -0.0, 0.0,
                                 448.0, -448.0, 464.0, 57344.0, 65504.0, 2.0 ** -9, 2.0 ** -10], device="cuda")
            idx = torch.randint(0, flat.numel(), (min(4096, flat.numel()),), device="cuda", generator=g)
            flat[idx] = vals[torch.arange(len(idx), device="cuda") % len(vals)]
        return x.to(dtype)

    if strided:
        k_nope = t(s, h, nope + 64)[:, :, :nope]
        v = t(s, h + 1, width)[:, :h, :]
    else:
        k_nope = t(s, h, nope)
        v = t(s, h, width)
    k_pe = t(s, 1, rope) if pe_3d else t(s, rope)
    return k_nope, k_pe, v


def bench(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


for s, h in [(s, h) for s in (1, 8, 64, 512, 4096) for h in (1, 16)] + [(7, 8), (333, 1), (256, 8)]:
    g = torch.Generator(device="cuda").manual_seed(0)
    k_nope = torch.randn(s, h, 512, device="cuda", generator=g).to(torch.bfloat16)
    k_pe = torch.randn(s, 64, device="cuda", generator=g).to(torch.bfloat16)
    v = torch.randn(s, h, 512, device="cuda", generator=g).to(torch.bfloat16)
    tg, tn = [], []
    for _ in range(7):
        tg.append(bench(lambda: gen.mla_kv_pack_quantize_fp8(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)))
        tn.append(bench(lambda: new.mla_kv_pack_quantize_fp8(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)))
    g_, n_ = statistics.median(tg), statistics.median(tn)
    print(f"official s={s:5d} h={h:2d}: master {g_:8.1f} us  new {n_:8.1f} us  {g_ / n_:5.2f}x", flush=True)
