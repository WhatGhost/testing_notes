"""fused_moe accuracy: bf16 vs fp32 storage of the down-projection rows (c3)."""

import importlib
import os
import sys

import torch

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401
from flaggems_sglang.reference import get_reference  # noqa: E402

ref = get_reference("fused_moe")
r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
g = importlib.import_module("flaggems_sglang.ops.fused_moe")
orig_empty = torch.empty


def run(variant, *args):
    if variant == "c3_fp32":
        def empty(*shape, dtype=None, device=None, **kw):
            if dtype in (torch.bfloat16, torch.float16) and len(shape) == 1 and len(shape[0]) == 2 \
                    and shape[0][1] == args[0].shape[1]:
                dtype = torch.float32
            return orig_empty(*shape, dtype=dtype, device=device, **kw)
        torch.empty = empty
    try:
        return r.fused_moe(*args)
    finally:
        torch.empty = orig_empty


for (T, H, I, E, topk) in ((64, 2048, 1408, 64, 8), (300, 2048, 1408, 64, 8), (200, 2048, 128, 256, 8),
                           (1024, 2048, 1408, 64, 8)):
    for rsf in (1.0, 2.5):
        gen = torch.Generator(device="cuda").manual_seed(T + E)
        x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
        w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
        to = (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None)
        want = ref(x, w1, w2, to, "silu", rsf).float()
        cells = []
        for name, fn in (("generic", lambda: g.fused_moe(x, w1, w2, to, "silu", rsf)),
                         ("bf16 c3", lambda: run("c3_bf16", x, w1, w2, to, "silu", rsf)),
                         ("fp32 c3", lambda: run("c3_fp32", x, w1, w2, to, "silu", rsf))):
            out = fn().float()
            d = (out - want).abs()
            excess = (d - (2e-2 + 2e-2 * want.abs())).max().item()
            ulps = (d / torch.where(want == 0, torch.ones_like(want), want.abs() * 2 ** -8)).max().item()
            cells.append(f"{name}: max {d.max().item():.3e} worst excess {excess:+.3e} max ulps {ulps:.1f}")
        print(f"T={T} I={I} E={E} rsf={rsf} | " + " | ".join(cells), flush=True)
