"""Sweep (BM, BN, warps) of the rdna3 recompute_w_u against the generic op
(gfx1100, do_bench median); also report whether outputs are bitwise equal."""

import importlib
import itertools
import os
import sys

import torch
from triton.testing import do_bench

B = open("/tmp/fgs-b3.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

gen = importlib.import_module("flaggems_sglang.ops.recompute_w_u")
new = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.recompute_w_u")

# (B, T, H, Hg, K, V, BT)
SHAPES = [(1, t, 16, 4, 128, 128, 64) for t in (64, 256, 1024, 4096)] + [
    (1, 8192, 12, 4, 128, 128, 64), (2, 4096, 8, 4, 128, 128, 64), (1, 2048, 16, 16, 64, 64, 64),
    (4, 512, 12, 4, 128, 128, 64)]
CFGS = [(16, 32, 2), (32, 32, 2), (32, 64, 4), (64, 32, 4), (64, 64, 8), (16, 64, 2), (64, 64, 4)]
rule_only = len(sys.argv) > 1 and sys.argv[1] == "rule"
orig = new._compact_config
for (b, t, h, hg, kd, vd, bt) in SHAPES:
    g = torch.Generator(device="cuda").manual_seed(9)

    def r(*s):
        return torch.randn(*s, device="cuda", generator=g).to(torch.bfloat16)

    k, v = r(b, t, hg, kd), r(b, t, h, vd)
    beta = torch.rand(b, t, h, device="cuda", generator=g).to(torch.bfloat16)
    gc = -torch.rand(b, t, h, device="cuda", generator=g).cumsum(1).float()
    A = (r(b, t, h, bt) * 0.1)
    args = (k, v, beta, gc, A, None)
    rw, ru = gen.recompute_w_u(*args)
    t_gen = do_bench(lambda: gen.recompute_w_u(*args), warmup=50, rep=400, return_mode="median") * 1e3
    res = []
    for cfg in ([None] if rule_only else CFGS):
        new._compact_config = orig if cfg is None else (lambda q, cfg=cfg: cfg)
        w, u = new.recompute_w_u(*args)
        same = torch.equal(w, rw) and torch.equal(u, ru)
        err = max((w.float() - rw.float()).abs().max().item(), (u.float() - ru.float()).abs().max().item())
        tm = do_bench(lambda: new.recompute_w_u(*args), warmup=50, rep=400, return_mode="median") * 1e3
        res.append((tm, cfg if cfg else orig(b * (t // bt) * h), same, err))
    new._compact_config = orig
    res.sort(key=lambda x: x[0])
    q = b * (t // bt) * h
    print(f"B{b} T{t} H{h} Hg{hg} K{kd} q={q:5d} | generic {t_gen:6.1f} | " + ", ".join(
        f"{c} {tm:.1f}{'' if s else f' (diff {e:.1e})'}" for tm, c, s, e in res[:4]) +
        (f" | -> {t_gen / res[0][0]:.2f}x" if rule_only else ""), flush=True)
