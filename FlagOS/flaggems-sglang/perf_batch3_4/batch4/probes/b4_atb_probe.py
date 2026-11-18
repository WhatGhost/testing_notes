"""apply_token_bitmask on gfx1100: block / warps / word-expansion variants."""

import importlib
import os
import sys

import torch
import triton
from triton.testing import do_bench

sys.path.insert(0, "/tmp")
B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401
import b4_atb_lib as L  # noqa: E402

gen = importlib.import_module("flaggems_sglang.ops.apply_token_bitmask")


def flat(logits, bitmask, block, nw):
    b, v = logits.shape
    out = torch.empty_like(logits)
    L._atb[(b, triton.cdiv(v, block))](logits, bitmask, out, v, bitmask.shape[1], BLOCK=block, num_warps=nw)
    return out


def words(logits, bitmask, wcount, nw):
    b, v = logits.shape
    out = torch.empty_like(logits)
    L._atb_words[(b, triton.cdiv(bitmask.shape[1], wcount))](logits, bitmask, out, v, bitmask.shape[1],
                                                             WORDS=wcount, num_warps=nw)
    return out


for b, v in ((1, 152064), (8, 152064), (64, 152064), (512, 152064), (4096, 152064), (64, 32000), (64, 128256),
             (256, 152064), (7, 1000)):
    g = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(b, v, generator=g, device="cuda")
    bitmask = torch.randint(-2**31, 2**31 - 1, (b, (v + 31) // 32), dtype=torch.int32, device="cuda", generator=g)
    ref = gen.apply_token_bitmask(logits, bitmask)
    t_cur = do_bench(lambda: gen.apply_token_bitmask(logits, bitmask), warmup=10, rep=100, return_mode="median") * 1e3
    res = []
    for blk in (1024, 2048, 4096, 8192):
        for nw in (4, 8, 16):
            o = flat(logits, bitmask, blk, nw)
            assert torch.equal(o, ref)
            res.append((do_bench(lambda: flat(logits, bitmask, blk, nw), warmup=10, rep=100,
                                 return_mode="median") * 1e3, f"flat{blk}/{nw}"))
    for wc in (32, 64, 128, 256):
        for nw in (4, 8, 16):
            o = words(logits, bitmask, wc, nw)
            assert torch.equal(o, ref)
            res.append((do_bench(lambda: words(logits, bitmask, wc, nw), warmup=10, rep=100,
                                 return_mode="median") * 1e3, f"words{wc}/{nw}"))
    res.sort()
    print(f"({b}, {v}) | current {t_cur:8.1f} | best: " + ", ".join(f"{n} {t:.1f}" for t, n in res[:4]), flush=True)
