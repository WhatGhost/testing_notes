"""ltx2_split_rotary on gfx1100: tiling variants vs the generic flat kernel."""

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
import b4_ltx_lib as L  # noqa: E402

gen = importlib.import_module("flaggems_sglang.ops.ltx2_split_rotary")


def make(batch, t, heads=16, hd=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(batch, t, heads * hd, device="cuda", generator=g).to(torch.bfloat16)
    ang = torch.randn(batch, heads, t, hd // 2, device="cuda", generator=g)
    return x, torch.cos(ang).to(torch.bfloat16), torch.sin(ang).to(torch.bfloat16)


def flat(x, c, s, blk, nw):
    b, h, t, d = c.shape
    out = torch.empty_like(x)
    n = b * t * h * d
    L._flat[(triton.cdiv(n, blk),)](x, c, s, out, n, t, h, d, BLOCK=blk, num_warps=nw)
    return out


def token_tile(x, c, s, bt, nw):
    b, h, t, d = c.shape
    out = torch.empty_like(x)
    L._token_tile[(triton.cdiv(t, bt), b)](x, c, s, out, t, h, d, BT=bt, BH=triton.next_power_of_2(h), num_warps=nw)
    return out


def head_tile(x, c, s, bt, nw):
    b, h, t, d = c.shape
    out = torch.empty_like(x)
    L._head_tile[(triton.cdiv(t, bt), b * h)](x, c, s, out, t, h, d, BT=bt, num_warps=nw)
    return out


cands = []
for blk, nw in ((512, 2), (512, 4), (1024, 4), (2048, 4), (2048, 8), (4096, 8)):
    cands.append((f"flat{blk}/{nw}", lambda x, c, s, blk=blk, nw=nw: flat(x, c, s, blk, nw)))
for bt, nw in ((1, 4), (2, 4), (2, 8), (4, 8), (8, 8), (4, 16)):
    cands.append((f"tok{bt}/{nw}", lambda x, c, s, bt=bt, nw=nw: token_tile(x, c, s, bt, nw)))
for bt, nw in ((16, 4), (32, 4), (32, 8), (64, 8), (128, 8), (64, 4)):
    cands.append((f"head{bt}/{nw}", lambda x, c, s, bt=bt, nw=nw: head_tile(x, c, s, bt, nw)))

for batch, t, heads, hd in ((1, 1, 16, 128), (1, 64, 16, 128), (1, 1024, 16, 128), (1, 8192, 16, 128),
                            (1, 4096, 16, 128), (2, 512, 8, 128), (1, 2048, 16, 64), (1, 129, 16, 128),
                            (1, 16384, 16, 128), (1, 8192, 32, 128)):
    x, c, s = make(batch, t, heads, hd)
    ref = gen.ltx2_split_rotary(x, c, s)
    t_gen = do_bench(lambda: gen.ltx2_split_rotary(x, c, s), warmup=25, rep=200, return_mode="median") * 1e3
    res = []
    for name, fn in cands:
        o = fn(x, c, s)
        if not torch.equal(o, ref):
            print("MISMATCH", (batch, t, heads, hd), name, (o.float() - ref.float()).abs().max().item())
            continue
        tt = do_bench(lambda: fn(x, c, s), warmup=25, rep=200, return_mode="median") * 1e3
        res.append((tt, name))
    res.sort()
    print(f"{str((batch, t, heads, hd)):20s} | generic {t_gen:7.1f} us | best: " +
          ", ".join(f"{n} {tt:.1f}" for tt, n in res[:5]), flush=True)
