"""ltx2_split_rotary: token tile sized by elements per program."""

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


def token_tile(x, c, s, elems, nw):
    b, h, t, d = c.shape
    bh = triton.next_power_of_2(h)
    bt = max(1, elems // (bh * d))
    out = torch.empty_like(x)
    L._token_tile[(triton.cdiv(t, bt), b)](x, c, s, out, t, h, d, BT=bt, BH=bh, num_warps=nw)
    return out


cands = [(f"E{e}/{nw}", lambda x, c, s, e=e, nw=nw: token_tile(x, c, s, e, nw))
         for e in (2048, 4096, 8192, 16384) for nw in (4, 8, 16, 32) if e // nw >= 128]

for batch, t, heads, hd in ((1, 1024, 16, 128), (1, 8192, 16, 128), (1, 4096, 16, 128), (2, 512, 8, 128),
                            (1, 2048, 16, 64), (1, 16384, 16, 128), (1, 8192, 32, 128), (1, 512, 16, 128),
                            (1, 256, 16, 128), (1, 4096, 24, 128), (4, 1024, 16, 128)):
    x, c, s = make(batch, t, heads, hd)
    ref = gen.ltx2_split_rotary(x, c, s)
    t_gen = do_bench(lambda: gen.ltx2_split_rotary(x, c, s), warmup=25, rep=200, return_mode="median") * 1e3
    res = []
    for name, fn in cands:
        o = fn(x, c, s)
        if not torch.equal(o, ref):
            print("MISMATCH", (batch, t, heads, hd), name)
            continue
        tt = do_bench(lambda: fn(x, c, s), warmup=25, rep=200, return_mode="median") * 1e3
        res.append((tt, name))
    d = dict((n, tt) for tt, n in res)
    res.sort()
    print(f"{str((batch, t, heads, hd)):20s} | generic {t_gen:7.1f} | E4096/16 {d.get('E4096/16', 0):7.1f} "
          f"E4096/8 {d.get('E4096/8', 0):7.1f} | best: " + ", ".join(f"{n} {tt:.1f}" for tt, n in res[:4]),
          flush=True)
