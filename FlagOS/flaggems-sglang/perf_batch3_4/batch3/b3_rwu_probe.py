"""recompute_w_u on gfx1100: current dispatch vs. forcing the single-kernel
_compact path with a few configs, on the official benchmark shapes."""

import importlib
import os
import sys

import torch
import triton
from triton.testing import do_bench

B = open("/tmp/fgs-b3.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
sys.path.insert(0, os.path.join(B, "baseline"))
import flaggems_sglang  # noqa: E402,F401

gen = importlib.import_module("flaggems_sglang.ops.recompute_w_u")
bl = importlib.import_module("recompute_w_u")


def compact(k, v, beta, g, A, bm, bn, nw, split, stages=1):
    Bsz, T, Hg, K = k.shape
    H, V = v.shape[-2:]
    BT = A.shape[-1]
    w = torch.empty((Bsz, T, H, K), dtype=k.dtype, device=k.device)
    u = torch.empty_like(v)
    q = Bsz * (T // BT) * H
    nt = triton.cdiv(K, bn) + triton.cdiv(V, bn) if split else triton.cdiv(max(K, V), bn)
    gen._compact[(q, triton.cdiv(BT, bm) * nt)](
        k, v, beta, g, A, w, u, H, Hg, K, V, BT, bm, bn,
        max(16, triton.next_power_of_2(BT)), split, False,
        num_warps=nw, num_stages=stages, enable_fp_fusion=False)
    return w, u


for b in bl.grid():
    args, kw = bl.build_inputs(b, torch.bfloat16, "cuda")
    k, v, beta, g, A = args[:5]
    ref_w, ref_u = gen.recompute_w_u(*args, **kw)
    t_cur = do_bench(lambda: gen.recompute_w_u(*args, **kw), warmup=50, rep=200, return_mode="median") * 1e3
    nbytes = sum(t.numel() * t.element_size() for t in (k, v, beta, g, A, ref_w, ref_u))
    src = torch.empty(nbytes // 2, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    t_copy = do_bench(lambda: dst.copy_(src), warmup=50, rep=200, return_mode="median") * 1e3
    res = []
    for bm in (16, 32, 64):
        for bn in (32, 64, 128):
            for nw in (2, 4, 8):
                for split in (True, False):
                    try:
                        w, u = compact(k, v, beta, g, A, bm, bn, nw, split)
                        if not (torch.equal(w, ref_w) and torch.equal(u, ref_u)):
                            ew = (w.float() - ref_w.float()).abs().max().item()
                            eu = (u.float() - ref_u.float()).abs().max().item()
                            if max(ew, eu) > 0.06:
                                continue
                        t = do_bench(lambda: compact(k, v, beta, g, A, bm, bn, nw, split),
                                     warmup=25, rep=100, return_mode="median") * 1e3
                        res.append((t, f"bm{bm} bn{bn} w{nw} {'split' if split else 'fused'}"))
                    except Exception as e:  # noqa: BLE001
                        pass
    res.sort()
    print(f"T={b['t']:5d} q={b['t'] // 64 * 16:5d} | current {t_cur:.1f} us | copy {t_copy:.1f} us "
          f"(ratio {t_cur / t_copy:.2f}) | best compact: " + ", ".join(f"{n} {t:.1f}" for t, n in res[:4]),
          flush=True)
