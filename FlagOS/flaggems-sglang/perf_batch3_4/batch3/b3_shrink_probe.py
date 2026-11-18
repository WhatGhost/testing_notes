"""chunked_sgmv_shrink on gfx1100: current dispatch vs. the generic module's own
candidate launchers, on the official benchmark shapes; plus a same-bytes copy."""

import importlib
import inspect
import os
import sys

import torch
from triton.testing import do_bench

B = open("/tmp/fgs-b3.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
sys.path.insert(0, os.path.join(B, "baseline"))
import flaggems_sglang  # noqa: E402,F401

gen = importlib.import_module("flaggems_sglang.ops.chunked_sgmv_shrink")
bl = importlib.import_module("chunked_sgmv_shrink")

cands = {n: f for n, f in vars(gen).items()
         if inspect.isfunction(f) and not n.startswith("_") and n != "chunked_sgmv_shrink"
         and list(inspect.signature(f).parameters)[:3] == ["x", "weights", "batch_info"]
         and not n.startswith(("serial_", "gcu_"))}
print(len(cands), "candidate launchers")

for shape in bl.SHAPES:
    x, w, kw = bl.build_case(shape, torch.bfloat16, "cuda")
    bi, ns = kw["batch_info"], kw["num_slices"]
    ref = gen.chunked_sgmv_shrink(x, w, bi, ns)
    t_cur = do_bench(lambda: gen.chunked_sgmv_shrink(x, w, bi, ns), warmup=50, rep=200, return_mode="median") * 1e3
    nbytes = x.numel() * 2 + w.numel() * 2 + ref.numel() * 2
    src = torch.empty(nbytes // 2, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    t_copy = do_bench(lambda: dst.copy_(src), warmup=50, rep=200, return_mode="median") * 1e3
    res = []
    for n, f in cands.items():
        try:
            out = f(x, w, bi, ns)
            err = (out.float() - ref.float()).abs().max().item()
            if not err < 0.5:
                continue
            t = do_bench(lambda: f(x, w, bi, ns), warmup=25, rep=100, return_mode="median") * 1e3
            res.append((t, n))
        except Exception:
            pass
    res.sort()
    print(f"shape {shape}: x {tuple(x.shape)} w {tuple(w.shape)} | current {t_cur:.1f} us | copy {t_copy:.1f} us "
          f"(ratio {t_cur / t_copy:.2f}) | best: " + ", ".join(f"{n} {t:.1f}" for t, n in res[:5]), flush=True)
