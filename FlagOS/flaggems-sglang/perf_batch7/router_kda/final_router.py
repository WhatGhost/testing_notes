"""Final router timings: master, the previous dispatch (split-K matrix path
from 1 token, master's kernel for < 128 experts and 8 tokens) and now,
3 rounds alternated, medians."""

import sys

import torch
import triton

sys.path.insert(0, "/tmp")
from opt_router_gpu_lib import amd, new, bench  # noqa: E402

g = torch.Generator(device="cuda").manual_seed(0)


def old_cfg(rows, experts):
    if rows <= 16:
        return (16, 64, 128, 4, 1, 8) if experts >= 384 else (16, 16, 128, 1, 1, 8)
    if rows <= 128:
        return 16, 32, 256, 2, 2, 4
    if rows <= 1024:
        return 128, 32, 128, 4, 2, 4
    return 128, 64, 128, 8, 1, 1


def previous(h, w):
    m, k = h.shape
    e = w.shape[0]
    if e < 128 or m == 8:
        return amd.dsv3_router_gemm(h, w, torch.float32)
    bm_, be, bk, warps, stages, split = old_cfg(m, e)
    out = torch.empty((m, e), dtype=torch.float32, device="cuda")
    part = out if split == 1 else torch.empty((split, m, e), dtype=torch.float32, device="cuda")
    new._router_dot_kernel[(triton.cdiv(m, bm_), triton.cdiv(e, be), split)](
        h, w, part, m, split, e, k, *h.stride(), *w.stride(), bm_, be, bk, num_warps=warps, num_stages=stages)
    if split > 1:
        new._router_sum_kernel[(triton.cdiv(m * e, 1024),)](part, out, m * e, split, 1024, num_warps=4)
    return out


official = [(m, e) for m in (1, 2, 4, 8, 16) for e in (256, 384)] + [(1, 128), (16, 512), (8, 64)]
extra = [(3, 256), (5, 256), (6, 384), (12, 256), (2, 128), (4, 64), (16, 64), (16, 128), (4, 512), (64, 256), (512, 256), (4096, 256)]
print(f"{'tokens, experts':16s} {'master':>8s} {'previous':>9s} {'now':>8s}  now vs master  now vs previous  kernel")
for label, shapes in (("official", official), ("extra", extra)):
    print(f"-- {label}")
    for m, e in shapes:
        h = torch.randn(m, 7168, device="cuda", generator=g).bfloat16()
        w = torch.randn(e, 7168, device="cuda", generator=g).bfloat16()
        tm, tp, tn = [], [], []
        for _ in range(3):
            tm.append(bench(lambda: amd.dsv3_router_gemm(h, w, torch.float32)))
            tp.append(bench(lambda: previous(h, w)))
            tn.append(bench(lambda: new.dsv3_router_gemm(h, w, torch.float32)))
        tm, tp, tn = (sorted(x)[1] for x in (tm, tp, tn))
        kind = new._config(m, e)[0]
        print(f"{str((m, e)):16s} {tm:8.1f} {tp:9.1f} {tn:8.1f}  {tm / tn:12.2f}x  {tp / tn:14.2f}x  {kind}", flush=True)
