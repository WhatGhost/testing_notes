"""Router mid token counts (5-8) and the 64-expert row: candidates
interleaved, 5 rounds, medians."""

import sys

import torch
import triton

sys.path.insert(0, "/tmp")
from opt_router_gpu_lib import gemv, scalar, new, bench  # noqa: E402

g = torch.Generator(device="cuda").manual_seed(1)
print(" M    E | shipped | scalar w2 | scalar w4 | scalar w8 |  gemv")
for e, ms in [(64, (1, 2, 4, 8, 12, 16)), (256, (5, 6, 7, 8)), (384, (5, 6, 7, 8)), (512, (5, 6, 7, 8))]:
    for m in ms:
        h = torch.randn(m, 7168, device="cuda", generator=g).bfloat16()
        w = torch.randn(e, 7168, device="cuda", generator=g).bfloat16()
        mb = triton.next_power_of_2(m)
        fns = {"shipped": lambda: new.dsv3_router_gemm(h, w, torch.float32), "w2": lambda: scalar(h, w, 2),
               "w4": lambda: scalar(h, w, 4), "w8": lambda: scalar(h, w, 8),
               "gemv": lambda: gemv(h, w, mb, 2, 1024, 4 if mb <= 4 else 8)}
        t = {k: [] for k in fns}
        for _ in range(5):
            for k, fn in fns.items():
                t[k].append(bench(fn))
        med = {k: sorted(v)[2] for k, v in t.items()}
        print(f"{m:2d} {e:4d} | {med['shipped']:7.1f} | {med['w2']:9.1f} | {med['w4']:9.1f} | {med['w8']:9.1f} | {med['gemv']:6.1f}", flush=True)
