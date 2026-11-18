import importlib, os, sys, itertools
import torch
from triton.testing import do_bench
sys.path.insert(0, "/tmp")
from b4_pgt_lib import cur, split
NUM_EXPERTS = 8
CFGS = [(1, 32, 32, 4), (2, 32, 32, 8), (4, 32, 32, 8), (8, 32, 32, 8), (4, 64, 64, 8), (8, 32, 128, 8), (16, 64, 128, 8), (4, 32, 64, 8), (8, 32, 64, 8)]
print("k rows | current | " + " | ".join(str(c) for c in CFGS))
for k, rows in [(k, n) for k in (128, 512, 4096) for n in (16, 128, 1024)]:
    a = torch.randn(rows * NUM_EXPERTS, k, device="cuda").to(torch.bfloat16)
    offs = torch.tensor([i * rows for i in range(NUM_EXPERTS + 1)], dtype=torch.int32, device="cuda")
    t_cur = do_bench(lambda: cur.per_group_transpose(a, offs), warmup=25, rep=200, return_mode="median") * 1e3
    ts = [do_bench(lambda: split(a, offs, *c), warmup=25, rep=200, return_mode="median") * 1e3 for c in CFGS]
    print(f"{k} {rows} | {t_cur:.1f} | " + " | ".join(f"{t:.1f}" for t in ts), flush=True)
