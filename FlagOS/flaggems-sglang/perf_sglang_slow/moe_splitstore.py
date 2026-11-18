import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-perf.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe_gemm as fm  # noqa: E402

orig = fm._split_store
# (T, E, top_k, N, K)
CASES = [
    (8, 64, 8, 2048, 1024), (32, 64, 8, 2048, 1024), (128, 64, 8, 2048, 1024),
    (8, 64, 8, 2048, 768), (32, 64, 8, 2048, 768), (128, 64, 8, 2048, 768),
    (16, 8, 2, 4096, 1024), (256, 8, 2, 4096, 1024), (2048, 8, 2, 4096, 1024),
]
_UNUSED = [
    (512, 8, 2, 4096, 4096), (4096, 8, 2, 4096, 4096), (64, 8, 2, 4096, 4096),
    (256, 256, 8, 2048, 128), (1024, 256, 8, 2048, 128), (4096, 256, 8, 2048, 128),
    (1024, 256, 8, 256, 2048), (4096, 256, 8, 256, 2048),
    (512, 8, 2, 4096, 512), (4096, 8, 2, 4096, 512),
    (512, 8, 2, 8192, 256), (4096, 8, 2, 8192, 256),
    (1024, 64, 6, 2048, 256), (4096, 64, 6, 2048, 256),
    (1024, 64, 8, 2048, 768), (4096, 64, 8, 2048, 768),
    (1024, 64, 8, 2048, 1024), (4096, 64, 8, 2048, 1024),
    (4096, 128, 8, 4096, 1536),
]
g = torch.Generator(device="cuda").manual_seed(0)
for T, E, top_k, N, K in CASES:
    A = torch.randn(T, K, device="cuda", generator=g).to(torch.bfloat16)
    B = (torch.randn(E, N, K, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    logits = torch.randn(T, E, device="cuda", generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    res = {}
    outs = {}
    for flag in (False, True):
        fm._split_store = lambda K, flag=flag: flag
        fn = lambda: fm.fused_moe_gemm(A, B, w, ids, top_k)  # noqa: E731
        outs[flag] = fn()
        res[flag] = do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3
    fm._split_store = orig
    same = torch.equal(outs[False], outs[True])
    P = T * top_k
    print(f"T={T:5d} E={E:3d} top{top_k} N={N:5d} K={K:5d} rows/expert={P / E:7.1f} | "
          f"plain {res[False]:8.1f} split {res[True]:8.1f} -> {res[False] / res[True]:.2f}x"
          f" | rule={'split' if orig(K) else 'plain'} identical={same}", flush=True)
