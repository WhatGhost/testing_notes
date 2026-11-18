import os
import sys

import torch

W = open("/tmp/fgs-perf.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang.runtime.backend._amd.rdna3.ops.tiny_n_gemm as tng  # noqa: E402

SHAPES = [(3584, 5120), (4096, 5120), (8704, 5120), (5120, 4352), (5120, 1536),
          (62080, 5120), (2560, 2048), (3072, 2048), (2048, 1024), (62080, 2048),
          (2048, 128), (256, 2048), (512, 7168), (1536, 7168), (96, 300), (64, 2000),
          (512, 1000), (24, 5120), (1, 2048)]
MS = [1, 2, 3, 8, 16, 17, 32, 33, 48, 63, 64, 100]
g = torch.Generator(device="cuda").manual_seed(0)
worst = 0.0
fails = 0
configs = set()
for n, k in SHAPES:
    w = torch.randn(n, k, device="cuda", generator=g).to(torch.bfloat16)
    for m in MS:
        x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
        ref = x.float() @ w.float().T
        for out_dtype in (torch.bfloat16, torch.float32):
            if out_dtype == torch.float32 and m not in (2, 17, 64):
                continue
            out = tng.tiny_n_gemm(x, w, out_dtype)
            err = ((out.float() - ref).abs() / (1 + ref.abs())).max().item()
            worst = max(worst, err)
            tol = 1.6e-2 if out_dtype == torch.bfloat16 else 1e-3
            if not err <= tol:
                fails += 1
                print("FAIL", (m, n, k, out_dtype), err)
        if m > 1:
            block_k = min(256, 1 << (max(k, 16) - 1).bit_length())
            kb = -(-k // block_k)
            configs.add(tng._splitk_config(m, n, kb, n * k * 2))
print("worst rel err", f"{worst:.2e}", "fails", fails)
print("configs used:", sorted(configs))
