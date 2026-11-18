"""Correctness of the rdna3 chunked_sgmv_shrink against the generic op and a
float64 reference, including uncovered rows, empty segments and permutations."""

import importlib
import os
import sys

import torch

B = open("/tmp/fgs-b3.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
sys.path.insert(0, os.path.join(B, "baseline"))
import flaggems_sglang  # noqa: E402,F401

gen = importlib.import_module("flaggems_sglang.ops.chunked_sgmv_shrink")
new = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.chunked_sgmv_shrink")
from __lora_batch_utils import make_batch_info  # noqa: E402

torch.manual_seed(0)
fails = 0
n = 0
for dtype in (torch.bfloat16, torch.float16, torch.float32):
    for seg_lens in ([8], [3, 7, 0, 16], [64] * 8, [1] * 32, [256, 0, 1, 512], [2048]):
        for k in (64, 100, 1000, 2048, 4096, 5120):
            for r, ns in ((16, 1), (8, 3), (32, 3), (64, 3)):
                for perm in ("none", "identity", "shuffled"):
                    for extra in (0, 5):
                        num_lora = 3
                        s = sum(seg_lens)
                        if s == 0:
                            continue
                        x = torch.randn(s + extra, k, device="cuda").to(dtype)
                        w = torch.randn(num_lora, ns * r, k, device="cuda").to(dtype)
                        wi = [(i * 7) % num_lora for i in range(len(seg_lens))]
                        bi = make_batch_info(seg_lens, wi, device="cuda", lora_ranks=[r] * num_lora,
                                             permutation=perm)
                        out = new.chunked_sgmv_shrink(x, w, bi, ns)
                        # float64 reference over the covered positions
                        ref = torch.zeros(s + extra, ns * r, dtype=torch.float64, device="cuda")
                        p = bi.permutation if bi.permutation is not None else torch.arange(s, device="cuda")
                        for i, (a, b) in enumerate(zip(bi.seg_indptr[:-1].tolist(), bi.seg_indptr[1:].tolist())):
                            rows = p[a:b].long()
                            ref[rows] = x[rows].double() @ w[wi[i]].double().T
                        g = gen.chunked_sgmv_shrink(x, w, bi, ns)
                        scale = 1 + ref.abs()
                        e_new = ((out.double() - ref).abs() / scale).max().item()
                        e_gen = ((g.double() - ref).abs() / scale).max().item()
                        tol = 1e-4 if dtype == torch.float32 else 2e-2
                        n += 1
                        if not (e_new <= max(tol, 1.5 * e_gen)) or out.shape != g.shape or out.dtype != g.dtype:
                            fails += 1
                            print("FAIL", dtype, seg_lens, k, r, ns, perm, extra, f"new {e_new:.2e} gen {e_gen:.2e}")
print("cases", n, "fails", fails)
