import importlib.util
import os
import sys

import torch

W = open("/tmp/fgs-perf.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
spec = importlib.util.spec_from_file_location(
    "t_ext", os.path.join(W, "tests/test_extend_attention.py")
)
t_ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t_ext)

from flaggems_sglang.reference import get_reference  # noqa: E402
from flaggems_sglang.reference_candidate import get_candidate  # noqa: E402
import flaggems_sglang.runtime.backend._amd.rdna3.ops.extend_attention as ea  # noqa: E402

reference = get_reference("extend_attention")
candidate = get_candidate("extend_attention", reference=reference)
print("candidate module:", getattr(candidate, "__module__", candidate))

cases = []
for hq, hkv in ((6, 1), (4, 1), (8, 1), (16, 2), (2, 2), (24, 4), (8, 8)):
    for B, n_ctx in ((1, 64), (3, 300), (4, 1024), (2, 4096), (16, 512)):
        for D in (256, 192):
            if D == 192 and (B, n_ctx) not in ((3, 300), (4, 1024)):
                continue
            cases.append((B, n_ctx, hq, hkv, D, torch.bfloat16))
cases += [(4, 1024, 6, 1, 256, torch.float16), (2, 300, 4, 1, 256, torch.float32)]

orig = ea._PACKED_WIDE_WORK
worst = 0.0
fails = 0
for mode, thr in (("auto", orig), ("small", float("inf")), ("large", 0)):
    ea._PACKED_WIDE_WORK = thr
    for i, (B, n_ctx, hq, hkv, D, dtype) in enumerate(cases):
        case = t_ext._case(B, n_ctx, hq, hkv, D, dtype=dtype, seed=100 + i)
        check = case.pop("check")
        exp = reference(**case)
        out = (candidate if mode == "auto" else ea.extend_attention)(**case)
        err = (out.float() - exp.float()).abs().max().item()
        worst = max(worst, err)
        try:
            check(out, exp)
        except AssertionError as e:
            fails += 1
            print("FAIL", mode, (B, n_ctx, hq, hkv, D, dtype), str(e)[:200])
    print(mode, "done; worst abs err so far", f"{worst:.2e}")
ea._PACKED_WIDE_WORK = orig
print("fails", fails, "of", 3 * len(cases))
