"""rdna3 chunked_sgmv_expand without a max_len hint: same output as with it,
and no device-to-host copy on the way."""

import dataclasses
import os
import sys

import torch

B = open("/tmp/fgs-b3.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
sys.path.insert(0, os.path.join(B, "baseline"))
import flaggems_sglang  # noqa: E402,F401
from flaggems_sglang.reference import get_reference  # noqa: E402
from flaggems_sglang.reference_candidate import get_candidate  # noqa: E402
import flaggems_sglang.runtime.backend._amd.rdna3.ops.chunked_sgmv_expand as ex  # noqa: E402
from __lora_batch_utils import make_batch_info  # noqa: E402

for name in ("chunked_sgmv_shrink", "recompute_w_u", "chunked_sgmv_expand"):
    c = get_candidate(name, reference=get_reference(name))
    print(name, "->", getattr(c, "__module__", c))

torch.manual_seed(0)
cases = 0
for seg_lens in ([8], [3, 7, 0, 16], [64] * 8, [1] * 32, [256, 0, 1, 512]):
    for rank, n_slices, out in ((16, 1, 1024), (8, 3, 4096), (64, 3, 3072)):
        for perm in ("identity", "shuffled"):
            for dtype in (torch.bfloat16, torch.float32):
                s = sum(seg_lens)
                bi = make_batch_info(seg_lens, [i % 2 for i in range(len(seg_lens))], device="cuda",
                                     lora_ranks=[rank, rank], permutation=perm)
                x = torch.randn(s, n_slices * rank, device="cuda").to(dtype)
                w = torch.randn(2, out, rank, device="cuda").to(dtype)
                step = out // n_slices
                slice_offsets = torch.arange(0, out + 1, step, device="cuda", dtype=torch.int32)[: n_slices + 1]
                base = torch.randn(s, out, device="cuda").to(dtype)
                ref = ex.chunked_sgmv_expand(x, w, bi, slice_offsets, step, base)
                fields = {f.name: getattr(bi, f.name) for f in dataclasses.fields(bi)}
                no_hint = type("NoHint", (), {k: v for k, v in fields.items() if k != "max_len"})()
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    got = ex.chunked_sgmv_expand(x, w, no_hint, slice_offsets, step, base)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                assert torch.equal(got, ref), (seg_lens, rank, n_slices, perm, dtype)
                cases += 1
print("no-hint cases equal:", cases)
