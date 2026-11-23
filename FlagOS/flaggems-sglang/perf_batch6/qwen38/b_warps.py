import importlib.util, os, sys, torch
from triton.testing import do_bench
W = open("/tmp/fgs-a.path").read().strip()
spec = importlib.util.spec_from_file_location("new_topk", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/qwen4_qsa_topk.py"))
new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
import triton
def args(rows):
    logits = torch.arange(rows * 640, device="cuda", dtype=torch.float32).reshape(rows, 640)
    starts = torch.arange(rows, device="cuda", dtype=torch.int32) % 4
    lengths = torch.tensor([513, 0, 1, 511, 512, 600, 636], device="cuda", dtype=torch.int32)
    return logits, starts, starts + lengths[torch.arange(rows, device="cuda") % 7], 512
for rows in (1, 64, 4096):
    logits, s, e, k = args(rows)
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    row = []
    for nw in (1, 2, 4, 8, 16):
        fn = lambda: new._topk_kernel[(rows,)](logits, s, e, out, logits.stride(0), logits.stride(1), 1, 1, out.stride(0), 640, k, 1024, num_warps=nw)
        fn()
        row.append(f"w{nw}:{do_bench(fn, warmup=10, rep=50, return_mode='median') * 1e3:.1f}")
    print(f"rows={rows}: " + " ".join(row), flush=True)
