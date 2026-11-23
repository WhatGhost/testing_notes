import importlib.util, os, torch
from triton.testing import do_bench
W = open("/tmp/fgs-a.path").read().strip()
spec = importlib.util.spec_from_file_location("new_topk", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/qwen4_qsa_topk.py"))
new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
for rows in (1, 2):
    logits = torch.arange(rows * 640, device="cuda", dtype=torch.float32).reshape(rows, 640)
    s = torch.arange(rows, device="cuda", dtype=torch.int32) % 4
    e = s + torch.tensor([513, 0], device="cuda", dtype=torch.int32)[:rows]
    out = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
    common = (logits, s, e, out, logits.stride(0), logits.stride(1), 1, 1, out.stride(0), 640, 512, 1024)
    fns = {"rank(16,256,4)": lambda: new._rank_kernel[(rows, 64)](*common, 16, 256, num_warps=4),
           "sort w8": lambda: new._sort_kernel[(rows,)](*common, num_warps=8),
           "sort w16": lambda: new._sort_kernel[(rows,)](*common, num_warps=16)}
    for f in fns.values(): f()
    res = {k: [] for k in fns}
    for _ in range(3):
        for k, f in fns.items():
            res[k].append(do_bench(f, warmup=25, rep=100, return_mode="median") * 1e3)
    print(f"rows={rows}: " + "  ".join(f"{k} {sorted(v)[1]:.1f}" for k, v in res.items()), flush=True)
