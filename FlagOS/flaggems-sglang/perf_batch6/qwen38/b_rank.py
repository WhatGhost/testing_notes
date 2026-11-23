import importlib.util, itertools, os, sys, torch
from triton.testing import do_bench
W = open("/tmp/fgs-a.path").read().strip()
spec = importlib.util.spec_from_file_location("new_topk", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/qwen4_qsa_topk.py"))
new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
ref = importlib.util.spec_from_file_location("ref", os.path.join(W, "src/flaggems_sglang/reference/qwen4_qsa_topk.py"))
refm = importlib.util.module_from_spec(ref); ref.loader.exec_module(refm)
def args(rows, ties=False):
    if ties:
        logits = torch.randint(-3, 4, (rows, 640), device="cuda").float()
    else:
        logits = torch.arange(rows * 640, device="cuda", dtype=torch.float32).reshape(rows, 640)
    starts = torch.arange(rows, device="cuda", dtype=torch.int32) % 4
    lengths = torch.tensor([513, 0, 1, 511, 512, 600, 636], device="cuda", dtype=torch.int32)
    return logits, starts, starts + lengths[torch.arange(rows, device="cuda") % 7], 512
# exactness of the rank kernel
for rows, ties in ((7, False), (37, True)):
    logits, s, e, k = args(rows, ties)
    out = torch.full((rows, k), -7, dtype=torch.int32, device="cuda")
    new._rank_kernel[(rows, 1024 // 64)](logits, s, e, out, logits.stride(0), logits.stride(1), 1, 1, out.stride(0), 640, k, 1024, 64, 128, num_warps=4)
    exp = refm.reference(logits.cpu(), s.cpu(), e.cpu(), k)
    print(f"rank kernel rows={rows} ties={ties}: exact={torch.equal(out.cpu(), exp)}", flush=True)
for rows in (1, 7, 64, 256, 1024):
    logits, s, e, k = args(rows)
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    res = []
    for chunk, jb, nw in itertools.product((16, 32, 64, 128), (64, 128, 256), (1, 2, 4, 8)):
        if chunk * jb // (nw * 32) > 128:
            continue
        fn = lambda: new._rank_kernel[(rows, 1024 // chunk)](logits, s, e, out, logits.stride(0), logits.stride(1), 1, 1, out.stride(0), 640, k, 1024, chunk, jb, num_warps=nw)
        fn()
        res.append((do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3, (chunk, jb, nw)))
    res.sort()
    print(f"rows={rows}: " + " ".join(f"{c}:{t:.1f}" for t, c in res[:4]), flush=True)
