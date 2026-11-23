import importlib.util, itertools, os, torch
from triton.testing import do_bench

W = open("/tmp/fgs-a.path").read().strip()
spec = importlib.util.spec_from_file_location("new_topk", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/qwen4_qsa_topk.py"))
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)


def args(rows):
    logits = torch.arange(rows * 640, device="cuda", dtype=torch.float32).reshape(rows, 640)
    starts = torch.arange(rows, device="cuda", dtype=torch.int32) % 4
    lengths = torch.tensor([513, 0, 1, 511, 512, 600, 636], device="cuda", dtype=torch.int32)
    return logits, starts, starts + lengths[torch.arange(rows, device="cuda") % 7]


for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096):
    logits, s, e = args(rows)
    out = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
    common = (logits, s, e, out, logits.stride(0), logits.stride(1), 1, 1, out.stride(0), 640, 512, 1024)
    res = []
    if rows <= 256:
        for chunk, jb, nw in itertools.product((16, 32, 64), (128, 256), (1, 2, 4, 8)):
            if chunk * jb // (nw * 32) > 128:
                continue
            fn = lambda: new._rank_kernel[(rows, 1024 // chunk)](*common, chunk, jb, num_warps=nw)
            fn()
            res.append((do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3, f"rank{(chunk, jb, nw)}"))
    for nw in (1, 2, 4, 8, 16):
        fn = lambda: new._sort_kernel[(rows,)](*common, num_warps=nw)
        fn()
        res.append((do_bench(fn, warmup=10, rep=50, return_mode="median") * 1e3, f"sort(w{nw})"))
    res.sort()
    print(f"rows={rows:5d}: " + " ".join(f"{c}:{t:.1f}" for t, c in res[:4]), flush=True)
