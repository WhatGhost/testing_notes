import os, sys, importlib, torch, triton
from triton.testing import do_bench
sys.path.insert(0, "/tmp")
B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa
import b4_atb_lib as L
gen = importlib.import_module("flaggems_sglang.ops.apply_token_bitmask")
def flat(logits, bitmask, block, nw):
    b, v = logits.shape
    out = torch.empty_like(logits)
    L._atb[(b, triton.cdiv(v, block))](logits, bitmask, out, v, bitmask.shape[1], BLOCK=block, num_warps=nw)
    return out
for b, v in ((8, 152064), (10, 152064), (12, 152064), (14, 152064), (40, 32000), (48, 32000), (56, 32000), (64, 32000), (12, 128256), (16, 128256)):
    g = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(b, v, generator=g, device="cuda")
    bitmask = torch.randint(-2**31, 2**31 - 1, (b, (v + 31) // 32), dtype=torch.int32, device="cuda", generator=g)
    t_cur = do_bench(lambda: gen.apply_token_bitmask(logits, bitmask), warmup=10, rep=200, return_mode="median") * 1e3
    cells = []
    for blk, nw in ((8192, 4), (4096, 4), (2048, 4)):
        cells.append("%d/%d %.1f" % (blk, nw, do_bench(lambda: flat(logits, bitmask, blk, nw), warmup=10, rep=200, return_mode="median") * 1e3))
    print((b, v), "cur %.1f" % t_cur, " | ".join(cells), flush=True)
