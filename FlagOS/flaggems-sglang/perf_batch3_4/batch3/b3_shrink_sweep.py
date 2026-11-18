"""Sweep split/tile configs of the rdna3 chunked_sgmv_shrink draft against the
generic op (gfx1100, do_bench median)."""

import importlib
import itertools
import json
import os
import sys

import torch
from triton.testing import do_bench

B = open("/tmp/fgs-b3.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
sys.path.insert(0, os.path.join(B, "baseline"))
import flaggems_sglang  # noqa: E402,F401

gen = importlib.import_module("flaggems_sglang.ops.chunked_sgmv_shrink")
new = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.chunked_sgmv_shrink")
bl = importlib.import_module("chunked_sgmv_shrink")
from __lora_batch_utils import make_batch_info  # noqa: E402

# (seg_len, num_segs, num_lora, r, k, num_slices)
SHAPES = list(bl.SHAPES) + list(bl.MORE_SHAPES) + [
    (1, 32, 4, 16, 4096, 3), (1, 8, 2, 64, 5120, 1),
    (512, 8, 4, 16, 4096, 3), (2048, 4, 2, 64, 5120, 1),
    (1, 1, 1, 64, 2048, 3), (16, 8, 4, 64, 2048, 3), (128, 16, 4, 16, 2048, 1),
    (2048, 1, 1, 32, 4096, 1), (32, 4, 2, 64, 5120, 3),
]


def case(shape, perm="identity"):
    seg_len, num_segs, num_lora, r, k, ns = shape
    g = torch.Generator(device="cuda").manual_seed(0)
    seg_lens = [seg_len] * num_segs
    x = torch.randn(sum(seg_lens), k, generator=g, device="cuda").to(torch.bfloat16)
    w = torch.randn(num_lora, ns * r, k, generator=g, device="cuda").to(torch.bfloat16)
    bi = make_batch_info(seg_lens, [i % num_lora for i in range(num_segs)], device="cuda",
                         lora_ranks=[r] * num_lora, permutation=perm)
    return x, w, bi, ns


phase = sys.argv[1] if len(sys.argv) > 1 else "tiles"
if phase == "rule":
    cfgs = None
elif phase == "tiles":
    cfgs = [(s, bm, bn, bk, 4, 2) for s, (bm, bn), bk in itertools.product(
        (2, 4, 6, 8, 10, 12, 16), ((16, 32), (16, 64), (32, 32), (32, 64), (64, 32), (64, 64)), (64, 128))]
else:
    base = json.load(open("/tmp/b3_shrink_top.json"))
    cfgs = sorted({(s, bm, bn, bk, w, st) for (s, bm, bn, bk, _, _) in base
                   for w in (2, 4, 8) for st in (1, 2)})
orig = new._split_config
out = {}
for shape in SHAPES:
    x, w, bi, ns = case(shape)
    ref = gen.chunked_sgmv_shrink(x, w, bi, ns)
    t_gen = do_bench(lambda: gen.chunked_sgmv_shrink(x, w, bi, ns), warmup=25, rep=100, return_mode="median") * 1e3
    row = {"generic": t_gen}
    shape_cfgs = cfgs
    if cfgs is None:
        rb, bm, bn, bk, _, _ = orig(x.shape[0], max(1, bi.max_len), bi.bs, x.shape[1], w.shape[1])
        shape_cfgs = [("rule",)] + [(b2, bm, bn, bk, 4, 1) for b2 in (3, 5, 7, 11, 20) if b2 != rb]
    for cfg in shape_cfgs:
        new._split_config = orig if cfg == ("rule",) else (lambda *a, cfg=cfg: cfg)
        try:
            o = new.chunked_sgmv_shrink(x, w, bi, ns)
            err = ((o.float() - ref.float()).abs() / (1 + ref.float().abs())).max().item()
            if not err < 2e-2:
                print("BAD", shape, cfg, err)
                continue
            row[str(cfg)] = do_bench(lambda: new.chunked_sgmv_shrink(x, w, bi, ns),
                                     warmup=25, rep=100, return_mode="median") * 1e3
        except Exception as e:  # noqa: BLE001
            print("ERR", shape, cfg, str(e)[:120])
    new._split_config = orig
    out[str(shape)] = row
    best = sorted((v, k) for k, v in row.items() if k != "generic")[:4]
    if cfgs is None:
        print(f"   rule {row.get(str(('rule',)), float('nan')):.1f} us (generic/rule {t_gen / row.get(str(('rule',)), 1):.2f}x)", end="")
    print(f"{shape} x{tuple(x.shape)} N={w.shape[1]} | generic {t_gen:.1f} | " +
          ", ".join(f"{k} {v:.1f}" for v, k in best), flush=True)
json.dump(out, open(f"/tmp/b3_shrink_{phase}.json", "w"), indent=1)
