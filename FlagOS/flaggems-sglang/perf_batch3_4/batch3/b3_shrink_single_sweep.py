"""One-kernel path vs the split path of the rdna3 chunked_sgmv_shrink, per
shape and config (gfx1100, do_bench median)."""

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
from __lora_batch_utils import make_batch_info  # noqa: E402

SHAPES = [(64, 8, 4, 32, 4096, 1), (256, 4, 2, 64, 4096, 1), (8, 1, 1, 16, 64, 1), (16, 4, 2, 16, 128, 1),
          (16, 2, 2, 32, 256, 3), (1, 32, 4, 16, 4096, 3), (1, 8, 2, 64, 5120, 1), (1, 1, 1, 64, 2048, 3),
          (16, 8, 4, 64, 2048, 3), (128, 16, 4, 16, 2048, 1), (32, 4, 2, 64, 5120, 3), (1, 4, 2, 16, 4096, 1),
          (2, 16, 4, 32, 4096, 1), (4, 64, 4, 16, 2048, 3), (8, 8, 2, 64, 4096, 1)]
CFGS = list(itertools.product((16, 32), (32, 64), (64, 128, 256), (2, 4, 8), (1, 2)))


def case(shape, perm="identity"):
    seg_len, num_segs, num_lora, r, k, ns = shape
    g = torch.Generator(device="cuda").manual_seed(0)
    seg_lens = [seg_len] * num_segs
    x = torch.randn(sum(seg_lens), k, generator=g, device="cuda").to(torch.bfloat16)
    w = torch.randn(num_lora, ns * r, k, generator=g, device="cuda").to(torch.bfloat16)
    bi = make_batch_info(seg_lens, [i % num_lora for i in range(num_segs)], device="cuda",
                         lora_ranks=[r] * num_lora, permutation=perm)
    return x, w, bi, ns


orig = new._single_config
out = {}
for shape in SHAPES:
    x, w, bi, ns = case(shape)
    ref = gen.chunked_sgmv_shrink(x, w, bi, ns)
    t_gen = do_bench(lambda: gen.chunked_sgmv_shrink(x, w, bi, ns), warmup=25, rep=100, return_mode="median") * 1e3
    new._single_config = lambda *a: None
    t_split = do_bench(lambda: new.chunked_sgmv_shrink(x, w, bi, ns), warmup=25, rep=100, return_mode="median") * 1e3
    row = {"generic": t_gen, "split": t_split}
    for cfg in CFGS:
        if cfg[0] > max(16, x.shape[0]) * 2:
            continue
        new._single_config = lambda *a, cfg=cfg: cfg
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
    new._single_config = orig
    out[str(shape)] = row
    best = sorted((v, k) for k, v in row.items() if k not in ("generic", "split"))[:3]
    tiles = bi.bs * -(-max(1, bi.max_len) // 16)
    print(f"{shape} x{tuple(x.shape)} N={w.shape[1]} tiles16={tiles} | generic {t_gen:.1f} split {t_split:.1f} | single best: "
          + ", ".join(f"{k} {v:.1f}" for v, k in best), flush=True)
json.dump(out, open("/tmp/b3_shrink_single.json", "w"), indent=1)
