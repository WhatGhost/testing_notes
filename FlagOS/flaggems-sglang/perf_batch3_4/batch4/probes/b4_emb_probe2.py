"""embedding_lora_a: tile choice over rank / vocab / tokens / adapters."""

import importlib
import itertools
import os
import sys
from types import SimpleNamespace

import torch
import triton
from triton.testing import do_bench

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

gen = importlib.import_module("flaggems_sglang.ops.embedding_lora_a")


def launch(ids, w, bi, vocab, bt, br, nw, st):
    tokens = ids.shape[0]
    rank = w.shape[1]
    out = torch.empty((tokens, rank), dtype=w.dtype, device=w.device)
    tb = triton.cdiv(bi.max_len, bt)
    rb = triton.cdiv(rank, br)
    gen._embedding_lora_a_kernel[(bi.bs * tb * rb,)](
        ids, w, bi.seg_indptr, bi.weight_indices, bi.lora_ranks, w, out, vocab,
        rank, bi.max_len, tb * rb, rb, 0, False, bt, br, num_warps=nw, num_stages=st)
    return out


CFGS = [(16, 16, 4, 2), (128, 64, 2, 1), (64, 64, 4, 1), (128, 32, 2, 1), (64, 32, 4, 1), (32, 64, 4, 1),
        (128, 128, 4, 1), (16, 8, 4, 2), (64, 16, 2, 1)]
for rank, vocab, segs, adapters in itertools.product((16, 32, 64, 128), (32000, 128256), (1, 4, 8), (1, 4)):
    for seg_len in (512, 2048):
        n = segs * seg_len
        g = torch.Generator(device="cuda").manual_seed(0)
        ids = torch.randint(0, vocab, (n,), device="cuda", generator=g, dtype=torch.int64)
        w = torch.randn(adapters, rank, vocab, device="cuda", generator=g).to(torch.bfloat16)
        seg = torch.arange(0, n + 1, seg_len, device="cuda", dtype=torch.int32)
        bi = SimpleNamespace(bs=segs, max_len=seg_len, seg_indptr=seg,
                             weight_indices=(torch.arange(segs, device="cuda", dtype=torch.int32) % adapters),
                             lora_ranks=torch.full((adapters,), rank, device="cuda", dtype=torch.int32))
        ref = gen.embedding_lora_a(ids, w, bi, vocab)
        t_cur = do_bench(lambda: gen.embedding_lora_a(ids, w, bi, vocab), warmup=25, rep=100, return_mode="median") * 1e3
        cells = []
        for c in CFGS:
            if c[1] > max(16, triton.next_power_of_2(rank)):
                continue
            o = launch(ids, w, bi, vocab, *c)
            assert torch.equal(o, ref), c
            t = do_bench(lambda: launch(ids, w, bi, vocab, *c), warmup=25, rep=100, return_mode="median") * 1e3
            cells.append((t, c))
        best = sorted(cells)[:2]
        d = {c: t for t, c in cells}
        print(f"r{rank:3d} V{vocab:6d} tok{n:6d} A{adapters} | cur {t_cur:6.1f} | (128,64,2,1) {d.get((128, 64, 2, 1), 0):6.1f}"
              f" (64,32,4,1) {d.get((64, 32, 4, 1), 0):6.1f} | best " + ", ".join(f"{c} {t:.1f}" for t, c in best),
              flush=True)
