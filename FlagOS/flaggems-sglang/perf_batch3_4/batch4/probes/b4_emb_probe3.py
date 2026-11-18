"""embedding_lora_a: candidate tiles over rank / vocab / tokens / adapters -> JSON."""

import importlib
import itertools
import json
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


rows = []
for rank, vocab, segs, adapters, seg_len in itertools.product((16, 32, 64, 128), (32000, 128256), (1, 4, 8),
                                                               (1, 2, 4), (64, 512, 2048)):
    if adapters > segs:
        continue
    n = segs * seg_len
    full = min(128, max(16, triton.next_power_of_2(rank)))
    cfgs = sorted({(16, 16, 4, 2), (64, full, 4, 1), (128, full, 2, 1), (128, full, 4, 1), (64, 32, 4, 1),
                   (128, 32, 2, 1), (64, 16, 2, 1), (32, full, 4, 1), (64, full, 2, 1)})
    g = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(0, vocab, (n,), device="cuda", generator=g, dtype=torch.int64)
    w = torch.randn(adapters, rank, vocab, device="cuda", generator=g).to(torch.bfloat16)
    seg = torch.arange(0, n + 1, seg_len, device="cuda", dtype=torch.int32)
    bi = SimpleNamespace(bs=segs, max_len=seg_len, seg_indptr=seg,
                         weight_indices=(torch.arange(segs, device="cuda", dtype=torch.int32) % adapters),
                         lora_ranks=torch.full((adapters,), rank, device="cuda", dtype=torch.int32))
    ref = gen.embedding_lora_a(ids, w, bi, vocab)
    times = {}
    for c in cfgs:
        o = launch(ids, w, bi, vocab, *c)
        assert torch.equal(o, ref), c
        times[str(c)] = do_bench(lambda: launch(ids, w, bi, vocab, *c), warmup=25, rep=100,
                                 return_mode="median") * 1e3
    rows.append(dict(rank=rank, vocab=vocab, segs=segs, adapters=adapters, seg_len=seg_len, times=times))
    best = min(times, key=times.get)
    print(rank, vocab, segs, adapters, seg_len, "cur %.1f" % times[str((16, 16, 4, 2))], "best", best,
          "%.1f" % times[best], flush=True)
json.dump(rows, open("/tmp/b4_emb_probe3.json", "w"))
