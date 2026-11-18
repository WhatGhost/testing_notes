"""embedding_lora_a on gfx1100: the generic kernel with other tiles and warps."""

import importlib
import itertools
import os
import sys

import torch
import triton
from triton.testing import do_bench

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, B)
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

gen = importlib.import_module("flaggems_sglang.ops.embedding_lora_a")
tm = importlib.import_module("benchmark.test_embedding_lora_a")
wl = tm.workload


def launch(ids, w, bi, vocab, extra, bt, br, nw, st):
    tokens = ids.shape[0]
    rank = w.shape[1]
    out = torch.empty((tokens, rank), dtype=w.dtype, device=w.device)
    tb = triton.cdiv(bi.max_len, bt)
    rb = triton.cdiv(rank, br)
    ec = 0 if extra is None else extra.shape[1]
    gen._embedding_lora_a_kernel[(bi.bs * tb * rb,)](
        ids, w, bi.seg_indptr, bi.weight_indices, bi.lora_ranks, w if extra is None else extra, out, vocab,
        rank, bi.max_len, tb * rb, rb, ec, extra is not None, bt, br, num_warps=nw, num_stages=st)
    return out


for shape in (*wl.SHAPES, *wl.MORE_SHAPES):
    inp = wl.build_case(shape, torch.bfloat16, "cuda")
    args = [a for a in inp if not isinstance(a, dict)]
    kw = {}
    for a in inp:
        if isinstance(a, dict):
            kw.update(a)
    ref = gen.embedding_lora_a(*args, **kw)
    ids, w = args[0], args[1]
    bi, vocab, extra = kw.get("batch_info", None), kw.get("vocab_size", None), kw.get("extra_embeddings", None)
    t_cur = do_bench(lambda: gen.embedding_lora_a(*args, **kw), warmup=25, rep=200, return_mode="median") * 1e3
    res = []
    for bt, br, nw, st in itertools.product((16, 32, 64, 128), (8, 16, 32, 64), (2, 4, 8), (1, 2)):
        if br > max(8, triton.next_power_of_2(w.shape[1])):
            continue
        o = launch(ids, w, bi, vocab, extra, bt, br, nw, st)
        if not torch.equal(o, ref):
            print("MISMATCH", shape, bt, br, nw, st)
            continue
        t = do_bench(lambda: launch(ids, w, bi, vocab, extra, bt, br, nw, st), warmup=25, rep=200,
                     return_mode="median") * 1e3
        res.append((t, (bt, br, nw, st)))
    res.sort()
    print(f"{str(shape)[:44]:44s} | current {t_cur:6.1f} us | best (BT, BR, warps, stages): " +
          ", ".join(f"{c} {t:.1f}" for t, c in res[:4]), flush=True)
