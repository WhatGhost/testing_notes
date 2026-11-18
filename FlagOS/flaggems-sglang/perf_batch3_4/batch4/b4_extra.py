"""Batch-4 final rdna3 ops vs upstream master on additional shapes (do_bench
medians, same process, alternating order)."""

import importlib
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import torch
from triton.testing import do_bench

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

R = "flaggems_sglang.runtime.backend._amd.rdna3.ops."
G = "flaggems_sglang.ops."
only = sys.argv[1:]


def load_master_pgt():
    path = "/tmp/fgs-base-check.35O2N6/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/per_group_transpose.py"
    spec = importlib.util.spec_from_file_location("master_rdna3_pgt", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bench(fn):
    return do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3


def pair(new, old):
    t = []
    for _ in range(2):
        t.append((bench(old), bench(new)))
    return min(x[0] for x in t), min(x[1] for x in t)


out = []


def record(op, shape, new, old, check):
    check()
    tm, tn = pair(new, old)
    out.append(dict(op=op, shape=shape, master=tm, new=tn))
    print(f"{op:22s} {shape:48s} master {tm:8.1f} new {tn:8.1f} {tm / tn:5.2f}x", flush=True)


if not only or "fused_softmax_pool" in only:
    r, g = importlib.import_module(R + "fused_softmax_pool"), importlib.import_module(G + "fused_softmax_pool")
    for bs, k in ((8, 64), (16, 64), (32, 64), (128, 64), (256, 64), (512, 64), (2048, 16), (8192, 16),
                  (16384, 16)):
        x = torch.randn(bs, k, 256, device="cuda").to(torch.bfloat16)

        def chk():
            assert (r.fused_softmax_pool(x, 128).float() - g.fused_softmax_pool(x, 128).float()).abs().max() < 2e-2

        record("fused_softmax_pool", f"bs={bs} k={k} d=128", lambda: r.fused_softmax_pool(x, 128),
               lambda: g.fused_softmax_pool(x, 128), chk)

if not only or "per_group_transpose" in only:
    r, m = importlib.import_module(R + "per_group_transpose"), load_master_pgt()
    gen = torch.Generator().manual_seed(0)
    cases = [("bf16", 8, 128, [256] * 8), ("bf16", 8, 128, [512] * 8), ("bf16", 8, 128, [2048] * 8),
             ("bf16", 8, 256, [512] * 8), ("bf16", 8, 256, [1024] * 8), ("bf16", 8, 256, [2048] * 8),
             ("bf16", 8, 128, (torch.rand(8, generator=gen) * 512).int().tolist()),
             ("bf16", 8, 128, (torch.rand(8, generator=gen) * 2048).int().tolist()),
             ("bf16", 32, 128, [256] * 32), ("bf16", 8, 128, [128] * 8), ("bf16", 8, 256, [128] * 8),
             ("fp32", 8, 128, [128] * 8), ("fp32", 8, 128, [1024] * 8), ("fp32", 8, 128, [2048] * 8)]
    for dt, e, k, sizes in cases:
        dtype = torch.bfloat16 if dt == "bf16" else torch.float32
        a = torch.randn(sum(sizes), k, device="cuda").to(dtype)
        offs = torch.tensor([0] + torch.tensor(sizes).cumsum(0).tolist(), dtype=torch.int32, device="cuda")

        def chk():
            assert torch.equal(r.per_group_transpose(a, offs), m.per_group_transpose(a, offs))

        avg = sum(sizes) // e
        tag = "" if len(set(sizes)) == 1 else " (uneven)"
        record("per_group_transpose", f"{dt} E={e} k={k} rows/group~{avg}{tag}",
               lambda: r.per_group_transpose(a, offs), lambda: m.per_group_transpose(a, offs), chk)

if not only or "ltx2_split_rotary" in only:
    r, g = importlib.import_module(R + "ltx2_split_rotary"), importlib.import_module(G + "ltx2_split_rotary")
    for b, t, h, hd in ((1, 256, 16, 128), (1, 512, 16, 128), (1, 4096, 16, 128), (1, 16384, 16, 128),
                        (1, 8192, 32, 128), (4, 1024, 16, 128), (2, 512, 8, 128), (1, 2048, 16, 64),
                        (1, 4096, 24, 128)):
        x = torch.randn(b, t, h * hd, device="cuda").to(torch.bfloat16)
        ang = torch.randn(b, h, t, hd // 2, device="cuda")
        c, s = torch.cos(ang).to(torch.bfloat16), torch.sin(ang).to(torch.bfloat16)

        def chk():
            assert torch.equal(r.ltx2_split_rotary(x, c, s), g.ltx2_split_rotary(x, c, s))

        record("ltx2_split_rotary", f"b={b} t={t} heads={h} head_dim={hd}", lambda: r.ltx2_split_rotary(x, c, s),
               lambda: g.ltx2_split_rotary(x, c, s), chk)

if not only or "dsv3_fused_a_gemm" in only:
    r, g = importlib.import_module(R + "dsv3_fused_a_gemm"), importlib.import_module(G + "dsv3_fused_a_gemm")
    for k, n, mm in ((7168, 2112, 3), (7168, 1536, 4), (7168, 1536, 16), (7168, 576, 4), (7168, 576, 16),
                     (7168, 3072, 4), (7168, 3072, 16), (5120, 2112, 8), (8192, 2112, 8)):
        a = torch.randn(mm, k, device="cuda").to(torch.bfloat16)
        w = torch.randn(n, k, device="cuda").to(torch.bfloat16)
        bm = w.t()

        def chk():
            exact = a.double() @ bm.double()
            assert ((r.dsv3_fused_a_gemm(a, bm).double() - exact).abs() / (1 + exact.abs())).max() < 1e-2

        record("dsv3_fused_a_gemm", f"M={mm} K={k} N={n}", lambda: r.dsv3_fused_a_gemm(a, bm),
               lambda: g.dsv3_fused_a_gemm(a, bm), chk)

if not only or "embedding_lora_a" in only:
    r, g = importlib.import_module(R + "embedding_lora_a"), importlib.import_module(G + "embedding_lora_a")
    for rank, vocab, segs, seg_len, adapters in ((64, 32000, 4, 2048, 1), (64, 32000, 8, 2048, 1),
                                                  (128, 32000, 1, 4096, 1), (128, 32000, 8, 2048, 1),
                                                  (64, 128256, 8, 2048, 1), (128, 128256, 4, 2048, 4),
                                                  (64, 128256, 1, 2048, 1)):
        n_tok = segs * seg_len
        ids = torch.randint(0, vocab, (n_tok,), device="cuda")
        wt = torch.randn(adapters, rank, vocab, device="cuda").to(torch.bfloat16)
        seg = torch.arange(0, n_tok + 1, seg_len, device="cuda", dtype=torch.int32)
        bi = SimpleNamespace(bs=segs, max_len=seg_len, seg_indptr=seg,
                             weight_indices=torch.arange(segs, device="cuda", dtype=torch.int32) % adapters,
                             lora_ranks=torch.full((adapters,), rank, device="cuda", dtype=torch.int32))

        def chk():
            assert torch.equal(r.embedding_lora_a(ids, wt, bi, vocab), g.embedding_lora_a(ids, wt, bi, vocab))

        record("embedding_lora_a", f"rank={rank} vocab={vocab} tokens={segs}x{seg_len} adapters={adapters}",
               lambda: r.embedding_lora_a(ids, wt, bi, vocab), lambda: g.embedding_lora_a(ids, wt, bi, vocab), chk)

if not only or "apply_token_bitmask" in only:
    r, g = importlib.import_module(R + "apply_token_bitmask"), importlib.import_module(G + "apply_token_bitmask")
    for b, v in ((10, 152064), (16, 152064), (32, 152064), (256, 152064), (48, 32000), (64, 32000), (64, 128256)):
        logits = torch.randn(b, v, device="cuda")
        bitmask = torch.randint(-2**31, 2**31 - 1, (b, (v + 31) // 32), dtype=torch.int32, device="cuda")

        def chk():
            assert torch.equal(r.apply_token_bitmask(logits, bitmask), g.apply_token_bitmask(logits, bitmask))

        record("apply_token_bitmask", f"B={b} V={v} fp32", lambda: r.apply_token_bitmask(logits, bitmask),
               lambda: g.apply_token_bitmask(logits, bitmask), chk)

json.dump(out, open(f"/tmp/b4_extra_{'_'.join(only) or 'all'}.json", "w"), indent=1)
