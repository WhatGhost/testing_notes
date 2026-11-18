"""Batch-4 ops: SGLang implementation vs upstream master vs this change, same
process, do_bench medians. SGLang kernels are the upstream sources extracted
into /tmp/sglang_new/standalone (dsv3_fused_a_gemm: SGLang uses F.linear on
ROCm)."""

import importlib
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F
from triton.testing import do_bench

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, B)
sys.path.insert(0, os.path.join(B, "src"))
sys.path.insert(0, "/tmp/sglang_new/standalone")
import flaggems_sglang  # noqa: E402,F401

R = "flaggems_sglang.runtime.backend._amd.rdna3.ops."
G = "flaggems_sglang.ops."
only = sys.argv[1:]
rows = []


def bench(fn):
    return do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3


def measure(op, kind, label, sgl, master, new, check):
    check()
    t = {"sgl": [], "master": [], "new": []}
    for order in (("sgl", "master", "new"), ("new", "master", "sgl")):
        for name in order:
            t[name].append(bench({"sgl": sgl, "master": master, "new": new}[name]))
    r = {k: min(v) for k, v in t.items()}
    rows.append(dict(op=op, kind=kind, shape=label, **r))
    print(f"{op:20s} {kind:8s} {label:46s} sglang {r['sgl']:8.1f}  master {r['master']:8.1f}  new {r['new']:8.1f}"
          f" | vs sglang {r['sgl'] / r['new']:5.2f}x  vs master {r['master'] / r['new']:5.2f}x", flush=True)


def load_master_pgt():
    path = "/tmp/fgs-base-check.35O2N6/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/per_group_transpose.py"
    spec = importlib.util.spec_from_file_location("master_rdna3_pgt", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


if not only or "fused_softmax_pool" in only:
    import sgl_fsp

    wl = importlib.import_module("baseline.fused_softmax_pool")
    r, g = importlib.import_module(R + "fused_softmax_pool"), importlib.import_module(G + "fused_softmax_pool")
    cases = [("official", s) for s in (*wl.SHAPES, *wl.MORE_SHAPES)]
    cases += [("extra", (bs, k, 128)) for bs, k in ((8, 64), (16, 64), (32, 64), (128, 64), (256, 64), (512, 64),
                                                    (2048, 16), (8192, 16), (16384, 16))]
    for kind, shape in cases:
        kv, hd = wl.build_case(shape, torch.bfloat16, "cuda")

        def chk():
            ys, yn = sgl_fsp.fused_softmax_pool_triton(kv, hd), r.fused_softmax_pool(kv, hd)
            assert (ys.float() - yn.float()).abs().max() < 2e-2

        measure("fused_softmax_pool", kind, f"rows={shape[0]} k={shape[1]}",
                lambda: sgl_fsp.fused_softmax_pool_triton(kv, hd), lambda: g.fused_softmax_pool(kv, hd),
                lambda: r.fused_softmax_pool(kv, hd), chk)

if not only or "per_group_transpose" in only:
    import sgl_pgt

    wl = importlib.import_module("baseline.per_group_transpose")
    r, m = importlib.import_module(R + "per_group_transpose"), load_master_pgt()
    cases = []
    for k, n in wl.SHAPES:
        a, offs = wl.build_case((k, n), torch.bfloat16, "cuda")
        cases.append(("official", f"bf16 groups=8 k={k} rows/group={n}", a, offs))
    gen = torch.Generator().manual_seed(0)
    extra = [("bf16", 8, 128, [256] * 8), ("bf16", 8, 128, [512] * 8), ("bf16", 8, 128, [2048] * 8),
             ("bf16", 8, 256, [512] * 8), ("bf16", 8, 256, [1024] * 8), ("bf16", 8, 256, [2048] * 8),
             ("bf16", 8, 128, (torch.rand(8, generator=gen) * 512).int().tolist()),
             ("bf16", 8, 128, (torch.rand(8, generator=gen) * 2048).int().tolist()),
             ("bf16", 32, 128, [256] * 32), ("bf16", 8, 128, [128] * 8), ("bf16", 8, 256, [128] * 8),
             ("fp32", 8, 128, [128] * 8), ("fp32", 8, 128, [1024] * 8), ("fp32", 8, 128, [2048] * 8)]
    for dt, e, k, sizes in extra:
        dtype = torch.bfloat16 if dt == "bf16" else torch.float32
        a = torch.randn(sum(sizes), k, device="cuda").to(dtype)
        offs = torch.tensor([0] + torch.tensor(sizes).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        tag = "" if len(set(sizes)) == 1 else " uneven"
        cases.append(("extra", f"{dt} groups={e} k={k} rows/group~{sum(sizes) // e}{tag}", a, offs))
    for kind, label, a, offs in cases:

        def chk():
            assert torch.equal(sgl_pgt.per_group_transpose(a, offs, 1), r.per_group_transpose(a, offs))

        measure("per_group_transpose", kind, label, lambda: sgl_pgt.per_group_transpose(a, offs, 1),
                lambda: m.per_group_transpose(a, offs), lambda: r.per_group_transpose(a, offs), chk)

if not only or "ltx2_split_rotary" in only:
    import sgl_ltx2

    r, g = importlib.import_module(R + "ltx2_split_rotary"), importlib.import_module(G + "ltx2_split_rotary")
    cases = [("official", (1, t, 16, 128)) for t in (1, 64, 1024, 8192)]
    cases += [("extra", s) for s in ((1, 256, 16, 128), (1, 512, 16, 128), (1, 4096, 16, 128), (1, 16384, 16, 128),
                                     (1, 8192, 32, 128), (4, 1024, 16, 128), (2, 512, 8, 128), (1, 2048, 16, 64),
                                     (1, 4096, 24, 128))]
    for kind, (b, t, h, hd) in cases:
        gq = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(b, t, h * hd, device="cuda", generator=gq).to(torch.bfloat16)
        ang = torch.randn(b, h, t, hd // 2, device="cuda", generator=gq)
        c, s = torch.cos(ang).to(torch.bfloat16), torch.sin(ang).to(torch.bfloat16)

        def chk():
            ys, yn = sgl_ltx2.apply_ltx2_split_rotary_emb(x, c, s), r.ltx2_split_rotary(x, c, s)
            assert (ys.float() - yn.float()).abs().max() <= 0.05

        measure("ltx2_split_rotary", kind, f"batch={b} tokens={t} heads={h} head_dim={hd}",
                lambda: sgl_ltx2.apply_ltx2_split_rotary_emb(x, c, s), lambda: g.ltx2_split_rotary(x, c, s),
                lambda: r.ltx2_split_rotary(x, c, s), chk)

if not only or "dsv3_fused_a_gemm" in only:
    r, g = importlib.import_module(R + "dsv3_fused_a_gemm"), importlib.import_module(G + "dsv3_fused_a_gemm")
    cases = [("official", (mm, 7168, 2112)) for mm in (1, 2, 4, 8, 16)]
    cases += [("extra", s) for s in ((3, 7168, 2112), (4, 7168, 1536), (16, 7168, 1536), (4, 7168, 576),
                                     (16, 7168, 576), (4, 7168, 3072), (16, 7168, 3072), (8, 5120, 2112),
                                     (8, 8192, 2112))]
    for kind, (mm, k, n) in cases:
        gq = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(mm, k, device="cuda", generator=gq).to(torch.bfloat16)
        w = torch.randn(n, k, device="cuda", generator=gq).to(torch.bfloat16)
        bm = w.t()

        def chk():
            exact = a.double() @ bm.double()
            for y in (F.linear(a, w), r.dsv3_fused_a_gemm(a, bm)):
                assert ((y.double() - exact).abs() / (1 + exact.abs())).max() < 1e-2

        measure("dsv3_fused_a_gemm", kind, f"M={mm} K={k} N={n}", lambda: F.linear(a, w),
                lambda: g.dsv3_fused_a_gemm(a, bm), lambda: r.dsv3_fused_a_gemm(a, bm), chk)

if not only or "embedding_lora_a" in only:
    import sgl_emb

    wl = importlib.import_module("baseline.embedding_lora_a")
    mbi = importlib.import_module("baseline.__lora_batch_utils").make_batch_info
    r, g = importlib.import_module(R + "embedding_lora_a"), importlib.import_module(G + "embedding_lora_a")
    cases = []
    for shape in (*wl.SHAPES, *wl.MORE_SHAPES):
        ids, wt, kw = wl.build_case(shape, torch.bfloat16, "cuda")
        seg_lens, num_lora, rank, vocab, extra = shape
        label = f"segs={len(seg_lens)} tokens={sum(seg_lens)} adapters={num_lora} rank={rank} vocab={vocab}"
        cases.append(("official", label + (f" extra={extra}" if extra else ""), ids, wt, kw))
    for rank, vocab, segs, seg_len, adapters in ((64, 32000, 4, 2048, 1), (64, 32000, 8, 2048, 1),
                                                  (128, 32000, 1, 4096, 1), (128, 32000, 8, 2048, 1),
                                                  (64, 128256, 8, 2048, 1), (128, 128256, 4, 2048, 4),
                                                  (64, 128256, 1, 2048, 1)):
        n_tok = segs * seg_len
        ids = torch.randint(0, vocab, (n_tok,), device="cuda")
        wt = torch.randn(adapters, rank, vocab, device="cuda").to(torch.bfloat16)
        bi = mbi([seg_len] * segs, [i % adapters for i in range(segs)], device="cuda", lora_ranks=[rank] * adapters)
        kw = dict(batch_info=bi, vocab_size=vocab, extra_embeddings=None)
        cases.append(("extra", f"segs={segs} tokens={n_tok} adapters={adapters} rank={rank} vocab={vocab}", ids, wt,
                      kw))
    for kind, label, ids, wt, kw in cases:

        def chk():
            ys = sgl_emb.embedding_lora_a_fwd(ids, wt, kw["batch_info"], kw["vocab_size"], kw["extra_embeddings"])
            assert torch.equal(ys, r.embedding_lora_a(ids, wt, **kw))

        measure("embedding_lora_a", kind, label,
                lambda: sgl_emb.embedding_lora_a_fwd(ids, wt, kw["batch_info"], kw["vocab_size"],
                                                     kw["extra_embeddings"]),
                lambda: g.embedding_lora_a(ids, wt, **kw), lambda: r.embedding_lora_a(ids, wt, **kw), chk)

if not only or "apply_token_bitmask" in only:
    import sgl_bitmask

    wl = importlib.import_module("baseline.apply_token_bitmask")
    r, g = importlib.import_module(R + "apply_token_bitmask"), importlib.import_module(G + "apply_token_bitmask")
    cases = [("official", s) for s in (*wl.SHAPES, *wl.MORE_SHAPES)]
    cases += [("extra", s) for s in ((10, 152064), (16, 152064), (32, 152064), (256, 152064), (48, 32000),
                                     (64, 32000), (64, 128256))]
    for kind, shape in cases:
        logits, bitmask = wl.build_case(shape, torch.float32, "cuda")
        sgl_logits = logits.clone()

        def chk():
            ref = r.apply_token_bitmask(logits, bitmask)
            tmp = logits.clone()
            sgl_bitmask.apply_token_bitmask_inplace_triton(tmp, bitmask)
            assert torch.equal(tmp, ref)

        measure("apply_token_bitmask", kind, f"batch={shape[0]} vocab={shape[1]} fp32",
                lambda: sgl_bitmask.apply_token_bitmask_inplace_triton(sgl_logits, bitmask),
                lambda: g.apply_token_bitmask(logits, bitmask), lambda: r.apply_token_bitmask(logits, bitmask), chk)

json.dump(rows, open(f"/tmp/b4_vs_sglang_{'_'.join(only) or 'all'}.json", "w"), indent=1)
