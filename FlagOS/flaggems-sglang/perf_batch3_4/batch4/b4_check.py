"""Batch-4 rdna3 ops vs the generic ops (bitwise where the arithmetic is the
same) and vs float64 references."""

import importlib
import os
import sys

import torch

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

R = "flaggems_sglang.runtime.backend._amd.rdna3.ops."
G = "flaggems_sglang.ops."
fails = 0


def mod(p, name):
    return importlib.import_module(p + name)


def report(ok, what):
    global fails
    if not ok:
        fails += 1
        print("FAIL", what, flush=True)


# fused_softmax_pool: fp64 error of rdna3 vs generic
r, g = mod(R, "fused_softmax_pool"), mod(G, "fused_softmax_pool")
n = 0
for k in (1, 3, 4, 16, 64, 65, 100):
    for bs in (1, 3, 4, 5, 8, 63, 64, 100, 127, 128, 500, 1023, 1024, 2047, 2048, 5000, 16384):
        for hd in (128, 64):
            if hd == 64 and bs > 1024:
                continue
            for dt in (torch.bfloat16, torch.float16):
                x = torch.randn(bs, k, 2 * hd, device="cuda").to(dt) * 3
                exact = (x[..., :hd].double() * torch.softmax(x[..., hd:].double(), 1)).sum(1)
                yr, yg = r.fused_softmax_pool(x, hd), g.fused_softmax_pool(x, hd)
                er = (yr.double() - exact).abs().max().item()
                eg = (yg.double() - exact).abs().max().item()
                report(yr.dtype == yg.dtype and yr.shape == yg.shape and er <= max(1.5 * eg, 2e-2),
                       f"fsp k={k} bs={bs} hd={hd} {dt} err {er:.2e} vs {eg:.2e}")
                n += 1
x = torch.randn(4, 0, 256, device="cuda").to(torch.bfloat16)
report(torch.equal(r.fused_softmax_pool(x, 128), torch.zeros(4, 128, device="cuda", dtype=torch.bfloat16)),
       "fsp k=0")
x = torch.full((64, 64, 256), float("-inf"), device="cuda").to(torch.bfloat16)
x[..., :128] = 1
x[:, 5, 128:] = 0
yr = r.fused_softmax_pool(x, 128)
report(torch.allclose(yr.float(), torch.ones_like(yr.float())), "fsp -inf scores k=64 bs=64")
print("fused_softmax_pool cases", n, flush=True)

# per_group_transpose: bitwise vs generic
r, g = mod(R, "per_group_transpose"), mod(G, "per_group_transpose")
n = 0
gen = torch.Generator().manual_seed(0)
for dt in (torch.bfloat16, torch.float16, torch.float32):
    for k in (1, 7, 64, 100, 128, 200, 256, 257, 512, 4096):
        for sizes in ([256] * 8, [1024] * 8, [3, 0, 5, 2], [17, 33, 1, 49], [0] * 4, [2048, 0, 0, 300],
                      (torch.rand(8, generator=gen) * 1500).int().tolist(), [300] * 32, [255] * 8, [256] * 4,
                      [128] * 8, [127] * 8, [300, 0, 50, 250, 10, 290, 100, 200]):
            offs = torch.tensor([0] + torch.tensor(sizes).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
            a = torch.randn(sum(sizes), k, device="cuda").to(dt)
            yr, yg = r.per_group_transpose(a, offs), g.per_group_transpose(a, offs)
            report(torch.equal(yr, yg), f"pgt {dt} k={k} sizes={sizes[:6]}")
            n += 1
print("per_group_transpose cases", n, flush=True)

# ltx2_split_rotary: bitwise vs generic
r, g = mod(R, "ltx2_split_rotary"), mod(G, "ltx2_split_rotary")
n = 0
for b, t, h, hd in ((1, 1, 16, 128), (1, 129, 16, 128), (2, 512, 8, 128), (1, 2048, 16, 64), (3, 77, 24, 128),
                    (1, 300, 12, 64), (2, 4096, 16, 128), (1, 8192, 32, 128), (1, 5, 1, 128), (1, 33, 3, 96),
                    (1, 64, 128, 256), (1, 17, 16, 32), (4, 1000, 16, 128), (1, 3, 200, 128),
                    (1, 512, 16, 128), (1, 511, 16, 128), (1, 256, 16, 128), (2, 256, 16, 128)):
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        x = torch.randn(b, t, h * hd, device="cuda").to(dt)
        ang = torch.randn(b, h, t, hd // 2, device="cuda")
        c, s = torch.cos(ang).to(dt), torch.sin(ang).to(dt)
        yr, yg = r.ltx2_split_rotary(x, c, s), g.ltx2_split_rotary(x, c, s)
        report(torch.equal(yr, yg), f"ltx2 {(b, t, h, hd)} {dt}")
        n += 1
print("ltx2_split_rotary cases", n, flush=True)

# dsv3_fused_a_gemm: fp64 error of rdna3 vs generic
r, g = mod(R, "dsv3_fused_a_gemm"), mod(G, "dsv3_fused_a_gemm")
n = 0
for k, nn in ((7168, 2112), (4096, 1536), (4097, 1536), (5000, 2112), (7200, 2113), (8192, 576), (2048, 2112),
              (7168, 100)):
    for m in (1, 2, 3, 4, 5, 8, 15, 16):
        for layout in ("col", "row", "a_strided"):
            if layout != "col" and m not in (4, 16):
                continue
            a = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            w = torch.randn(nn, k, device="cuda").to(torch.bfloat16)
            bmat = w.t()
            if layout == "row":
                bmat = w.t().contiguous()
            if layout == "a_strided":
                a = torch.randn(m, 2 * k, device="cuda").to(torch.bfloat16)[:, ::2]
            exact = a.double() @ bmat.double()
            yr, yg = r.dsv3_fused_a_gemm(a, bmat), g.dsv3_fused_a_gemm(a, bmat)
            er = ((yr.double() - exact).abs() / (1 + exact.abs())).max().item()
            eg = ((yg.double() - exact).abs() / (1 + exact.abs())).max().item()
            report(yr.dtype == yg.dtype and er <= max(1.5 * eg, 1e-2),
                   f"dsv3 K={k} N={nn} M={m} {layout} err {er:.2e} vs {eg:.2e}")
            n += 1
for m in (17, 32):
    a = torch.randn(m, 7168, device="cuda").to(torch.bfloat16)
    w = torch.randn(2112, 7168, device="cuda").to(torch.bfloat16)
    exact = a.double() @ w.t().double()
    yr = r.dsv3_fused_a_gemm(a, w.t())
    er = ((yr.double() - exact).abs() / (1 + exact.abs())).max().item()
    report(er < 1e-2, f"dsv3 M={m} split-K err {er:.2e}")
    n += 1
print("dsv3_fused_a_gemm cases", n, flush=True)

# apply_token_bitmask: bitwise vs generic
r, g = mod(R, "apply_token_bitmask"), mod(G, "apply_token_bitmask")
n = 0
for b, v in ((1, 152064), (8, 152064), (14, 152064), (16, 152064), (64, 152064), (3, 100), (7, 1000), (65, 32001),
             (64, 32000), (1, 1)):
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        logits = torch.randn(b, v, device="cuda").to(dt)
        words = (v + 31) // 32 + (1 if b == 3 else 0)
        bitmask = torch.randint(-2**31, 2**31 - 1, (b, words), dtype=torch.int32, device="cuda")
        report(torch.equal(r.apply_token_bitmask(logits, bitmask), g.apply_token_bitmask(logits, bitmask)),
               f"atb {(b, v)} {dt}")
        n += 1
print("apply_token_bitmask cases", n, flush=True)

# embedding_lora_a: bitwise vs generic
from types import SimpleNamespace  # noqa: E402

r, g = mod(R, "embedding_lora_a"), mod(G, "embedding_lora_a")
n = 0
for rank, vocab, seg_lens, adapters, extra, ranks in (
        (64, 128256, (2048,) * 4, 2, 0, None), (64, 32000, (3000, 0, 17, 1), 2, 0, None),
        (128, 32000, (4096,), 1, 0, (100,)), (96, 1000, (1500, 600), 2, 8, (96, 40)),
        (16, 32000, (512,) * 8, 4, 0, None), (256, 5000, (2048, 5), 3, 4, (256, 200, 64)),
        (64, 256, (9, 4), 2, 8, None), (64, 50000, (2048,), 1, 16, (64,))):
    n_tok = sum(seg_lens)
    ids = torch.randint(0, vocab + extra, (n_tok,), device="cuda")
    w = torch.randn(adapters, rank, vocab, device="cuda").to(torch.bfloat16)
    seg = torch.tensor([0] + torch.tensor(seg_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
    bi = SimpleNamespace(bs=len(seg_lens), max_len=max(seg_lens), seg_indptr=seg,
                         weight_indices=torch.arange(len(seg_lens), device="cuda", dtype=torch.int32) % adapters,
                         lora_ranks=torch.tensor(ranks or [rank] * adapters, device="cuda", dtype=torch.int32))
    ex = torch.randn(adapters, extra, rank, device="cuda").to(torch.bfloat16) if extra else None
    yr = r.embedding_lora_a(ids, w, bi, vocab, ex)
    yg = g.embedding_lora_a(ids, w, bi, vocab, ex)
    report(torch.equal(yr, yg), f"emb rank={rank} vocab={vocab} segs={seg_lens} extra={extra}")
    n += 1
print("embedding_lora_a cases", n, flush=True)
print("FAILS", fails)
