"""per_group_transpose on gfx1100: split each group's rows over S programs
(grid-stride inside the group) vs the current rdna3 launch."""

import importlib
import itertools
import os
import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401

cur = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.per_group_transpose")


@triton.jit
def _pgt_split(input_ptr, output_ptr, expert_offsets_ptr, K: tl.constexpr, S: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    expert = tl.program_id(0)
    k_block = tl.program_id(1)
    part = tl.program_id(2)
    group_start = tl.load(expert_offsets_ptr + expert)
    group_rows = tl.load(expert_offsets_ptr + expert + 1) - group_start
    group_base = group_start.to(tl.int64) * K
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    for row_start in tl.range(part * BLOCK_M, group_rows, S * BLOCK_M):
        row_offsets = row_start + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < group_rows
        values = tl.load(input_ptr + group_base + row_offsets[:, None] * K + k_offsets[None, :],
                         mask=row_mask[:, None] & k_mask[None, :], other=0)
        tl.store(output_ptr + group_base + k_offsets[:, None] * group_rows + row_offsets[None, :],
                 tl.trans(values), mask=k_mask[:, None] & row_mask[None, :])


def split(a, offs, s, bm, bk, nw):
    _, k = a.shape
    e = offs.numel() - 1
    out = torch.empty_like(a)
    _pgt_split[(e, triton.cdiv(k, bk), s)](a, out, offs, K=k, S=s, BLOCK_M=bm, BLOCK_K=bk, num_warps=nw)
    return out


NUM_EXPERTS = 8
for k, rows in [(k, n) for k in (128, 512, 4096) for n in (16, 128, 1024)]:
    a = torch.randn(rows * NUM_EXPERTS, k, device="cuda").to(torch.bfloat16)
    offs = torch.tensor([i * rows for i in range(NUM_EXPERTS + 1)], dtype=torch.int32, device="cuda")
    ref = cur.per_group_transpose(a, offs)
    t_cur = do_bench(lambda: cur.per_group_transpose(a, offs), warmup=25, rep=200, return_mode="median") * 1e3
    res = []
    for s, (bm, bk), nw in itertools.product((1, 2, 4, 8, 16), ((32, 32), (32, 64), (64, 64), (64, 128), (32, 128)), (4, 8)):
        if bm * s > max(rows, bm) * 2 and s > 1:
            continue
        o = split(a, offs, s, bm, bk, nw)
        if not torch.equal(o, ref):
            print("MISMATCH", k, rows, s, bm, bk, nw)
            continue
        t = do_bench(lambda: split(a, offs, s, bm, bk, nw), warmup=25, rep=200, return_mode="median") * 1e3
        res.append((t, (s, bm, bk, nw)))
    res.sort()
    print(f"k={k:5d} rows={rows:5d} ({a.numel() * 2 / 1e6:.1f} MB) | current {t_cur:6.1f} us | best (S, BM, BK, warps): "
          + ", ".join(f"{c} {t:.1f}" for t, c in res[:3]), flush=True)
