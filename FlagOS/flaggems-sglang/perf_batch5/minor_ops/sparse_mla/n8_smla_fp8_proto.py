# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""triton_sparse_mla_fwd for RDNA3 (gfx1100).

Both kernels read q and the selected kv rows as fp8 and widen them to fp16
in registers (exact for e4m3), so the gathers move half the bytes of a bf16
copy and the whole fp8 pool can stay in L2. The softmax and the product
with the values are one kernel with an online softmax over blocks of the
selected keys.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp16(code, FNUZ: tl.constexpr):
    c = code.to(tl.uint16)
    y = ((c & 0x80) << 8 | (c & 0x7F) << 7).to(tl.float16, bitcast=True)
    if FNUZ:
        y = y * tl.full((1,), 128.0, tl.float16)
        nan = c == 0x80
    else:
        y = y * tl.full((1,), 256.0, tl.float16)
        nan = (c & 0x7F) == 0x7F
    return tl.where(nan, float("nan"), y.to(tl.float32)).to(tl.float16)


@triton.jit
def _scores(
    QN,
    QR,
    KV,
    I,
    A,
    ROWS,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    QN0: tl.constexpr,
    QN1: tl.constexpr,
    QN2: tl.constexpr,
    QR0: tl.constexpr,
    QR1: tl.constexpr,
    QR2: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    SCALE: tl.constexpr,
    FNUZ: tl.constexpr,
    PARTS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Part j of PARTS sums its share of the BK-column steps over D; the
    # values kernel adds the parts.
    part = tl.program_id(2) % PARTS
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(2) // PARTS * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    h_ok = h < H
    idx = tl.load(I + s * I0 + n * I2, n < T, -1)
    valid = (n < T) & (idx >= 0)
    page = tl.maximum(idx, 0)
    acc = tl.zeros((16, BN), tl.float32)
    STEPS: tl.constexpr = triton.cdiv(triton.cdiv(D, BK), PARTS)
    for i in range(STEPS):
        d = (part * STEPS + i) * BK + r
        qn = tl.load(
            QN + s * QN0 + h[:, None] * QN1 + d[None, :] * QN2,
            h_ok[:, None] & (d[None, :] < V),
            0,
        )
        qr = tl.load(
            QR + s * QR0 + h[:, None] * QR1 + (d[None, :] - V) * QR2,
            h_ok[:, None] & (d[None, :] >= V) & (d[None, :] < D),
            0,
        )
        k = tl.load(
            KV + page[None, :] * K0 + d[:, None] * K2,
            valid[None, :] & (d[:, None] < D),
            0,
        )
        q = _fp16(tl.where(d[None, :] < V, qn, qr), FNUZ)
        acc = tl.dot(q, _fp16(k, FNUZ), acc)
    tl.store(
        A + part * ROWS * T + (s * H + h[:, None]) * T + n[None, :],
        tl.where(valid[None, :], acc * SCALE, -float("inf")),
        h_ok[:, None] & (n[None, :] < T),
    )


@triton.jit
def _values(
    A,
    KV,
    I,
    O,
    W,
    ROWS,
    H: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    FNUZ: tl.constexpr,
    PARTS: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
    BV: tl.constexpr,
):
    # A holds PARTS partial scores per (row, key), -inf where the key is
    # not selected. The softmax runs over blocks of BN keys with a running
    # maximum and sum. With SPLIT > 1, split j takes the j-th of SPLIT
    # chunks of keys and leaves its unnormalized output, maximum and sum
    # in W for _combine.
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.program_id(2) % triton.cdiv(V, BV) * BV + tl.arange(0, BV)
    split = tl.program_id(2) // triton.cdiv(V, BV)
    n = tl.arange(0, BN)
    h_ok = h < H
    rows = s * H + h
    m_i = tl.full((16,), -float("inf"), tl.float32)
    l_i = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, BV), tl.float32)
    CHUNK: tl.constexpr = triton.cdiv(triton.cdiv(T, SPLIT), BN) * BN
    for start in range(split * CHUNK, tl.minimum(T, (split + 1) * CHUNK), BN):
        t = start + n
        t_ok = t < T
        a = tl.zeros((16, BN), tl.float32)
        for part in tl.static_range(PARTS):
            a += tl.load(
                A + part * ROWS * T + rows[:, None] * T + t[None, :],
                h_ok[:, None] & t_ok[None, :],
                -float("inf"),
            )
        m_new = tl.maximum(m_i, tl.max(a, 1))
        m_use = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_use)
        p = tl.exp(a - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        idx = tl.load(I + s * I0 + t * I2, t_ok, -1)
        v = tl.load(
            KV + tl.maximum(idx, 0)[:, None] * K0 + d[None, :] * K2,
            (t_ok & (idx >= 0))[:, None] & (d[None, :] < V),
            0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), _fp16(v, FNUZ))
        m_i = m_new
    mask = h_ok[:, None] & (d[None, :] < V)
    if SPLIT == 1:
        out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        tl.store(O + rows[:, None] * V + d[None, :], out, mask)
    else:
        part_rows = split * ROWS + rows
        tl.store(W + part_rows[:, None] * (V + 2) + d[None, :], acc, mask)
        if tl.program_id(2) % triton.cdiv(V, BV) == 0:
            tl.store(W + part_rows * (V + 2) + V, m_i, h_ok)
            tl.store(W + part_rows * (V + 2) + V + 1, l_i, h_ok)


@triton.jit
def _combine(
    W,
    O,
    ROWS,
    V: tl.constexpr,
    SPLIT: tl.constexpr,
    BV: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BV)
    d_ok = d < V
    m = tl.full((), -float("inf"), tl.float32)
    for split in tl.static_range(SPLIT):
        m = tl.maximum(m, tl.load(W + (split * ROWS + row) * (V + 2) + V))
    m = tl.where(m == -float("inf"), 0.0, m)
    acc = tl.zeros((BV,), tl.float32)
    total = tl.zeros((), tl.float32)
    for split in tl.static_range(SPLIT):
        base = W + (split * ROWS + row) * (V + 2)
        w = tl.exp(tl.load(base + V) - m)
        total += w * tl.load(base + V + 1)
        acc += w * tl.load(base + d, d_ok, 0.0)
    tl.store(O + row * V + d, acc / tl.where(total > 0, total, 1.0), d_ok)


def _scores_config(s, h, t):
    """BN, BK, num_warps, num_stages of the scores kernel."""
    return 64, 64, 4, 2


def _values_config(s, h, t):
    """BN, BV, num_warps, num_stages and key splits of the values kernel."""
    return 64, 128, 4, 1, 1


def triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v):
    s, h, _ = q_nope.shape
    d, t = (kv.shape[-1], indices.shape[-1])
    parts = (
        3
        if d == 576 and s * triton.cdiv(h, 16) * triton.cdiv(t, 64) < 64
        else 1
    )
    fnuz = q_nope.dtype == torch.float8_e4m3fnuz
    a = torch.empty(
        (parts, s, h, t), device=q_nope.device, dtype=torch.float32
    )
    out = torch.empty(
        (1, s, h, d_v), device=q_nope.device, dtype=torch.bfloat16
    )
    bn, bk, warps, stages = _scores_config(s, h, t)
    _scores[s, triton.cdiv(h, 16), triton.cdiv(t, bn) * parts](
        q_nope.view(torch.uint8),
        q_rope.view(torch.uint8),
        kv.view(torch.uint8),
        indices,
        a,
        s * h,
        h,
        d,
        d_v,
        t,
        *q_nope.stride(),
        *q_rope.stride(),
        kv.stride(0),
        kv.stride(2),
        indices.stride(0),
        indices.stride(2),
        sm_scale,
        fnuz,
        parts,
        bn,
        bk,
        num_warps=warps,
        num_stages=stages,
    )
    bn, bv, warps, stages, splits = _values_config(s, h, t)
    work = (
        torch.empty(
            (splits, s * h, d_v + 2), device=q_nope.device, dtype=torch.float32
        )
        if splits > 1
        else out
    )
    _values[s, triton.cdiv(h, 16), triton.cdiv(d_v, bv) * splits](
        a,
        kv.view(torch.uint8),
        indices,
        out,
        work,
        s * h,
        h,
        d_v,
        t,
        kv.stride(0),
        kv.stride(2),
        indices.stride(0),
        indices.stride(2),
        fnuz,
        parts,
        splits,
        bn,
        bv,
        num_warps=warps,
        num_stages=stages,
    )
    if splits > 1:
        _combine[(s * h,)](
            work,
            out,
            s * h,
            d_v,
            splits,
            triton.next_power_of_2(d_v),
            num_warps=4,
        )
    return out


__all__ = ["triton_sparse_mla_fwd"]
