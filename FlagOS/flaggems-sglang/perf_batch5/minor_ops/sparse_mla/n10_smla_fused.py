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

"""triton_sparse_mla_fwd for RDNA3 (gfx1100), fused prototype.

q and the kv rows are widened as in the generic op. One kernel then walks
the selected keys in blocks: each block of widened rows is read once, used
for the scores against q and, through its first d_v columns, as the values,
with an online softmax in between. With few query rows the keys are split
across programs and the partial results combined.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode(q, FNUZ: tl.constexpr):
    q = q.to(tl.uint32)
    a = q & 127
    e = a >> 3
    m = a & 7
    if FNUZ:
        normal = (e + 119 << 23 | m << 20).to(tl.float32, bitcast=True)
        v = tl.where(e == 0, m.to(tl.float32) * 0.0009765625, normal)
    else:
        normal = (e + 120 << 23 | m << 20).to(tl.float32, bitcast=True)
        v = tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal)
    v = (v.to(tl.uint32, bitcast=True) | (q & 128) << 24).to(
        tl.float32, bitcast=True
    )
    if FNUZ:
        return tl.where(q == 128, float("nan"), v)
    else:
        return tl.where(a == 127, float("nan"), v)


@triton.jit
def _packboth(
    Q,
    R,
    K,
    I,
    QP,
    KP,
    S: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    T: tl.constexpr,
    GATHER: tl.constexpr,
    FNUZ: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    nq = triton.cdiv(S * H * D, B)
    if pid < nq:
        x = pid * B + tl.arange(0, B)
        s = x // (H * D)
        h = x // D % H
        d = x % D
        q = tl.load(Q + s * Q0 + h * Q1 + d * Q2, (s < S) & (d < V), 0)
        r = tl.load(R + s * R0 + h * R1 + (d - V) * R2, (s < S) & (d >= V), 0)
        tl.store(QP + x, _decode(tl.where(d < V, q, r), FNUZ), x < S * H * D)
    else:
        x = (pid - nq) * B + tl.arange(0, B)
        if GATHER:
            row = x // D
            idx = tl.load(I + row // T * I0 + row % T * I2, row < N, -1)
            q = tl.load(
                K + tl.maximum(idx, 0) * K0 + x % D * K2,
                (x < N * D) & (idx >= 0),
                0,
            )
        else:
            q = tl.load(K + x // D * K0 + x % D * K2, x < N * D, 0)
        tl.store(KP + x, _decode(q, FNUZ), x < N * D)


@triton.jit
def _values(
    acc, alpha, p, KP, page, valid, d, D: tl.constexpr, V: tl.constexpr
):
    v = tl.load(
        KP + page[:, None] * D + d[None, :],
        valid[:, None] & (d[None, :] < V),
        0,
    )
    return acc * alpha[:, None] + tl.dot(p, v)


@triton.jit
def _store(
    acc,
    norm,
    m_i,
    l_i,
    O,
    W,
    rows,
    h_ok,
    split,
    ROWS,
    d,
    V: tl.constexpr,
    SPLIT: tl.constexpr,
):
    mask = h_ok[:, None] & (d[None, :] < V)
    if SPLIT == 1:
        tl.store(O + rows[:, None] * V + d[None, :], acc * norm[:, None], mask)
    else:
        part_rows = split * ROWS + rows
        tl.store(W + part_rows[:, None] * (V + 2) + d[None, :], acc, mask)
        if d.min() == 0:
            tl.store(W + part_rows * (V + 2) + V, m_i, h_ok)
            tl.store(W + part_rows * (V + 2) + V + 1, l_i, h_ok)


@triton.jit
def _attend(
    QP,
    KP,
    I,
    O,
    W,
    ROWS,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    SCALE: tl.constexpr,
    GATHER: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
    BV: tl.constexpr,
    BD: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    split = tl.program_id(2)
    h_ok = h < H
    rows = s * H + h
    dv = tl.arange(0, BV)
    m_i = tl.full((16,), -float("inf"), tl.float32)
    l_i = tl.zeros((16,), tl.float32)
    acc0 = tl.zeros((16, BV), tl.float32)
    acc1 = tl.zeros((16, BV), tl.float32)
    acc2 = tl.zeros((16, BV), tl.float32)
    acc3 = tl.zeros((16, BV), tl.float32)
    CHUNK: tl.constexpr = triton.cdiv(triton.cdiv(T, SPLIT), BN) * BN
    for start in range(split * CHUNK, tl.minimum(T, (split + 1) * CHUNK), BN):
        t = start + tl.arange(0, BN)
        t_ok = t < T
        idx = tl.load(I + s * I0 + t * I2, t_ok, -1)
        valid = t_ok & (idx >= 0)
        page = tl.maximum(idx, 0)
        if GATHER:
            page = s * T + t
        sc = tl.zeros((16, BN), tl.float32)
        for c in range(triton.cdiv(D, BD)):
            d = c * BD + tl.arange(0, BD)
            q = tl.load(
                QP + rows[:, None] * D + d[None, :],
                h_ok[:, None] & (d[None, :] < D),
                0,
            )
            k = tl.load(
                KP + page[None, :] * D + d[:, None],
                valid[None, :] & (d[:, None] < D),
                0,
            )
            sc = tl.dot(q, k, sc)
        sc = tl.where(valid[None, :], sc * SCALE, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_use = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_use)
        p = tl.exp(sc - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        p = p.to(tl.bfloat16)
        acc0 = _values(acc0, alpha, p, KP, page, valid, dv, D, V)
        acc1 = _values(acc1, alpha, p, KP, page, valid, dv + BV, D, V)
        acc2 = _values(acc2, alpha, p, KP, page, valid, dv + 2 * BV, D, V)
        acc3 = _values(acc3, alpha, p, KP, page, valid, dv + 3 * BV, D, V)
        m_i = m_new
    norm = 1.0 / tl.where(l_i > 0, l_i, 1.0)
    _store(acc0, norm, m_i, l_i, O, W, rows, h_ok, split, ROWS, dv, V, SPLIT)
    _store(
        acc1, norm, m_i, l_i, O, W, rows, h_ok, split, ROWS, dv + BV, V, SPLIT
    )
    _store(
        acc2,
        norm,
        m_i,
        l_i,
        O,
        W,
        rows,
        h_ok,
        split,
        ROWS,
        dv + 2 * BV,
        V,
        SPLIT,
    )
    _store(
        acc3,
        norm,
        m_i,
        l_i,
        O,
        W,
        rows,
        h_ok,
        split,
        ROWS,
        dv + 3 * BV,
        V,
        SPLIT,
    )


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


def _attend_config(s, h, t):
    """BN, num_warps, num_stages and key splits of the attention kernel."""
    programs = s * triton.cdiv(h, 16)
    splits = max(
        1, min(64, t // 64, triton.next_power_of_2(triton.cdiv(384, programs)))
    )
    return 64, 8, 1, splits


def triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v):
    s, h, _ = q_nope.shape
    d, t = (kv.shape[-1], indices.shape[-1])
    gather = s * t < kv.shape[0]
    nk = s * t if gather else kv.shape[0]
    fnuz = q_nope.dtype == torch.float8_e4m3fnuz
    qp = torch.empty((s, h, d), device=q_nope.device, dtype=torch.bfloat16)
    kp = torch.empty((nk, d), device=q_nope.device, dtype=torch.bfloat16)
    out = torch.empty(
        (1, s, h, d_v), device=q_nope.device, dtype=torch.bfloat16
    )
    _packboth[triton.cdiv(s * h * d, 1024) + triton.cdiv(nk * d, 1024),](
        q_nope.view(torch.uint8),
        q_rope.view(torch.uint8),
        kv.view(torch.uint8),
        indices,
        qp,
        kp,
        s,
        h,
        nk,
        d,
        d_v,
        *q_nope.stride(),
        *q_rope.stride(),
        kv.stride(0),
        kv.stride(2),
        indices.stride(0),
        indices.stride(2),
        t,
        gather,
        fnuz,
        1024,
        num_warps=4,
    )
    bn, warps, stages, splits = _attend_config(s, h, t)
    work = (
        torch.empty(
            (splits, s * h, d_v + 2), device=q_nope.device, dtype=torch.float32
        )
        if splits > 1
        else out
    )
    _attend[s, triton.cdiv(h, 16), splits](
        qp,
        kp,
        indices,
        out,
        work,
        s * h,
        h,
        d,
        d_v,
        t,
        indices.stride(0),
        indices.stride(2),
        sm_scale,
        gather,
        splits,
        bn,
        128,
        64,
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
