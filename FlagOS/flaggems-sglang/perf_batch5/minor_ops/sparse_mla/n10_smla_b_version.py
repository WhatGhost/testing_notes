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

q and the kv rows are widened and the scores computed as in the generic op.
The softmax and the product with the values are one kernel with an online
softmax over blocks of the selected keys, which replaces the separate
softmax kernel and the stored probabilities. When there are few query rows
and many keys, the keys are also split across programs and the partial
results combined, so the long key loop does not run on a handful of
programs.
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
def _scores(
    Q,
    K,
    I,
    A,
    H: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    SCALE: tl.constexpr,
    GATHER: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    idx = tl.load(I + s * I0 + n * I2, n < T, -1)
    valid = (n < T) & (idx >= 0)
    page = tl.maximum(idx, 0)
    if GATHER:
        page = s * T + n
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(triton.cdiv(D, BK)):
        d = start * BK + r
        q = tl.load(
            Q + (s * H + h[:, None]) * D + d[None, :],
            (h[:, None] < H) & (d[None, :] < D),
            0,
        )
        k = tl.load(
            K + page[None, :] * D + d[:, None],
            valid[None, :] & (d[:, None] < D),
            0,
        )
        acc = tl.dot(q, k, acc)
    tl.store(
        A + (s * H + h[:, None]) * T + n[None, :],
        tl.where(valid[None, :], acc * SCALE, -float("inf")),
        (h[:, None] < H) & (n[None, :] < T),
    )


@triton.jit
def _scores_split(
    Q,
    K,
    I,
    A,
    H: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    SCALE: tl.constexpr,
    GATHER: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ROWS: tl.constexpr,
    PARTS: tl.constexpr,
):
    part = tl.program_id(2) % PARTS
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(2) // PARTS * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    idx = tl.load(I + s * I0 + n * I2, n < T, -1)
    valid = (n < T) & (idx >= 0)
    page = tl.maximum(idx, 0)
    if GATHER:
        page = s * T + n
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(triton.cdiv(D, PARTS * BK)):
        d = (part * triton.cdiv(D, PARTS * BK) + start) * BK + r
        q = tl.load(
            Q + (s * H + h[:, None]) * D + d[None, :],
            (h[:, None] < H) & (d[None, :] < D),
            0,
        )
        k = tl.load(
            K + page[None, :] * D + d[:, None],
            valid[None, :] & (d[:, None] < D),
            0,
        )
        acc = tl.dot(q, k, acc)
    tl.store(
        A + part * ROWS * T + (s * H + h[:, None]) * T + n[None, :],
        tl.where(valid[None, :], acc * SCALE, -float("inf")),
        (h[:, None] < H) & (n[None, :] < T),
    )


@triton.jit
def _values(
    A,
    K,
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
    GATHER: tl.constexpr,
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
        page = tl.maximum(idx, 0)
        if GATHER:
            page = s * T + t
        v = tl.load(
            K + page[:, None] * D + d[None, :],
            (t_ok & (idx >= 0))[:, None] & (d[None, :] < V),
            0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
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


def _values_config(s, h, t):
    """BN, BV, num_warps, num_stages and key splits of the values kernel."""
    return 64, 128, 4, 1, 1


def triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v):
    s, h, _ = q_nope.shape
    d, t = (kv.shape[-1], indices.shape[-1])
    gather = s * t < kv.shape[0]
    nk = s * t if gather else kv.shape[0]
    parts = (
        3
        if d == 576 and s * triton.cdiv(h, 16) * triton.cdiv(t, 64) < 64
        else 1
    )
    fnuz = q_nope.dtype == torch.float8_e4m3fnuz
    qp = torch.empty((s, h, d), device=q_nope.device, dtype=torch.bfloat16)
    kp = torch.empty((nk, d), device=q_nope.device, dtype=torch.bfloat16)
    a = torch.empty(
        (parts, s, h, t), device=q_nope.device, dtype=torch.float32
    )
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
    if parts > 1:
        _scores_split[s, triton.cdiv(h, 16), triton.cdiv(t, 64) * parts](
            qp,
            kp,
            indices,
            a,
            h,
            d,
            t,
            indices.stride(0),
            indices.stride(2),
            sm_scale,
            gather,
            64,
            64,
            s * h,
            parts,
            num_warps=4,
            num_stages=2,
        )
    else:
        _scores[s, triton.cdiv(h, 16), triton.cdiv(t, 64)](
            qp,
            kp,
            indices,
            a,
            h,
            d,
            t,
            indices.stride(0),
            indices.stride(2),
            sm_scale,
            gather,
            64,
            64,
            num_warps=4,
            num_stages=2,
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
        gather,
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
