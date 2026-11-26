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

"""chunk_kda_scaled_dot_kkt for RDNA3 (gfx1100).

Each program owns 16 rows of a 64-row chunk of one head. For an earlier
16-row block J, exp2(g_i - g_j) = exp2(g_i - g_a) * exp2(g_a - g_j) with g_a
the first row of the program's block, so both factors are at most 1 when g
decreases down the chunk; q and k scaled by them go to the matrix units as
fp16 high and low parts (three products, about 22 bits). The diagonal block
is summed exactly in fp32, as is any block whose factors exceed 2 or whose
scaled values leave the fp16 range. The head dimension is walked in 32-wide
slices to keep the registers small.
"""

import torch
import triton
import triton.language as tl

from flaggems_sglang.ops.chunk_kda_scaled_dot_kkt import (
    chunk_kda_scaled_dot_kkt as _generic,
)


@triton.jit
def _split16(x):
    hi = x.to(tl.float16)
    return hi, (x - hi.to(tl.float32)).to(tl.float16)


@triton.jit
def _exact(
    Q,
    K,
    G,
    b,
    h,
    ri,
    c0,
    T,
    SQ0,
    SQ1,
    SQ2,
    SQ3,
    SK0,
    SK1,
    SK2,
    SK3,
    SG0,
    SG1,
    SG2,
    SG3,
    gs,
    D: tl.constexpr,
    BD: tl.constexpr,
    BC: tl.constexpr,
    DK: tl.constexpr,
    DIAG: tl.constexpr,
):
    # sum_d k_i k_j exp2(g_i - g_j) as written, one column per step; masked
    # entries get an exponent of -inf.
    jj = tl.arange(0, BC)
    acc_k = tl.zeros((BC, BC), tl.float32)
    acc_q = tl.zeros((BC, BC), tl.float32)
    for d0 in tl.static_range(0, BD, DK):
        d = d0 + tl.arange(0, DK)
        mi = (ri[:, None] < T) & (d[None, :] < D)
        ki = tl.load(
            K + b * SK0 + ri[:, None] * SK1 + h * SK2 + d[None, :] * SK3,
            mi,
            0,
        ).to(tl.float32)
        qi = tl.load(
            Q + b * SQ0 + ri[:, None] * SQ1 + h * SQ2 + d[None, :] * SQ3,
            mi,
            0,
        ).to(tl.float32)
        gi = tl.load(
            G + b * SG0 + ri[:, None] * SG1 + h * SG2 + d[None, :] * SG3,
            mi,
            0,
        ).to(tl.float32)
        for j in range(BC):
            cj = c0 + j
            mj = (d < D) & (cj < T)
            kj = tl.load(
                K + b * SK0 + cj * SK1 + h * SK2 + d * SK3, mj, 0
            ).to(tl.float32)
            gj = tl.load(
                G + b * SG0 + cj * SG1 + h * SG2 + d * SG3, mj, 0
            ).to(tl.float32)
            keep = (ri < T) & (cj < T)
            if DIAG:
                keep = keep & (ri >= cj)
            expo = tl.where(
                keep[:, None], (gi - gj[None, :]) * gs, float("-inf")
            )
            wk = kj[None, :] * tl.exp2(expo)
            hit = jj[None, :] == j
            acc_k += tl.where(hit, tl.sum(ki * wk, 1)[:, None], 0.0)
            acc_q += tl.where(hit, tl.sum(qi * wk, 1)[:, None], 0.0)
    return acc_k, acc_q


@triton.jit
def _kkt_kernel(
    Q,
    K,
    G,
    BETA,
    A,
    AQ,
    T,
    SQ0,
    SQ1,
    SQ2,
    SQ3,
    SK0,
    SK1,
    SK2,
    SK3,
    SG0,
    SG1,
    SG2,
    SG3,
    SB0,
    SB1,
    SB2,
    scale,
    gs,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    DK: tl.constexpr,
):
    NB: tl.constexpr = BT // BC
    pid = tl.program_id(0)
    blk = pid % NB
    h = pid // NB % H
    rest = pid // (NB * H)
    nt = tl.cdiv(T, BT)
    chunk = rest % nt
    b = rest // nt
    c0 = chunk * BT
    r0 = c0 + blk * BC
    if r0 < T:
        ri = r0 + tl.arange(0, BC)
        jj = tl.arange(0, BC)
        beta = tl.load(BETA + b * SB0 + ri * SB1 + h * SB2, ri < T, 0).to(
            tl.float32
        )
        out = ((b * T + ri[:, None]) * H + h) * BT + jj[None, :]
        rows = ri[:, None] < T
        for blk_j in range(0, blk):
            cj = c0 + blk_j * BC + jj
            acc_k = tl.zeros((BC, BC), tl.float32)
            acc_q = tl.zeros((BC, BC), tl.float32)
            ok = True
            for d0 in tl.static_range(0, BD, DK):
                d = d0 + tl.arange(0, DK)
                dm = d < D
                mr = (ri[:, None] < T) & dm[None, :]
                ga = tl.load(
                    G + b * SG0 + r0 * SG1 + h * SG2 + d * SG3, dm, 0
                ).to(tl.float32)
                gr = tl.load(
                    G
                    + b * SG0
                    + ri[:, None] * SG1
                    + h * SG2
                    + d[None, :] * SG3,
                    mr,
                    0,
                ).to(tl.float32)
                fr = tl.exp2(tl.where(mr, gr - ga[None, :], 0.0) * gs)
                kr = (
                    tl.load(
                        K
                        + b * SK0
                        + ri[:, None] * SK1
                        + h * SK2
                        + d[None, :] * SK3,
                        mr,
                        0,
                    ).to(tl.float32)
                    * fr
                )
                qr = (
                    tl.load(
                        Q
                        + b * SQ0
                        + ri[:, None] * SQ1
                        + h * SQ2
                        + d[None, :] * SQ3,
                        mr,
                        0,
                    ).to(tl.float32)
                    * fr
                )
                gc = tl.load(
                    G
                    + b * SG0
                    + cj[:, None] * SG1
                    + h * SG2
                    + d[None, :] * SG3,
                    dm[None, :],
                    0,
                ).to(tl.float32)
                fc = tl.exp2((ga[None, :] - gc) * gs)
                kc = (
                    tl.load(
                        K
                        + b * SK0
                        + cj[:, None] * SK1
                        + h * SK2
                        + d[None, :] * SK3,
                        dm[None, :],
                        0,
                    ).to(tl.float32)
                    * fc
                )
                big = tl.maximum(
                    tl.maximum(tl.max(tl.max(fr, 1), 0), tl.max(tl.max(fc, 1), 0))
                    * 15000.0,
                    tl.maximum(
                        tl.max(tl.max(tl.abs(kc), 1), 0),
                        tl.maximum(
                            tl.max(tl.max(tl.abs(kr), 1), 0),
                            tl.max(tl.max(tl.abs(qr), 1), 0),
                        ),
                    ),
                )
                ok = ok & (big < 30000.0)
                kr_hi, kr_lo = _split16(kr)
                qr_hi, qr_lo = _split16(qr)
                kc_hi, kc_lo = _split16(kc)
                kc_hi = tl.trans(kc_hi)
                kc_lo = tl.trans(kc_lo)
                acc_k = tl.dot(kr_hi, kc_hi, acc_k)
                acc_q = tl.dot(qr_hi, kc_hi, acc_q)
            if not ok:
                acc_k, acc_q = _exact(
                    Q,
                    K,
                    G,
                    b,
                    h,
                    ri,
                    c0 + blk_j * BC,
                    T,
                    SQ0,
                    SQ1,
                    SQ2,
                    SQ3,
                    SK0,
                    SK1,
                    SK2,
                    SK3,
                    SG0,
                    SG1,
                    SG2,
                    SG3,
                    gs,
                    D,
                    BD,
                    BC,
                    DK,
                    False,
                )
            tl.store(A + out + blk_j * BC, acc_k * beta[:, None], rows)
            tl.store(AQ + out + blk_j * BC, acc_q * scale, rows)
        acc_k, acc_q = _exact(
            Q,
            K,
            G,
            b,
            h,
            ri,
            r0,
            T,
            SQ0,
            SQ1,
            SQ2,
            SQ3,
            SK0,
            SK1,
            SK2,
            SK3,
            SG0,
            SG1,
            SG2,
            SG3,
            gs,
            D,
            BD,
            BC,
            DK,
            True,
        )
        strict = jj[:, None] > jj[None, :]
        tl.store(
            A + out + blk * BC,
            tl.where(strict, acc_k * beta[:, None], 0.0),
            rows,
        )
        tl.store(AQ + out + blk * BC, acc_q * scale, rows)
        zero = tl.zeros((BC, BC), tl.float32)
        for blk_j in range(blk + 1, NB):
            tl.store(A + out + blk_j * BC, zero, rows)
            tl.store(AQ + out + blk_j * BC, zero, rows)


def chunk_kda_scaled_dot_kkt(q, k, gk, beta, scale, gk_scale, chunk_size):
    b, t, h, d = k.shape
    if int(chunk_size) != 64 or d > 256:
        return _generic(q, k, gk, beta, scale, gk_scale, chunk_size)
    bt = 64
    a = torch.empty((b, t, h, bt), device=k.device, dtype=torch.float32)
    aq = torch.empty_like(a)
    grid = b * h * triton.cdiv(t, bt) * (bt // 16)
    if grid:
        _kkt_kernel[(grid,)](
            q,
            k,
            gk,
            beta,
            a,
            aq,
            t,
            *q.stride(),
            *k.stride(),
            *gk.stride(),
            *beta.stride(),
            float(scale),
            float(gk_scale),
            h,
            d,
            max(32, triton.next_power_of_2(d)),
            bt,
            16,
            32,
            num_warps=4,
        )
    return a, aq


__all__ = ["chunk_kda_scaled_dot_kkt"]
