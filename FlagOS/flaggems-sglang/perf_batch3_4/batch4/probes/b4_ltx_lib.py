import triton
import triton.language as tl


@triton.jit
def _flat(X, C, S, O, N, T, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    q = i // D
    d = i % D
    h = q % H
    row = q // H
    b = row // T
    t = row % T
    xo = q * (2 * D) + d
    co = ((b * H + h) * T + t) * D + d
    a = tl.load(X + xo, m, eviction_policy="evict_last").to(tl.float32)
    z = tl.load(X + xo + D, m, eviction_policy="evict_last").to(tl.float32)
    c = tl.load(C + co, m, eviction_policy="evict_last").to(tl.float32)
    s = tl.load(S + co, m, eviction_policy="evict_last").to(tl.float32)
    y1 = (a * c).to(tl.bfloat16).to(tl.float32) - z * s
    y2 = (z * c).to(tl.bfloat16).to(tl.float32) + a * s
    tl.store(O + xo, y1, m, eviction_policy="evict_first")
    tl.store(O + xo + D, y2, m, eviction_policy="evict_first")


@triton.jit
def _token_tile(X, C, S, O, T, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr):
    """Program = BT tokens x all heads (x rows contiguous)."""
    pid_t = tl.program_id(0)
    b = tl.program_id(1)
    t = pid_t * BT + tl.arange(0, BT)
    h = tl.arange(0, BH)
    d = tl.arange(0, D)
    m = (t[:, None, None] < T) & (h[None, :, None] < H)
    xo = ((b * T + t[:, None, None]).to(tl.int64) * H + h[None, :, None]) * (2 * D) + d[None, None, :]
    co = ((b * H + h[None, :, None]) * T + t[:, None, None]).to(tl.int64) * D + d[None, None, :]
    a = tl.load(X + xo, m).to(tl.float32)
    z = tl.load(X + xo + D, m).to(tl.float32)
    c = tl.load(C + co, m).to(tl.float32)
    s = tl.load(S + co, m).to(tl.float32)
    y1 = (a * c).to(tl.bfloat16).to(tl.float32) - z * s
    y2 = (z * c).to(tl.bfloat16).to(tl.float32) + a * s
    tl.store(O + xo, y1, m)
    tl.store(O + xo + D, y2, m)


@triton.jit
def _head_tile(X, C, S, O, T, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    """Program = one (b, h) x BT tokens (cos/sin rows contiguous)."""
    pid_t = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    t = pid_t * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    m = t[:, None] < T
    xo = ((b * T + t[:, None]).to(tl.int64) * H + h) * (2 * D) + d[None, :]
    co = (bh * T + t[:, None]).to(tl.int64) * D + d[None, :]
    a = tl.load(X + xo, m).to(tl.float32)
    z = tl.load(X + xo + D, m).to(tl.float32)
    c = tl.load(C + co, m).to(tl.float32)
    s = tl.load(S + co, m).to(tl.float32)
    y1 = (a * c).to(tl.bfloat16).to(tl.float32) - z * s
    y2 = (z * c).to(tl.bfloat16).to(tl.float32) + a * s
    tl.store(O + xo, y1, m)
    tl.store(O + xo + D, y2, m)
