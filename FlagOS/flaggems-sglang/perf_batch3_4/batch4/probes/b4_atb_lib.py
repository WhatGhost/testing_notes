import triton
import triton.language as tl


@triton.jit
def _atb(logits_ptr, bitmask_ptr, out_ptr, V, NW, BLOCK: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_v = pid_v * BLOCK + tl.arange(0, BLOCK)
    m_v = offs_v < V
    word = offs_v // 32
    bit = offs_v % 32
    w = tl.load(bitmask_ptr + pid_b * NW + word, mask=m_v, other=0)
    keep = ((w >> bit) & 1) != 0
    x = tl.load(logits_ptr + pid_b * V + offs_v, mask=m_v, other=0.0)
    neg_inf = tl.full(x.shape, float("-inf"), x.dtype)
    y = tl.where(keep, x, neg_inf)
    tl.store(out_ptr + pid_b * V + offs_v, y, mask=m_v)


@triton.jit
def _atb_words(logits_ptr, bitmask_ptr, out_ptr, V, NW, WORDS: tl.constexpr):
    """Each program loads WORDS bitmask words once and expands them."""
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)
    wi = pid_v * WORDS + tl.arange(0, WORDS)
    w = tl.load(bitmask_ptr + pid_b * NW + wi, mask=wi < NW, other=0)
    bit = tl.arange(0, 32)
    keep = ((w[:, None] >> bit[None, :]) & 1) != 0
    offs = wi[:, None] * 32 + bit[None, :]
    m = offs < V
    x = tl.load(logits_ptr + pid_b.to(tl.int64) * V + offs, mask=m, other=0.0)
    y = tl.where(keep, x, float("-inf"))
    tl.store(out_ptr + pid_b.to(tl.int64) * V + offs, y, mask=m)
