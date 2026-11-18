import torch
import triton
import triton.language as tl


@triton.jit
def _fused_softmax_pool_kernel(
    kv_score_ptr,
    out_ptr,
    stride_bs: tl.constexpr,
    stride_k: tl.constexpr,
    K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * stride_bs

    offs = tl.arange(0, HEAD_BLOCK)
    mask = offs < HEAD_DIM

    max_val = tl.full([HEAD_BLOCK], float("-inf"), dtype=tl.float32)
    for k in range(K):
        s = tl.load(
            kv_score_ptr + base + k * stride_k + HEAD_DIM + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        max_val = tl.maximum(max_val, s)

    sum_exp = tl.zeros([HEAD_BLOCK], dtype=tl.float32)
    weighted = tl.zeros([HEAD_BLOCK], dtype=tl.float32)
    for k in range(K):
        s = tl.load(
            kv_score_ptr + base + k * stride_k + HEAD_DIM + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        v = tl.load(
            kv_score_ptr + base + k * stride_k + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.exp(s - max_val)
        sum_exp += w
        weighted += v * w

    result = weighted / sum_exp
    tl.store(
        out_ptr + pid * HEAD_DIM + offs, result.to(out_ptr.dtype.element_ty), mask=mask
    )


def fused_softmax_pool_triton(
    kv_score: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Fused softmax-weighted-sum: out = (kv * softmax(score, dim=1)).sum(dim=1).

    Replaces the generic cunn_SpatialSoftMaxForward + elementwise multiply + sum
    with a single Triton kernel.

    Args:
        kv_score: [bs, K, 2 * head_dim] where first head_dim is kv, second is score.
        head_dim: dimension of each of kv and score.
    Returns:
        output: [bs, head_dim]
    """
    assert kv_score.dim() == 3
    bs, K, last = kv_score.shape
    assert last == 2 * head_dim
    assert kv_score.is_contiguous()

    out = torch.empty(bs, head_dim, dtype=kv_score.dtype, device=kv_score.device)
    if bs == 0:
        return out

    HEAD_BLOCK = triton.next_power_of_2(head_dim)
    grid = (bs,)
    _fused_softmax_pool_kernel[grid](
        kv_score,
        out,
        stride_bs=kv_score.stride(0),
        stride_k=kv_score.stride(1),
        K=K,
        HEAD_DIM=head_dim,
        HEAD_BLOCK=HEAD_BLOCK,
    )
    return out
