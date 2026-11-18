import triton
import triton.language as tl


@triton.jit
def _read_rows(w_ptr, ids_ptr, out_ptr, R, H, stride_we, stride_wn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # streaming read of R rows of each selected expert (ceiling for the access pattern)
    pid_n = tl.program_id(0)
    pair = tl.program_id(1)
    expert = tl.load(ids_ptr + pair).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    ptrs = w_ptr + expert * stride_we + offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None]
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for k in range(0, H, BLOCK_K):
        acc += tl.load(ptrs + k, mask=(offs_n < R)[None, :], other=0.0).to(tl.float32)
    tl.store(out_ptr + pair * R + offs_n, tl.sum(acc, axis=0), mask=offs_n < R)
