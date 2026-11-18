import triton
import triton.language as tl


@triton.jit
def _split_k(A, B, P, M, N, K, A0, A1, B0, B1, k_split, SPLITS: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    pid_n = pid % num_n
    part = (pid // num_n + pid_n) % SPLITS
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    k_start = part * k_split
    k_end = tl.minimum(k_start + k_split, K)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(k_start, k_end, BK):
        kk = start + k
        if EVEN_K:
            a = tl.load(A + m[:, None] * A0 + kk[None, :] * A1, m[:, None] < M, 0)
            b = tl.load(B + kk[:, None] * B0 + n[None, :] * B1, n[None, :] < N, 0)
        else:
            a = tl.load(A + m[:, None] * A0 + kk[None, :] * A1, (m[:, None] < M) & (kk[None, :] < k_end), 0)
            b = tl.load(B + kk[:, None] * B0 + n[None, :] * B1, (kk[:, None] < k_end) & (n[None, :] < N), 0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(P + (part * M + m[:, None]) * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _split_k_il(A, B, P, M, N, K, A0, A1, B0, B1, SPLITS: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    pid_n = pid % num_n
    part = pid // num_n
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(part * BK, K, SPLITS * BK):
        kk = start + k
        if EVEN_K:
            a = tl.load(A + m[:, None] * A0 + kk[None, :] * A1, m[:, None] < M, 0)
            b = tl.load(B + kk[:, None] * B0 + n[None, :] * B1, n[None, :] < N, 0)
        else:
            a = tl.load(A + m[:, None] * A0 + kk[None, :] * A1, (m[:, None] < M) & (kk[None, :] < K), 0)
            b = tl.load(B + kk[:, None] * B0 + n[None, :] * B1, (kk[:, None] < K) & (n[None, :] < N), 0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(P + (part * M + m[:, None]) * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce(P, C, total, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.load(P + offs, mask, 0.0)
    for part in tl.static_range(1, SPLITS):
        acc += tl.load(P + part * total + offs, mask, 0.0)
    tl.store(C + offs, acc, mask)
