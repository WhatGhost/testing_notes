"""tiny_n_gemm (rdna3) split-K tiles for M = 16/32/64 on the Qwen3.6 TP=4 shapes."""

import itertools
import sys

import torch
import triton

W = open("/tmp/fgs-perf.path").read().strip()
sys.path.insert(0, f"{W}/src")
import flaggems_sglang as fgs  # noqa: E402
import flaggems_sglang.runtime.backend._amd.rdna3.ops.tiny_n_gemm as tng  # noqa: E402

SHAPES = {
    "27B qkv": (3584, 5120), "27B gdn_qkvz": (4096, 5120), "27B gate_up": (8704, 5120),
    "27B down": (5120, 4352), "27B o_proj": (5120, 1536), "35B qkv": (2560, 2048),
    "35B gdn_qkvz": (3072, 2048), "35B o_proj": (2048, 1024),
    "27B lm_head": (62080, 5120), "35B lm_head": (62080, 2048),
}
MS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [16, 32, 64]


def run(x, w, bm, bn, bk, steps, nw):
    m, k = x.shape
    n = w.shape[0]
    split_k = triton.cdiv(triton.cdiv(k, bk), steps)
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    part = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    tng._tiny_n_dot_splitk_kernel[(tiles * split_k,)](
        x, w, part, m, n, k, split_k, steps, x.stride(0), w.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, EVEN_K=(k % bk == 0), IEEE=False,
        num_warps=nw, num_stages=1)
    total = m * n
    tng._tiny_n_reduce_kernel[(triton.cdiv(total, 256),)](part, out, total, split_k, BLOCK=256, num_warps=1, num_stages=1)
    return out


def bench(fn, rep=40):
    return triton.testing.do_bench(fn, warmup=10, rep=rep) * 1e3


if __name__ == "__main__":
    for name, (N, K) in SHAPES.items():
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * K**-0.5
        for M in MS:
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            ref = torch.nn.functional.linear(x, w)
            t_hip = bench(lambda: torch.nn.functional.linear(x, w))
            t_cur = bench(lambda: fgs.tiny_n_gemm(x, w, torch.bfloat16))
            res = []
            for bm, bn, bk, steps, nw in itertools.product(
                    (16, 32, 64), (16, 32, 64), (64, 128, 256), (1, 2, 3, 4, 6, 8), (2, 4, 8)):
                if bm > max(16, triton.next_power_of_2(M)) or bm * bk > 64 * 128 * (nw // 2 if nw > 2 else 1) * 2:
                    continue
                if bk * steps > K * 2:
                    continue
                try:
                    out = run(x, w, bm, bn, bk, steps, nw)
                    if not torch.allclose(out.float(), ref.float(), atol=3e-2, rtol=3e-2):
                        continue
                    res.append((bench(lambda: run(x, w, bm, bn, bk, steps, nw), rep=15), (bm, bn, bk, steps, nw)))
                except Exception:  # noqa: BLE001
                    pass
            res.sort()
            top = sorted((bench(lambda c=c: run(x, w, *c)), c) for _, c in res[:4])
            print(f"{name:13s} M={M:2d}: hipBLASLt {t_hip:7.1f}  current {t_cur:7.1f}  best " +
                  "; ".join(f"{t:7.1f} {c}" for t, c in top[:3]), flush=True)
        del w
        torch.cuda.empty_cache()
