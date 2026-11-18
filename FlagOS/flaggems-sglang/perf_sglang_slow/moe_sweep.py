"""fused_moe_gemm (rdna3) on the Qwen3.6-35B-A3B TP=4 MoE shapes: per-pair GEMV
path vs grouped path with swept tiles, against SGLang fused_experts_impl."""

import itertools
import json
import sys

import torch
import triton

W = open("/tmp/fgs-perf.path").read().strip()
sys.path.insert(0, f"{W}/src")
import flaggems_sglang  # noqa: E402,F401
import flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe_gemm as fm  # noqa: E402

E, TOPK, H, INTER = 256, 8, 2048, 128
BF16 = torch.bfloat16
MS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 4, 8, 16, 32, 64, 256, 1024, 4096]


def bench(fn, rep=40):
    return triton.testing.do_bench(fn, warmup=10, rep=rep) * 1e3


def sglang_ref():
    import sglang_fl.dispatch.backends.vendor.amd.patch  # noqa: F401  (FlagGems align/silu, as in the server)
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="/target/models/Qwen3.6-35B-A3B"))
    return fused_experts_impl


def run_grouped(A, B, tw, tid, top_k, cfg):
    T, K = A.shape
    Ex, N, _ = B.shape
    out = torch.empty((T, top_k, N), dtype=A.dtype, device=A.device)
    orig = fm._grouped_config
    fm._grouped_config = lambda P, E_, N_, K_, o: cfg
    try:
        return fm._fused_moe_gemm_grouped(A, B, tw.contiguous(), tid.to(torch.int32).contiguous(), top_k, out)
    finally:
        fm._grouped_config = orig


def run_gemv(A, B, tw, tid, top_k):
    T, K = A.shape
    Ex, N, _ = B.shape
    out = torch.empty((T, top_k, N), dtype=A.dtype, device=A.device)
    grid = lambda meta: (T, top_k, triton.cdiv(N, meta["BLOCK_N"]))
    fm._fused_moe_gemm_single_token_kernel[grid](
        A, B, tw.contiguous(), tid.to(torch.int32).contiguous(), out, T, Ex, N, K, top_k,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), B.stride(2),
        out.stride(0), out.stride(1), out.stride(2))
    return out


def configs(N, K):
    for bm, bn, bk, nw, ns in itertools.product((16, 32, 64), (32, 64, 128, 256), (32, 64, 128, 256), (2, 4, 8), (1, 2)):
        if bn > N or bk > K or (bk == 256 and bm == 64) or (bn == 256 and bm == 64 and bk > 64):
            continue
        yield bm, bn, bk, nw, ns


def sweep(name, A, B, tw, tid, top_k):
    ref = run_gemv(A, B, tw, tid, top_k)
    t_gemv = bench(lambda: run_gemv(A, B, tw, tid, top_k))
    best = []
    for cfg in configs(B.shape[1], B.shape[2]):
        try:
            out = run_grouped(A, B, tw, tid, top_k, cfg)
            if not torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2):
                print(f"   {name} cfg {cfg} MISMATCH", flush=True)
                continue
            best.append((bench(lambda: run_grouped(A, B, tw, tid, top_k, cfg), rep=20), cfg))
        except Exception as e:  # noqa: BLE001
            print(f"   {name} cfg {cfg} fail {type(e).__name__}: {str(e)[:60]}", flush=True)
    best.sort()
    # re-time the top few more carefully
    top = [(bench(lambda c=c: run_grouped(A, B, tw, tid, top_k, c)), c) for _, c in best[:5]]
    top.sort()
    return t_gemv, top


if __name__ == "__main__":
    fx = sglang_ref()
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * INTER, H, dtype=BF16, device="cuda") * 0.03
    w2 = torch.randn(E, H, INTER, dtype=BF16, device="cuda") * 0.03
    results = {}
    for M in MS:
        h = torch.randn(M, H, dtype=BF16, device="cuda")
        tw, tid = torch.topk(torch.softmax(torch.randn(M, E, device="cuda"), -1), TOPK, -1)
        tw = (tw / tw.sum(-1, keepdim=True)).float().contiguous()
        tid = tid.to(torch.int32).contiguous()
        t_ref = bench(lambda: fx(h, w1, w2, tw, tid, activation="silu"))
        ones = torch.ones_like(tw)
        up = run_gemv(h, w1, ones, tid, TOPK)
        act = flaggems_sglang.silu_and_mul(up.view(M * TOPK, 2 * INTER))
        t_silu = bench(lambda: flaggems_sglang.silu_and_mul(up.view(M * TOPK, 2 * INTER)))
        g_up, top_up = sweep("up", h, w1, ones, tid, TOPK)
        g_dn, top_dn = sweep("down", act, w2, tw.view(-1, 1), tid.view(-1, 1), 1)
        down = run_gemv(act, w2, tw.view(-1, 1), tid.view(-1, 1), 1)
        t_sum = bench(lambda: down.view(M, TOPK, H).sum(dim=1))
        best_up = min(g_up, top_up[0][0]) if top_up else g_up
        best_dn = min(g_dn, top_dn[0][0]) if top_dn else g_dn
        print(f"M={M}: SGLang {t_ref:.0f}us | up gemv {g_up:.0f} grouped {[(round(t), c) for t, c in top_up[:3]]} | "
              f"down gemv {g_dn:.0f} grouped {[(round(t), c) for t, c in top_dn[:3]]} | silu {t_silu:.0f} sum {t_sum:.0f} | "
              f"best total ~{best_up + best_dn + t_silu + t_sum:.0f}us", flush=True)
        results[M] = dict(ref=t_ref, gemv_up=g_up, gemv_down=g_dn, up=top_up, down=top_dn, silu=t_silu, sum=t_sum)
    json.dump({str(k): v for k, v in results.items()}, open("/tmp/moe_sweep.json", "w"), indent=1, default=str)
