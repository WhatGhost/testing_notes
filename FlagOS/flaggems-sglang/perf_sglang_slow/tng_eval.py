"""tiny_n_gemm candidate configs on every decode shape, CUDA-graph timed like
bench_kernels.graph_us (L2/IC flushed before each call). Saves the full grid."""

import json
import sys

import torch
import triton

sys.path.insert(0, "/tmp")
from tng_sweep import run  # noqa: E402

W = open("/tmp/fgs-perf.path").read().strip()
sys.path.insert(0, f"{W}/src")
import flaggems_sglang as fgs  # noqa: E402

CFGS = [(16, 16, 256, 3, 2), (16, 64, 256, 3, 4), (16, 64, 256, 6, 4), (16, 32, 256, 3, 2),
        (32, 64, 256, 3, 8), (32, 64, 256, 6, 8), (64, 64, 256, 6, 8), (16, 16, 256, 1, 2),
        (16, 16, 256, 8, 4), (32, 16, 256, 1, 4), (32, 32, 256, 1, 4), (64, 16, 256, 8, 4),
        (32, 32, 256, 3, 4), (16, 64, 256, 1, 4), (16, 16, 256, 2, 2), (32, 16, 256, 8, 4),
        (64, 32, 256, 3, 8), (32, 64, 256, 1, 8), (16, 64, 256, 8, 4), (16, 32, 256, 1, 2),
        (32, 16, 256, 3, 4), (64, 64, 256, 3, 8), (16, 16, 128, 1, 2), (16, 64, 128, 1, 4)]
SHAPES = {
    "27B qkv": (3584, 5120), "27B gdn_qkvz": (4096, 5120), "27B gate_up": (8704, 5120),
    "27B down": (5120, 4352), "27B o_proj": (5120, 1536), "27B lm_head": (62080, 5120),
    "35B qkv": (2560, 2048), "35B gdn_qkvz": (3072, 2048), "35B o_proj": (2048, 1024),
    "35B lm_head": (62080, 2048), "35B shared_down": (2048, 128), "35B router": (256, 2048),
    "repo n512": (512, 7168), "repo n1536": (1536, 7168),
}
MS = [1, 2, 4, 8, 16, 32, 64]
_FLUSH = []


def graph_us(fn, reps=20):
    if not _FLUSH:
        _FLUSH.append(torch.empty(32 * 1024 * 1024, dtype=torch.int32, device="cuda"))
    flush = _FLUSH[0]

    def build(with_fn):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                flush.zero_()
                if with_fn:
                    fn()
        return g

    def timed(g):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(7):
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return sorted(ts)[len(ts) // 2]

    return (timed(build(True)) - timed(build(False))) / reps * 1e3


if __name__ == "__main__":
    out = {}
    for name, (N, K) in SHAPES.items():
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * K**-0.5
        for M in MS:
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            ref = torch.nn.functional.linear(x, w)
            row = {"hip": graph_us(lambda: torch.nn.functional.linear(x, w)),
                   "cur": graph_us(lambda: fgs.tiny_n_gemm(x, w, torch.bfloat16))}
            for c in CFGS:
                if c[0] > max(16, triton.next_power_of_2(M)) or c[2] > max(16, triton.next_power_of_2(K)):
                    continue
                if c[2] == 128 and K > 1024:
                    continue
                o = run(x, w, *c)
                if not torch.allclose(o.float(), ref.float(), atol=3e-2, rtol=3e-2):
                    continue
                row[str(c)] = graph_us(lambda c=c: run(x, w, *c))
            out[f"{name}|{M}"] = row
            cands = [(v, k) for k, v in row.items() if k not in ("hip", "cur")]
            best = min(cands) if cands else (float("nan"), "-")
            print(f"{name:15s} M={M:2d}: hip {row['hip']:7.1f} cur {row['cur']:7.1f} best {best[0]:7.1f} {best[1]}", flush=True)
            json.dump(out, open("/tmp/tng_eval.json", "w"), indent=1)
        del w
        torch.cuda.empty_cache()
    json.dump(out, open("/tmp/tng_eval.json", "w"), indent=1)
