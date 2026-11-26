"""rdna3 kkt with num_warps 1 / 2 / 4 / 8: cold and warm time on the benchmark
shapes against the generic op, plus a tolerance check."""

import importlib.util
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
from flaggems_sglang.ops.chunk_kda_scaled_dot_kkt import chunk_kda_scaled_dot_kkt as gen  # noqa: E402
from flaggems_sglang.reference.chunk_kda_scaled_dot_kkt import reference  # noqa: E402

bm = __import__("benchmark.test_chunk_kda_scaled_dot_kkt", fromlist=["x"])
src = open(f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/chunk_kda_scaled_dot_kkt.py").read()
assert "num_warps=1" in src
mods = {}
for nw in (1, 2, 4, 8):
    path = f"/tmp/kkt_nw{nw}.py"
    open(path, "w").write(src.replace("num_warps=1", f"num_warps={nw}"))
    spec = importlib.util.spec_from_file_location(f"kkt_nw{nw}", path)
    mods[nw] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mods[nw])


def warm(fn, n=50):
    for _ in range(5):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


for s in list(bm.SHAPES) + list(bm.MORE_SHAPES):
    args = next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))
    ref = reference(*args)
    line = f"{str(s):18s} generic cold {do_bench(lambda: gen(*args), warmup=25, rep=200, return_mode='median') * 1e3:6.1f} warm {warm(lambda: gen(*args)):6.1f} |"
    for nw, m in mods.items():
        out = m.chunk_kda_scaled_dot_kkt(*args)
        ok = all(torch.allclose(o, r, atol=2e-2, rtol=2e-2) for o, r in zip(out, ref))
        tc = do_bench(lambda: m.chunk_kda_scaled_dot_kkt(*args), warmup=25, rep=200, return_mode="median") * 1e3
        line += f" w{nw} {tc:6.1f}/{warm(lambda: m.chunk_kda_scaled_dot_kkt(*args)):6.1f}{'' if ok else ' BAD'}"
    print(line, flush=True)
