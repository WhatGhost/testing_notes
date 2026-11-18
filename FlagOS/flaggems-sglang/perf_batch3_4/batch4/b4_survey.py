"""Fourth-batch survey on gfx1100 with the official benchmark's own inputs:
op time, a copy moving the same total bytes (read half, write half), the host
share in back-to-back calls, and kernels per call."""

import importlib
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile
from triton.testing import do_bench

B = open("/tmp/fgs-b4a.path").read().strip()
sys.path.insert(0, B)
sys.path.insert(0, os.path.join(B, "src"))
import flaggems_sglang  # noqa: E402,F401
from flaggems_sglang.reference import get_reference  # noqa: E402
from flaggems_sglang.reference_candidate import get_candidate  # noqa: E402

OPS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "add3", "fused_softmax_pool", "embedding_lora_a", "fused_qk_gemma_rmsnorm_rope_gate",
    "ltx2_split_rotary", "per_group_transpose", "dsv3_fused_a_gemm", "residual_gate_add",
    "state_passing", "hc_head", "moe_sum_reduce", "dcp_lse_combine", "moe_fused_mul_sum",
    "apply_token_bitmask", "sgemm_lora_b", "silu_and_mul_masked"]


def unpack(inp):
    args, kwargs = [], {}
    for item in inp if isinstance(inp, tuple) else (inp,):
        if isinstance(item, (torch.Tensor, int, float, list, tuple, torch.dtype)) or item is None:
            args.append(item)
        elif isinstance(item, dict):
            kwargs.update(item)
    return args, kwargs


def tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from tensors(o)
    elif hasattr(obj, "__dataclass_fields__"):
        for f in obj.__dataclass_fields__:
            yield from tensors(getattr(obj, f))


rows = []
for op in OPS:
    try:
        tm = importlib.import_module(f"benchmark.test_{op}")
        wl = tm.workload
        cand = get_candidate(op, reference=get_reference(op))
    except Exception as e:  # noqa: BLE001
        print(f"{op}: cannot load ({str(e)[:120]})", flush=True)
        continue
    impl = "rdna3" if "rdna3" in getattr(cand, "__module__", "") else "generic"
    inplace = getattr(wl, "IS_INPLACE", False)
    for shape in (*wl.SHAPES, *getattr(wl, "MORE_SHAPES", [])):
        for dtype in wl.DTYPES:
            for plan in tm._case_fn(shape, dtype):
                key = str(shape)
                try:
                    args, kwargs = unpack(tm._build_inputs(plan, dtype, "cuda"))
                    fn = lambda: cand(*args, **kwargs)  # noqa: E731
                    out = fn()
                    torch.cuda.synchronize()
                    in_bytes = sum(t.numel() * t.element_size() for t in tensors((args, kwargs)))
                    outs = list(tensors(out))
                    if outs:
                        out_bytes = sum(t.numel() * t.element_size() for t in outs)
                    else:
                        first = next(tensors((args, kwargs)), None)
                        out_bytes = first.numel() * first.element_size() if inplace and first is not None else 0
                    nbytes = max(in_bytes + out_bytes, 8)
                    t_op = do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3
                    src = torch.empty(max(nbytes // 4, 1), dtype=torch.bfloat16, device="cuda")
                    dst = torch.empty_like(src)
                    t_copy = do_bench(lambda: dst.copy_(src), warmup=25, rep=100, return_mode="median") * 1e3
                    for _ in range(5):
                        fn()
                    torch.cuda.synchronize()
                    n = 50
                    t0 = time.perf_counter()
                    for _ in range(n):
                        fn()
                    torch.cuda.synchronize()
                    t_cont = (time.perf_counter() - t0) / n * 1e6
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        fn()
                        torch.cuda.synchronize()
                    kernels = sum(1 for e in prof.events() if e.device_type.name == "CUDA")
                    row = dict(op=op, impl=impl, dtype=str(dtype).split(".")[-1], shape=key, t_op=t_op,
                               t_copy=t_copy, ratio=t_op / t_copy, t_cont=t_cont,
                               host=max(0.0, 1 - t_op / t_cont) if t_cont > 0 else 0.0,
                               kernels=kernels, mbytes=nbytes / 1e6)
                    rows.append(row)
                    print(f"{op:34s} {impl:7s} {row['dtype']:8s} {key[:40]:40s} {t_op:8.1f}us copy {t_copy:7.1f} "
                          f"ratio {row['ratio']:5.2f} cont {t_cont:7.1f} host {row['host']:4.0%} k{kernels} "
                          f"{row['mbytes']:.2f}MB", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{op} {key[:40]}: ERROR {str(e)[:140]}", flush=True)
json.dump(rows, open("/tmp/b4_survey2.json", "w"), indent=1)
