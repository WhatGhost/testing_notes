#!/usr/bin/env python3
"""Survey batch10 / competition ops with the round-3 W7900D metric.

Primary metric: op do_bench median / same-byte copy do_bench median.
Also reports host-side fraction for shapes with wall >= 30 µs, and max
VGPR spill bytes/lane from the Triton AMDGCN listing after the run.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import re
import statistics
import sys
import time
import traceback
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import triton

ROOT = pathlib.Path("/tmp/fgs-master-new")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import flaggems_sglang  # noqa: E402

OPS = [
    "causal_conv3d_cat_pad",
    "chunk_kda_scaled_dot_kkt",
    "fused_append_shared_experts",
    "gemma_dual_rmsnorm_residual_scalar",
    "gemma_qkv_rmsnorm",
    "gemma_rmsnorm",
    "gemma_rmsnorm_residual_scalar",
    "grouped_gemm",
    "kda_gate_chunk_cumsum",
    "kda_prologue",
    "normalize_and_patchify",
    "step_a_q",
    "step_a_v",
    "step_b_q",
    "step_b_v",
    "swiglu_oai_split",
]

# Ops whose cost is primarily compute (not bandwidth). Ratio vs copy is
# diagnostic only; they go into the compute bucket like round-3.
COMPUTE_OPS = {
    "grouped_gemm",
    "chunk_kda_scaled_dot_kkt",
    "step_a_q",
    "step_a_v",
    "step_b_q",
    "step_b_v",
    "kda_prologue",
    "kda_gate_chunk_cumsum",
}

MIN_US = 30.0
WARMUP = 100
REP = 100
HOST_ITERS = 20
OUT = pathlib.Path("/tmp/survey_batch10_results.jsonl")
CACHE_DIR = pathlib.Path(
    os.environ.get("TRITON_CACHE_DIR", str(pathlib.Path.home() / ".triton" / "cache"))
)


def do_bench(fn: Callable[[], Any], warmup: int = WARMUP, rep: int = REP) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def tensor_nbytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_nbytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(tensor_nbytes(v) for v in obj.values())
    return 0


def collect_tensors(obj: Any) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out.extend(collect_tensors(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(collect_tensors(v))
    return out


def unpack_call_args(sample: Any) -> Tuple[tuple, dict]:
    """Match FlagGems unpack: trailing dict becomes kwargs."""
    if not isinstance(sample, (list, tuple)):
        return (sample,), {}
    args = list(sample)
    kwargs = {}
    if args and isinstance(args[-1], dict):
        kwargs = args.pop()
    return tuple(args), kwargs


def copy_latency_for_bytes(n_bytes: int, device: torch.device) -> float:
    """Time a device copy of roughly n_bytes (bf16 vector round-trip-ish)."""
    if n_bytes <= 0:
        return float("nan")
    n = max(int(n_bytes // 2), 1)  # bf16 elements
    # Cap to keep memory sane on tiny ops; floor so copy isn't launch-dominated
    # differently from the op for large shapes.
    n = min(n, 256 * 1024 * 1024)  # <= 512 MiB bf16
    src = torch.empty(n, device=device, dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    return do_bench(lambda: dst.copy_(src))


def host_fraction(fn: Callable[[], Any], iters: int = HOST_ITERS) -> Tuple[float, float, float]:
    """Return (wall_us, gpu_us, host_frac) using CUDA events for GPU time."""
    # Warm
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start_evt.record()
    for _ in range(iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    wall_us = (t1 - t0) * 1e6 / iters
    gpu_us = start_evt.elapsed_time(end_evt) * 1e3 / iters  # ms -> µs
    host = max(wall_us - gpu_us, 0.0)
    frac = host / wall_us if wall_us > 0 else 0.0
    return wall_us, gpu_us, frac


def max_spill_bytes(cache_dir: pathlib.Path, since_mtime: float) -> Optional[int]:
    """Parse AMDGCN listings written after since_mtime for max VGPR spill B/lane."""
    if not cache_dir.exists():
        return None
    # Patterns seen in AMD backend dumps
    pats = [
        re.compile(r"ScratchSize:\s*(\d+)"),
        re.compile(r"vgpr_spill_count:\s*(\d+)"),
        re.compile(r";\s*spilledvgprs:\s*(\d+)", re.I),
        re.compile(r"NumBytesOfScratchWorkItems:\s*(\d+)"),
    ]
    best = 0
    found = False
    for path in cache_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < since_mtime:
                continue
        except OSError:
            continue
        if path.suffix not in {".amdgcn", ".ll", ".ttir", ".ttgir", ""} and "amdgcn" not in path.name.lower():
            # still try .json meta
            if path.suffix not in {".json", ".txt"}:
                continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for pat in pats:
            for m in pat.finditer(text):
                found = True
                best = max(best, int(m.group(1)))
        # wave32: ScratchSize is often total per workitem already in bytes
    return best if found else None


def load_bench_module(op: str):
    return importlib.import_module(f"benchmark.test_{op}")


def shapes_of(mod) -> List[tuple]:
    shapes = [tuple(s) for s in getattr(mod, "SHAPES", [])]
    more = [tuple(s) for s in getattr(mod, "MORE_SHAPES", [])]
    # comprehensive = both
    return list(dict.fromkeys(shapes + more))


def dtype_of(mod) -> torch.dtype:
    # Most new ops bench bf16; a few may only expose via OpBenchmark.dtypes
    # Heuristic: prefer bf16.
    return torch.bfloat16


def survey_op(op: str, device: torch.device) -> List[dict]:
    rows = []
    mod = load_bench_module(op)
    input_fn = getattr(mod, "_input_fn", None)
    if input_fn is None:
        raise RuntimeError(f"{op}: no _input_fn")
    fn = getattr(flaggems_sglang, op)
    dtype = dtype_of(mod)
    shapes = shapes_of(mod)
    print(f"\n== {op}  shapes={len(shapes)}  compute={op in COMPUTE_OPS}", flush=True)

    for shape in shapes:
        row = {
            "op": op,
            "shape": list(shape) if not isinstance(shape, (str,)) else shape,
            "dtype": str(dtype).replace("torch.", ""),
            "compute": op in COMPUTE_OPS,
        }
        try:
            sample = next(input_fn(shape, dtype, device))
            args, kwargs = unpack_call_args(sample)

            # Probe once for output bytes / correctness of call
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            in_bytes = tensor_nbytes(args) + tensor_nbytes(kwargs)
            out_bytes = tensor_nbytes(out)
            # Round-3 "real bytes": prefer write volume; use max(in, out) as
            # conservative traffic proxy when both present.
            traffic = max(in_bytes + out_bytes, out_bytes, 1)

            def run():
                return fn(*args, **kwargs)

            # Flush / warmup lightly then official-ish do_bench
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            lat_us = do_bench(run) * 1e3  # ms -> µs
            copy_us = copy_latency_for_bytes(traffic, device) * 1e3
            ratio = lat_us / copy_us if copy_us > 0 else float("nan")

            wall_us = gpu_us = host_frac = float("nan")
            if lat_us >= MIN_US * 0.5:  # still measure near-threshold
                wall_us, gpu_us, host_frac = host_fraction(run)

            row.update(
                {
                    "ok": True,
                    "latency_us": lat_us,
                    "copy_us": copy_us,
                    "ratio": ratio,
                    "in_bytes": in_bytes,
                    "out_bytes": out_bytes,
                    "traffic_bytes": traffic,
                    "wall_us": wall_us,
                    "gpu_us": gpu_us,
                    "host_frac": host_frac,
                }
            )
            flag = ""
            if lat_us < MIN_US:
                flag = "small"
            elif op in COMPUTE_OPS:
                flag = "compute"
            elif ratio <= 1.15:
                flag = "near_copy"
            elif ratio < 1.5:
                flag = "1.15-1.5"
            elif ratio < 2.0:
                flag = "1.5-2"
            else:
                flag = ">2"
            row["bucket"] = flag
            print(
                f"  {shape}: {lat_us:8.1f} µs  copy {copy_us:7.1f}  "
                f"ratio {ratio:5.2f}  host {host_frac:4.0%}  [{flag}]",
                flush=True,
            )
        except Exception as e:
            row.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
            print(f"  {shape}: FAIL {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        rows.append(row)
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
    return rows


def summarize(all_rows: List[dict]) -> dict:
    by_op = {}
    for r in all_rows:
        by_op.setdefault(r["op"], []).append(r)

    summary = {}
    for op, rows in by_op.items():
        ok = [r for r in rows if r.get("ok")]
        big = [r for r in ok if r.get("latency_us", 0) >= MIN_US]
        ratios = [r["ratio"] for r in big if r.get("ratio") == r.get("ratio")]
        hosts = [r["host_frac"] for r in big if r.get("host_frac") == r.get("host_frac")]
        if not ok:
            bucket = "fail"
        elif not big:
            bucket = "small_only"
        elif op in COMPUTE_OPS:
            bucket = "compute"
        else:
            # Use worst (max) ratio among large shapes — matches "有空间" sorting.
            worst = max(ratios) if ratios else float("nan")
            if worst <= 1.15:
                bucket = "near_copy"
            elif worst < 1.5:
                bucket = "batch4_1.15-1.5"
            elif worst < 2.0:
                bucket = "batch3_1.5-2"
            else:
                bucket = "batch3_>2"
        summary[op] = {
            "bucket": bucket,
            "n_shapes": len(rows),
            "n_ok": len(ok),
            "n_ge_30us": len(big),
            "max_ratio": (max(ratios) if ratios else None),
            "min_ratio": (min(ratios) if ratios else None),
            "max_host_frac": (max(hosts) if hosts else None),
            "max_latency_us": (max(r["latency_us"] for r in ok) if ok else None),
            "errors": [r.get("error") for r in rows if not r.get("ok")],
        }
    return summary


def main():
    device = torch.device("cuda:0")
    assert torch.cuda.is_available(), "cuda required"
    print("device:", torch.cuda.get_device_name(0), flush=True)
    print("cache:", CACHE_DIR, flush=True)
    if OUT.exists():
        OUT.unlink()

    t0 = time.time()
    all_rows: List[dict] = []
    for op in OPS:
        all_rows.extend(survey_op(op, device))

    summary = summarize(all_rows)
    spill = max_spill_bytes(CACHE_DIR, t0)
    out_summary = {
        "spill_bytes_lane_max_seen": spill,
        "ops": summary,
        "device": torch.cuda.get_device_name(0),
        "min_us": MIN_US,
        "warmup": WARMUP,
        "rep": REP,
    }
    pathlib.Path("/tmp/survey_batch10_summary.json").write_text(
        json.dumps(out_summary, indent=2)
    )
    print("\n==== SUMMARY ====", flush=True)
    for op, s in summary.items():
        print(
            f"{op:40s} {s['bucket']:16s} "
            f"max_ratio={s['max_ratio']}  max_host={s['max_host_frac']}  "
            f"max_lat={s['max_latency_us']}",
            flush=True,
        )
    print("spill_max:", spill, flush=True)
    print("wrote", OUT, "and /tmp/survey_batch10_summary.json", flush=True)


if __name__ == "__main__":
    main()
