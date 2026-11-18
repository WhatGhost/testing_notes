## Summary
Speed up four gfx1100 (`_amd/rdna3`) ops at Qwen3.6-27B / 35B-A3B shapes. Against upstream master: MoE experts up to 9.4×, head_dim=256 extend attention 1.9–3.6×, decode GEMMs up to 2.0× (LM head up to 2.1×), `silu_and_mul` up to 1.7×. Public APIs, shared tests and references unchanged; the official benchmark shapes stay at 0.99–1.09×.

## Changes
- **silu_and_mul**: tile the flattened output (512 elements per 16-warp program) instead of one program per row, so short rows (d=128) share a program and long rows (d=4352) need no padding to a power of two. Bit-identical.
- **fused_moe_gemm**:
  - Small experts (`N·K < 2^20`, e.g. up 256×2048, down 2048×128) now take the grouped GEMM once the pairs read more than 16 MB of weights and either K ≥ 1024 or there are ≥ 8 pairs per expert, with their own tile configs. Existing configs unchanged.
  - For K ≤ 1024 the grouped kernel regroups the WMMA accumulator with `tl.split`/`tl.join` before the store (16-byte instead of 2-byte stores). Bit-identical.
- **tiny_n_gemm**: choose the split-K tile and K steps per call (`_splitk_config`) from the weight size and the rows, instead of a fixed 16×16 tile with 3 K steps. Weights well past the 96 MB Infinity Cache (e.g. an LM head) take one row block for up to 64 rows; small weights or ≤ 16 rows take the widest N tile that still leaves enough work. The M=1 GEMV path and the official benchmark shapes keep their configs.
- **extend_attention**: the packed GQA kernel now also covers bf16/fp16 with head_dim ≤ 256 (was ≤ 128), choosing a 16-row or a 128-row tile from the estimated K/V work. head_dim ≤ 128 and fp32 unchanged.

## Perf
W7900D / gfx1100, bf16, vs upstream master `5c2ed0f`. Ops called directly at Qwen3.6-27B / 35B-A3B per-GPU shapes; master and this PR alternated for 3 rounds, medians. eager: `triton.testing.do_bench` with cache flush. graph: per-call time inside a CUDA graph with a 128 MB cache flush before each call (decode sizes).

### MoE experts: `fused_moe_gemm` (up) → `silu_and_mul` → `fused_moe_gemm` (down) → top-k sum
E=256, top-8, H=2048, inter=128 (Qwen3.6-35B-A3B per GPU).

| M | master | this PR | speedup (eager) | speedup (graph) |
|---:|---:|---:|---:|---:|
| 1 | 79.8 µs | 79.7 µs | 1.00× | 1.01× |
| 2 | 128 µs | 129 µs | 1.00× | 0.99× |
| 4 | 228 µs | 199 µs | **1.14×** | **1.26×** |
| 8 | 429 µs | 325 µs | **1.32×** | **1.36×** |
| 16 | 1120 µs | 433 µs | **2.58×** | **2.87×** |
| 32 | 1679 µs | 574 µs | **2.92×** | **3.10×** |
| 64 | 1491 µs | 723 µs | **2.06×** | **2.17×** |
| 128 | 1689 µs | 811 µs | **2.08×** | — |
| 256 | 2359 µs | 851 µs | **2.77×** | — |
| 1024 | 5460 µs | 1008 µs | **5.42×** | — |
| 4096 | 19479 µs | 2067 µs | **9.42×** | — |

### `extend_attention`, head_dim=256, bf16
Per GPU: Qwen3.6-27B has 6 query / 1 KV heads, Qwen3.6-35B-A3B 4 / 1. Eager timing.

| model | batch | extend | prefix | master | this PR | speedup |
|---|---:|---:|---:|---:|---:|---:|
| 27B | 1 | 1024 | 0 | 592 µs | 227 µs | **2.61×** |
| 27B | 1 | 4096 | 0 | 3092 µs | 1355 µs | **2.28×** |
| 27B | 4 | 1024 | 0 | 1157 µs | 603 µs | **1.92×** |
| 27B | 1 | 4096 | 12288 | 19921 µs | 9901 µs | **2.01×** |
| 35B-A3B | 1 | 1024 | 0 | 543 µs | 152 µs | **3.57×** |
| 35B-A3B | 1 | 4096 | 0 | 2283 µs | 1078 µs | **2.12×** |
| 35B-A3B | 4 | 1024 | 0 | 920 µs | 421 µs | **2.19×** |
| 35B-A3B | 1 | 4096 | 12288 | 12036 µs | 6400 µs | **1.88×** |

### `tiny_n_gemm`, decode GEMMs
Each cell: speedup with CUDA graph / eager timing. M=1 keeps the unchanged GEMV path.

| model | layer | N × K | M=2 | M=4 | M=8 | M=16 | M=32 | M=64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 27B | qkv | 3584 × 5120 | 1.01 / 0.98 | 1.02 / 0.98 | 1.04 / 1.00 | 1.05 / 1.03 | 1.08 / 1.11 | 1.19 / 1.13 |
| 27B | o_proj | 5120 × 1536 | 1.33 / 1.11 | 1.31 / 1.11 | 1.20 / 1.08 | 1.21 / 1.05 | 1.30 / 1.07 | 1.22 / 1.25 |
| 27B | gdn_qkvz | 4096 × 5120 | 0.96 / 1.01 | 0.96 / 1.01 | 0.96 / 1.00 | 0.95 / 1.05 | 1.04 / 1.30 | 1.37 / 1.24 |
| 27B | gdn_ba | 24 × 5120 | 0.98 / 1.00 | 1.00 / 1.00 | 0.99 / 1.01 | 1.00 / 1.01 | 0.99 / 1.01 | 1.02 / 1.01 |
| 27B | gate_up | 8704 × 5120 | 0.99 / 1.01 | 0.99 / 1.03 | 0.99 / 1.06 | 1.03 / 1.11 | 1.24 / 1.11 | 1.92 / 1.53 |
| 27B | down | 5120 × 4352 | 0.97 / 1.02 | 0.97 / 1.03 | 0.98 / 1.03 | 0.98 / 1.08 | 1.11 / 1.13 | 1.56 / 1.35 |
| 35B-A3B | qkv | 2560 × 2048 | 1.25 / 1.15 | 1.30 / 1.17 | 1.37 / 1.20 | 1.39 / 1.19 | 1.80 / 1.31 | 1.91 / 1.35 |
| 35B-A3B | o_proj | 2048 × 1024 | 1.69 / 1.35 | 1.70 / 1.36 | 1.71 / 1.38 | 1.74 / 1.37 | 2.00 / 1.37 | 1.67 / 1.18 |
| 35B-A3B | gdn_qkvz | 3072 × 2048 | 1.57 / 1.27 | 1.58 / 1.29 | 1.60 / 1.30 | 1.63 / 1.29 | 1.69 / 1.22 | 1.35 / 1.09 |
| 35B-A3B | gdn_ba | 16 × 2048 | 1.11 / 1.08 | 1.11 / 1.08 | 1.09 / 1.08 | 1.09 / 1.08 | 1.12 / 1.07 | 1.10 / 1.08 |
| 35B-A3B | router / shared gate_up | 256 × 2048 | 1.10 / 1.06 | 1.10 / 1.07 | 1.12 / 1.08 | 1.15 / 1.10 | 1.15 / 1.09 | 1.16 / 1.07 |
| 35B-A3B | shared_down | 2048 × 128 | 1.01 / 0.99 | 1.00 / 1.00 | 0.99 / 1.00 | 0.99 / 1.00 | 0.98 / 1.01 | 0.99 / 1.00 |
| 35B-A3B | shared expert gate | 1 × 2048 | 1.14 / 1.11 | 1.18 / 1.11 | 1.16 / 1.11 | 1.15 / 1.11 | 1.10 / 1.07 | 1.09 / 1.08 |

27B `gdn_qkvz` at M=2–16 is about 4% slower with graph timing (1.00–1.05× eager): a long K keeps three K steps per program, which is the faster choice with eager timing.

### `tiny_n_gemm`, LM head (N=62080)

| model | K | M | master (graph) | this PR (graph) | speedup (graph) | speedup (eager) |
|---|---:|---:|---:|---:|---:|---:|
| 27B | 5120 | 1 | 836 µs | 835 µs | 1.00× | 1.00× |
| 27B | 5120 | 4 | 942 µs | 914 µs | 1.03× | 1.04× |
| 27B | 5120 | 16 | 1058 µs | 1016 µs | 1.04× | **1.07×** |
| 27B | 5120 | 32 | 2046 µs | 1074 µs | **1.91×** | **1.83×** |
| 35B-A3B | 2048 | 1 | 380 µs | 381 µs | 1.00× | 1.00× |
| 35B-A3B | 2048 | 4 | 607 µs | 406 µs | **1.49×** | **1.36×** |
| 35B-A3B | 2048 | 16 | 654 µs | 480 µs | **1.36×** | **1.22×** |
| 35B-A3B | 2048 | 32 | 957 µs | 459 µs | **2.09×** | **1.87×** |

### `silu_and_mul`, bf16

| input | M | master | this PR | speedup (eager) | speedup (graph) |
|---|---:|---:|---:|---:|---:|
| 27B MLP, d=4352 | 1 | 13.5 µs | 12.4 µs | **1.09×** | **1.19×** |
| 27B MLP, d=4352 | 8 | 13.8 µs | 12.8 µs | **1.08×** | **1.09×** |
| 27B MLP, d=4352 | 64 | 17.8 µs | 15.8 µs | **1.12×** | **1.13×** |
| 27B MLP, d=4352 | 1024 | 106 µs | 80.2 µs | **1.32×** | — |
| 27B MLP, d=4352 | 4096 | 320 µs | 280 µs | **1.15×** | — |
| 35B-A3B experts, d=128 | 1 | 12.3 µs | 12.3 µs | 1.00× | 1.01× |
| 35B-A3B experts, d=128 | 8 | 12.3 µs | 12.3 µs | 1.00× | 1.00× |
| 35B-A3B experts, d=128 | 64 | 12.4 µs | 12.4 µs | 1.00× | 1.00× |
| 35B-A3B experts, d=128 | 1024 | 13.8 µs | 13.8 µs | 1.00× | — |
| 35B-A3B experts, d=128 | 4096 | 27.1 µs | 19.8 µs | **1.37×** | — |
| 35B-A3B experts, d=128 | 32768 | 134 µs | 76.5 µs | **1.74×** | — |

### Official benchmark (`benchmark/`, comprehensive, `--mode kernel`, bf16)

| op | shapes | geomean | range |
|---|---:|---:|---:|
| `silu_and_mul` | 15 | 1.024× | 0.99–1.09× |
| `tiny_n_gemm` | 10 | 0.998× | 0.99–1.00× |
| `fused_moe_gemm` | 8 | 0.999× | 0.99–1.00× |
| `extend_attention` | 7 | 0.998× | 0.99–1.00× |

## Test plan
- [x] `tests/test_{silu_and_mul,fused_moe_gemm,tiny_n_gemm,extend_attention}.py` on gfx1100, with the rdna3 ops and with `ARCH=none`
- [x] Extra checks against the references: `extend_attention` head_dim 192/256 (group 1–8, varlen, both tiles forced); `tiny_n_gemm` 19 shapes × M=1–100, including non-power-of-two M and K, vs an fp32 matmul; the split/join store is bit-identical
- [x] Official comprehensive bench (table above)
- [x] black / flake8 / isort

Made with [Cursor](https://cursor.com)
