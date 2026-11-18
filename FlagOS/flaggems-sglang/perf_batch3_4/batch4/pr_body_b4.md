## Summary
Add `_amd/rdna3` (gfx1100) implementations of `fused_softmax_pool`, `ltx2_split_rotary`, `dsv3_fused_a_gemm`, `embedding_lora_a` and `apply_token_bitmask`, which on master run the generic ops, and split rows in the existing gfx1100 `per_group_transpose` for narrow, tall groups. Against upstream master on the official benchmark: `fused_softmax_pool` up to 2.87×, `per_group_transpose` up to 1.58×, `ltx2_split_rotary` up to 1.30×, `dsv3_fused_a_gemm` 1.16–1.29× for M ≥ 4, `apply_token_bitmask` up to 1.15×, `embedding_lora_a` up to 1.09× (up to 1.69× at larger ranks). Public APIs, shared tests and references unchanged.

## Changes
- **fused_softmax_pool** (new `_amd/rdna3` file; the generic kernels, trimmed to the HIP paths): for k = 64 the two-pass kernel runs one program per row and takes ~42 µs at any row count up to 64. 5–127 rows now use the online-merge kernel over 32-column slices, and its 128-column form (already used from 1024 rows) starts at 128 rows. k = 16 uses the 128-column online merge from 2048 rows. Other paths unchanged.
- **per_group_transpose** (existing `_amd/rdna3` file): with k ≤ 256 each group has at most two column blocks, so groups averaging 128+ rows are split over 8 programs (64×128 tiles, 8 warps); the gain is largest with uneven groups, where the largest group sets the time. Bitwise identical; other shapes keep the current launch.
- **ltx2_split_rotary** (new `_amd/rdna3` file): `x` is token-major and `cos`/`sin` are head-major, so blocks in `x` order read `cos`/`sin` as 128-byte pieces far apart. From 2^19 elements each program covers whole tokens (all heads), which reads both as long runs. Same arithmetic, bitwise identical.
- **dsv3_fused_a_gemm** (new `_amd/rdna3` file): for M ≥ 3, K > 4096 and N ≤ 3072 the generic op runs one program per 16 columns (132 for N = 2112), each streaming all of K. K is now split four ways over interleaved 256-wide blocks, followed by a small reduction. Interleaving keeps concurrent programs in one K window; contiguous splits whose starts were an even number of blocks apart ran up to 2.5× slower on gfx1100. M ≤ 2 and other shapes keep the generic kernels.
- **embedding_lora_a** (new `_amd/rdna3` file): the weights are rank-major, so every value is a separate gather. With rank ≥ 64 and 2048+ tokens, 128-token tiles over the whole rank (2 warps) keep more gathers in flight. Bitwise identical.
- **apply_token_bitmask** (new `_amd/rdna3` file): from about 1.3M elements, 8192-wide blocks with 4 warps instead of 2048-wide with 8. Bitwise identical.

## Perf
W7900D / gfx1100, vs upstream master `5c2ed0f`. Official benchmark: comprehensive, `--mode kernel`, master and this PR alternated for 3 rounds, medians. Additional shapes: `triton.testing.do_bench` medians, master and this PR in the same process.

### `fused_softmax_pool`, official benchmark (head_dim 128)

| rows | k | master | this PR | speedup |
|---:|---:|---:|---:|---:|
| 1 | 4 | 8.9 µs | 8.9 µs | 1.00× |
| 1 | 16 | 9.1 µs | 9.0 µs | 1.02× |
| 1 | 64 | 9.5 µs | 9.5 µs | 1.00× |
| 64 | 4 | 9.1 µs | 9.0 µs | 1.01× |
| 64 | 16 | 10.2 µs | 10.0 µs | 1.02× |
| 64 | 64 | 43.7 µs | 15.2 µs | **2.87×** |
| 1024 | 4 | 13.9 µs | 14.3 µs | 0.97× |
| 1024 | 16 | 31.6 µs | 31.7 µs | 1.00× |
| 1024 | 64 | 98.0 µs | 98.3 µs | 1.00× |
| 4096 | 4 | 32.3 µs | 32.3 µs | 1.00× |
| 4096 | 16 | 109.0 µs | 97.4 µs | **1.12×** |
| 4096 | 64 | 345.7 µs | 345.7 µs | 1.00× |

### `fused_softmax_pool`, additional shapes (head_dim 128)

| rows | k | master | this PR | speedup |
|---:|---:|---:|---:|---:|
| 8 | 64 | 41.6 µs | 10.9 µs | **3.82×** |
| 16 | 64 | 41.5 µs | 11.2 µs | **3.72×** |
| 32 | 64 | 42.0 µs | 11.9 µs | **3.52×** |
| 128 | 64 | 47.2 µs | 23.8 µs | **1.98×** |
| 256 | 64 | 67.4 µs | 35.2 µs | **1.92×** |
| 512 | 64 | 120.5 µs | 58.9 µs | **2.05×** |
| 2048 | 16 | 61.0 µs | 54.1 µs | **1.13×** |
| 8192 | 16 | 248.2 µs | 195.9 µs | **1.27×** |
| 16384 | 16 | 428.2 µs | 353.8 µs | **1.21×** |

### `per_group_transpose`, official benchmark (8 groups, bf16)

| k | rows per group | master | this PR | speedup |
|---:|---:|---:|---:|---:|
| 128 | 16 | 9.3 µs | 9.3 µs | 1.00× |
| 128 | 128 | 10.8 µs | 10.3 µs | **1.05×** |
| 128 | 1024 | 26.9 µs | 17.0 µs | **1.58×** |
| 512 | 16 | 9.4 µs | 9.5 µs | 0.99× |
| 512 | 128 | 12.5 µs | 12.4 µs | 1.00× |
| 512 | 1024 | 57.4 µs | 57.4 µs | 1.00× |
| 4096 | 16 | 12.5 µs | 12.6 µs | 0.99× |
| 4096 | 128 | 46.1 µs | 46.2 µs | 1.00× |
| 4096 | 1024 | 508.8 µs | 509.9 µs | 1.00× |

### `per_group_transpose`, additional shapes

| dtype | groups | k | rows per group | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 8 | 128 | 256 | 12.5 µs | 10.8 µs | **1.16×** |
| bf16 | 8 | 128 | 512 | 17.0 µs | 13.0 µs | **1.31×** |
| bf16 | 8 | 128 | 2048 | 46.6 µs | 27.4 µs | **1.70×** |
| bf16 | 8 | 256 | 512 | 19.9 µs | 16.8 µs | **1.19×** |
| bf16 | 8 | 256 | 1024 | 32.0 µs | 26.8 µs | **1.20×** |
| bf16 | 8 | 256 | 2048 | 55.0 µs | 48.5 µs | **1.13×** |
| bf16 | 8 | 128 | 243 (uneven) | 15.4 µs | 11.0 µs | **1.41×** |
| bf16 | 8 | 128 | 726 (uneven) | 27.7 µs | 15.0 µs | **1.85×** |
| bf16 | 32 | 128 | 256 | 16.8 µs | 16.6 µs | 1.01× |
| bf16 | 8 | 128 | 128 | 10.7 µs | 10.3 µs | 1.04× |
| bf16 | 8 | 256 | 128 | 11.0 µs | 10.8 µs | 1.03× |
| fp32 | 8 | 128 | 128 | 11.1 µs | 11.2 µs | 0.99× |
| fp32 | 8 | 128 | 1024 | 31.0 µs | 26.0 µs | **1.19×** |
| fp32 | 8 | 128 | 2048 | 53.2 µs | 46.0 µs | **1.16×** |

### `ltx2_split_rotary`, official benchmark (batch 1, 16 heads, head_dim 128)

| tokens | master | this PR | speedup |
|---:|---:|---:|---:|
| 1 | 9.1 µs | 8.9 µs | 1.02× |
| 64 | 10.5 µs | 10.2 µs | 1.04× |
| 1024 | 42.0 µs | 36.6 µs | **1.15×** |
| 8192 | 400.9 µs | 308.0 µs | **1.30×** |

### `ltx2_split_rotary`, additional shapes

| batch | tokens | heads | head_dim | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 16 | 128 | 14.7 µs | 14.9 µs | 0.99× |
| 1 | 512 | 16 | 128 | 24.8 µs | 22.4 µs | **1.11×** |
| 1 | 4096 | 16 | 128 | 192.3 µs | 150.8 µs | **1.28×** |
| 1 | 16384 | 16 | 128 | 537.4 µs | 446.6 µs | **1.20×** |
| 1 | 8192 | 32 | 128 | 537.6 µs | 441.3 µs | **1.22×** |
| 4 | 1024 | 16 | 128 | 171.2 µs | 144.8 µs | **1.18×** |
| 2 | 512 | 8 | 128 | 23.7 µs | 23.2 µs | 1.02× |
| 1 | 2048 | 16 | 64 | 39.5 µs | 36.0 µs | **1.10×** |
| 1 | 4096 | 24 | 128 | 308.8 µs | 257.8 µs | **1.20×** |

### `dsv3_fused_a_gemm`, official benchmark (K 7168, N 2112)

| M | master | this PR | speedup |
|---:|---:|---:|---:|
| 1 | 87.4 µs | 87.2 µs | 1.00× |
| 2 | 90.0 µs | 90.2 µs | 1.00× |
| 4 | 114.8 µs | 99.3 µs | **1.16×** |
| 8 | 125.6 µs | 98.9 µs | **1.27×** |
| 16 | 128.4 µs | 99.7 µs | **1.29×** |

### `dsv3_fused_a_gemm`, additional shapes

| M | K | N | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|
| 3 | 7168 | 2112 | 112.8 µs | 99.1 µs | **1.14×** |
| 4 | 7168 | 1536 | 90.0 µs | 84.7 µs | **1.06×** |
| 16 | 7168 | 1536 | 106.0 µs | 84.7 µs | **1.25×** |
| 4 | 7168 | 576 | 45.1 µs | 38.1 µs | **1.18×** |
| 16 | 7168 | 576 | 42.7 µs | 38.8 µs | **1.10×** |
| 4 | 7168 | 3072 | 151.6 µs | 150.0 µs | 1.01× |
| 16 | 7168 | 3072 | 189.8 µs | 149.3 µs | **1.27×** |
| 8 | 5120 | 2112 | 114.8 µs | 84.8 µs | **1.35×** |
| 8 | 8192 | 2112 | 122.0 µs | 108.5 µs | **1.12×** |

### `embedding_lora_a`, official benchmark

| segments × length | adapters | rank | vocab | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 × 512 | 4 | 32 | 32000 | 35.8 µs | 35.2 µs | 1.02× |
| 4 × 2048 | 2 | 64 | 128256 | 114.3 µs | 105.0 µs | **1.09×** |
| 1 × 5 | 1 | 16 | 128 | 9.2 µs | 9.3 µs | 1.00× |
| 4 segments, 22 tokens | 2 | 32 | 512 | 9.2 µs | 9.3 µs | 0.98× |
| 2 segments, 13 tokens, 8 extra | 2 | 16 | 256 | 9.5 µs | 9.4 µs | 1.01× |

### `embedding_lora_a`, additional shapes

| segments × length | adapters | rank | vocab | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 4 × 2048 | 1 | 64 | 32000 | 34.5 µs | 23.6 µs | **1.46×** |
| 8 × 2048 | 1 | 64 | 32000 | 46.7 µs | 33.5 µs | **1.40×** |
| 1 × 4096 | 1 | 128 | 32000 | 49.6 µs | 36.8 µs | **1.35×** |
| 8 × 2048 | 1 | 128 | 32000 | 93.6 µs | 55.5 µs | **1.69×** |
| 8 × 2048 | 1 | 64 | 128256 | 98.6 µs | 70.1 µs | **1.41×** |
| 4 × 2048 | 4 | 128 | 128256 | 335.3 µs | 317.6 µs | **1.06×** |
| 1 × 2048 | 1 | 64 | 128256 | 55.0 µs | 47.7 µs | **1.15×** |

### `apply_token_bitmask`, official benchmark (fp32 logits)

| batch | vocab | master | this PR | speedup |
|---:|---:|---:|---:|---:|
| 1 | 152064 | 10.4 µs | 10.5 µs | 0.98× |
| 8 | 152064 | 27.7 µs | 27.8 µs | 1.00× |
| 64 | 152064 | 269.6 µs | 233.8 µs | **1.15×** |
| 512 | 152064 | 1135.3 µs | 1080.3 µs | **1.05×** |
| 4096 | 152064 | 7708.8 µs | 7552.1 µs | 1.02× |
| 3 | 100 | 8.7 µs | 8.5 µs | 1.02× |
| 7 | 1000 | 8.8 µs | 8.6 µs | 1.02× |
| 1 | 32000 | 9.4 µs | 9.2 µs | 1.02× |

### `apply_token_bitmask`, additional shapes (fp32 logits)

| batch | vocab | master | this PR | speedup |
|---:|---:|---:|---:|---:|
| 10 | 152064 | 35.6 µs | 33.0 µs | **1.08×** |
| 16 | 152064 | 60.7 µs | 49.6 µs | **1.22×** |
| 32 | 152064 | 130.7 µs | 102.0 µs | **1.28×** |
| 256 | 152064 | 670.9 µs | 632.9 µs | **1.06×** |
| 48 | 32000 | 37.4 µs | 34.2 µs | **1.09×** |
| 64 | 32000 | 51.2 µs | 43.3 µs | **1.18×** |
| 64 | 128256 | 227.6 µs | 194.3 µs | **1.17×** |

## Test plan
- [x] `tests/test_{fused_softmax_pool,per_group_transpose,ltx2_split_rotary,dsv3_fused_a_gemm,embedding_lora_a,apply_token_bitmask}.py` on gfx1100, with the rdna3 ops and with `ARCH=none`
- [x] Bitwise equal to master: `per_group_transpose` 390 cases (bf16 / fp16 / fp32, empty and uneven groups), `ltx2_split_rotary` 54 cases (bf16 / fp16 / fp32, head counts that are not a power of two, fallback shapes), `embedding_lora_a` 8 cases (extra embeddings, active rank below the full rank), `apply_token_bitmask` 30 cases
- [x] Against float64 references, errors no larger than master's: `fused_softmax_pool` 420 cases (including -inf scores and k = 0), `dsv3_fused_a_gemm` 98 cases (K not a multiple of 256, row-major B, strided A, M = 17 / 32)
- [x] Official comprehensive bench (tables above)
- [x] black / flake8 / isort

Made with [Cursor](https://cursor.com)
