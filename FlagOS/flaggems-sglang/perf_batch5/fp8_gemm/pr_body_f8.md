## Summary
Add gfx1100 (`_amd/rdna3`) versions of the two FP8 GEMMs. Against upstream master on the official benchmark's shapes and inputs: `w8a8_block_fp8_matmul` 1.00–5.16× (1.55–2.17× from 8 to 64 rows, 3.16–5.16× from 512 rows, 16.5× and 23.0× on the two K = 512 shapes), `triton_scaled_mm` 1.00–2.82× (1.27–1.90× from 64 rows). Products stay exact (e4m3 decodes exactly to fp16) and scales are applied in float32, as before. Two input cases the generic ops get wrong now work: a strided `A` or `B` in `w8a8_block_fp8_matmul`, and a broadcast `[1, N]` bias in `triton_scaled_mm`. Public APIs, shared tests and references unchanged.

## Changes
- **w8a8_block_fp8_matmul** (new `_amd/rdna3` file). With 128 × 128 scale blocks and K, N multiples of 128:
  - Up to 192 rows (384 when N ≤ 2048): `A` is decoded to fp16 once and `B` stays fp8, decoded in registers four K values per 32-bit load as the left operand of `B @ A^T`. Each 128-wide K block is summed on the matrix units on its own, scaled by `As * Bs` and added with an fma; K blocks are split across programs when the grid is small, and a second kernel adds the parts.
  - Above: both operands are decoded to fp16 once (rows padded by 32 elements) and multiplied in 128 × 128 tiles. The per-block partial sum is what kept master's tiles at 128 × 64 with 4 warps and one stage; here the accumulator is kept as two 64-column halves that share the A tile, so the partial needs half a tile of registers.
  - One row against at most 2048 columns, other block sizes, and K or N not multiples of 128 use the generic kernels. Master's small-K path (K ≤ 1024) took 300–490 µs on the two benchmark shapes with K = 512; those now take the paths above.
  - `A` and `B` are made contiguous first: the generic kernels index them as dense rows, so a strided operand gave wrong results.
- **triton_scaled_mm** (new `_amd/rdna3` file). FP8 inputs with K, N multiples of 128:
  - Up to 640 rows: the weight stays fp8 and is widened in registers, four columns per 32-bit load, against an activation widened to fp16 once; small grids split K across programs and add the parts in a second kernel.
  - Above 640 rows: both operands are widened once and multiplied in 128 × 128 × 64 tiles with a two-stage pipeline (master: one stage).
  - N and K are compile-time constants (weight shapes are fixed), M is not.
  - A `[N]`, `[1, N]` or `[M, 1]` bias broadcasts like the reference; master reads past the end of a `[1, N]` bias for M > 1 (wrong values, or a memory fault).
  - One row, int8 and other shapes use the generic kernels.

## Perf
W7900D / gfx1100, vs upstream master `87e9acc`. The official benchmark harness cannot run these two ops at the moment: their `_input_fn` yields the keyword arguments as a dict, which `unpack_to_args_kwargs` iterates as keys, so the op is called without arguments (31 other ops fail the same way). The numbers below use the official benchmark's shapes and inputs, timed with `triton.testing.do_bench`, master and this PR alternated in the same process for 3 rounds, medians.

### `w8a8_block_fp8_matmul` (official benchmark shapes and inputs, bf16 output, 128 x 128 blocks)

| M | N | K | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 4096 | 35.3 µs | 35.2 µs | 1.00× |
| 1 | 4096 | 4096 | 128.1 µs | 113.2 µs | **1.13×** |
| 1 | 7168 | 4096 | 254.4 µs | 161.3 µs | **1.58×** |
| 8 | 1024 | 4096 | 61.4 µs | 39.3 µs | **1.56×** |
| 8 | 4096 | 4096 | 179.7 µs | 116.2 µs | **1.55×** |
| 8 | 7168 | 4096 | 298.0 µs | 164.5 µs | **1.81×** |
| 64 | 1024 | 4096 | 89.8 µs | 48.2 µs | **1.86×** |
| 64 | 4096 | 4096 | 273.7 µs | 133.3 µs | **2.05×** |
| 64 | 7168 | 4096 | 406.9 µs | 187.5 µs | **2.17×** |
| 512 | 1024 | 4096 | 501.1 µs | 158.7 µs | **3.16×** |
| 512 | 4096 | 4096 | 1657.0 µs | 419.6 µs | **3.95×** |
| 512 | 7168 | 4096 | 2726.6 µs | 677.2 µs | **4.03×** |
| 4096 | 1024 | 4096 | 2696.9 µs | 666.4 µs | **4.05×** |
| 4096 | 4096 | 4096 | 9783.9 µs | 2107.9 µs | **4.64×** |
| 4096 | 7168 | 4096 | 18265.2 µs | 3540.0 µs | **5.16×** |
| 7 | 256 | 512 | 301.2 µs | 18.3 µs | **16.48×** |
| 64 | 1024 | 512 | 483.9 µs | 21.0 µs | **23.00×** |
| 256 | 1024 | 4096 | 136.0 µs | 111.6 µs | **1.22×** |

### `triton_scaled_mm` (official benchmark shapes and inputs, FP8 inputs, fp16 output)

| M | K | N | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 4096 | 67.9 µs | 67.9 µs | 1.00× |
| 1 | 4096 | 7168 | 120.6 µs | 120.3 µs | 1.00× |
| 8 | 4096 | 4096 | 189.2 µs | 67.2 µs | **2.82×** |
| 8 | 4096 | 7168 | 102.6 µs | 102.7 µs | 1.00× |
| 64 | 4096 | 4096 | 128.1 µs | 93.2 µs | **1.37×** |
| 64 | 4096 | 7168 | 163.8 µs | 129.3 µs | **1.27×** |
| 512 | 4096 | 4096 | 528.2 µs | 380.2 µs | **1.39×** |
| 512 | 4096 | 7168 | 876.6 µs | 603.8 µs | **1.45×** |
| 4096 | 4096 | 4096 | 3337.4 µs | 1946.5 µs | **1.71×** |
| 4096 | 4096 | 7168 | 6124.4 µs | 3227.3 µs | **1.90×** |
| 32 | 32 | 32 | 18.1 µs | 18.1 µs | 1.00× |
| 17 | 64 | 96 | 18.0 µs | 18.0 µs | 1.00× |

## Test plan
- [x] `tests/test_{w8a8_block_fp8_matmul,triton_scaled_mm}.py` with `ARCH` unset and `ARCH=none` (both import the generic modules)
- [x] Dispatch on gfx1100 resolves both ops to the new `_amd/rdna3` files
- [x] `w8a8_block_fp8_matmul` against the reference, 56 cases: the unit-test cases and benchmark shapes, M from 1 to 4096 around the path boundaries, bf16 / fp16 / fp32 output, other block sizes and odd K / N (generic fallback), strided `A` and `B`; all within the unit-test check, with max error at most one output ulp of the largest magnitude (or 1.5× master's)
- [x] `triton_scaled_mm` against the reference and master, 78 cases: the unit-test cases and benchmark shapes, M edges, fp16 / bf16 / fp32 output, scalar scales, `[1, N]` and `[M, N]` bias, odd K / N, int8; all within the unit-test tolerance, max error at most 4e-3 of the largest magnitude, and at most 1.3e-4 from master
- [x] black / flake8 / isort
