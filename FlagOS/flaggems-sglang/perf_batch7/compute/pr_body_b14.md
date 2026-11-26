Add gfx1100 (`_amd/rdna3`) versions of three ops that have no AMD version yet, so master runs the generic ops on gfx1100: `chunk_kda_scaled_dot_kkt`, `grouped_gemm` and `absorbed_bmm_concat_cast_q_fp8`.

On W7900D / gfx1100 against upstream master `010bd92`:

- `chunk_kda_scaled_dot_kkt`: 3.01× and 2.94× on the official shapes with 1024 and 4096 tokens, 2.51–6.49× on five more shapes; the two smaller official shapes keep the generic op. From 4096 (batch, time, head) rows the generic op computes each 64 × 64 chunk, upper half included, in one 4-warp program, factoring exp2(g_i − g_j) around the chunk's middle row so that fp16 high and low parts can go to the matrix units. On gfx1100 that kernel spills 361 registers to scratch, and with 8 heads and 1024 tokens it runs 128 programs. Here each program computes one 16 × 16 (row block, column block) pair of the lower triangle in one warp without spills, 10 pairs per chunk, factoring exp2(g_i − g_j) into exp2(g_i − g_a) · exp2(g_a − g_j) around a row of its own block (its first row, or its middle row on the diagonal). The three fp16 products (about 22 bits) are each added to the fp32 sum on their own, because accumulating inside the matrix instruction truncates; this also lowers the error against the reference. A block whose factors or values would leave the fp16 range (strong decay, rising gates) is summed exactly in fp32. Below 4096 rows, for other chunk sizes, head_dim above 256 and strided inputs, the generic op runs.
- `grouped_gemm`: 1.25–1.45× on the four official shapes with 1024 and 4096 tokens, 1.13–1.83× on nine more shapes with 512–8192 tokens, 8–128 experts and (N, K) of (512, 2048), (1536, 2048) and (2048, 768); the two 128-token official shapes keep the generic op. For most shapes from 512 tokens the generic op splits K up to four ways on HIP and writes fp32 partial sums that a second kernel reads back and adds. Here each program computes one (row block of an expert, column block) tile over the whole K, the column blocks of a row block are launched next to each other so they share its rows of A in the cache, and the tile follows the rows per expert, N, K and the number of experts, fitted on a grid of 45 such shapes where none is slower than the generic op. Below 512 tokens the generic op's GEMV paths stay.
- `absorbed_bmm_concat_cast_q_fp8`: 2.18× on the official (1024 tokens, 16 heads) shape and 1.55–2.93× on four more shapes from 12288 (token, head) rows up to 2^24 outputs; the other six official shapes keep the generic op. q_nope is (token, head, k), and below 2^24 outputs the generic op walks one head over all tokens at a time, which reads q_nope in 256-byte pieces. Here the programs for the heads of a token block run next to each other. The fp8 codes come from the generic op's encoder, four per 32-bit store. The gain is in cold-cache runs, which is what `do_bench` measures; with q_nope and w_kc already in the 96 MB Infinity Cache, back-to-back calls are at parity. Below 12288 rows the CPU side of the launch dominates, and from 2^24 outputs the generic op orders the heads this way itself, so both keep the generic op, as do fp16 inputs (the shared encoder takes bf16 values).

Public APIs, shared tests and references unchanged.

## Correctness

- `pytest tests/test_chunk_kda_scaled_dot_kkt.py tests/test_grouped_gemm.py tests/test_absorbed_bmm_concat_cast_q_fp8.py`: 23 passed with the rdna3 ops and with `ARCH=none`; `get_op` resolves the three ops to the rdna3 files.
- `chunk_kda_scaled_dot_kkt`: both outputs within the unit-test tolerance of the fp32 reference on 18 cases, with max errors on par with the generic op's (2.1e-5 against 4.8e-5 for `A` at 1024 tokens): the unit-test shapes, the four official shapes, a tail chunk (1100 tokens), batch 2 with 3 heads, head_dim 32, 96 and 256, gk_scale = log2 e, strided inputs (generic path), strong and medium decay, small and large rising gates, and k scaled by 100. Strong decay and large rising gates send blocks to the exact fp32 path.
- `grouped_gemm`: within the unit-test tolerance of the reference on 23 cases, with the same max error as the generic op (lower for a single expert): the unit-test shapes, the six official shapes, ragged N = 200 / K = 300, 256 experts with many empty, unassigned rows before the first and after the last expert, fp16, a single expert, 4096 rows over 8 experts with K = 768 and 2048, 128 experts with (N, K) = (1536, 2048), (2048, 768) and (512, 2048), and the tile boundaries (512 rows with N = 1536, 512 and 768 rows over 32 experts, K = 300 with 256-wide K blocks).
- `absorbed_bmm_concat_cast_q_fp8`: on 10 bf16 cases the rope part is bit-for-bit equal to the reference, the nope part is within the unit-test tolerance, and the rest of the padded buffer is untouched: 512 × 32, 1000 × 32, 1024 × 16, 1536 × 16, 2047 × 16 (just below 2^24 outputs), 2048 × 8 in a 4096 × 16 buffer, 768 × 16 and 767 × 16 (the row threshold), and 2048 × 16 and 4096 × 16 (generic op). Where the kernel runs, about 2 in a million fp8 codes differ from the generic op's, because the fp32 sums round differently.

## Performance

W7900D / gfx1100, bf16, µs. For `chunk_kda_scaled_dot_kkt` and `grouped_gemm` the tables come from the official harness in kernel mode (`--level comprehensive --mode kernel --warmup 1000 --iter 100`), with `ARCH=none` (the generic ops that master runs) and this PR alternated for 3 rounds, medians. The official harness cannot run `absorbed_bmm_concat_cast_q_fp8` at the moment: its `_input_fn` yields the keyword arguments as a dict, which `unpack_to_args_kwargs` iterates as keys. Its numbers, and the extra shapes of all three ops, use the benchmark's inputs with `triton.testing.do_bench` (which kernel mode also uses), master and this PR alternated in one process for 3 rounds, medians. Rows marked "generic" run the same kernel on both sides.

### `chunk_kda_scaled_dot_kkt` (official harness)

| batch, seq, heads, head_dim | master | this PR | speedup |
|---:|---:|---:|---:|
| 1, 64, 8, 128 | 14.60 | 14.52 | 1.01× (generic) |
| 1, 256, 8, 128 | 21.64 | 21.76 | 0.99× (generic) |
| 1, 1024, 8, 128 | 155.12 | 51.48 | 3.01× |
| 1, 4096, 8, 128 | 479.72 | 163.18 | 2.94× |

More shapes (`do_bench`): (1, 512, 8, 128) 3.17×, (2, 256, 8, 128) 3.14×, (1, 2048, 4, 64) 6.49×, (4, 1024, 16, 128) 2.51×, (1, 8192, 32, 128) 2.58×. Back to back with the inputs in cache: 1.98× and 3.05× at 1024 and 4096 tokens, 1.70–4.05× on the extra shapes.

### `grouped_gemm` (official harness)

| tokens, N, K, experts | master | this PR | speedup |
|---:|---:|---:|---:|
| 128, 512, 2048, 8 | 80.88 | 81.04 | 1.00× (generic) |
| 128, 512, 2048, 32 | 178.80 | 178.88 | 1.00× (generic) |
| 1024, 512, 2048, 8 | 146.60 | 114.08 | 1.29× |
| 1024, 512, 2048, 32 | 269.50 | 216.30 | 1.25× |
| 4096, 512, 2048, 8 | 317.96 | 246.60 | 1.29× |
| 4096, 512, 2048, 32 | 456.68 | 314.28 | 1.45× |

More shapes (`do_bench`; tokens, N, K, experts): (512, 512, 2048, 8) 1.21×, (768, 512, 2048, 32) 1.23×, (2048, 512, 2048, 32) 1.47×, (8192, 512, 2048, 8) 1.36×, (4096, 512, 2048, 128) 1.35×, (512, 1536, 2048, 128) 1.40×, (1024, 1536, 2048, 64) 1.13×, (4096, 1536, 2048, 128) 1.35×, (4096, 2048, 768, 128) 1.83×. Back to back with the inputs in cache: 1.12–1.30× on the four official shapes above, 1.17–1.87× on the extra shapes.

### `absorbed_bmm_concat_cast_q_fp8` (official shapes, `do_bench`)

| tokens, heads | master | this PR | speedup |
|---:|---:|---:|---:|
| 16, 16 | 16.8 | 16.8 | 1.00× (generic) |
| 128, 16 | 20.6 | 20.5 | 1.00× (generic) |
| 1024, 16 | 127.8 | 58.7 | 2.18× |
| 4096, 16 | 186.5 | 186.2 | 1.00× (generic) |
| 16, 8 | 14.0 | 13.9 | 1.00× (generic) |
| 512, 16 | 50.0 | 50.0 | 1.00× (generic) |
| 64, 8 | 14.8 | 14.8 | 1.00× (generic) |

More shapes: (768, 16) 1.76×, (1536, 16) 2.69×, (2048, 8) 2.93×, (512, 32) 1.55×. Back to back with the inputs in cache, these and the 1024-token shape are at parity (0.99–1.08×).
