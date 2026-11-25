Add gfx1100 (`_amd/rdna3`) versions of two ops added in #418: `dsv3_router_gemm` and `fused_recurrent_kda`.

On W7900D / gfx1100 against upstream master `2667764`:

- `dsv3_router_gemm`: 1.00–3.28× on the official benchmark shapes (1.89–3.28× from 1 to 4 tokens, 1.20–1.30× at 16 tokens, 1.86× with 128 experts), 2.58–14.69× at 64, 512 and 4096 tokens. Up to 16 tokens the op mostly streams the router weights, so 16-bit inputs take the kernel that reads them fastest: a few experts per program for 1 to 4 tokens, and the AMD per-(token, expert) kernel with more warps for small expert counts and for 5 to 8 tokens. From 9 tokens on they run on the matrix units: each program multiplies a block of token rows by a block of experts over a slice of the hidden dimension, so each weight is read once per row block; small grids split the hidden dimension and a second kernel adds the parts. Other dtypes keep the AMD kernels.
- `fused_recurrent_kda`: same kernels and arithmetic as the generic op, bit-for-bit identical output and final state on every shape below; only the tiles change. 1.00–1.24× on the official shapes (L2 norm off), 1.07–1.32× with the L2 norm on, 1.33× and 1.21× on decode with 64 and 256 sequences. Each row of the state evolves independently, so mid-size batches split the state into more, smaller programs and decode gives each program more warps; with the L2 norm, from 128 token rows the preparation kernel gives each row one warp so its sums stay within the warp.

Public APIs, shared tests and references unchanged.

## Correctness

- `pytest tests/test_dsv3_router_gemm.py tests/test_fused_recurrent_kda.py`: 22 passed with the rdna3 ops and with `ARCH=none`; `get_op` resolves both ops to the rdna3 files.
- `dsv3_router_gemm`: max error against a float32 reference on par with master on the benchmark shapes, odd token counts, fp16, bf16 output, strided hidden states, 100 experts and hidden size 1000.
- `fused_recurrent_kda`: output and final state bit-for-bit equal to the generic op on the 7 benchmark shapes and 4 decode batch sizes, each with and without the L2 norm, and on the unit-test shapes, headwise and per-value beta, fp16, and V = 96.

## Performance

W7900D / gfx1100, bf16. The official harness cannot run these two ops at the moment: their `_input_fn` yields the keyword arguments as a dict, which `unpack_to_args_kwargs` iterates as keys, so the reference is called without arguments. The numbers below use the official benchmark's shapes and inputs, timed with `triton.testing.do_bench`, master and this PR alternated in the same process for 3 rounds, medians, in µs.

### `dsv3_router_gemm` (official benchmark shapes)

| tokens, experts | master | this PR | speedup |
|---:|---:|---:|---:|
| 1, 256 | 38.6 | 20.5 | 1.89× |
| 1, 384 | 55.9 | 25.2 | 2.22× |
| 2, 256 | 49.3 | 21.0 | 2.35× |
| 2, 384 | 64.2 | 25.6 | 2.51× |
| 4, 256 | 70.4 | 21.4 | 3.28× |
| 4, 384 | 80.9 | 26.2 | 3.09× |
| 8, 256 | 22.4 | 21.3 | 1.05× |
| 8, 384 | 26.9 | 26.9 | 1.00× |
| 16, 256 | 31.6 | 24.4 | 1.30× |
| 16, 384 | 35.9 | 29.9 | 1.20× |
| 1, 128 | 26.0 | 14.0 | 1.86× |
| 16, 512 | 45.0 | 35.2 | 1.28× |
| 8, 64 | 13.0 | 12.6 | 1.03× |

More token counts with 256 experts: 3 → 2.65×, 5 → 1.15×, 12 → 0.98×, 64 → 2.58×, 512 → 8.66×, 4096 → 14.69×. With 64 experts: 4 → 2.23×, 16 → 1.49×.

### `fused_recurrent_kda` (8 heads, K = V = 128)

| batch, t | generic | this PR | speedup | with L2 norm: generic | this PR | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1, 64 | 40.4 | 40.0 | 1.01× | 45.9 | 42.7 | 1.07× |
| 4, 64 | 112.8 | 102.9 | 1.10× | 132.0 | 106.8 | 1.24× |
| 16, 64 | 320.6 | 305.6 | 1.05× | 402.6 | 329.8 | 1.22× |
| 64, 64 | 1096.5 | 1096.1 | 1.00× | 1394.1 | 1157.2 | 1.20× |
| 2, 64 | 73.8 | 59.8 | 1.24× | 83.7 | 63.4 | 1.32× |
| 8, 32 | 112.0 | 102.4 | 1.09× | 130.8 | 107.7 | 1.21× |
| 32, 128 | 1098.6 | 1099.0 | 1.00× | 1399.8 | 1164.1 | 1.20× |
| 1, 1 (decode) | 14.8 | 14.8 | 1.00× | 15.6 | 15.8 | 0.99× |
| 16, 1 | 49.2 | 46.8 | 1.05× | 50.6 | 47.7 | 1.06× |
| 64, 1 | 206.3 | 155.4 | 1.33× | 210.2 | 156.6 | 1.34× |
| 256, 1 | 548.8 | 454.6 | 1.21× | 571.2 | 467.0 | 1.22× |

In the 1, 1 decode case both ops launch the same tiles; the difference is run-to-run noise.
