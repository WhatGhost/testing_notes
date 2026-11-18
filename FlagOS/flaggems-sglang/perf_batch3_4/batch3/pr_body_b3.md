## Summary
Add gfx1100 (`_amd/rdna3`) versions of `chunked_sgmv_shrink` and `recompute_w_u`, and drop a host sync from the gfx1100 `chunked_sgmv_expand`. Against upstream master: `chunked_sgmv_shrink` 1.32–1.88× on the official benchmark and up to 2.0× on larger batches, `recompute_w_u` up to 2.06×. Public APIs, shared tests and references unchanged.

## Changes
- **chunked_sgmv_shrink** (new gfx1100 version):
  - A hidden dimension of up to 1024 takes one kernel, whose last programs zero the rows no segment covers, instead of a zero fill, a split kernel and a reduction.
  - A longer one still splits the hidden dimension and adds the partial sums in a second kernel, but the split follows the batch instead of a fixed 16 splits, keeps the split starts off a power of two of K blocks apart (4 or 8 splits of K = 4096 ran up to 3x slower on gfx1100), and skips the zero fill when the reduction writes every row.
  - Shapes are runtime arguments, so a new `max_len` does not compile new kernels.

  float32 keeps an IEEE-mode direct kernel, as in the generic op.
- **recompute_w_u** (new gfx1100 version): from 512 chunk-heads on, the generic op on HIP stages k·β·exp(g), v·β and A in packed buffers and multiplies them in a second kernel. Here one kernel computes every tile from the inputs at all sizes. Bitwise identical to master.
- **chunked_sgmv_expand** (gfx1100): without a `max_len` hint, the token count bounds the segment length instead of reading the longest segment back with `.item()`. Output unchanged.

## Perf
W7900D / gfx1100, bf16, vs upstream master `5c2ed0f`. Official benchmark: comprehensive, `--mode kernel`, master and this PR alternated for 3 rounds, medians. Additional shapes: `triton.testing.do_bench` medians, master and this PR in the same process.

### `chunked_sgmv_shrink`, official benchmark

| segments × length | adapters | hidden | rank × slices | master | this PR | speedup |
|---|---:|---:|---:|---:|---:|---:|
| 8 × 64 | 4 | 4096 | 32 × 1 | 41.6 µs | 31.4 µs | **1.32×** |
| 4 × 256 | 2 | 4096 | 64 × 1 | 74.9 µs | 46.9 µs | **1.60×** |
| 1 × 8 | 1 | 64 | 16 × 1 | 17.0 µs | 9.5 µs | **1.79×** |
| 4 × 16 | 2 | 128 | 16 × 1 | 17.8 µs | 9.8 µs | **1.82×** |
| 2 × 16 | 2 | 256 | 32 × 3 | 19.4 µs | 10.4 µs | **1.88×** |

### `chunked_sgmv_shrink`, additional shapes

| segments × length | adapters | hidden | rank × slices | master | this PR | speedup |
|---|---:|---:|---:|---:|---:|---:|
| 32 × 1 | 4 | 4096 | 16 × 3 | 38.1 µs | 23.4 µs | **1.63×** |
| 8 × 1 | 2 | 5120 | 64 × 1 | 24.6 µs | 19.9 µs | **1.24×** |
| 1 × 1 | 1 | 2048 | 64 × 3 | 19.9 µs | 17.9 µs | **1.11×** |
| 8 × 16 | 4 | 2048 | 64 × 3 | 29.8 µs | 27.3 µs | **1.09×** |
| 4 × 32 | 2 | 5120 | 64 × 3 | 41.6 µs | 29.8 µs | **1.40×** |
| 16 × 128 | 4 | 2048 | 16 × 1 | 42.7 µs | 42.5 µs | 1.00× |
| 1 × 2048 | 1 | 4096 | 32 × 1 | 110.9 µs | 66.1 µs | **1.68×** |
| 8 × 512 | 4 | 4096 | 16 × 3 | 245.3 µs | 121.7 µs | **2.02×** |
| 4 × 2048 | 2 | 5120 | 64 × 1 | 596.5 µs | 321.4 µs | **1.86×** |

### `recompute_w_u`, official benchmark (B=1, H=16, Hg=4, K=V=128, BT=64)

| T | master | this PR | speedup |
|---:|---:|---:|---:|
| 64 | 14.3 µs | 14.0 µs | 1.03× |
| 256 | 25.3 µs | 25.2 µs | 1.01× |
| 1024 | 66.7 µs | 62.2 µs | **1.07×** |
| 4096 | 421.9 µs | 204.9 µs | **2.06×** |

### `recompute_w_u`, additional shapes (BT=64)

| B | T | H | Hg | K = V | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8192 | 12 | 4 | 128 | 592.2 µs | 310.3 µs | **1.91×** |
| 2 | 4096 | 8 | 4 | 128 | 426.6 µs | 214.2 µs | **1.99×** |
| 1 | 2048 | 16 | 16 | 64 | 123.6 µs | 66.1 µs | **1.87×** |
| 4 | 512 | 12 | 4 | 128 | 96.2 µs | 90.4 µs | **1.06×** |

### `chunked_sgmv_expand`, official benchmark

1.00× on all 5 shapes; the benchmark passes `max_len`, so it does not reach the changed path.

## Test plan
- [x] `tests/test_{chunked_sgmv_shrink,recompute_w_u,chunked_sgmv_expand}.py` on gfx1100, with the rdna3 ops and with `ARCH=none`
- [x] `chunked_sgmv_shrink` against a float64 reference: 2592 cases (permutation none / identity / shuffled, empty segments, rows no segment covers, bf16 / fp16 / fp32, hidden not a multiple of the K block)
- [x] `recompute_w_u` bitwise equal to master on 8 shapes
- [x] `chunked_sgmv_expand` without `max_len`: bitwise equal to the result with it (60 cases), and no synchronizing op under `torch.cuda.set_sync_debug_mode("error")`
- [x] Official comprehensive bench (tables above)
- [x] black / flake8 / isort

Made with [Cursor](https://cursor.com)
