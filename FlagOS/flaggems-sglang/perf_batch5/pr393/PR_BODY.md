## Summary
Make the gfx1100 (`_amd/rdna3`) `decode_attention` and `decode_grouped_attention` correct and fast when `q` or the KV buffers are not contiguous, and add a gfx1100 launch rule for `fused_recurrent_gated_delta_rule_packed_decode`. Against upstream master: with a strided `q` (a view into a fused qkv projection, batch ≥ 2) the two decode ops go from wrong outputs (max error 0.08–0.34, or NaN) and 3.9–45× slower runs to correct outputs at the speed of a contiguous `q`; contiguous inputs are unchanged (official benchmark 0.97–1.04×). `fused_recurrent_gated_delta_rule_packed_decode` is 1.04–1.19× faster on the official benchmark, bitwise identical. Public APIs, shared tests and references unchanged.

## Changes
- **decode_attention / decode_grouped_attention** (existing `_amd/rdna3` files): the split-KV path required contiguous `q`, `k_buffer` and `v_buffer`, and otherwise fell back to kernels that also index all three as contiguous, so a strided `q` (e.g. `qkv[:, :H * D].view(B, H, D)`) read the wrong rows; the `decode_attention` fallback also looped over `k_buffer.shape[0] // batch` tokens, i.e. the whole KV pool, for every head. The split-KV and fallback kernels now take the batch and head strides of `q` and the token and head strides of the KV buffers, and only need unit stride along the head dimension (otherwise the inputs are made contiguous first). The strides are compile-time constants: as runtime values, the multiplies in the per-token fallback loop cost it 40%. The `decode_attention` fallback loops over each sequence's own length, and offsets switch to 64 bits when the KV pool passes 2^31 elements.
- **fused_recurrent_gated_delta_rule_packed_decode** (new `_amd/rdna3` file): 32-row state tiles, with 16 warps from 128 (sequence, value head) pairs on and 8 below, instead of 64-row tiles with 8 warps. Rows are independent, so the result is bitwise identical.

## Perf
W7900D / gfx1100, vs upstream master `567d379`. Official benchmark: comprehensive, `--mode kernel`, master and this PR alternated for 3 rounds, medians. Additional shapes: `triton.testing.do_bench` medians, master and this PR in the same process.

Contiguous inputs run the same kernels with the same launch parameters as master. The official benchmarks of the two decode ops only have contiguous inputs, so their tables check that nothing regresses; the gains are for a strided `q` (first table).

### Strided `q` (a view into a fused qkv projection), bf16, 200K-token KV pool

| op | batch × seq | q / kv heads | head_dim | master | max error | this PR | max error | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `decode_attention` | 1 × 4096 | 16 / 4 | 128 | 42.2 µs | 1.2e-04 | 42.3 µs | 1.2e-04 | 1.00× |
| `decode_grouped_attention` | 1 × 4096 | 16 / 4 | 128 | 42.1 µs | 1.2e-04 | 42.2 µs | 1.2e-04 | 1.00× |
| `decode_attention` | 8 × 2048 | 16 / 4 | 128 | 1751.2 µs | 1.8e-01 | 117.4 µs | 2.2e-04 | **14.91×** |
| `decode_grouped_attention` | 8 × 2048 | 16 / 4 | 128 | 1543.0 µs | 1.8e-01 | 117.6 µs | 2.2e-04 | **13.12×** |
| `decode_attention` | 32 × 1024 | 16 / 4 | 128 | 1146.0 µs | nan | 205.6 µs | 3.6e-04 | **5.57×** |
| `decode_grouped_attention` | 32 × 1024 | 16 / 4 | 128 | 812.6 µs | 3.4e-01 | 206.4 µs | 3.6e-04 | **3.94×** |
| `decode_attention` | 4 × 8192 | 6 / 1 | 256 | 3523.6 µs | 8.1e-02 | 138.9 µs | 9.5e-05 | **25.38×** |
| `decode_grouped_attention` | 4 × 8192 | 6 / 1 | 256 | 6252.6 µs | 8.1e-02 | 137.5 µs | 9.5e-05 | **45.47×** |
| `decode_attention` | 16 × 2048 | 24 / 4 | 256 | 3232.7 µs | 2.0e-01 | 390.4 µs | 2.4e-04 | **8.28×** |
| `decode_grouped_attention` | 16 × 2048 | 24 / 4 | 256 | 1702.8 µs | 2.0e-01 | 390.1 µs | 2.4e-04 | **4.37×** |

With batch 1 the view is contiguous to PyTorch (`is_contiguous()` ignores the stride of a size-1 dimension), so master already took the split-KV path.

### `decode_attention`, official benchmark (contiguous inputs)

| batch | q heads | kv heads | head_dim | KV tokens | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 8 | 128 | 2048 | 36.7 µs | 36.6 µs | 1.00× |
| 8 | 32 | 8 | 128 | 16384 | 184.0 µs | 183.9 µs | 1.00× |
| 64 | 32 | 8 | 128 | 32768 | 319.0 µs | 318.8 µs | 1.00× |
| 512 | 32 | 8 | 128 | 65536 | 515.2 µs | 515.4 µs | 1.00× |
| 4096 | 32 | 8 | 128 | 524288 | 3142.1 µs | 3143.5 µs | 1.00× |
| 2 | 4 | 4 | 64 | 20 | 9.9 µs | 10.1 µs | 0.97× |
| 2 | 4 | 2 | 64 | 20 | 10.0 µs | 10.1 µs | 0.99× |
| 2 | 4 | 4 | 80 | 20 | 10.5 µs | 10.6 µs | 0.98× |
| 2 | 16 | 1 | 512 | 256 | 26.0 µs | 25.0 µs | 1.04× |

### `decode_grouped_attention`, official benchmark (contiguous inputs)

| batch | q heads | kv heads | head_dim | KV tokens | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1 | 128 | 2048 | 25.8 µs | 25.6 µs | 1.01× |
| 8 | 128 | 1 | 128 | 16384 | 58.1 µs | 58.2 µs | 1.00× |
| 64 | 128 | 1 | 128 | 32768 | 87.4 µs | 87.3 µs | 1.00× |
| 512 | 128 | 1 | 128 | 65536 | 257.2 µs | 256.9 µs | 1.00× |
| 4096 | 128 | 1 | 128 | 524288 | 1105.2 µs | 1105.3 µs | 1.00× |
| 2 | 32 | 1 | 64 | 20 | 10.1 µs | 10.2 µs | 0.99× |
| 2 | 128 | 1 | 64 | 20 | 10.1 µs | 10.1 µs | 1.00× |
| 2 | 32 | 4 | 80 | 20 | 10.6 µs | 10.8 µs | 0.99× |
| 2 | 128 | 8 | 512 | 256 | 67.0 µs | 67.1 µs | 1.00× |

### `fused_recurrent_gated_delta_rule_packed_decode`, official benchmark (32 value heads, 16 key heads, K = V = 128)

| batch | master | this PR | speedup |
|---:|---:|---:|---:|
| 1 | 16.9 µs | 16.3 µs | 1.04× |
| 8 | 67.5 µs | 56.5 µs | **1.19×** |
| 64 | 383.1 µs | 363.5 µs | **1.05×** |
| 256 | 1513.7 µs | 1408.4 µs | **1.07×** |
| 512 | 2990.2 µs | 2788.4 µs | **1.07×** |

The kernel reads and writes the whole float32 state (64 KiB per sequence and value head) at every step. From batch 64 on, this PR moves 738–770 GB/s, about the rate of a plain in-place fp32 update on this GPU (754 GB/s), up from 701–718 GB/s on master, so little is left there; batch 1 is launch-bound.

### `fused_recurrent_gated_delta_rule_packed_decode`, additional shapes (K = V = 128)

| key heads | value heads | batch | master | this PR | speedup |
|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 1 | 12.5 µs | 12.1 µs | 1.03× |
| 4 | 8 | 16 | 39.9 µs | 35.6 µs | **1.12×** |
| 4 | 8 | 64 | 121.4 µs | 104.7 µs | **1.16×** |
| 4 | 8 | 256 | 393.5 µs | 367.9 µs | **1.07×** |
| 4 | 12 | 1 | 13.2 µs | 12.7 µs | 1.04× |
| 4 | 12 | 8 | 32.2 µs | 30.4 µs | **1.06×** |
| 4 | 12 | 32 | 95.8 µs | 82.8 µs | **1.16×** |
| 4 | 12 | 128 | 301.5 µs | 279.1 µs | **1.08×** |
| 16 | 48 | 4 | 58.4 µs | 47.3 µs | **1.23×** |
| 16 | 48 | 32 | 301.1 µs | 277.7 µs | **1.08×** |

## Test plan
- [x] `tests/test_{decode_attention,decode_grouped_attention,fused_recurrent_gated_delta_rule_packed_decode}.py` on gfx1100, with the rdna3 ops and with `ARCH=none`
- [x] Decode attention against a float64 reference, 2164 cases: `q` contiguous / view into qkv / head-major / batch-strided / head-dim-strided; KV buffers contiguous / k and v interleaved in one pool / padded rows; bf16 / fp16 / fp32; head_dim 64–512; 1–16 query heads per KV head; empty sequences; a KV pool past 2^31 elements. With unit stride along the head dimension, outputs for a strided `q` are bitwise equal to those for the same `q` made contiguous
- [x] `fused_recurrent_gated_delta_rule_packed_decode` bitwise equal to master: 336 cases (head counts, batch sizes, K = V = 64 / 128, padded slots, with and without L2 norm)
- [x] Official comprehensive bench (tables above)
- [x] black / flake8 / isort
