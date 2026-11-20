## Summary
Add gfx1100 (`_amd/rdna3`) versions of `awq_gemm` and `mla_kv_pack_quantize_fp8`. Against upstream master: `awq_gemm` 1.00–1.78× on the official benchmark (fp32 activations) and 1.06–32.9× with fp16 / bf16 activations (2.5–32.9× from 8 rows), which master ran one row at a time on its scalar kernel; `mla_kv_pack_quantize_fp8` 1.16–1.19× from 512 tokens with 16 heads, other shapes unchanged. fp32 `awq_gemm` results stay fp32-accurate, fp16 / bf16 weights are dequantized exactly as in the reference, and `mla_kv_pack_quantize_fp8` output is bit-for-bit identical. Public APIs, shared tests and references unchanged.

## Changes
- **awq_gemm** (new `_amd/rdna3` file)
  - fp16 / bf16 activations (scales of the same dtype): each weight tile is dequantized in registers to `(q - z) * scale` in the activation dtype, rounded once as in the reference, and multiplied on the matrix units, one dot per K block. Master ran these dtypes through its one-row scalar kernel, once per row. The bf16 product is formed in fp32 and rounded explicitly, because an elementwise bf16 multiply in Triton on gfx1100 truncates.
  - fp32 activations keep master's scheme (`q - z` is exact in fp16, the activations are split into high and low fp16 halves, two dots per K block, each group's sum scaled in fp32), with tile sizes, warps and K splits chosen per shape. From 64 rows master and this PR are both bound by the two fp16 dots (about 19 TFLOPs at 512 × 4096), so those shapes stay on par.
  - One activation row: an fp32 multiply-add kernel that loads each group's scales and zeros once per tile (master loaded them per weight element).
  - K splits are added by a second kernel, as in master.
- **mla_kv_pack_quantize_fp8** (new `_amd/rdna3` file): the same encoding as the generic op. From 256 tokens, with power-of-two widths of at least 4, the fp8 codes are packed four per 32-bit store, and all heads of a token share one program up to 8192 lanes (16 warps). Fewer tokens run the same kernel as master.

## Perf
W7900D / gfx1100, vs upstream master `87e9acc`. The official harness cannot run these two ops: their `_input_fn` yields the keyword arguments as a dict, which `unpack_to_args_kwargs` iterates as keys, so the op is called without arguments. The numbers below use the official benchmark's shapes and inputs, timed with `triton.testing.do_bench`, master and this PR alternated in the same process, medians of 3 rounds (`awq_gemm`) and 7 rounds (`mla_kv_pack_quantize_fp8`).

### `awq_gemm`, official benchmark (fp32 activations, K = 4096, group size 128)

| M | N | master | this PR | speedup |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 22.8 µs | 21.0 µs | **1.09×** |
| 1 | 4096 | 43.2 µs | 38.2 µs | **1.13×** |
| 8 | 1024 | 28.2 µs | 27.1 µs | 1.04× |
| 8 | 4096 | 56.6 µs | 49.3 µs | **1.15×** |
| 64 | 1024 | 53.2 µs | 51.3 µs | 1.04× |
| 64 | 4096 | 140.3 µs | 138.5 µs | 1.01× |
| 512 | 1024 | 253.2 µs | 252.8 µs | 1.00× |
| 512 | 4096 | 930.2 µs | 895.9 µs | 1.04× |
| 16 | 32 | 20.4 µs | 19.5 µs | 1.05× |
| 24 | 64 | 28.7 µs | 20.2 µs | **1.42×** |
| 32 | 128 | 38.0 µs | 21.4 µs | **1.78×** |

### `awq_gemm`, fp16 / bf16 activations (same shapes and weights, scales in the activation dtype)

| M | N | fp16 master | fp16 this PR | fp16 speedup | bf16 master | bf16 this PR | bf16 speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 23.3 µs | 21.2 µs | **1.10×** | 23.2 µs | 21.8 µs | **1.06×** |
| 1 | 4096 | 44.9 µs | 38.0 µs | **1.18×** | 44.6 µs | 38.6 µs | **1.16×** |
| 8 | 1024 | 57.0 µs | 21.2 µs | **2.68×** | 54.8 µs | 22.0 µs | **2.50×** |
| 8 | 4096 | 201.8 µs | 40.2 µs | **5.01×** | 195.4 µs | 44.8 µs | **4.36×** |
| 64 | 1024 | 349.6 µs | 32.3 µs | **10.82×** | 326.0 µs | 36.4 µs | **8.95×** |
| 64 | 4096 | 1434.3 µs | 66.0 µs | **21.75×** | 1389.2 µs | 73.9 µs | **18.80×** |
| 512 | 1024 | 2669.4 µs | 126.1 µs | **21.17×** | 2503.8 µs | 130.5 µs | **19.18×** |
| 512 | 4096 | 11078.4 µs | 336.4 µs | **32.93×** | 10734.9 µs | 357.1 µs | **30.06×** |
| 16 | 32 | 18.8 µs | 16.1 µs | **1.17×** | 19.0 µs | 16.4 µs | **1.15×** |
| 24 | 64 | 29.0 µs | 16.2 µs | **1.80×** | 28.9 µs | 17.2 µs | **1.69×** |
| 32 | 128 | 35.7 µs | 17.2 µs | **2.08×** | 34.4 µs | 18.0 µs | **1.91×** |

### `mla_kv_pack_quantize_fp8`, official benchmark (bf16 inputs, e4m3 output, nope 512 + rope 64, value 512)

| tokens | heads | master | this PR | speedup |
|---:|---:|---:|---:|---:|
| 1 | 1 | 9.1 µs | 9.2 µs | 0.99× |
| 1 | 16 | 9.3 µs | 9.4 µs | 0.99× |
| 8 | 1 | 9.0 µs | 9.1 µs | 0.99× |
| 8 | 16 | 9.9 µs | 10.1 µs | 0.98× |
| 64 | 1 | 9.2 µs | 9.1 µs | 1.01× |
| 64 | 16 | 15.7 µs | 15.8 µs | 1.00× |
| 512 | 1 | 11.8 µs | 11.9 µs | 0.99× |
| 512 | 16 | 83.3 µs | 71.5 µs | **1.16×** |
| 4096 | 1 | 38.5 µs | 38.8 µs | 0.99× |
| 4096 | 16 | 558.2 µs | 469.7 µs | **1.19×** |
| 7 | 8 | 9.5 µs | 9.6 µs | 0.99× |
| 333 | 1 | 10.7 µs | 10.7 µs | 1.00× |
| 256 | 8 | 21.9 µs | 20.9 µs | **1.05×** |

Below 256 tokens this PR runs master's kernel; the 0.98–1.01× there is run-to-run noise (0.99–1.02× on a second GPU). With one head the packed stores make no measurable difference.

## Test plan
- [x] `tests/test_{awq_gemm,mla_kv_pack_quantize_fp8}.py` with `ARCH` unset and `ARCH=none` (both import the generic modules)
- [x] Dispatch on gfx1100 resolves both ops to the new `_amd/rdna3` files
- [x] `awq_gemm` against an fp64 product with the reference's dequantized weight, 60 cases: the unit-test cases and benchmark shapes with fp32 / fp16 / bf16 activations, group sizes 32 / 64 / 128 / 256 and whole K, odd M and N, strided activations. Max error relative to the largest magnitude: fp32 1.1e-6 (the fp32 reference itself is off by up to 3.7e-6), fp16 4.4e-4 and bf16 3.6e-3, the same as the reference's own error
- [x] `mla_kv_pack_quantize_fp8` against master byte for byte, 46 cases: the benchmark shapes, NaN / inf / out-of-range / subnormal values, negative and extreme scales, e4m3 and e5m2 output, 3-D `k_pe`, strided inputs, non-power-of-two widths, fp16 inputs
- [x] black / flake8 / isort
