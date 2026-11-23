## Summary
Add gfx1100 (`_amd/rdna3`) versions of six ops added in #395: `qsa_sparse_paged_attention`, `qwen4_qsa_topk`, `ple_short_conv_`, `ple_prefill_short_conv_`, `qwen4_hc_inject_combine` and `qwen4_hc_gate_reduce`. On the official benchmark against upstream master: sparse attention 4.7–15.3×, top-k 7.9–9.7×, PLE decode convolution 1.4–2.8×, PLE prefill convolution 2.8× on the four-request shape, HC inject 2.8–3.9× and HC gate reduce 1.7–2.2× from 1024 rows. The PLE and HC ops are bit-for-bit identical to the generic ops, top-k is bit-for-bit identical to the reference, and sparse attention runs the generic kernels. Public APIs, shared tests and references unchanged.

## Changes
- **qsa_sparse_paged_attention**: without a caller workspace the generic op runs one program per (row, KV head) over the whole TopK list, with an 8-row query tile below the 16-row WMMA minimum. Here the query tile is 16 rows and the TopK axis is always split, in 2 to 32 parts (about 64 programs per row group), with the partials in a workspace allocated by the op when the caller passes none; a caller workspace still sets the split count. Keys are visited 16 at a time with 4 warps. The split, merge and gate kernels are the generic ones.
- **qwen4_qsa_topk**: the generic op is a chain of PyTorch ops (19 kernels per call). Here one kernel sorts each row: every column becomes a 64-bit key, an order-preserving image of the negated logit above the column index, so tied logits keep column order as in the reference (which `torch.topk` does not guarantee). Columns outside the row window sort last and are written as -1.
- **ple_short_conv_**: decode (one token per cache row) is one kernel instead of two: each program reads its rows' old state slice once, writes the convolution + SiLU output, then the shifted state, with the generic arithmetic, tap order and rounding points. Ragged, speculative and profiling calls keep the generic two launches, where the state can only be written after every output has read it.
- **ple_prefill_short_conv_**: the state is strided per channel, so its loads are one element each; the convolution now issues them only in the time block that reaches back before the request start instead of as masked loads in every block. Channel blocks are 1024 wide with 8 warps when the grid has at least 512 request x time blocks, else 256 wide with 4 warps (master: 128 wide, 8 warps).
- **qwen4_hc_inject_combine**: from 256 rows each program covers all HC branches of a 4096-wide hidden slice, so the block output is read once instead of once per branch and the grid has fewer, larger programs. Fewer rows run the generic kernel and launch.
- **qwen4_hc_gate_reduce**: from 256 rows, 4096-wide hidden slices with 8 warps instead of 256-wide slices with 4. Fewer rows keep the generic launch.

## Perf
W7900D / gfx1100, vs upstream master `2667764`, official benchmark harness (comprehensive level, kernel mode), master and this PR alternated for 3 rounds, medians.

### `qsa_sparse_paged_attention` (3 query heads per KV head, head dim 256, page size 1)

| rows | TopK | master | this PR | speedup |
|---:|---:|---:|---:|---:|
| 1 | 512 | 244.7 µs | 17.0 µs | **14.39×** |
| 7 | 2048 | 926.6 µs | 60.7 µs | **15.27×** |
| 64 | 2048 | 1687.6 µs | 361.7 µs | **4.67×** |

The benchmark's indices walk consecutive pages. With page size 16 and tokens in random order (do_bench, 3 rounds): 1 row x 513: 318.0 → 27.7 µs (11.49×), 7 x 2051: 1391.0 → 118.4 µs (11.75×), 64 x 2051: 2271.7 → 686.6 µs (3.31×), 300 x 513: 1666.7 → 685.5 µs (2.43×).

### `qwen4_qsa_topk` (640 logits per row, top 512)

| rows | master | this PR | speedup |
|---:|---:|---:|---:|
| 1 | 120.5 µs | 15.2 µs | **7.92×** |
| 7 | 146.4 µs | 15.0 µs | **9.73×** |
| 64 | 160.2 µs | 18.3 µs | **8.74×** |

1024 and 4096 rows (do_bench, 3 rounds): 417.4 → 83.6 µs (4.99×) and 867.8 → 231.4 µs (3.75×).

### `ple_short_conv_` (decode, 10240 channels, width 4, dilation 3)

| rows | master | this PR | speedup |
|---:|---:|---:|---:|
| 1 | 113.2 µs | 64.9 µs | **1.74×** |
| 7 | 117.1 µs | 74.8 µs | **1.57×** |
| 64 | 134.1 µs | 97.2 µs | **1.38×** |
| 1024 | 1714.7 µs | 609.8 µs | **2.81×** |

The harness times in-place ops as one call between events after a synchronize, so the small rows include the call's host time. Back-to-back with do_bench: 15.2 → 10.5 µs (1.45×), 20.3 → 15.1 µs (1.34×), 85.0 → 56.5 µs (1.50×), 1769.0 → 652.9 µs (2.71×).

### `ple_prefill_short_conv_` (10240 channels, width 4, dilation 3)

| requests x tokens | master | this PR | speedup |
|---:|---:|---:|---:|
| 1 x 1024 | 173.6 µs | 174.9 µs | 0.99× |
| 4 x 1024 | 1323.6 µs | 469.2 µs | **2.82×** |

The single request is host-bound in this timing: its convolution kernel takes 64.0 µs on master and 33.9 µs here (profiler).

### `qwen4_hc_inject_combine` (hidden 2560, 4 branches)

| rows | master | this PR | speedup |
|---:|---:|---:|---:|
| 1 | 9.5 µs | 9.5 µs | 1.00× |
| 7 | 9.8 µs | 9.9 µs | 0.99× |
| 64 | 14.7 µs | 14.9 µs | 0.99× |
| 1024 | 418.4 µs | 148.0 µs | **2.83×** |
| 16384 | 4683.8 µs | 1201.9 µs | **3.90×** |

### `qwen4_hc_gate_reduce` (hidden 2560, 4 branches)

| rows | master | this PR | speedup |
|---:|---:|---:|---:|
| 1 | 9.5 µs | 9.5 µs | 1.00× |
| 7 | 9.8 µs | 9.9 µs | 0.99× |
| 64 | 17.2 µs | 17.4 µs | 0.99× |
| 1024 | 242.8 µs | 145.2 µs | **1.67×** |
| 16384 | 2699.1 µs | 1226.3 µs | **2.20×** |

Below 256 rows both HC ops run the generic kernel and launch, so those rows differ only by run-to-run noise. At 256 / 512 rows (do_bench, 3 rounds): inject 57.4 → 37.3 µs (1.54×) and 144.1 → 72.4 µs (1.99×), gate reduce 55.4 → 42.4 µs (1.31×) and 103.6 → 81.4 µs (1.27×).

## Test plan
- [x] `tests/test_{qsa_sparse_paged_attention,qwen4_qsa_topk,ple_short_conv_,ple_prefill_short_conv_,qwen4_hc_inject_combine,qwen4_hc_gate_reduce}.py` with `ARCH` unset and `ARCH=none`: 58 passed in each mode (these tests call the dispatched ops, so they cover the new files, including the CUDA graph replay tests)
- [x] Dispatch on gfx1100 resolves all six ops to the new `_amd/rdna3` files
- [x] `qsa_sparse_paged_attention` against the reference (float, CPU) and the generic op, 21 cases: benchmark inputs; page size 16 with reversed tables and sentinel / out-of-table tokens; 1–300 rows; TopK 33–2051; head dims 64 / 128 / 256; 1 / 3 / 4 / 8 query heads per KV head with 1 or 2 KV heads; gate on and off; a caller workspace; invalid request rows (zero output). Error against the reference equals the generic op's, and the difference from the generic op is within the unit test's split-error gates
- [x] `qwen4_qsa_topk` against the reference, 21 cases, all bit-exact: benchmark inputs and their negation; random and tied logits; -inf and ±0 inside the window; windows of 0 / 1 / 511 / 512 / 513 / all columns; 100–2048 columns; topk 100–1024 including topk above the column count; int64 bounds; strided logits. The generic op differs from the reference on the tied cases
- [x] `ple_short_conv_` against the generic op, output and state bit-for-bit, 20 cases: benchmark inputs; ±0; fp16 / fp32; fp32 state with bf16 input; initial-state flags; null and out-of-range rows; permuted rows; strided input and output; extra state capacity; 1000 channels; widths 2–8 and dilations 1–3; ragged and profiling calls
- [x] `ple_prefill_short_conv_` against the generic op, output and state bit-for-bit, 13 cases: benchmark inputs; ragged requests including empty and short ones; initial-state flags and null rows; ±0; fp16 / fp32; fp32 state with bf16 input; 1000 channels; widths 2–8 and dilations 1–3
- [x] `qwen4_hc_inject_combine` and `qwen4_hc_gate_reduce` against the generic ops, bit-for-bit, 7 cases each: benchmark inputs (1–16384 rows), fp16, 3-D inputs
- [x] black / flake8 / isort
