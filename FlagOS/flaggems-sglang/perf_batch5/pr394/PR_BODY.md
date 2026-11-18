## Summary
Add a gfx1100 (`_amd/rdna3`) `fused_moe`: the unquantized experts of a MoE layer (gate/up projection, SiLU or GELU, down projection, weighted sum over the top-k experts). Against upstream master (the generic op): 1.54–10.03× on the official benchmark (geomean 3.72×), 1.58–9.85× on more token counts of the same layer, and 1.78–6.94× on a layer with 256 small experts, from 1 to 4096 tokens. Against SGLang's own Triton kernel on this GPU it is faster on 22 of the 26 shapes below (1.17–6.10×) and runs at 0.82–0.89× on the other 4, the largest batches, because it keeps float32 accuracy there; SGLang's max error is larger on 25 of the 26 shapes. Errors against a float32 reference are on par with master's. Public APIs, shared tests and references unchanged.

## Changes
- **fused_moe** (new `_amd/rdna3` file). Two paths, by the number P of (token, expert) pairs and E of experts:
  - **P ≤ 2E** (decode and small batches): two launches, no sorting. The gate/up kernel runs one program per pair and column block and accumulates gate and up in float32 lanes; with fewer than 512 programs, K is split into up to 8 interleaved parts. The down kernel runs one program per token and column block: it adds up the parts, forms the activation of each of the token's top-k pairs times its routing weight, and accumulates the down projections of all top-k pairs before a single reduction.
  - **More pairs**: the pairs are sorted by expert on the device (the same counting and placement kernels as `fused_moe_gemm` in [#371](https://github.com/flagos-ai/FlagGems-sglang/pull/371); no host synchronization), gate/up and down run as grouped GEMMs on the matrix units (activation in the gate/up epilogue, routing weight in the down epilogue), and a last kernel adds up the top-k rows of each token. The intermediate is stored as a 16-bit value plus its 16-bit residual and the down GEMM multiplies both, which keeps the accuracy of a float32 intermediate (a plain 16-bit intermediate doubled the error). WMMA results are added to the float32 accumulators with `tl.fma`, since WMMA truncates what it adds into a float32 accumulator; programs run in `GROUP_M` groups so that rows and expert weights are reread from L2; 16-bit tiles are stored through a `tl.split`/`tl.join` round trip, without which the store keeps the WMMA layout and writes two bytes per lane.
  - Block sizes come from sweeps on the shapes below: `BLOCK_M` is 32, or 64 from 256 rows per expert when inter ≥ 512.

## Perf
W7900D / gfx1100, bf16, vs upstream master `567d379`. Official benchmark: comprehensive, `--mode kernel`, master and this PR alternated for 3 rounds, medians. Additional shapes: `triton.testing.do_bench` medians, master and this PR in the same process.

### Official benchmark (64 experts, hidden 2048, inter 1408, top-8)

| tokens | master | this PR | speedup |
|---:|---:|---:|---:|
| 1 | 931.9 µs | 364.6 µs | **2.56×** |
| 8 | 2366.9 µs | 1143.2 µs | **2.07×** |
| 64 | 2900.7 µs | 1883.2 µs | **1.54×** |
| 512 | 21198.1 µs | 2416.7 µs | **8.77×** |
| 4096 | 112217.2 µs | 11189.2 µs | **10.03×** |

### Additional shapes (64 experts, hidden 2048, inter 1408, top-8)

| tokens | master | this PR | speedup |
|---:|---:|---:|---:|
| 2 | 1483.3 µs | 504.8 µs | **2.94×** |
| 4 | 1842.1 µs | 764.9 µs | **2.41×** |
| 16 | 2607.4 µs | 1470.1 µs | **1.77×** |
| 32 | 2836.4 µs | 1794.1 µs | **1.58×** |
| 128 | 6930.8 µs | 1922.8 µs | **3.60×** |
| 256 | 8454.2 µs | 2010.1 µs | **4.21×** |
| 1024 | 34924.8 µs | 3835.6 µs | **9.11×** |
| 2048 | 61277.9 µs | 6218.3 µs | **9.85×** |

### Additional shapes (256 experts, hidden 2048, inter 128, top-8)

| tokens | master | this PR | speedup |
|---:|---:|---:|---:|
| 1 | 146.0 µs | 47.1 µs | **3.10×** |
| 2 | 169.6 µs | 79.3 µs | **2.14×** |
| 4 | 301.0 µs | 156.9 µs | **1.92×** |
| 8 | 581.0 µs | 274.7 µs | **2.12×** |
| 16 | 1093.7 µs | 389.8 µs | **2.81×** |
| 32 | 1470.2 µs | 513.3 µs | **2.86×** |
| 64 | 1766.1 µs | 696.6 µs | **2.54×** |
| 128 | 1782.4 µs | 979.7 µs | **1.82×** |
| 256 | 1806.3 µs | 1016.9 µs | **1.78×** |
| 512 | 3977.4 µs | 1061.7 µs | **3.75×** |
| 1024 | 4342.4 µs | 1122.5 µs | **3.87×** |
| 2048 | 10229.9 µs | 1473.3 µs | **6.94×** |
| 4096 | 15512.0 µs | 2578.6 µs | **6.02×** |

### Comparison with SGLang's Triton kernel
SGLang 0.5.11's Triton `fused_moe`, run as the FlagOS SGLang plugin runs it on this GPU: the default config (SGLang has no tuned table for gfx1100), FlagGems `moe_align_block_size` and `silu_and_mul` (the plugin's AMD patch), and the `torch.compile` top-k sum up to 32 tokens, the Triton one above. Same process and inputs, `triton.testing.do_bench` medians; max error against a float32 reference rounded to bf16, as in the unit test.

#### 64 experts, hidden 2048, inter 1408, top-8 (the official benchmark layer)

| tokens | SGLang | this PR | speedup | SGLang max error | this PR max error |
|---:|---:|---:|---:|---:|---:|
| 1 | 854.7 µs | 359.1 µs | **2.38×** | 1.6e-02 | 2.4e-04 |
| 2 | 1113.3 µs | 504.8 µs | **2.21×** | 1.6e-02 | 3.9e-03 |
| 4 | 1365.9 µs | 764.9 µs | **1.79×** | 3.1e-02 | 3.9e-03 |
| 8 | 1677.1 µs | 1148.7 µs | **1.46×** | 1.6e-02 | 3.9e-03 |
| 16 | 1962.6 µs | 1470.1 µs | **1.33×** | 3.1e-02 | 7.8e-03 |
| 32 | 2108.1 µs | 1794.1 µs | **1.18×** | 6.2e-02 | 1.6e-02 |
| 64 | 2164.3 µs | 1844.3 µs | **1.17×** | 3.1e-02 | 1.6e-02 |
| 128 | 2972.5 µs | 1922.8 µs | **1.55×** | 6.2e-02 | 1.6e-02 |
| 256 | 3017.3 µs | 2010.1 µs | **1.50×** | 3.1e-02 | 1.6e-02 |
| 512 | 2911.3 µs | 2472.6 µs | **1.18×** | 3.1e-02 | 1.6e-02 |
| 1024 | 3162.8 µs | 3835.6 µs | 0.82× | 6.2e-02 | 1.6e-02 |
| 2048 | 5302.6 µs | 6218.3 µs | 0.85× | 6.2e-02 | 1.6e-02 |
| 4096 | 10124.9 µs | 11356.9 µs | 0.89× | 6.2e-02 | 3.1e-02 |

#### 256 experts, hidden 2048, inter 128, top-8

| tokens | SGLang | this PR | speedup | SGLang max error | this PR max error |
|---:|---:|---:|---:|---:|---:|
| 1 | 287.3 µs | 47.1 µs | **6.10×** | 3.9e-03 | 7.6e-06 |
| 2 | 318.3 µs | 79.3 µs | **4.01×** | 2.4e-03 | 2.4e-04 |
| 4 | 311.3 µs | 156.9 µs | **1.98×** | 3.9e-03 | 9.8e-04 |
| 8 | 547.1 µs | 274.7 µs | **1.99×** | 3.9e-03 | 9.8e-04 |
| 16 | 970.5 µs | 389.8 µs | **2.49×** | 3.9e-03 | 4.9e-04 |
| 32 | 1087.9 µs | 513.3 µs | **2.12×** | 3.9e-03 | 9.8e-04 |
| 64 | 1241.8 µs | 696.6 µs | **1.78×** | 3.9e-03 | 9.8e-04 |
| 128 | 1272.1 µs | 979.7 µs | **1.30×** | 3.9e-03 | 3.9e-03 |
| 256 | 1292.3 µs | 1016.9 µs | **1.27×** | 7.8e-03 | 2.0e-03 |
| 512 | 1833.1 µs | 1061.7 µs | **1.73×** | 4.2e-03 | 2.0e-03 |
| 1024 | 1924.3 µs | 1122.5 µs | **1.71×** | 1.6e-02 | 2.0e-03 |
| 2048 | 1772.4 µs | 1473.3 µs | **1.20×** | 7.8e-03 | 3.9e-03 |
| 4096 | 2275.0 µs | 2578.6 µs | 0.88× | 7.8e-03 | 3.9e-03 |

- From 1024 tokens on the 64-expert layer, and at 4096 tokens on the 256-expert layer, this PR runs at 0.82–0.89× SGLang's speed. That is the price of accuracy: the intermediate keeps float32 accuracy (a 16-bit value plus its 16-bit residual), so the down GEMM runs two dots, about a third more math, while SGLang keeps the intermediate and the rows before the top-k sum in bf16.
- SGLang's max error is 2–10× this PR's from 2 tokens on (equal once, on the 256-expert layer at 128 tokens), and 64× / 512× for a single token, where the pair path runs in float32 throughout. At the unit-test tolerances (bf16 2e-2, fp16 1e-2), on 32 cases with 1024–8192 tokens (bf16 / fp16; SiLU with `routed_scaling_factor` 1 and GELU with 2.5), SGLang exceeds the tolerance in 8 cases, by up to 3.1×, including all 6 bf16 cases of the 64-expert layer; this PR passes all 32, using at most 35% of the tolerance.
- With a bf16 intermediate, this PR would run at 1.09–1.15× SGLang's speed from 1024 tokens on the 64-expert layer, but would exceed the tolerance in 5 of those 32 cases.

## Test plan
- [x] `tests/test_fused_moe.py` on gfx1100, with the rdna3 op and with `ARCH=none`
- [x] Against a float32 reference at the unit-test tolerances, 105 cases through both paths: bf16 / fp16, SiLU / GELU, `routed_scaling_factor` 1 / 2.5, half of the experts without tokens, 8–256 experts, top-2 to top-8, hidden 128–2048, inter 80–1408, 1–513 tokens; mean error relative to master's: median 0.98×
- [x] 12 cases with 2048–8192 tokens (`BLOCK_M` 64) against a per-expert float32 reference: same max and mean error as master
- [x] Official comprehensive bench (tables above)
- [x] black / flake8 / isort
