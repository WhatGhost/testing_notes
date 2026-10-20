# W7900D（RDNA3）与 R9700（RDNA4）上 FlagGems-sglang 算子的优化空间（合并）

本文合并两份文档：`R9700_AMD_optimization.md`（Radeon AI PRO R9700，RDNA4，gfx1201）和 `W7900D_AMD_optimization.md`（Radeon PRO W7900D，RDNA3，gfx1100），都以 RTX 4090 为基准，只关注哪些算子值得优化、潜力有多大。先分别列出两张卡原来的分批，再把 56 个算子合成一个总的分批（见「合并后的总批次」）。数字都取自这两份文档及其依据的三份报告（同样 484 个形状的 kernel 模式延迟），本文没有新的实验；各项改法的细节、正确性和原始数据见两份文档。

共 56 个算子：44 个有大形状（4090 上不少于 50 µs 的形状，共 100 个），12 个只有小形状。两张卡上做过的初步测试不一样多：

- R9700：对 4 个算子的改法做了初步测试，`context_attention`、`fused_moe_router_tensorcore`、`causal_conv1d_fn` 明显变快，`gelu_and_mul` 调分块没有收益。其余是推算。
- W7900D：做了两轮初步测试，18 个算子有收益；另对全部算子扫了一遍 num_warps。其余是推算。
- 两张卡之间：R9700 第一批的三项改法在 W7900D 上也测过，`fused_moe_router_tensorcore`、`causal_conv1d_fn` 同样变快，`context_attention` 的 bf16x3 在 W7900D 上反而变慢；W7900D 上的改法都没有在 R9700 上测过。

## 硬件和整体情况

| 项目 | Radeon AI PRO R9700 | Radeon PRO W7900D | RTX 4090 |
|---|---|---|---|
| 架构 | RDNA4（gfx1201） | RDNA3（gfx1100） | Ada（sm_89） |
| CU / SM | 64 CU | 96 CU | 128 SM |
| 峰值显存带宽 | 640 GB/s | 864 GB/s | 1008 GB/s |
| FP32 向量（TFLOPS） | 47.8（单发射 23.9） | 53（单发射 26.5） | 82.6 |
| FP16 / BF16 矩阵，FP32 累加（TFLOPS） | 191 | 106 | 165.2 |
| INT8 矩阵（TOPS） | 383 | 106 | 660.6 |
| 计时下限（报告中最短的算子延迟） | 6.48 µs | 8.80 µs | 3.07 µs |
| WMMA kernel 的写回 | 主要是 16 字节向量写 | 拆成 2 或 4 字节的标量写 | - |
| scratch 门槛（折合每条 lane） | 约 2.2 KB（按 CU 数推算，没有验证） | 1493 B | - |

- 规格出自 AMD 官网规格页和 NVIDIA 的 Ada 架构白皮书，都是不含结构化稀疏的稠密峰值；W7900D 一列与 `W7900D_report.md` 的规格表相同，其中 BF16、INT8 是按与 FP16 矩阵同速推算的。AMD 的 FP32 峰值按双发射计算，Triton 生成的代码很少用到双发射。
- 以 W7900D 为 1，R9700 的带宽是 0.74 倍，BF16 矩阵峰值是 1.80 倍，INT8 是 3.6 倍。R9700 带宽最低、BF16 矩阵最强，同一个改法在两张卡上的收益可能差很多，例如 `context_attention` 的 bf16x3。
- scratch 门槛：每条 lane 溢出超过它的 kernel，ROCm 每次启动都要重新分配 scratch，W7900D 上每次多花约 150–200 µs；用环境变量 `HSA_SCRATCH_SINGLE_LIMIT` 可以调高。

| 项目 | R9700 | W7900D |
|---|---:|---:|
| 按峰值带宽预期，访存型算子的耗时是 4090 的 | 1.57× | 1.17× |
| 大形状耗时比（100 个形状的几何平均） | 2.34× | 2.51× |
| 达到预期的比例 | 67% | 46% |
| 带宽归一化效率 < 0.5 / 0.5–0.8 / ≥ 0.8 的算子数 | 4 / 26 / 14 | 15 / 28 / 1 |
| 简单访存型算子的峰值带宽利用率（4090 为 87%–91%） | 61%–84% | 45%–65% |
| 延迟 ≤ 1.5 × 计时下限的形状 | 220 / 484 | 238 / 484 |
| 初步测试改后的大形状耗时比 | 没有汇总（只有 3 个算子有改后的数据） | 1.67×（18 个算子改后；达到预期的 70%，带宽归一化效率 6 / 32 / 6） |

带宽归一化效率 = (4090 耗时 ÷ AMD 耗时) ÷ (AMD 峰值带宽 ÷ 4090 峰值带宽)，只看大形状。等于 1 表示带宽利用率与 4090 相同；对计算型算子只能用来排序。

两张卡慢的原因大体相同：AMD 后端没有专门实现，56 个算子都走通用实现，参数是按别的芯片选的；fp32 输入的矩阵乘在 ROCm 上编译成 FMA；`fused_moe_gemm` 是逐 token 的 GEMV（4090 上也一样）；访存型算子的分块和 warp 数没有针对 AMD 调过。W7900D 原来的效率更低，但做过初步测试的改动更多：18 个算子改后，大形状达到按带宽预期的 70%，与 R9700 改动前（67%）相当。

## 各卡原来的分批

两份文档都是先把有大形状的 44 个算子按能快多少分批，再另列只有小形状的算子。同一批内按能快的倍数排序。第一批括号内是初步测试中各算子的加速比，不加说明时指大形状。

### R9700（`R9700_AMD_optimization.md`）

| 批次 | 判断依据 | 个数 | 算子 |
|---|---|---:|---|
| 第一批 | 已做初步测试，大形状大约快 2–14 倍 | 3 | `fused_moe_router_tensorcore`（5.0–14.3×）、`causal_conv1d_fn`（2.2–4.1×）、`context_attention`（2.2–3.1×） |
| 第二批 | 要改写 kernel，推算可快 1.7 倍以上 | 4 | `fused_moe_gemm`、`chain_speculative_sampling`、`embedding_lora_a`、`per_group_transpose` |
| 第三批 | 推算可快 1.2–1.7 倍，差距应来自分块、num_warps 等参数 | 25 | `chunk_state_varlen`、`silu_and_mul`、`fla_layernorm_gated`、`log_scaling_tau`、`bmm_chunk`、`gate_up_lora_b`、`moe_sum_reduce`、`fused_rmsnorm`、`silu_and_mul_masked`、`w8a8_block_int8_matmul`、`merge_state`、`per_token_group_quant_int8`、`mamba_layernorm_gated`、`chunked_sgmv_expand`、`per_token_quant_int8`、`moe_fused_mul_sum`、`mrope_fused`、`ernie45_rope_fused`、`qkv_lora_b`、`rotary_embedding`、`apply_token_bitmask`、`chunk_local_cumsum_vector`、`draft_topk1`、`fused_norm_rope_stacked`、`state_passing` |
| 第四批 | 已接近 4090，推算不到 1.2 倍，或初步测试没有收益 | 12 | `fused_dual_residual_rmsnorm`、`softcap_inplace_logits`、`fused_moe_router_cudacore`、`chunk_scaled_dot_kkt`、`chunk_state`、`softcap_out`、`extend_attention`、`selective_state_update`、`decode_attention`、`hc_head`、`gelu_and_mul`、`decode_grouped_attention` |
| 只有小形状 | 4090 上都不到 50 µs，以固定开销为主：R9700 上是 4090 的 1.60–2.01 倍，不超过计时下限之比（2.11 倍），单个 kernel 能省的不多 | 12 | `l2norm`、`chunk_local_cumsum_scalar`、`fused_gdn_gating`、`chunked_embedding_lora_a`、`chunked_sgmv_shrink`、`sgemm_lora_a`、`sgemm_lora_b`、`causal_conv1d_update`、`chunk_cumsum`、`moe_fused_gate`、`sigmoid_gate_topk_renorm`、`interleaved_rope` |

### W7900D（`W7900D_AMD_optimization.md`）

| 批次 | 判断依据 | 个数 | 算子 |
|---|---|---:|---|
| 第一批 | 已做初步测试，按算子不同大约快 1.2–18.5 倍 | 18 | `context_attention`（4.6–18.5×）、`fused_moe_gemm`（T ≥ 8 时 2.0–17.2×）、`fused_moe_router_tensorcore`（7.8–11.0×）、`causal_conv1d_fn`（4.5–9.4×）、`decode_grouped_attention`（batch 1、8 时 5.5×、2.0×，batch ≥ 64 时 1.0–1.1×）、`decode_attention`（batch 1 时 3.2×，其余 1.1–1.2×）、`extend_attention`（1.3–2.2×）、`chunk_state_varlen`（1.5–2.4×）、`w8a8_block_int8_matmul`（1.4–1.5×）、`chunked_sgmv_expand`（1.8–1.9×）、`gate_up_lora_b`（1.1–1.6×）、`sgemm_lora_b`（只有小形状，1.1–1.7×）、`per_group_transpose`（最大形状 1.4×）、`bmm_chunk`（1.2×）、`fused_rmsnorm`（1.2×）、`silu_and_mul`（1.2–1.4×）、`log_scaling_tau`（最大形状 1.2×）、`fused_moe_router_cudacore`（大形状不变，两个小形状 1.8×、14.5×） |
| 第二批 | 要改编译器或数据布局。推算可快 1.47 倍，与第四批相当；初步测试中跳过 `optimize_epilogue` 只快 1.03–1.11 倍，改分块没有收益 | 1 | `chunk_state` |
| 第三批 | 要改写 kernel，推算可快 1.7 倍以上 | 12 | `chain_speculative_sampling`、`merge_state`、`fla_layernorm_gated`、`mrope_fused`、`ernie45_rope_fused`、`silu_and_mul_masked`、`per_token_quant_int8`、`qkv_lora_b`、`state_passing`、`mamba_layernorm_gated`、`per_token_group_quant_int8`、`embedding_lora_a` |
| 第四批 | 推算可快 1.2–1.7 倍 | 10 | `chunk_local_cumsum_vector`、`fused_norm_rope_stacked`、`fused_dual_residual_rmsnorm`、`moe_fused_mul_sum`、`rotary_embedding`、`apply_token_bitmask`、`chunk_scaled_dot_kkt`、`selective_state_update`、`hc_head`、`draft_topk1` |
| 第五批 | 已接近上限，或初步测试没有收益：推算 1.15–1.67 倍，测过的 3 个最多只快 1.03–1.16 倍 | 4 | `softcap_out`、`moe_sum_reduce`、`gelu_and_mul`、`softcap_inplace_logits` |
| 只有小形状 | 4090 上都不到 50 µs，以固定开销为主：只改 num_warps 最多快 1.14 倍，例外是 `chunked_embedding_lora_a`，调高 scratch 门槛快 3.27 倍（`sgemm_lora_b` 已在第一批） | 11 | `l2norm`、`chunk_local_cumsum_scalar`、`fused_gdn_gating`、`chunked_embedding_lora_a`、`chunked_sgmv_shrink`、`sgemm_lora_a`、`causal_conv1d_update`、`chunk_cumsum`、`moe_fused_gate`、`sigmoid_gate_topk_renorm`、`interleaved_rope` |

## 合并后的总批次

合并的规则：先看是否做过初步测试，再看潜力大小。W7900D 上的初步测试说明推算值偏乐观：调参数的 7 个访存型算子中，只有问题明确的 `causal_conv1d_fn` 接近推算值，其余 6 个只快 1.0–1.4 倍（推算 1.6–3.1 倍）。所以做过初步测试、已经看到收益的排在只有推算的前面。

| 总批次 | 潜力 | 判断依据 | 个数 | 算子 |
|---|---|---|---:|---|
| 第一批 | 高 | 已做初步测试，大形状大约快 2–18.5 倍；慢的原因（fp32 矩阵乘走 FMA、逐 token 的 GEMV、分块参数）两张卡相同 | 4 | `context_attention`、`fused_moe_router_tensorcore`、`causal_conv1d_fn`、`fused_moe_gemm` |
| 第二批 | 中 | W7900D 上已做初步测试，大形状几何平均快 1.18–1.79 倍（`fused_moe_router_cudacore`、`sgemm_lora_b` 的收益在小形状上）；R9700 上没有测 | 14 | `chunked_sgmv_expand`、`decode_grouped_attention`、`extend_attention`、`gate_up_lora_b`、`chunk_state_varlen`、`w8a8_block_int8_matmul`、`per_group_transpose`、`decode_attention`、`silu_and_mul`、`fused_rmsnorm`、`bmm_chunk`、`log_scaling_tau`、`fused_moe_router_cudacore`、`sgemm_lora_b` |
| 第三批 | 中到高，只有推算 | 没有测试；至少一张卡推算可快 1.7 倍以上（W7900D 1.74–3.10 倍，R9700 1.22–2.04 倍），大多要改写 kernel | 12 | `chain_speculative_sampling`、`merge_state`、`fla_layernorm_gated`、`mrope_fused`、`ernie45_rope_fused`、`embedding_lora_a`、`silu_and_mul_masked`、`per_token_quant_int8`、`qkv_lora_b`、`state_passing`、`mamba_layernorm_gated`、`per_token_group_quant_int8` |
| 第四批 | 较小 | 推算不到 1.7 倍（W7900D 1.37–1.69 倍，R9700 1.04–1.35 倍），没有发现具体问题或已知的问题改不动 | 11 | `chunk_local_cumsum_vector`、`fused_norm_rope_stacked`、`fused_dual_residual_rmsnorm`、`moe_fused_mul_sum`、`rotary_embedding`、`apply_token_bitmask`、`chunk_scaled_dot_kkt`、`chunk_state`、`selective_state_update`、`hc_head`、`draft_topk1` |
| 第五批 | 很少 | 预期不到 1.2 倍：测过的最多只快 1.00–1.16 倍，没有测的最大形状已到峰值带宽的 78%–87% | 4 | `softcap_out`、`moe_sum_reduce`、`gelu_and_mul`、`softcap_inplace_logits` |
| 只有小形状 | 很少（单个 kernel） | 4090 上都不到 50 µs，以 kernel 启动和计时的固定开销为主；W7900D 上只改 num_warps 最多快 1.14 倍，例外是 `chunked_embedding_lora_a`（调高 scratch 门槛快 3.27 倍）。要靠合并 kernel、去掉主机同步、用 HIP graph | 11 | `chunked_embedding_lora_a`、`fused_gdn_gating`、`chunked_sgmv_shrink`、`sigmoid_gate_topk_renorm`、`l2norm`、`chunk_local_cumsum_scalar`、`moe_fused_gate`、`chunk_cumsum`、`interleaved_rope`、`causal_conv1d_update`、`sgemm_lora_a` |

合计 4 + 14 + 12 + 11 + 4 + 11 = 56 个。与两份文档的分批相比，变动较大的是：

- 上调：`fused_moe_gemm`（R9700 第二批）在 W7900D 上初步测试快到 17 倍，进第一批。`extend_attention`、两个 decode 算子、`fused_moe_router_cudacore` 在 R9700 文档中是第四批；W7900D 上查到的原因与硬件无关，或两张卡上都有，进第二批。
- 下调：`chain_speculative_sampling`、`embedding_lora_a`（R9700 第二批）两张卡上都没有测试，放第三批。`moe_sum_reduce`（R9700 第三批）在 W7900D 上搜索 45 种分块最多快 1.09 倍，R9700 上最大形状已到峰值带宽的 78%，放第五批。`chunk_state`（W7900D 第二批）在 W7900D 上改分块没有收益，放第四批。

下面逐批列出各算子，各表的列含义如下：

- 「大形状是 4090 的倍数」：各算子大形状耗时相对 4090 的几何平均。「→」后面是初步测试中改后的值。
- 「可快」：标「初测」的是初步测试中相对同一张卡上上游实现的加速比，不加说明时指大形状。标「推算」的是做到与 4090 相同的带宽利用率时的上限，即「大形状是 4090 的倍数」÷ 按带宽预期的倍数（R9700 1.57、W7900D 1.17）；`fused_moe_gemm`、`w8a8_block_int8_matmul` 按算力利用率推算。括号里是该算子在各自文档中的批次。
- 「说明」一栏空着的，两份文档都没有发现具体问题。

### 第一批：潜力高，已做初步测试（4 个）

同一批内按 W7900D 上大形状几何平均的改善排序。

| 算子 | 用途 | R9700：大形状是 4090 的倍数 | R9700：可快 | W7900D：大形状是 4090 的倍数 | W7900D：可快 | 改法和两张卡的差别 |
|---|---|---:|---|---:|---|---|
| `context_attention` | 注意力 | 10.68 → 4.16 | 初测 2.23–3.11×（第一批） | 8.67 → 0.87 | 初测 4.64–18.53×（第一批） | fp32 输入的 `tl.dot` 编译成 FMA，只用到 FP32 峰值的 3.8%–4.4%。R9700 改用 bf16x3，改后仍是 4090 的 3.6 倍，还有空间。W7900D 上 bf16x3 反而慢（0.44–0.69×），改用 fp16 WMMA（fp32 累加），再去掉主机同步、让同一 (序列, 头) 的 query 块在网格中相邻、K/V 先整体转成 fp16，最大形状比 4090 快 3.0 倍。后几项与 dot 的精度无关，R9700 上应同样有收益，没有测 |
| `fused_moe_router_tensorcore` | MoE | 7.88 → 1.31（bf16x3）、0.64（分块求和） | 初测 5.04–7.17×（bf16x3）、10.56–14.31×（分块求和）（第一批） | 9.97 → 1.06 | 初测 7.76–11.01×（第一批） | 上游把 bf16 输入提升成 fp32 再按 FMA 算。两张卡都可以直接用 bf16 WMMA、按 K 分块求和。R9700 上分块求和在 M=4096 的测试用例上略超容差（容差占比 1.01，W7900D 上 0.88），只用 bf16x3 时是 5.04–7.17× |
| `causal_conv1d_fn` | Mamba | 4.84 → 1.61 | 初测 2.24–4.07×（第一批） | 10.00 → 1.52 | 初测 4.54–9.35×（第一批） | 两张卡用同一规则：BLOCK_D=1，BLOCK_T 按最大序列长度取，只改启动参数。最大形状的峰值带宽利用率 R9700 从 21% 升到 84%，W7900D 从 7% 升到 65% |
| `fused_moe_gemm` | MoE | 2.25 | 推算约 22×（第二批） | 1.57 → 0.40 | 初测 T ≥ 8 时 2.00–17.18×，T=1 不变（第一批） | 上游每个 (token, slot) 单独做 GEMV，三张卡都只用到矩阵峰值的 1.5%–3.6%。改成先按专家排序、再用 `tl.dot` 的分组 GEMM，与硬件无关；W7900D 上 T=4096 到 BF16 峰值的 63%，比 4090 快 13 倍 |

### 第二批：潜力中等，W7900D 上已做初步测试（14 个）

大多只改参数（num_warps、分块、流水级数），是按 gfx1100 测出来的，R9700 上的收益不确定；`extend_attention` 和两个 decode 算子改的是算法和网格，R9700 上应同样有收益。同一批内按 W7900D 上大形状几何平均的改善（现在 ÷ 改后）排序。

| 算子 | 用途 | R9700：大形状是 4090 的倍数 | R9700：可快 | W7900D：大形状是 4090 的倍数 | W7900D：可快 | 改法和两张卡的差别 |
|---|---|---:|---|---:|---|---|
| `chunked_sgmv_expand` | LoRA | 2.24 | 推算 1.42×（第三批） | 2.87 → 1.60 | 初测 1.79–1.88×（第一批） | num_warps 从 4 改成 16。两张卡上都有 2 字节标量读写；W7900D 上跳过 `optimize_epilogue` 也快 1.89×，两者能否叠加没有测 |
| `decode_grouped_attention` | 注意力 | 1.35 | 带宽利用率已高于 4090（第四批） | 1.71 → 1.03 | 初测 batch 1、8：5.49×、1.99×；batch 64–4096：0.98–1.11×；head_dim 512：1.70×（第一批） | batch 1 时上游只有 8 个程序，改成按序列切分（split-KV）。R9700 上大 batch 已高于 4090，小 batch 网格太小的问题应同样存在 |
| `extend_attention` | 注意力 | 1.81 | 推算 1.15×（第四批） | 2.73 → 1.73 | 初测 1.28–2.20×（第一批） | 上游先用单独的 kernel 把前缀 K/V 拷成连续的缓冲区（只到峰值带宽的 4%–21%），每个 query 头又各读一遍 K/V。改成在 attention kernel 里经 `kv_indices` 直接读前缀，共享一个 KV 头的 query 头放进同一个 tile。与硬件无关，R9700 上应也有收益 |
| `gate_up_lora_b` | LoRA | 2.33 | 推算 1.48×（第三批） | 3.80 → 2.41 | 初测 1.59×，8 段 × 64 个 token 1.10×（第一批） | num_warps 从 4 改成 16 |
| `chunk_state_varlen` | Mamba | 2.46 | 推算 1.56×（第三批） | 4.54 → 3.01 | 初测 1.51–2.42×（第一批） | gfx1100 上两个 dot 的操作数都要经 LDS 转置，编译成逐个元素的 2 字节 LDS 读。改成每块 1 个头、BLOCK_K 16、1 级流水。gfx1201 上有没有同样的问题没有核对；R9700 上 2 / 5 个版本溢出（最多 120 B） |
| `w8a8_block_int8_matmul` | 量化 | 2.40 | 推算 1.28–1.43×（第三批） | 6.03 → 4.23 | 初测 1.35–1.50×，[64, 512] × [1024, 512] 12.88×（第一批） | gfx1100 上 BLOCK_M=64 的版本全部溢出（480–1564 B），改成 BLOCK_M ≤ 32、2 级流水，到 INT8 峰值的 64%。R9700 上不溢出、只到 INT8 峰值的 33%，要另调 AMD 的 autotune 配置 |
| `per_group_transpose` | 量化 | 2.76 | 推算 1.75×（第二批） | 3.62 → 2.59 | 初测最大形状 1.42×（第一批） | 数据量不少于 2^24 个元素时用 8 个 warp。两张卡上转置写回都是 2 字节标量写；R9700 文档建议的向量化写回，W7900D 上只再快 4.5%，最大形状也只到峰值带宽的 34% |
| `decode_attention` | 注意力 | 1.66 | 推算 1.05×，最大形状到峰值带宽的 88%（第四批） | 1.77 → 1.28 | 初测 batch 1 3.23×，batch 8–4096 1.08–1.20×（第一批） | split-KV，num_warps 从 8 改成 4。R9700 上大 batch 已接近带宽上限，小 batch 网格太小的问题应同样存在 |
| `silu_and_mul` | 激活、归一化 | 2.37 | 推算 1.51×（第三批） | 2.35 → 1.82 | 初测 1.21–1.39×（第一批） | 大形状走按行循环的 kernel，网格最多 512 个程序；都改用二维 kernel、16 个 warp。R9700 上读取已是 16 字节向量，最大形状只到峰值带宽的 61% |
| `fused_rmsnorm` | 激活、归一化 | 2.28 | 推算 1.45×（第三批） | 2.44 → 1.99 | 初测 1.21–1.22×（第一批） | 上游每行一个程序、2 个 warp，大行宽时占用率只有 7。按行宽选 num_warps（R9700 文档的建议） |
| `bmm_chunk` | Mamba | 2.34 | 推算 1.49×（第三批） | 2.52 → 2.07 | 初测 1.22–1.24×（第一批） | 1 个 warp；W7900D 上跳过 `optimize_epilogue` 只快 1.08–1.10× |
| `log_scaling_tau` | 注意力 | 2.35 | 推算 1.50×（第三批） | 2.35 → 1.99 | 初测 1.18×（第一批） | 程序数不少于 4096 时用 16 个 warp，benchmark 中只有最大的形状用到 |
| `fused_moe_router_cudacore` | MoE | 1.85；[83, 512] × 32 个专家是 4090 的 5.0 倍 | 推算 1.17×（第四批） | 1.84 → 1.86；[83, 512] 从 4090 的 19.2 倍降到 1.31 倍 | 初测大形状不变，[83, 512] × 32 个专家 14.50×，[37, 256] × 16 个专家 1.75×（第一批） | 专家数 ≤ 32 时 1 个 warp 处理 32×512 的 tile，两张卡都溢出约 2 KB（R9700 2068 B、W7900D 2056 B），改成 block_h > 128 时用 4 个 warp。W7900D 上这个溢出超过 scratch 门槛，该形状 261.6 µs（R9700 67.88 µs、4090 13.66 µs） |
| `sgemm_lora_b` | LoRA | 只有小形状 | - | 只有小形状 | 初测两个较大的形状 1.73×、1.13×（第一批） | num_warps 从 4 改成 16 |

### 第三批：潜力中到高，只有推算（12 个）

两张卡上都没有测试过改写 kernel。W7900D 上只改 num_warps 最多快 1.17 倍，要拿到推算的收益，大概要改写 kernel 或先查清原因。「W7900D：可快」一栏同时列出只改 num_warps 的初测结果，各大形状都不到 1.05 倍的记为没有收益。同一批内按两张卡中较大的推算值排序。

| 算子 | 用途 | R9700：大形状是 4090 的倍数 | R9700：可快 | W7900D：大形状是 4090 的倍数 | W7900D：可快 | 说明 |
|---|---|---:|---|---:|---|---|
| `chain_speculative_sampling` | 采样、语法约束 | 3.21 | 推算 2.04×（第二批） | 3.62 | 推算 3.10×；num_warps 1.00–1.17×（第三批） | ROCm 路径上一次调用要执行多个 kernel（缓存中 11 种），含按顺序认领、发布的步骤，要合并 kernel。两张卡上都没有做原型 |
| `merge_state` | 注意力 | 2.25 | 推算 1.43×（第三批） | 2.58 | 推算 2.21×；num_warps 1.06×（第三批） |  |
| `fla_layernorm_gated` | 线性注意力（FLA） | 2.36 | 推算 1.50×（第三批） | 2.30 | 推算 1.97×；num_warps 1.13×（第三批） |  |
| `mrope_fused` | RoPE | 2.12 | 推算 1.34×（第三批） | 2.29 | 推算 1.96×；num_warps 1.12×（第三批） |  |
| `ernie45_rope_fused` | RoPE | 2.11 | 推算 1.34×（第三批） | 2.27 | 推算 1.94×；num_warps 1.11×（第三批） | 两张卡上都有 2 字节标量读 |
| `embedding_lora_a` | LoRA | 2.97 | 推算 1.88×（第二批） | 2.03 | 推算 1.74×；num_warps 没有收益（第三批） | 按行 gather，有 2 字节标量读 |
| `silu_and_mul_masked` | MoE | 2.28 | 推算 1.45×（第三批） | 2.19 | 推算 1.88×；带 autotune，没有扫（第三批） | 两张卡上 autotune 选中的版本都不溢出 |
| `per_token_quant_int8` | 量化 | 2.18 | 推算 1.38×（第三批） | 2.16 | 推算 1.85×；num_warps 1.07–1.08×（第三批） |  |
| `qkv_lora_b` | LoRA | 2.03 | 推算 1.29×（第三批） | 2.15 | 推算 1.84×；num_warps 没有收益（第三批） | 封装里有主机同步，最小的形状 4090 上也要 41.98 µs；2 字节标量读写，W7900D 上经 LDS 写回只快 1.05–1.07× |
| `state_passing` | Mamba | 1.92 | 推算 1.22×（第三批） | 2.12 | 推算 1.82×；带 autotune，没有扫（第三批） | 两张卡上 autotune 选中的版本都不溢出 |
| `mamba_layernorm_gated` | Mamba | 2.24 | 推算 1.42×（第三批） | 2.08 | 推算 1.78×；num_warps 没有收益（第三批） | 有 2 字节标量读写 |
| `per_token_group_quant_int8` | 量化 | 2.25 | 推算 1.43×（第三批） | 2.07 | 推算 1.78×；num_warps 1.07–1.10×（第三批） |  |

### 第四批：潜力较小（11 个）

推算不到 1.7 倍。按 W7900D 上访存型算子的初步测试，调参数大概只能快 1.0–1.4 倍；W7900D 上只改 num_warps，只有 `chunk_local_cumsum_vector` 快到 1.17 倍。`chunk_state` 是例外：问题已经查明，但改分块没有收益，要改数据布局。同一批内按两张卡中较大的推算值排序。

| 算子 | 用途 | R9700：大形状是 4090 的倍数 | R9700：可快 | W7900D：大形状是 4090 的倍数 | W7900D：可快 | 说明 |
|---|---|---:|---|---:|---|---|
| `chunk_local_cumsum_vector` | 线性注意力（FLA） | 1.97 | 推算 1.25×（第三批） | 1.98 | 推算 1.69×；1 个 warp 1.17×（第四批） |  |
| `fused_norm_rope_stacked` | 投机解码 | 1.94 | 推算 1.23×（第三批） | 1.93 | 推算 1.65×；num_warps 没有收益（第四批） |  |
| `fused_dual_residual_rmsnorm` | 激活、归一化 | 1.87 | 推算 1.19×（第四批） | 1.92 | 推算 1.65×；num_warps 没有收益（第四批） |  |
| `moe_fused_mul_sum` | MoE | 2.13 | 推算 1.35×（第三批） | 1.88 | 推算 1.61×；1 个 warp 1.00–1.06×（第四批） |  |
| `rotary_embedding` | 扩散模型 | 1.99 | 推算 1.26×（第三批） | 1.83 | 推算 1.57×；16 个 warp 1.04–1.09×（第四批） |  |
| `apply_token_bitmask` | 采样、语法约束 | 1.99 | 推算 1.26×（第三批） | 1.82 | 推算 1.56×；1 个 warp 1.00–1.08×（第四批） |  |
| `chunk_scaled_dot_kkt` | 线性注意力（FLA） | 1.84 | 推算 1.17×（第四批） | 1.74 | 推算 1.49×；num_warps 没有收益（第四批） | R9700 上 gstore kernel 的 2 / 2 个版本溢出（最多 84 B）；W7900D 上 storeopt kernel 的写回是 4 字节标量写，改成向量写后耗时不变 |
| `chunk_state` | Mamba | 1.82 | 推算 1.16×（第四批） | 1.71 | 推算 1.47×；跳过 `optimize_epilogue` 初测 1.03–1.11×，13 种分块组合没有收益（第二批） | gfx1100 上与 `chunk_state_varlen` 一样，两个 dot 的操作数都经 LDS 转置；要改数据布局（让 token 方向连续），或用 Gluon 显式指定布局 |
| `selective_state_update` | Mamba | 1.79 | 推算 1.14×（第四批） | 1.65 | 推算 1.41×；num_warps 没有收益（第四批） |  |
| `hc_head` | 激活、归一化 | 1.64 | 推算 1.04×（第四批） | 1.60 | 推算 1.37×；num_warps 没有收益（第四批） | R9700 上带宽归一化效率 0.96，已接近 4090 |
| `draft_topk1` | 投机解码 | 1.94 | 推算 1.23×（第三批） | 1.60 | 推算 1.37×；num_warps 没有收益（1.03–1.04×）（第四批） |  |

### 第五批：潜力很少（4 个）

都是简单的访存型算子，已接近峰值带宽，或调参数的初步测试没有收益。W7900D 上同样数据量的拷贝也只跑到峰值带宽的 70%（读写共 256 MiB）到 84%（1 GiB）。

| 算子 | 用途 | R9700：大形状是 4090 的倍数 | R9700：可快 | W7900D：大形状是 4090 的倍数 | W7900D：可快 | 说明 |
|---|---|---:|---|---:|---|---|
| `softcap_out` | 激活、归一化 | 1.82 | 推算 1.15×，最大形状到峰值带宽的 84%（第四批） | 1.85 | 推算 1.59×，初测 16 种分块最多 1.16×（第五批） | W7900D 上最快的分块只在 [512, 128256] 上快 1.16 倍，另两个大形状 1.03–1.06 倍；只改成 16 个 warp，那两个形状快 1.10–1.12 倍 |
| `moe_sum_reduce` | MoE | 2.31 | 推算 1.46×，但最大形状已到峰值带宽的 78%（第三批） | 1.94 | 推算 1.67×，初测 45 种分块最多 1.09×（第五批） | R9700 上利用率最高的访存型形状是 88%，按这个上限，最大形状最多再快约 1.13 倍 |
| `gelu_and_mul` | 激活、归一化 | 2.12 | 初测 14 种配置最多 1.00×（第四批） | 1.87 | 推算 1.61×，初测 15 种分块最多 1.03×（第五批） | 两张卡上调分块都没有收益，R9700 上还试了向量化读写和按行二维索引；最大形状到峰值带宽的 72%（R9700）、61%（W7900D） |
| `softcap_inplace_logits` | 激活、归一化 | 1.87 | 推算 1.19×，最大形状到峰值带宽的 85%（第四批） | 1.34 | 推算 1.15×，最大形状到峰值带宽的 87%（第五批） | 两张卡上都已接近峰值带宽 |

### 只有小形状（11 个）

这些算子在 4090 上都不到 50 µs。R9700 上是 4090 的 1.60–2.01 倍（R9700 文档只给了范围），不超过两者计时下限之比（2.11 倍）；W7900D 上是 2.01–3.02 倍，除 `chunked_embedding_lora_a` 外不超过两者计时下限之比（2.87 倍）。耗时以 kernel 启动和计时的固定开销为主，单个 kernel 能省的不多，要靠合并 kernel、去掉主机同步、在 SGLang 里用 HIP graph。同样只有小形状的 `sgemm_lora_b` 在 W7900D 上做过初步测试，列在第二批。

W7900D 一列是各算子全部形状的几何平均，取自 `opt_w7900d/results/analysis.json`，按它从大到小排序。

| 算子 | 用途 | W7900D：全部形状是 4090 的倍数 | 说明 |
|---|---|---:|---|
| `chunked_embedding_lora_a` | LoRA | 3.02 | 唯一的版本每条 lane 溢出 1764 B，超过 W7900D 的 scratch 门槛；调高门槛后最大的形状快 3.27 倍（223.6 → 68.4 µs，4090 34.78 µs）。按推算的门槛，R9700 上不超过。改参数消除溢出没有试 |
| `fused_gdn_gating` | 线性注意力（FLA） | 2.58 |  |
| `chunked_sgmv_shrink` | LoRA | 2.36 |  |
| `sigmoid_gate_topk_renorm` | MoE | 2.34 |  |
| `l2norm` | 激活、归一化 | 2.32 |  |
| `chunk_local_cumsum_scalar` | 线性注意力（FLA） | 2.29 |  |
| `moe_fused_gate` | MoE | 2.27 |  |
| `chunk_cumsum` | Mamba | 2.16 |  |
| `interleaved_rope` | RoPE | 2.15 |  |
| `causal_conv1d_update` | Mamba | 2.12 | W7900D 上只改成 8 个 warp 快 1.09–1.14 倍 |
| `sgemm_lora_a` | LoRA | 2.01 |  |

## 两张卡上改法的差别

第一批、第二批中有的改法与硬件有关，同一个改法在两张卡上的收益可能不同：

| 改法 | R9700（gfx1201） | W7900D（gfx1100） |
|---|---|---|
| `context_attention` 的 dot | bf16x3，初测 2.23–3.11× | fp16 单次 WMMA，初测 2.55–3.59×；bf16x3 反而慢（0.44–0.69×） |
| `context_attention` 的网格顺序、去掉主机同步、K/V 先转换 | 没有测；与 dot 的精度无关，应同样有收益 | 在 fp16 写法的基础上再快 1.8–6.4× |
| `fused_moe_router_tensorcore` 改用 bf16 WMMA | bf16x3 初测 5.04–7.17×，分块求和 10.56–14.31× | 分块求和初测 7.76–11.01×，bf16x3 3.15–4.29× |
| `causal_conv1d_fn` 的分块规则 | 同一规则，初测 2.24–4.07× | 同一规则，初测 4.54–9.35× |
| `fused_moe_gemm` 的分组 GEMM、`extend_attention` 去掉 gather 和合并头、decode 的 split-KV | 没有测；与硬件无关，应同样有收益 | 初测：`fused_moe_gemm` T ≥ 8 时 2.00–17.18×，`extend_attention` 1.28–2.20×，decode batch 1 时 3.23–5.49× |
| `w8a8_block_int8_matmul` | 不溢出，只到 INT8 峰值的 33%；要另调 autotune 配置，没有测 | BLOCK_M=64 的版本溢出，改成 BLOCK_M ≤ 32、2 级流水后到 INT8 峰值的 64% |
| `fused_moe_router_cudacore` 的 4 个 warp | 同样溢出约 2 KB，应同样有收益，没有测 | 初测 [83, 512] × 32 个专家 14.50× |
| 各算子的 num_warps（LoRA 16 个、`bmm_chunk` 1 个、`silu_and_mul` 16 个、decode 4 个等） | 要重新扫 | 按 gfx1100 测出 |
| `chunk_state_varlen` 的分块 | 有没有 LDS 转置的问题没有核对 | 两个 dot 的操作数都经 LDS 转置；改分块初测 1.51–2.42× |
| WMMA kernel 的写回 | 主要是 16 字节向量写 | 2 或 4 字节标量写；跳过 `optimize_epilogue` 因 kernel 而异（0.90–2.04×），要改 Triton |
| scratch 门槛 | 约 2.2 KB（推算）；`context_attention` 上游版本溢出 2648 B，也超过，4 头的小形状可能也受影响 | 1493 B；超过的 kernel 每次启动多花约 150–200 µs |

## 潜力判断的不确定性

- 「推算」都是上限，偏乐观：W7900D 上调参数的访存型算子除 `causal_conv1d_fn` 外只快 1.0–1.4 倍（推算 1.6–3.1 倍），R9700 上 `gelu_and_mul` 也没有兑现。第三、四批的实际收益要改写后才知道。
- W7900D 上的改法都没有在 R9700 上测过，第二批在 R9700 上能快多少还不知道。
- `context_attention` 在 W7900D 上的收益依赖 fp16：q、k、v 中绝对值超过 65504 的元素会溢出成 inf，真实模型的激活是否在这个范围内没有确认；不能用 fp16 时，只改分块大形状快 1.10–2.21 倍。K/V 先转换另需显存，最大的形状 2.1 GB。
- `fused_moe_router_tensorcore` 的分块求和在 M=4096 的测试用例上余量很小（fp32 参考本身相对 fp64 已用掉 93%–97% 的容差），R9700 上这个用例略超容差；只能用 bf16x3 时，R9700 上的收益约为分块求和的一半（5.04–7.17×）。
- 初步测试只在 benchmark 的形状上比较过耗时（`causal_conv1d_fn` 的分块规则、split-KV 的配置、`per_group_transpose` 的 2^24 门槛等），实际模型的形状上收益可能不同。
