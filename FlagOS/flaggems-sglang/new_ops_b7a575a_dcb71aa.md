# FlagGems-sglang 新增算子（b7a575a → dcb71aa）

本目录 RTX 4090、W7900D、R9700 的测试都基于 flagos-ai/FlagGems-sglang 的 b7a575a，当时仓库注册了 56 个算子。到 9580d88 新增了 32 个（2026-10-21 记录），到 2aaa9b0 又新增 36 个（2026-10-29 记录），到 7e9943f 再新增 17 个（2026-11-02 记录），到 dcb71aa 又新增 6 个（2026-11-05 记录），共 147 个，其间没有删除或改名的算子。四段分开记录，这份清单记录的是还没有在各张卡上测完的算子，某个算子在所有卡上都测过之后就可以从这里去掉。

## 测试覆盖（2026-11-05）

| | RTX 4090 | W7900D | R9700 |
|---|---|---|---|
| b7a575a 的 56 个算子 | 已测 | 已测 | 已测 |
| 第一段新增 32 个 | 未测 | 已摸底（单测 + 计时） | 未测 |
| 第二段新增 36 个 | 未测 | 已摸底（单测 + 计时） | 未测 |
| 第三段新增 17 个 | 未测 | 已摸底（单测 + 计时） | 未测 |
| 第四段新增 6 个 | 未测 | 只跑了单测，未计时 | 未测 |

- 三张卡的报告（`RTX4090_report.md`、`W7900D_report.md`、`R9700_report.md`）都停在 b7a575a 的 56 个算子，这 91 个新增算子**在 RTX 4090 和 R9700 上一个都没有测过**，所以目前没有可以从这份清单里去掉的算子。
- W7900D 上前三段的 85 个全部测了耗时，也都跑了单测。第三段的 17 个原来没有参考实现和单测，后来由 batch9（[#270](https://github.com/flagos-ai/FlagGems-sglang/pull/270)）补上，单测全部通过。只有第一段的 `residual_gate_add` 不通过，gfx1100 上的修复在 [#290](https://github.com/flagos-ai/FlagGems-sglang/pull/290)（在审）。第四段的 6 个单测全部通过，还没有计时。
- W7900D 上做的是摸底，不是三份报告那种正式测试：用的是 benchmark 自己的形状和 `do_bench`，没有跑官方 benchmark 的全量形状，也没有做多轮稳定性复测。要把某一段标成「已测」并从清单里去掉，应当按三份报告的口径重跑。
- 逐算子数据和在 W7900D 上的优化分批见 `W7900D_AMD_optimization.md` 的「第三轮」；摸底的做法和结论见文末。

## b7a575a → 9580d88：新增 32 个

### 提交范围

| | 提交 | 日期 | 提交说明 |
|---|---|---|---|
| 起点 | [b7a575a](https://github.com/flagos-ai/FlagGems-sglang/commit/b7a575a32632d718eb1b6edae07b4f0e57beb224) | 2026-10-09 | Merge pull request #136 from flagos-ai/ci/uv-env-prep |
| 终点 | [9580d88](https://github.com/flagos-ai/FlagGems-sglang/commit/9580d883ad32b75aee4c5c87333e738e554ab6b1) | 2026-10-20 | Merge pull request #162 from wuwentao/thead-ppu-sdk-probe（记录时的 master） |

两者之间的全部改动见 [b7a575a...9580d88](https://github.com/flagos-ai/FlagGems-sglang/compare/b7a575a...9580d88)。

### 来源

32 个算子分两批加入，batch5 有 17 个，batch6 有 15 个。

- 参考实现、单测和 benchmark 由 liuhycs 提交：batch5 是 [4728ea6](https://github.com/flagos-ai/FlagGems-sglang/commit/4728ea6e8ca6286e10b9ca6bfa74306d7d0ae2ac)，batch6 是 [26c74ab](https://github.com/flagos-ai/FlagGems-sglang/commit/26c74ab563a216805e0e0362c17d597ddfdb8495)，分别经 [#160](https://github.com/flagos-ai/FlagGems-sglang/pull/160) 和 [#155](https://github.com/flagos-ai/FlagGems-sglang/pull/155) 在 2026-10-15 合入 master。
- Triton 实现来自 FlagOS 竞赛 Track1 的参赛 PR（标题都是「[FlagOS Competition-Track1] Add … Triton Kernel for sglang」），每个算子一个 PR。batch5 的在 2026-10-12 合入，batch6 的在 2026-10-20 合入。
- 每个算子还有 Ascend、Enflame、Hygon、Iluvatar、Kunlunxin、MetaX 中 1 到 6 家的厂商版本（20 个算子六家都有）。没有 AMD 版本，在 AMD GPU 上都走通用实现 `flaggems_sglang.ops`。

### 算子列表

类别取自 `conf/operators.yaml` 的标签（去掉所有算子都有的 fused 和 sglang）。`dsv3_fused_a_gemm` 和 `tiny_k_gemm` 没有其他标签，按描述记为 GEMM。说明是 `operators.yaml` 中描述的概括。

#### batch5（17 个）

| 算子 | 类别 | 说明 | 实现 PR | 作者 |
|---|---|---|---|---|
| `build_trtllm_mha_page_table` | 注意力 | 在设备上构建 TRT-LLM MHA 的页表，不做主机同步 | [#122](https://github.com/flagos-ai/FlagGems-sglang/pull/122) | c2flowDS |
| `clamp_position` | 逐元素、注意力 | decode 步的位置：`clamp(seq_lens - 1, min=0)` | [#120](https://github.com/flagos-ai/FlagGems-sglang/pull/120) | yzw1128 |
| `compute_src2dst` | MoE | 求 MoE 路由排列的逆：由排序后的 reorder ids 写出每个 (token, slot) 的目标位置 | [#119](https://github.com/flagos-ai/FlagGems-sglang/pull/119) | yzw1128 |
| `concat_mla_k` | 注意力 | 由 NoPE 和 RoPE 两部分拼出 MLA 的 key，单头的 RoPE 部分广播到所有头 | [#128](https://github.com/flagos-ai/FlagGems-sglang/pull/128) | xiaojiu1414 |
| `create_flashinfer_kv_indices` | 注意力 | 把 `req_to_token` 中各请求的 token 槽位展平成 FlashInfer paged attention 用的 `kv_indices` | [#126](https://github.com/flagos-ai/FlagGems-sglang/pull/126) | c2flowDS |
| `deepep_permute` | MoE | DeepEP MoE 前处理：按路由把每个 token 的隐状态行复制、重排成分组 GEMM 的输入 | [#123](https://github.com/flagos-ai/FlagGems-sglang/pull/123) | c2flowDS |
| `deepep_post_reorder` | MoE | DeepEP MoE 后处理：取回每个 token 的 top-k 专家输出，按路由权重和缩放因子合并 | [#118](https://github.com/flagos-ai/FlagGems-sglang/pull/118) | yzw1128 |
| `dsv3_fused_a_gemm` | GEMM | DeepSeek-V3 decode 形状（token 数 ≤ 16）的 QKV-A 下投影 GEMM | [#125](https://github.com/flagos-ai/FlagGems-sglang/pull/125) | c2flowDS |
| `fill_padded_rows` | MoE | 把 MoE 激活的填充行置为常数，非填充行数从显存读取 | [#137](https://github.com/flagos-ai/FlagGems-sglang/pull/137) | yzw1128 |
| `fused_eh_norm` | 归一化、投机解码 | EAGLE 草稿模型的预处理：输入 embedding 和上一步隐状态分别做 RMSNorm 后拼接 | [#124](https://github.com/flagos-ai/FlagGems-sglang/pull/124) | c2flowDS |
| `fused_moe_dispatch_index` | MoE | 一次启动生成 masked/DeepGEMM MoE 分组 GEMM 的目标行索引 | [#117](https://github.com/flagos-ai/FlagGems-sglang/pull/117) | yzw1128 |
| `gate_topk` | MoE | k ≤ 32 的 top-k，沿最后一维，可直接替换 `torch.topk` | [#111](https://github.com/flagos-ai/FlagGems-sglang/pull/111) | goldenfox2025 |
| `gelu_tanh_and_mul` | 逐元素、激活 | tanh 近似的门控 GELU：`gelu_tanh(x[..., :d]) * x[..., d:]` | [#121](https://github.com/flagos-ai/FlagGems-sglang/pull/121) | yzw1128 |
| `group_norm_silu` | 归一化 | 扩散模型 U-Net、VAE 中的 GroupNorm + SiLU，统计量和仿射变换用 fp32 计算 | [#127](https://github.com/flagos-ai/FlagGems-sglang/pull/127) | c2flowDS |
| `residual_gate_add` | 逐元素 | LTX-2 transformer 块的残差门控更新：`residual + update * gate`，乘积先舍入到张量类型 | [#113](https://github.com/flagos-ai/FlagGems-sglang/pull/113) | hangglider5 |
| `seqlens_expand` | 注意力 | 把 extend batch 各请求的 (qo_len, kv_len) 展开成每个 query token 的 KV 长度 | [#115](https://github.com/flagos-ai/FlagGems-sglang/pull/115) | yzw1128 |
| `sigmoid_gate_mul` | 逐元素 | 同形状的 sigmoid 门控乘法：`x * sigmoid(gate)` | [#114](https://github.com/flagos-ai/FlagGems-sglang/pull/114) | c2flowDS |

#### batch6（15 个）

| 算子 | 类别 | 说明 | 实现 PR | 作者 |
|---|---|---|---|---|
| `compute_position` | 注意力 | extend（chunked prefill）batch 的位置和起始偏移，前缀和在 kernel 内完成 | [#143](https://github.com/flagos-ai/FlagGems-sglang/pull/143) | c2flowDS |
| `concat_and_cast_mha_k` | 注意力 | 由逐头的 NoPE 和广播的单头 RoPE 拼出 MHA 的 key，写入时转换类型 | [#148](https://github.com/flagos-ai/FlagGems-sglang/pull/148) | c2flowDS |
| `create_flashmla_kv_indices` | 注意力 | 构建 FlashMLA 的 block table，每页一项 | [#138](https://github.com/flagos-ai/FlagGems-sglang/pull/138) | yzw1128 |
| `fixup_zero_kv` | 注意力 | TRT-LLM ragged attention 中 KV 长度为 0 的请求：输出清零，LSE 置为 -inf | [#140](https://github.com/flagos-ai/FlagGems-sglang/pull/140) | yzw1128 |
| `fused_gate_sigmoid_mul_add` | 激活、MoE | MoE 共享专家的门控：求 hidden 与 gate 权重的行点积，再把 `sigmoid(gate) * shared_output` 加到累加结果上 | [#142](https://github.com/flagos-ai/FlagGems-sglang/pull/142) | c2flowDS |
| `indexed_scale_shift` | 逐元素 | 逐行的 AdaLN 调制，scale 和 shift 按索引从表中取 | [#149](https://github.com/flagos-ai/FlagGems-sglang/pull/149) | c2flowDS |
| `moe_align_block_size` | MoE | fused-MoE GEMM 前的排序和补齐：按专家分组，每个专家补到 block_size 的整数倍 | [#152](https://github.com/flagos-ai/FlagGems-sglang/pull/152) | c2flowDS |
| `moe_align_single_token` | MoE | `moe_align_block_size` 的单 token decode 特化 | [#150](https://github.com/flagos-ai/FlagGems-sglang/pull/150) | xychenzhicheng |
| `moe_topk_sum` | MoE | 每个 token 的 top-k 专家输出求和，fp32 累加 | [#141](https://github.com/flagos-ai/FlagGems-sglang/pull/141) | c2flowDS |
| `pack_topk_ids` | MoE | 把（专家 id，路由权重）打包成一个 int32，供 FlashInfer TRT-LLM MoE 使用 | [#146](https://github.com/flagos-ai/FlagGems-sglang/pull/146) | SweetDeathh |
| `post_reorder_cutlass` | MoE | CUTLASS MoE 后处理：用 fp32 合并每个 token 的路由专家输出 | [#151](https://github.com/flagos-ai/FlagGems-sglang/pull/151) | c2flowDS |
| `relu2` | 逐元素、激活 | 平方 ReLU：`max(0, x) ** 2` | [#159](https://github.com/flagos-ai/FlagGems-sglang/pull/159) | gameayagame |
| `sigmoid_gate_mul_broadcast` | 逐元素 | 按行广播的 sigmoid 门控：`out[n, :] = x[n, :] * sigmoid(gate[n])` | [#139](https://github.com/flagos-ai/FlagGems-sglang/pull/139) | yzw1128 |
| `tiny_k_gemm` | GEMM | 小 K、大 N 的 GEMM：`x @ w.T`，m ≤ 16，k 为 128 或 256 | [#147](https://github.com/flagos-ai/FlagGems-sglang/pull/147) | HAi-WORLD |
| `unpad_draft_extend_output` | 注意力、投机解码 | 投机解码中 `pad_draft_extend_query` 的逆操作：从填充后的注意力输出取出有效行，组成 ragged 张量 | [#144](https://github.com/flagos-ai/FlagGems-sglang/pull/144) | c2flowDS |

### 状态

这 32 个算子只在 W7900D 上做过摸底，RTX 4090 和 R9700 都没测（见文末，逐算子数据在 `W7900D_AMD_optimization.md` 的「第三轮」）。W7900D 上：单测除 `residual_gate_add` 外都通过，耗时上 15 个只有小形状、12 个已接近拷贝带宽、3 个按整张量算字节数会高估、`dsv3_fused_a_gemm` 和 `residual_gate_add` 有优化空间。另外：

- 每个算子都有单测 `tests/test_<算子>.py` 和 benchmark `benchmark/test_<算子>.py`，benchmark 只测 bfloat16。
- 静态检查发现下面 7 个 benchmark 生成的输入个数与算子的参数个数对不上（参考实现的参数与算子相同）。benchmark 把输入直接传给两者，上卡确认都会报 `TypeError`，如 `compute_src2dst() missing 1 required positional argument: 'num_toks'`。

| 算子 | benchmark 生成的输入 | 算子参数 |
|---|---|---|
| `compute_src2dst` | 1 个 | `(reorder_ids, num_toks)` |
| `concat_mla_k` | 2 个 | `(k, k_nope, k_rope)` |
| `deepep_permute` | 2 个 | `(input, gateup_input, src2dst, topk_ids, topk, hidden_size)` |
| `dsv3_fused_a_gemm` | 7 个 | `(mat_a, mat_b)` |
| `fill_padded_rows` | 2 个 | `(x, num_token_non_padded, fill_value)` |
| `fused_moe_dispatch_index` | 2 个 | `(topk_ids, num_local_experts, m_max)` |
| `seqlens_expand` | 1 个 | `(extend_seq_lens, seq_lens, total_len, max_q_len)` |

- 上卡还发现另外 3 个 benchmark 也调不通，静态检查没查出来是因为多出来的输入放在末尾的 dict 里、按关键字参数传入：`build_trtllm_mha_page_table` 多传 `block_size`、`create_flashinfer_kv_indices` 多传 `page_size`、`deepep_post_reorder` 多传 `scaling`，算子和参考实现都不接受这些参数。这 10 个算子的摸底改用单测里的用例构造输入，按 benchmark 的形状测；其中 `dsv3_fused_a_gemm` 的形状是 `(m, 4096, 64, 8)`，带专家数和 top-k，像是从某个 MoE 算子的 benchmark 抄来的，与这个矩阵乘无关，摸底时只取了 m。这 10 个 benchmark 后来由 [#272](https://github.com/flagos-ai/FlagGems-sglang/pull/272) 修好（2026-11-02 合入），W7900D 上都能跑通。
- `residual_gate_add` 在 W7900D 上过不了自己的单测，12 个用例失败 11 个，都是 bfloat16 下结果接近 0、出现大幅抵消时差一个 ulp（0.015625，绝对容差 0.015）。参考实现是 `(r.float() + (u.float() * g.float()).to(dtype).float()).to(dtype)`，通用实现是 `product = (u * g).to(O.dtype); result = r + product`。看生成的汇编，编译器把这一串映射成了 `v_dot2_bf16_bf16`——这条指令内部按 fp32 算、只在最后舍入一次，跳过了参考要求的中间舍入，所以抵消之后整个结果就等于那 1 个 ulp。全仓只有这个算子和 `rmsnorm_scale` 编译出了这条指令；后者后来由 batch9 补了参考实现，单测在 W7900D 上全部通过。gfx1100 上的修复见 [#290](https://github.com/flagos-ai/FlagGems-sglang/pull/290)（在审）。
- 已测的 56 个算子在这期间没有变化：通用实现、参考实现、单测和 benchmark 文件都没有改，`_amd` 后端也没有改动。改动的只有测试框架：`benchmark/conftest.py`、`benchmark/performance_utils.py`、`benchmark/attri_util.py` 和 `tests/conftest.py` 新增了按用例列出和重放、profiling、只验证参考实现、`--base-data` 对比列和运行时替换算子等选项。计时函数 `get_latency` 没有改，`--level` 默认仍为 comprehensive。

## 9580d88 → 2aaa9b0：新增 36 个

### 提交范围

| | 提交 | 日期 | 提交说明 |
|---|---|---|---|
| 起点 | [9580d88](https://github.com/flagos-ai/FlagGems-sglang/commit/9580d883ad32b75aee4c5c87333e738e554ab6b1) | 2026-10-20 | Merge pull request #162 from wuwentao/thead-ppu-sdk-probe |
| 终点 | [2aaa9b0](https://github.com/flagos-ai/FlagGems-sglang/commit/2aaa9b0801f26a0e4eb880b240392b26513dd1af) | 2026-10-29 | Merge pull request #242 from flagos-ai/feat/daily-kernelgen-sync（记录时的 master） |

两者之间的全部改动见 [9580d88...2aaa9b0](https://github.com/flagos-ai/FlagGems-sglang/compare/9580d88...2aaa9b0)。36 个算子的实现最晚在 2026-10-26 合入，之后到 2aaa9b0 没有再新增算子。

### 来源

- 34 个算子的参考实现、单测和 benchmark 来自两次批量提交，都由 liuhycs 提交：batch7 是 [64e4bc7](https://github.com/flagos-ai/FlagGems-sglang/commit/64e4bc7fd33f2b082fb2a5238f623c2f9e801218)（17 个），经 [#210](https://github.com/flagos-ai/FlagGems-sglang/pull/210) 在 2026-10-23 合入；batch8 是 [4af8db6](https://github.com/flagos-ai/FlagGems-sglang/commit/4af8db66d2e490968f93f7541dbe9dd67a07a7e0)（17 个），经 [#221](https://github.com/flagos-ai/FlagGems-sglang/pull/221) 在 2026-10-26 合入。
- 另外 2 个 `add3`、`hash_topk` 属于 batch6：参考实现、单测和 benchmark 在上一段的 [26c74ab](https://github.com/flagos-ai/FlagGems-sglang/commit/26c74ab563a216805e0e0362c17d597ddfdb8495) 里就有，当时还没有实现，没有计入上一段的 32 个，实现在这一段合入。
- Triton 实现同样来自 FlagOS 竞赛 Track1 的参赛 PR（标题都是「[FlagOS Competition-Track1] Add … Triton Kernel for sglang」），每个算子一个 PR，在 2026-10-21 到 10-26 之间合入，多数在 10-23。
- 每个算子有 Ascend、Enflame、Hygon、Iluvatar、Kunlunxin、MetaX 中 0 到 6 家的厂商版本（4 家和 5 家的各 14 个，`store_cache_4d` 六家都有，`fused_sigmoid_mul`、`hash_topk` 一家都没有）。`assign_draft_cache_locs_contiguous`、`store_cache_4d`、`wide_row_softmax` 的实现 PR 还带了 AMD（`_amd/ops/`）和 NVIDIA 版本，W7900D 上实测这三个走 `_amd/ops/` 的版本；其余 33 个在 AMD GPU 上走通用实现 `flaggems_sglang.ops`。

### 算子列表

类别取自 `conf/operators.yaml` 的标签（去掉 fused 和 sglang）。`memcpy_triton`、`recompute_w_u`、`tiny_n_gemm`、`vocab_parallel_embedding` 没有其他标签，按描述分别记为数据搬运、线性注意力、GEMM、Embedding。说明是 `operators.yaml` 中描述的概括。

#### batch6 中这一段补上实现的（2 个）

| 算子 | 类别 | 说明 | 实现 PR | 作者 |
|---|---|---|---|---|
| `add3` | 逐元素 | 三路逐元素相加，保留两次舍入：`bf16(bf16(a + b) + c)`，不能先用 fp32 累加再舍入一次，结果与不融合的写法逐位相同 | [#134](https://github.com/flagos-ai/FlagGems-sglang/pull/134) | liuthe |
| `hash_topk` | MoE | 按 token id 查表得到专家集合的 hash 路由门控，不做 top-k 搜索：取出对应的 logits，用 `sqrt(softplus(·))` 打分、按行归一化，再追加融合的共享专家 | [#145](https://github.com/flagos-ai/FlagGems-sglang/pull/145) | SweetDeathh |

#### batch7（17 个）

| 算子 | 类别 | 说明 | 实现 PR | 作者 |
|---|---|---|---|---|
| `add_constant` | 逐元素 | 加一个编译期整数常量：`dst = src + constant`，常量折叠进 kernel，每个不同的值都要单独 JIT 编译一次 | [#188](https://github.com/flagos-ai/FlagGems-sglang/pull/188) | SweetDeathh |
| `concat_mla_absorb_q` | 注意力 | 沿最后一维拼接 absorbed MLA query 的 NoPE 和 RoPE 两半，两个步长不同的源行写进一个连续的目标行 | [#168](https://github.com/flagos-ai/FlagGems-sglang/pull/168) | yzw1128 |
| `create_chunked_prefix_cache_kv_indices` | 注意力 | `create_flashinfer_kv_indices` 的 chunked prefill 版本，每个请求只取池中 `[chunk_start_idx, chunk_start_idx + chunk_seq_len)` 这一段 | [#164](https://github.com/flagos-ai/FlagGems-sglang/pull/164) | love530love |
| `fused_pack_qkv` | 注意力 | 用同一个 gather 索引把填充的 [B, S, H, D] Q/K/V 打包成变长的 [total_valid, H, D]，一次启动代替三次高级索引 gather | [#204](https://github.com/flagos-ai/FlagGems-sglang/pull/204) | febyeji |
| `fused_sigmoid_mul` | 逐元素、激活 | 注意力输出门控：`out = attn_output * sigmoid(gate)`，按步长直接读 3 维的 gate，不做连续化拷贝 | [#201](https://github.com/flagos-ai/FlagGems-sglang/pull/201) | Truman9999 |
| `get_mla_kv_buffer` | 注意力 | `set_mla_kv_buffer` 的反操作：读出分页 MLA KV cache 的第 `loc[i]` 行，拆回 NoPE 和 RoPE 两半 | [#173](https://github.com/flagos-ai/FlagGems-sglang/pull/173) | yzw1128 |
| `ltx2_split_rotary` | RoPE | LTX-2 的 split-half 旋转位置编码：旋转前后两半而不是相邻元素对，每个头有自己的 cos、sin 表 | [#184](https://github.com/flagos-ai/FlagGems-sglang/pull/184) | c2flowDS |
| `pad_draft_extend_query` | 注意力、投机解码 | 按 `cu_seqlens_q` 把变长的草稿 query 写进填充的 [bs, max_seq_len, H, D] 缓冲区，超出 `seq_lens_q[b]` 的行不动 | [#187](https://github.com/flagos-ai/FlagGems-sglang/pull/187) | SweetDeathh |
| `post_reorder_deepgemm` | MoE | DeepGEMM 版的 MoE 后处理，算术与 `post_reorder_cutlass` 相同；有效性判断是 `expert_id >= 0` 而不是 `!= num_local_experts`，融合的共享专家得以保留 | [#176](https://github.com/flagos-ai/FlagGems-sglang/pull/176) | yzw1128 |
| `pre_reorder_cutlass` | MoE | CUTLASS MoE 的输入重排，融合按张量的反量化缩放：每个 token 行复制到各个路由目标，途中乘以 `1 / a1_scales` | [#177](https://github.com/flagos-ai/FlagGems-sglang/pull/177) | yzw1128 |
| `recompute_w_u` | 线性注意力 | 分块 Gated DeltaNet 前向中重算 w 和 u 的一步：用每块求逆后的 UT 变换作用于 beta 缩放的 value 块和门控缩放的 key 块 | [#197](https://github.com/flagos-ai/FlagGems-sglang/pull/197) | xuanzhengdu-eng |
| `rmsnorm_hf` | 归一化 | HuggingFace LlamaRMSNorm 语义的 RMSNorm：归一化结果先转回输入类型再乘权重，这一点与 `fused_rmsnorm` 不同 | [#178](https://github.com/flagos-ai/FlagGems-sglang/pull/178) | yzw1128 |
| `set_mla_kv_buffer` | 注意力 | 分页 MLA KV cache 的 scatter 写：每个 token 的 NoPE、RoPE 两部分 key 拼成一行写到第 `loc[i]` 槽 | [#198](https://github.com/flagos-ai/FlagGems-sglang/pull/198) | xuanzhengdu-eng |
| `tiny_n_gemm` | GEMM | decode 阶段小投影用的窄 M bf16 GEMM：`out = x @ w.T`，最多 max_m 行；N 分成 split_n 组，使 n / split_n 个块正好填满一波 SM | [#180](https://github.com/flagos-ai/FlagGems-sglang/pull/180) | yzw1128 |
| `tma_align_input_scale` | MoE、量化 | 把按 token 的 FP8 激活缩放因子转置成 DeepGEMM TMA 要求的列主序、16 字节对齐布局，M 补齐到 16 / 元素字节数的倍数 | [#195](https://github.com/flagos-ai/FlagGems-sglang/pull/195) | liangkw16 |
| `topk_sigmoid` | MoE | 融合的 sigmoid MoE 门控：对 router logits 求 sigmoid、选 top-k 专家、可选重新归一化，原地写入预分配的 fp32 `topk_weights` 和 int32 `topk_ids` | [#181](https://github.com/flagos-ai/FlagGems-sglang/pull/181) | yzw1128 |
| `zero_experts_identity` | MoE | 零（恒等）专家：专家 id ≥ num_experts 的路由槽不做计算，贡献是该 token 的隐状态乘以这些槽的路由权重之和 | [#183](https://github.com/flagos-ai/FlagGems-sglang/pull/183) | yzw1128 |

#### batch8（17 个）

| 算子 | 类别 | 说明 | 实现 PR | 作者 |
|---|---|---|---|---|
| `assign_draft_cache_locs_contiguous` | 投机解码 | 从 `req_to_token` 池中取出每个请求新分配的草稿 KV 槽位，紧凑写入草稿前向用的扁平 `out_cache_loc` | [#202](https://github.com/flagos-ai/FlagGems-sglang/pull/202) | sinpeyw |
| `conv_window_scatter_with_mask` | Mamba、投机解码 | `state_scatter_with_mask` 的卷积窗口版本，用于投机解码中 Mamba 状态的提交；源是重叠的 as_strided 视图，kernel 按步长索引 (dim, win) 元素 | [#165](https://github.com/flagos-ai/FlagGems-sglang/pull/165) | love530love |
| `dcp_lse_combine` | 注意力 | 按 log-sum-exp 合并 N 个 decode context parallel rank 的部分注意力输出；NaN 和 +inf 的 lse 视为 -inf，失效的分片不贡献 | [#169](https://github.com/flagos-ai/FlagGems-sglang/pull/169) | yzw1128 |
| `fill_accept_out_cache_loc` | 投机解码 | 紧凑保存投机解码中被接受的草稿 token 的 KV cache 槽位：`accept_index` 给出每个被接受槽位的草稿位置，-1 表示被拒绝 | [#170](https://github.com/flagos-ai/FlagGems-sglang/pull/170) | yzw1128 |
| `fill_bonus_tokens` | 投机解码 | 从接受 token 矩阵中取出每个请求的 bonus token：`bonus_tokens[i] = accept_tokens[i, accept_lens[i] - 1]` | [#185](https://github.com/flagos-ai/FlagGems-sglang/pull/185) | c2flowDS |
| `fused_qkvzba_split_reshape_cat` | 注意力 | 一次启动把打包的 GDN 投影拆成线性注意力 kernel 需要的四个张量；只搬数据、不做运算，受带宽限制，难点在地址计算 | [#171](https://github.com/flagos-ai/FlagGems-sglang/pull/171) | yzw1128 |
| `fused_softmax_pool` | 注意力 | 沿 K 轴做 softmax 加权池化，值和分数打包在同一个张量里；一遍代替 softmax、逐元素乘和归约 | [#186](https://github.com/flagos-ai/FlagGems-sglang/pull/186) | c2flowDS |
| `gather_spec_extras` | 投机解码 | 用同一个行索引一次 gather 投机解码的四个辅助缓冲区：`topk_p`、`topk_index`、`bonus_tokens` 和可选的 `hidden_states` | [#172](https://github.com/flagos-ai/FlagGems-sglang/pull/172) | yzw1128 |
| `get_target_cache_loc` | 投机解码 | 把 verify 步的 `out_cache_loc` 分成保留的槽位和归还给分配器的槽位，两部分按每个请求接受的草稿数各自紧凑排列 | [#174](https://github.com/flagos-ai/FlagGems-sglang/pull/174) | yzw1128 |
| `linear_decode_forward` | 注意力 | Lightning attention 的 decode 步：每条序列一个 token，更新各槽位 [D, D] 的线性注意力状态；衰减只作用于之前的状态，`slot_idx` 为 -1 的填充行不动 | [#175](https://github.com/flagos-ai/FlagGems-sglang/pull/175) | yzw1128 |
| `memcpy_triton` | 数据搬运 | 偏移和大小都放在显存里的设备内拷贝，每步搬运量变化时仍能被 CUDA graph 捕获 | [#196](https://github.com/flagos-ai/FlagGems-sglang/pull/196) | hangglider5 |
| `set_token_filter` | 采样 | 语法约束和结构化输出的 token 掩码：重置 int32 词表位掩码的一行，再用 atomic_or、atomic_and 按允许、禁止列表置位或清位 | [#179](https://github.com/flagos-ai/FlagGems-sglang/pull/179) | yzw1128 |
| `state_scatter_with_mask` | Mamba、投机解码 | 投机解码中 Mamba 状态提交的融合 masked gather-scatter：一次启动代替 nonzero、两次 index_select 和一次高级索引 scatter，覆盖所有层 | [#167](https://github.com/flagos-ai/FlagGems-sglang/pull/167) | yzw1128 |
| `store_cache_4d` | 注意力 | 一次启动、按 token 并行把 K、V 写入按页排列的 4 维 KV cache 视图，扁平槽位号拆成页号和页内位置 | [#200](https://github.com/flagos-ai/FlagGems-sglang/pull/200) | goldenfox2025 |
| `top_p_renorm` | 采样 | 精确的 top-p（nucleus）截断和重新归一化，沿用 FlashInfer 的 pivot 语义：与 pivot 相等的元素全部保留 | [#189](https://github.com/flagos-ai/FlagGems-sglang/pull/189) | c2flowDS |
| `vocab_parallel_embedding` | Embedding | 词表并行的 embedding 查表：每个 TP rank 持有一个词表窗口和一个新增 token 窗口，窗口外的 token 读零行 | [#182](https://github.com/flagos-ai/FlagGems-sglang/pull/182) | yzw1128 |
| `wide_row_softmax` | 投机解码 | 超宽 fp32 行的列并行 softmax，可选按行温度：每行分到多个块，合并各块的部分最大值和和，只写前 V 列 | [#166](https://github.com/flagos-ai/FlagGems-sglang/pull/166) | love530love |

### 状态

这 36 个算子同样只在 W7900D 上做过摸底，RTX 4090 和 R9700 都没测（见文末，逐算子数据在 `W7900D_AMD_optimization.md` 的「第三轮」）。W7900D 上：单测全部通过，耗时上 8 个只有小形状、8 个已接近拷贝带宽、12 个按整张量算字节数会高估，4 个有优化空间（`add3`、`fused_softmax_pool`、`ltx2_split_rotary`、`dcp_lse_combine`），另外 4 个明显偏慢（`get_target_cache_loc`、`tiny_n_gemm`、`top_p_renorm`、`recompute_w_u`），是这 85 个里最值得优化的。其中 `tiny_n_gemm`、`top_p_renorm`（及通用 `tiny_n_gemm` 的 K 尾部掩码）已合入 [#298](https://github.com/flagos-ai/FlagGems-sglang/pull/298)（2026-11-09），见 `W7900D_AMD_optimization.md` 的「`tiny_n_gemm` / `top_p_renorm`（2026-11-09）」；还剩 `get_target_cache_loc`、`recompute_w_u`。另外：

- 每个算子都有单测和 benchmark。benchmark 只测一种 dtype：`conv_window_scatter_with_mask`、`linear_decode_forward`、`state_scatter_with_mask`、`top_p_renorm`、`wide_row_softmax` 只测 float32，其余 31 个只测 bfloat16。
- 用与上一段相同的静态检查没有发现新的参数个数不匹配。`hash_topk`、`topk_sigmoid`、`zero_experts_identity` 的 benchmark 把后几个参数放在末尾的 dict 里，框架按关键字参数传入，个数对得上；`create_chunked_prefix_cache_kv_indices` 的签名是 `(*args, **kwargs)`，静态检查不适用。
- 上一段列出的 7 个 benchmark 在这一段只改了调用方式（见下），生成的输入没变，`seqlens_expand` 的签名也没变，上卡确认问题都还在。
- 摸底时 `assign_draft_cache_locs_contiguous`、`store_cache_4d`、`wide_row_softmax` 测到的是 `_amd/ops/` 的版本，不是通用实现。

已记录的 88 个算子在这期间的变化：

- `seqlens_expand` 的通用实现删掉了会让编译器崩溃的分组 kernel（[#222](https://github.com/flagos-ai/FlagGems-sglang/pull/222)，加 6 行、删 80 行），参考实现和单测没改。
- 8 个算子在 AMD 后端新增了 gfx1100 版本（`_amd/rdna3/ops/`）：`context_attention`、`w8a8_block_int8_matmul`（[#199](https://github.com/flagos-ai/FlagGems-sglang/pull/199)），`fused_moe_gemm`、`fused_moe_router_tensorcore`、`fused_moe_router_cudacore`（[#205](https://github.com/flagos-ai/FlagGems-sglang/pull/205)），`causal_conv1d_fn`、`chunk_state_varlen`、`bmm_chunk`（[#231](https://github.com/flagos-ai/FlagGems-sglang/pull/231)）。#199 同时把 `ARCH_MAP` 改成按 GPU 型号精确匹配，`_amd` 只把 gfx1100 映射到 rdna3，所以 W7900D 上这 8 个算子不再走通用实现。
- `silu_and_mul` 的单测改用新的 `reference_test` 写法（[#240](https://github.com/flagos-ai/FlagGems-sglang/pull/240)），容差和输入不变。
- 其余算子的通用实现、参考实现、单测和厂商版本都没有改。

测试框架和运行时的变化：

- benchmark 改成先由 case planner 列出用例、再按需生成输入，可以按用例列出和重放（[#234](https://github.com/flagos-ai/FlagGems-sglang/pull/234)、[#239](https://github.com/flagos-ai/FlagGems-sglang/pull/239)）。各 benchmark 只改了调用方式：`input_fn` 换成 `case_fn` 加 `build_inputs_fn`，`set_gems` 换成 `set_gems_from_module`，形状和生成的输入没变。`benchmark/core_shapes.yaml` 没改，新增的厂商形状覆盖目前只有 Enflame 的 `core_shapes.yaml`（[#232](https://github.com/flagos-ai/FlagGems-sglang/pull/232)），AMD 上仍用默认形状。
- 新增把候选实现注入单测和 benchmark 的机制（[#233](https://github.com/flagos-ai/FlagGems-sglang/pull/233) 的 candidate injection，以及 dummy candidate、[#240](https://github.com/flagos-ai/FlagGems-sglang/pull/240) 的 reference-as-candidate）。计时仍调用 `get_latency`，这个函数没改，只是被计时的函数改由 `_candidate_call` 选出；`--level`、`--mode`、`--dtypes` 等选项没有删改。
- 后端模块改用带包名的导入，避免 `sys.modules` 冲突（[#236](https://github.com/flagos-ai/FlagGems-sglang/pull/236)）；`src/flaggems_sglang/__init__.py` 把设备信息的初始化挪到注册算子之前（[#196](https://github.com/flagos-ai/FlagGems-sglang/pull/196)）。

## 2aaa9b0 → 7e9943f：新增 17 个

### 提交范围

| | 提交 | 日期 | 提交说明 |
|---|---|---|---|
| 起点 | [2aaa9b0](https://github.com/flagos-ai/FlagGems-sglang/commit/2aaa9b0801f26a0e4eb880b240392b26513dd1af) | 2026-10-29 | Merge pull request #242 from flagos-ai/feat/daily-kernelgen-sync |
| 终点 | [7e9943f](https://github.com/flagos-ai/FlagGems-sglang/commit/7e9943fb837c4baff7c3b56874be2e578b659167) | 2026-11-02 | Merge pull request #256 from WhatGhost/feat/amd-gfx1100-lora-memory（记录时的 master） |

两者之间的全部改动见 [2aaa9b0...7e9943f](https://github.com/flagos-ai/FlagGems-sglang/compare/2aaa9b0...7e9943f)。17 个算子都在 2026-11-02 合入。

### 来源

- 17 个算子都来自 FlagOS 竞赛 Track1 的参赛 PR（标题都是「[FlagOS Competition-Track1] Add … Triton Kernel for sglang」），每个算子一个 PR，都在 2026-11-02 合入：14 个由 c2flowDS 提交（提交者署名 dongsheng.wang），3 个由 yzw1128 提交。
- 与前两段不同，这一段没有配套的批量提交：仓库里没有这 17 个算子的参考实现、单测和 benchmark，`conf/operators.yaml` 的目录里也没有它们，PR 只带了通用实现和厂商版本。
- 16 个有 Ascend、Hygon、Iluvatar、Kunlunxin、MetaX 五家的厂商版本，`qk_head_layernorm` 没有 Iluvatar；都没有 AMD 版本，在 AMD GPU 上走通用实现 `flaggems_sglang.ops`。

### 算子列表

目录里没有这 17 个算子，类别按功能归；说明取自各实现 PR 的描述。

| 算子 | 类别 | 说明 | 实现 PR | 作者 |
|---|---|---|---|---|
| `ep_gather` | MoE（专家并行） | 专家并行 MoE 的合并：从重排后的缓冲区按 token 取回各专家的输出，按路由权重加权求和；专家 id 为负的槽位不参与，累加用 FP32 | [#212](https://github.com/flagos-ai/FlagGems-sglang/pull/212) | c2flowDS |
| `ep_scatter` | MoE（专家并行） | 专家并行 MoE 的分发：把 token 数据和量化缩放因子写进按专家填充的缓冲区，同时生成专家映射和目标索引 | [#219](https://github.com/flagos-ai/FlagGems-sglang/pull/219) | c2flowDS |
| `fused_qk_gemma_rmsnorm_rope_gate` | 归一化、RoPE | 对 Q、K 逐头做 Gemma RMSNorm 和 NeoX 风格的部分 RoPE，同时把可选的 gate 从 Q 缓冲区里拆出来 | [#216](https://github.com/flagos-ai/FlagGems-sglang/pull/216) | c2flowDS |
| `fused_qk_inv_rms` | 归一化 | 从打包的 QKV 张量算出 Q、K 的 RMS 倒数，每个 token 在全部头和 head 维上归约，归约和输出都用 FP32 | [#209](https://github.com/flagos-ai/FlagGems-sglang/pull/209) | c2flowDS |
| `gemma_fused_add_rmsnorm` | 归一化 | 残差相加与 Gemma RMSNorm 融合，和与 RMS 归约用 FP32，权重按 `(1 + weight)` 作用 | [#211](https://github.com/flagos-ai/FlagGems-sglang/pull/211) | c2flowDS |
| `hadamard_transform` | 变换 | 沿最后一维做 Walsh-Hadamard 变换再乘 `scale`，即 `x @ H_n.T * scale`，`H_n` 是未归一化的 Sylvester Hadamard 矩阵 | [#220](https://github.com/flagos-ai/FlagGems-sglang/pull/220) | c2flowDS |
| `layernorm_modulate` | 归一化 | 不带仿射参数的 LayerNorm 后接 AdaLN 调制：`LN(x) * (1 + scale) + shift`，scale、shift 按 batch 给出、沿序列维广播 | [#218](https://github.com/flagos-ai/FlagGems-sglang/pull/218) | c2flowDS |
| `layernorm_scale_shift_gate_select01` | 归一化 | LayerNorm（可选仿射权重和偏置）后做 scale-shift 调制，每个 token 按索引从两组参数中选一组，对应的 gate 一并选出写回 | [#215](https://github.com/flagos-ai/FlagGems-sglang/pull/215) | c2flowDS |
| `one_pass_rms_norm` | 归一化 | 沿最后一维做 RMSNorm 再乘权重，归约和乘权重用 FP32，最后转回输入类型 | [#207](https://github.com/flagos-ai/FlagGems-sglang/pull/207) | c2flowDS |
| `pack_qkv_destination_major` | 注意力、数据搬运 | 为 Ulysses all-to-all 把 Q、K、V 打包成按目标 rank 排列的布局：每个 rank 的连续块内，按头依次拼接 Q、K、V | [#217](https://github.com/flagos-ai/FlagGems-sglang/pull/217) | c2flowDS |
| `qk_gemma_rmsnorm_rope` | 归一化、RoPE | 对 Q、K 逐头做 Gemma RMSNorm 和旋转位置编码，支持 NeoX 和交错两种布局 | [#214](https://github.com/flagos-ai/FlagGems-sglang/pull/214) | c2flowDS |
| `qk_head_layernorm` | 归一化 | 对 Q、K 的每个头独立做不带仿射参数的 LayerNorm，均值和有偏方差用 FP32 算，结果转回输入类型；Q、K 一次启动处理 | [#225](https://github.com/flagos-ai/FlagGems-sglang/pull/225) | yzw1128 |
| `rmsnorm_scale` | 归一化 | BF16 的 RMSNorm 后乘 `weight` 和 `scale`，每一行 `scale` 可以对应连续的一组输入行 | [#230](https://github.com/flagos-ai/FlagGems-sglang/pull/230) | c2flowDS |
| `rmsnorm_scale_shift` | 归一化 | 带权重的 RMSNorm 后按 batch 做 scale-shift 调制，[B, 1, D] 的 scale、shift 沿 [B, S, D] 输入的序列维广播 | [#224](https://github.com/flagos-ai/FlagGems-sglang/pull/224) | yzw1128 |
| `rmsnorm_tanh_residual` | 归一化 | RMSNorm、乘权重、tanh 门控和残差相加融合在一起，gate 的一行可以广播到连续的多行输入 | [#223](https://github.com/flagos-ai/FlagGems-sglang/pull/223) | yzw1128 |
| `scale_residual_rmsnorm_scale_shift` | 归一化 | 带门控的残差相加、RMSNorm 和 scale-shift 调制在一次启动中完成 | [#213](https://github.com/flagos-ai/FlagGems-sglang/pull/213) | c2flowDS |
| `scatter_to_padded` | 注意力、数据搬运 | 用逆索引把打包的变长注意力输出还原成填充张量，填充位置写 0 | [#208](https://github.com/flagos-ai/FlagGems-sglang/pull/208) | c2flowDS |

### 状态

这 17 个算子在 W7900D 上先只测了耗时；batch9（[#270](https://github.com/flagos-ai/FlagGems-sglang/pull/270)，2026-11-02 合入）补上参考实现、单测和 benchmark 后，单测在 W7900D 上全部通过。RTX 4090 和 R9700 都没测（见文末，逐算子数据在 `W7900D_AMD_optimization.md` 的「第三轮」）。W7900D 上：15 个已接近拷贝带宽，`fused_qk_gemma_rmsnorm_rope_gate` 有优化空间，`ep_scatter` 按整张量算字节数会高估。另外：

- 摸底时仓库里还没有参考实现、单测和 benchmark，输入按通用实现的签名和代码里写死的形状分支自己构造，只能发现报错和 NaN。
- 在 AMD GPU 上走通用实现 `flaggems_sglang.ops`。
- 上卡发现 3 个问题。`rmsnorm_scale_shift` 在 hidden 不是 2 的幂时编译失败，测的 6 个 3072 形状全部报 `arange's range must be a power of 2`（`_rmsnorm_scale_shift_kernel` 第 54 行用 hidden 当 `tl.arange` 的长度），2048 的形状正常；`qk_head_layernorm` 在 q、k 头数不同时非法访存，按 q 的行数去处理 k，GQA 形状（如 32 个 q 头、8 个 kv 头）越界，q、k 头数相同的 4 个形状正常；`ep_scatter` 在一个形状上输出带 NaN，看代码是填充行没有初始化，可能是有意为之。这些都是通用实现本身的问题，与硬件无关。`rmsnorm_scale_shift`、`qk_head_layernorm` 已在 [#289](https://github.com/flagos-ai/FlagGems-sglang/pull/289) 修复（在审）；batch9 的单测只用 2 的幂宽度和相同的 q、k 头数，覆盖不到这两个问题。`ep_scatter` 未处理。
- `scatter_to_padded`、`ep_gather` 把行数和 stride 声明成了 `tl.constexpr`，每换一个形状都要重新编译。benchmark 有预热看不出来，真实推理里序列长度一直在变。
- 几个归一化算子的通用实现带 HIP 专用的形状分支（`layernorm_modulate` 的 d == 2240、`rmsnorm_scale_shift` 的 hidden == 2048 且行数 ≥ 2048、`rmsnorm_tanh_residual`、`hadamard_transform`），竞赛时应该在 AMD 上按 DiT 的形状调过，实测这些形状确实已经跑在拷贝带宽附近。
- 12 个是归一化的融合变体（RMSNorm 或 LayerNorm 加调制、门控、残差或 RoPE），与已测的 `fused_rmsnorm`、`fused_dual_residual_rmsnorm`、`rmsnorm_hf` 同属访存型；`ep_gather`、`ep_scatter`、`pack_qkv_destination_major`、`scatter_to_padded` 是数据搬运；`hadamard_transform` 的通用实现有 1510 行，是计算较重的变换。

已记录的 124 个算子在这期间的变化：

- 6 个算子在 AMD 后端新增了 gfx1100 版本：`chunked_sgmv_expand`、`gate_up_lora_b`、`sgemm_lora_b`、`per_group_transpose`、`silu_and_mul`、`fused_rmsnorm`（[#256](https://github.com/flagos-ai/FlagGems-sglang/pull/256)）。加上上一段的 8 个，W7900D 上共 14 个算子走 `_amd/rdna3/ops/`；`decode_attention`、`decode_grouped_attention`、`extend_attention`、`log_scaling_tau` 的 gfx1100 版本在 [#271](https://github.com/flagos-ai/FlagGems-sglang/pull/271) 中，记录时还没合入（2026-11-02 已合入）。
- 125 个单测文件都改成先声明参考实现、再用 `get_candidate(op, reference=reference)` 取被测算子（[#241](https://github.com/flagos-ai/FlagGems-sglang/pull/241)，基于 [#244](https://github.com/flagos-ai/FlagGems-sglang/pull/244)，`reference_candidate.py` 相应调整），并新增 `tests/core/test_correctness_reference_declarations.py` 检查每个单测都声明了参考实现。输入和容差都没变；`get_candidate` 默认返回注册的算子，W7900D 上就是 rdna3 版本。上一段改用 `reference_test` 写法的 `silu_and_mul` 也一并改成了这种写法。
- Kunlunxin 的 `fill_bonus_tokens` 适配了新的 `tle.raw.dialect` 接口（[#248](https://github.com/flagos-ai/FlagGems-sglang/pull/248)）。
- 其余算子的通用实现、参考实现、benchmark 和厂商版本都没有改；测试框架（`benchmark/conftest.py`、`benchmark/performance_utils.py`、`benchmark/attri_util.py`、`tests/conftest.py`）也没有改。

## 7e9943f → dcb71aa：新增 6 个

### 提交范围

| | 提交 | 日期 | 提交说明 |
|---|---|---|---|
| 起点 | [7e9943f](https://github.com/flagos-ai/FlagGems-sglang/commit/7e9943fb837c4baff7c3b56874be2e578b659167) | 2026-11-02 | Merge pull request #256 from WhatGhost/feat/amd-gfx1100-lora-memory |
| 终点 | [dcb71aa](https://github.com/flagos-ai/FlagGems-sglang/commit/dcb71aab0263cc48b40c02bd824c89820bb45be6) | 2026-11-03 | Merge pull request #287 from liuhycs/fix/one-pass-rms-norm-w-kwarg（记录时的 master） |

两者之间的全部改动见 [7e9943f...dcb71aa](https://github.com/flagos-ai/FlagGems-sglang/compare/7e9943f...dcb71aa)。

### 来源

- 6 个算子都来自 liuhycs 的 [#279](https://github.com/flagos-ai/FlagGems-sglang/pull/279)（2026-11-03 合入），实现、参考实现、单测和 benchmark 一起提交；[#284](https://github.com/flagos-ai/FlagGems-sglang/pull/284) 补全了 `conf/operators.yaml` 的字段。
- 厂商版本：`chunk_gated_delta_rule_fwd` 有 Hygon、Iluvatar、MThreads、T-Head；`fused_moe` 只有 Hygon；`fused_recurrent_gated_delta_rule` 有 Ascend、Hygon、Iluvatar、MThreads、T-Head；`fused_recurrent_gated_delta_rule_packed_decode`、`sglang_fl_rotary_embedding` 有 Ascend、Hygon、Iluvatar、MetaX、MThreads、T-Head；`topk_softmax` 只有 Iluvatar。都没有 AMD 版本，在 AMD GPU 上走通用实现。

### 算子列表

类别取自 `conf/operators.yaml` 的标签（去掉 fused 和 sglang），说明是其中描述的概括。

| 算子 | 类别 | 说明 | 实现 PR | 作者 |
|---|---|---|---|---|
| `chunk_gated_delta_rule_fwd` | 线性注意力 | 分块的 gated delta-rule 前向（Qwen3-Next），按衰减窗口分段调度递推，可在 kernel 内对 q、k 做 L2 归一化 | [#279](https://github.com/flagos-ai/FlagGems-sglang/pull/279) | liuhycs |
| `fused_moe` | MoE | 未量化的融合 MoE 专家计算（SGLang `fused_experts` 语义）：排序后的分组 GEMM 做 gate/up 投影和激活，再做 down 投影和 top-k 合并 | [#279](https://github.com/flagos-ai/FlagGems-sglang/pull/279) | liuhycs |
| `fused_recurrent_gated_delta_rule` | 线性注意力 | 逐步递推的 gated delta-rule（Qwen3-Next、FLA），每步一个标量门控，可在 kernel 内对 q、k 做 L2 归一化 | [#279](https://github.com/flagos-ai/FlagGems-sglang/pull/279) | liuhycs |
| `fused_recurrent_gated_delta_rule_packed_decode` | 线性注意力 | 直接从打包的 QKV 投影做单 token decode：一步递推写入分页的 SSM 状态，按请求索引并处理填充槽 | [#279](https://github.com/flagos-ai/FlagGems-sglang/pull/279) | liuhycs |
| `sglang_fl_rotary_embedding` | RoPE | 对打包的 query、key 原地做 RoPE，按位置索引 cos/sin 缓存，支持 neox（前后两半）和交错两种布局，多出的 head lane 原样保留 | [#279](https://github.com/flagos-ai/FlagGems-sglang/pull/279) | liuhycs |
| `topk_softmax` | MoE | MoE top-k 门控：对专家做 fp32 softmax，取每个 token 最大的 k 个，按降序写入预分配的权重和 id，可选重新归一化 | [#279](https://github.com/flagos-ai/FlagGems-sglang/pull/279) | liuhycs |

### 状态

这 6 个算子只在 W7900D 上跑了单测（每个 4 个用例，全部通过），还没有计时；RTX 4090 和 R9700 都没测。

已记录的 141 个算子在这期间的变化：

- `decode_attention`、`decode_grouped_attention`、`extend_attention`、`log_scaling_tau` 的 gfx1100 版本合入（[#271](https://github.com/flagos-ai/FlagGems-sglang/pull/271)），W7900D 上共 18 个算子走 `_amd/rdna3/ops/`。
- batch9（[#270](https://github.com/flagos-ai/FlagGems-sglang/pull/270)）补上第三段 17 个算子的参考实现、单测和 benchmark；[#272](https://github.com/flagos-ai/FlagGems-sglang/pull/272) 修好第一段那 10 个调不通的 benchmark。
- `fused_qk_gemma_rmsnorm_rope_gate` 的通用实现把 `next_power_of_2` 移到 kernel 外（[#285](https://github.com/flagos-ai/FlagGems-sglang/pull/285)）；MThreads、T-Head 新增了 `one_pass_rms_norm` 的厂商版本，参数名在 [#287](https://github.com/flagos-ai/FlagGems-sglang/pull/287) 对齐。
- 其余算子的通用实现、参考实现、单测和厂商版本都没有改。
- 在审：[#289](https://github.com/flagos-ai/FlagGems-sglang/pull/289) 修复 `rmsnorm_scale_shift`、`qk_head_layernorm`、`silu_and_mul_masked` 的通用实现，[#290](https://github.com/flagos-ai/FlagGems-sglang/pull/290) 新增 `residual_gate_add`、`silu_and_mul_masked` 的 gfx1100 版本，见 `W7900D_AMD_optimization.md` 的「正确性修复（2026-11-04）」。

## W7900D 摸底

这 85 个算子在 W7900D（gfx1100）上测过一轮：68 个有 benchmark 的用 benchmark 自己的输入，其中 10 个调不通的改用单测里的用例，最新 17 个没有 benchmark、按通用实现的签名和代码里写死的形状分支自己构造输入。单测除 `residual_gate_add` 外都通过。结论是大多数已跑在这张卡的可达带宽附近，当时最值得优化的是 `tiny_n_gemm`、`get_target_cache_loc`、`top_p_renorm`、`recompute_w_u`。其中 `tiny_n_gemm`、`top_p_renorm` 已合入 [#298](https://github.com/flagos-ai/FlagGems-sglang/pull/298)：rdna3 `tiny_n_gemm` 对 M>1 一律 split-K，官方 benchmark n=1536 快 5.42–5.61 倍、n=512 快 1.44–1.56 倍；rdna3 `top_p_renorm` 用探测点 + Anderson–Björck 减少搜索遍数，官方 benchmark 快 1.35–2.32 倍；同 PR 还修了通用 split-K 对 K 尾部的掩码。还剩 `get_target_cache_loc`（封装里三处主机取值，连续调用 70.4 µs 里只有 19.7 µs 在 GPU 上）和 `recompute_w_u`（`_packed_dot` 有 64 条 2 字节访存对 12 条 16 字节）。详情见 `W7900D_AMD_optimization.md` 的「`tiny_n_gemm` 复核」「`tiny_n_gemm` / `top_p_renorm`（2026-11-09）」。

这些算子后来和早先的 56 个放在同一口径下重新测过、重新分了批，口径（含为什么按整张量算字节数会把 KV cache 这类算子算出 2263% 的带宽）、逐算子数据和新的分批都在 `W7900D_AMD_optimization.md` 的「第三轮」，完整排名表在 `opt_w7900d/survey/results/batches.md`。
