# R9700（RDNA4）优化记录：进展、PR 候选和后续

R9700 上投机解码和推理性能的工作记录：做了什么、结论和数据、PR 候选，以及后续能做的。本文件原名 `pr_candidates.md`。

- 最后更新：2026-11-03。
- 机器：8 张 AMD Radeon AI PRO R9700（gfx1201，64 CU，32 GB）。
- 行号对应 vLLM main `852989f58`（2026-10-19）和 AITER 0.1.25。
- 第 1–4 条的测试环境：单卡 R9700；Qwen3-8B 目标模型 + `dspark_qwen3_8b_block7` 草稿（草稿长度 7）；ShareGPT 数据集，`--max-model-len 16384`。第 5–7 条的模型和设置写在各条里。

## 维护方式

- **GitHub 上的位置**：[`WhatGhost/testing_notes` 的 `SpeculativeDecoding/r9700_optim_notes.md`](https://github.com/WhatGhost/testing_notes/blob/main/SpeculativeDecoding/r9700_optim_notes.md)。MI355X 的进度在同目录的 `mi355_spec_optim_progress.md`。
- **R9700 机器上的对应关系**：仓库克隆在 `/target/spec_test/testing_notes`。`/target/spec_test/dspark_aiter_debug/r9700_optim_notes.md` 是指向仓库里这份文件的软链接，改哪个路径都是同一个文件。
- **每次更新**：每做完一项优化或测量，就更新以下几处：
  - 对应条目的内容；
  - "当前状态"表、"后续能做的"和"更新记录"；
  - 开头的"最后更新"日期。

  然后在仓库里提交（commit message 写一句话），推送到 `main`。
- **换机器**：`gh repo clone WhatGhost/testing_notes`，在这份文件上接着更新。文中的数据和脚本都在 R9700 机器的 `/target/spec_test/dspark_aiter_debug/` 下，没有上传。

## 当前状态（2026-11-03）

| # | 事项 | 类型 | 结论和收益 | 状态 |
|---|---|---|---|---|
| 1 | RDNA4 加 6–128 行的 Triton skinny GEMM | vLLM PR | DSpark conc 1/4/16 吞吐 +28–29%，不开投机 conc 16–64 +20–25% | [#62799](https://github.com/vllm-project/vllm/pull/62799) 10-21 提交。无人 review；10-30 起和 main 冲突，需要 rebase；`pre-run-check` 要维护者加 `ready` 或 `verified` 标签 |
| 2 | 报告 hipBLASLt 在 gfx1201 小行数时选错 kernel | 上游 issue | 库里已有快 1.1–1.4 倍的 kernel | 未报 |
| 3 | 用 R9700 数据支持 #61433 | 评审数据 | 开投机的默认配置 +15–80%，混合模型 +3–7% | #61433 仍 open；评审评论草稿（第 3 条末尾）未发 |
| 4 | AITER Unified 后端支持 non-causal | vLLM（可选 AITER） | R9700 上走 vLLM 的 Triton 回退，预计没有性能收益 | R9700 暂不做；MI355 侧在做（`mi355_spec_optim_progress.md` 的候选 4） |
| 5 | R9700 的 fused_moe 调优配置（Qwen3.6-35B-A3B） | 配置文件 | 单卡 int4：kernel 1.18–1.63 倍，端到端 conc 4–32 +8–12%。TP=4 BF16：kernel 1.01–1.16 倍 | int4 和 [#64045](https://github.com/vllm-project/vllm/pull/64045) 重复，11-03 已评论支持并建议补 `"512"`；BF16 TP=4 不提交 |
| 6 | ROCm 后端支持 fused multi-step draft decode | 评估 | R9700 上没有收益，和 MI355X 的结论一致 | 不做 |
| 7 | 投机解码的快/慢两种状态和 `GPU_MAX_HW_QUEUES` | 根因排查 | 默认环境下每段随机落在快态或慢态（差约 12%）；`GPU_MAX_HW_QUEUES=1` 或 `2` 稳定在快态 | 未报 issue。TP=4 加 `GPU_MAX_HW_QUEUES=1` 时机器死机过一次 |

## 后续能做的

按建议顺序：

1. #62799 rebase 解决冲突，跟进 review（第 1 条）。
2. 跟进 #64045 对 `"512"` 条目的回复（第 5 条）。
3. 在同一时段对比默认环境和 `GPU_MAX_HW_QUEUES=1`（Qwen3.6 int4 单卡）。目前 conc 1 的 69 对 53 tok/s 是跨日期、跨卡的对比（第 7 条）。
4. 决定是否报 hipBLASLt 的 issue（第 2 条）、是否发 #61433 的评审评论（第 3 条）、是否给 ROCm 报快/慢状态的 issue（第 7 条）。
5. skinny GEMM 扩到 129–256 行（按形状查表；原型在 DSpark conc 32 每步 −16%），384–512 行另想方案，并验证 gfx1200 和 RDNA3（第 1 条"后续"）。
6. 调 MTP 草稿层用的 BF16 MoE 配置 `E=256,N=512,device_name=AMD_Radeon_R9700.json`（第 5 条）。
7. MI355 侧的第 4 条方案落地后，在 R9700 上确认 DSpark 草稿仍选 TRITON_ATTN、性能不回退。

## 更新记录

> **2026-11-03**：文档由 `pr_candidates.md` 改名，加了"当前状态"和"后续能做的"。第 1 条已提交为 #62799；第 5 条完成（int4 配置和 #64045 重复，已评论）；新增第 6 条（fused multi-step draft decode）和第 7 条（快/慢状态）。
>
> **2026-10-20 更新（二）**：补完了之前因为机器卡住没跑的测试。
>
> - 第 1 条：正式实现在 R9700 上跑完了 GPU 单测、微基准、端到端 A/B 和冷启动。按补测结果调整了 tile 规则：N 的下限从 256 提到 512，65–128 行遇到窄权重时改用小 tile。
> - 第 3 条：在 R9700 上实测了 #61433 本身（默认配置、开 AITER、混合模型），补测了不开投机时 ROCM_ATTN vs TRITON_ATTN，并附上评审评论草稿（未发）。
>
> **2026-10-20 更新（一）**：对照 zhentaocc 最近两周的 vLLM PR 调整了清单（详见文末"和 zhentaocc 近期 PR 的关系"）。
>
> - 原第 3、4 条（attention 后端选择）已被 [#61433](https://github.com/vllm-project/vllm/pull/61433) 覆盖，合并成新的第 3 条：用 R9700 数据支持 #61433，不另开 PR。
> - 原第 5 条（AITER Unified non-causal）降级为第 4 条，补充它和 #61433 的相互影响。
> - 原第 6 条（ROCM_ATTN 草稿元数据原地更新）删除：#61433 落地后草稿默认不再用 ROCM_ATTN。
> - 原第 7 条（fused_moe 配置）顺延为第 5 条。
> - 第 1 条注明和 RFC [#61761](https://github.com/vllm-project/vllm/issues/61761) 的分工；代码已在本地分支 `gfx12-skinny-gemm` 写好（未提交），行数范围从 6–256 收窄到 6–128。

## 和路线图 issue #9 的对应

路线图：<https://github.com/AMD-AIM/speculative-bench/issues/9>。

| # | 对应的路线图条目（Radeon 部分） |
|---|---|
| 1 | Optimize GEMM for baseline, drafting, and target verification（P0）；Optimize dense MLP and MoE execution（P0） |
| 2 | Optimize GEMM（P0） |
| 3 | Improve speculative-aware target/draft backend selection（P1） |
| 4 | Support required non-causal attention in AITER Unified（P2） |
| 5 | Optimize dense MLP and MoE execution（P0）；Support W4A16 on Radeon for larger models（P3） |
| 6 | Extend HIP Graph coverage to draft models（P1） |
| 7 | 没有直接对应的条目；影响所有 R9700 上的性能测量 |

---

## 1. gfx12 加 6–128 行的 Triton skinny GEMM 路径

### 问题

`vllm/model_executor/layers/utils.py` 的 `rocm_unquantized_gemm_impl` 在 gfx12 上按行数分两条路（行数即这一步要算的 token 数）：

- **1–5 行**：走 vLLM 自己的 `wvSplitK`，带宽约 600 GB/s。
- **6 行以上**：直接走 `F.linear`，也就是 hipBLASLt。Qwen3-8B 的形状在 6–512 行都选同一个 `MT128x128x32` kernel，小行数时只有 420–470 GB/s。别的形状要到 128 行以上 hipBLASLt 才换成更合适的 kernel（见下面的 tile 扫描）。

所以 5→6 行有断崖，比如 gate_up 从 323 µs 跳到 473 µs。`csrc/rocm/skinny_gemms.cu` 里 `wvSplitK` 只有 1–5 行的模板，其他行数直接报错，不能只改阈值。

各场景的行数：

- **不开投机**：行数等于并发数。
- **DSpark 验证**：每个请求 8 个 token，conc 1 就是 8 行，从单请求开始就落在断崖之后。

DSpark conc 1 每步 GPU 忙碌 44.8 ms，其中 GEMM 占 42.1 ms（94%）：

| 位置 | 每步耗时（ms） |
|---|---|
| 目标模型 gate_up | 17.3 |
| 目标模型 down | 7.5 |
| 目标模型 qkv | 3.9 |
| 目标模型 o_proj | 3.0 |
| 目标模型 lm_head | 2.3 |
| 草稿模型合计 | 8.1 |

### 证据

**微基准（原型阶段）**：冷权重，每次调用的耗时（µs），6–64 行。

| 形状 | 5 行 `wvSplitK` | hipBLASLt 默认 | TunableOp 最优 | Triton 原型 |
|---|---|---|---|---|
| gate_up 4096→24576 | 323 | 473–587 | 335–351 | 334–343 |
| down 12288→4096 | 163 | 237–318 | 182–197 | 168–172 |
| qkv 4096→6144 | 85 | 107–110 | 97–101 | 89–93 |
| o_proj 4096→4096 | 58 | 76–83 | 69–72 | 62 |
| lm_head 4096→151936 | 1988 | 2289–2569 | 1989–2012（只测了 6–14 行） | 1990–2078 |

**tile 扫描（用来定正式实现的规则）**：

- 覆盖 36 个形状：Qwen3 4B/8B/14B、Qwen3-32B TP=2、Llama-3-8B、gpt-oss、Gemma 词表投影、DSpark 和 Markov 草稿、两个 MoE router，以及 N=256–2048 的 7 个补测形状。
- 6–128 行测了 13 个行数（6、8、12、16、24、32、40、48、64、72、80、96、128）。每个点试所有 tile 组合，并和 hipBLASLt 对比。共 17694 条记录，最大相对误差 1.8e-3。
- 下表是最终规则。"最优"指这个点所有 tile 组合里最快的那个；只统计规则覆盖的点（N ≥ 512、K 是 64 的倍数）。

| 行数 | N | tile（BM, BN, BK 上限, warps） | 点数 | 相对最优（几何平均） | 最差相对最优 | 相对 hipBLASLt（几何平均） | 最差相对 hipBLASLt | 慢于 hipBLASLt 的点 |
|---|---|---|---|---|---|---|---|---|
| 6–16 | ≥ 512 | 16, 64, 256, 4 | 132 | 1.021 | 1.152 | 0.785 | 0.953 | 0 |
| 17–32 | ≥ 512 | 32, 32, 256, 4 | 66 | 1.014 | 1.058 | 0.746 | 0.967 | 0 |
| 33–64 | ≥ 512 | 64, 32, 256, 4 | 93 | 1.022 | 1.231 | 0.743 | 1.012 | 1 |
| 65–128 | ≥ 4096 | 128, 64, 128, 8 | 96 | 1.046 | 1.192 | 0.743 | 1.033 | 8 |
| 65–128 | 512–4095 | 64, 32, 256, 4 | 36 | 1.054 | 1.233 | 0.744 | 0.967 | 0 |
| 合计 | | | 423 | 1.029 | 1.233 | 0.757 | 1.033 | 9 |

- 慢于 hipBLASLt 的 9 个点都在 3.3% 以内：Qwen3-14B 的 o_proj（64/72/80/128 行）和 qkv（72/80/96/128 行），以及 Qwen3-8B qkv 的 96 行。
- 补测得出的结论：
  - **N 的下限取 512**：N=256 在 64 行时比 hipBLASLt 慢 8%；N=512–2048 在 6–128 行都更快。N=32、128 的 MoE router 不走 Triton。
  - **65–128 行按 N 分两种 tile**：N < 4096 时，128×64 的 tile 凑不出足够的 workgroup 占满 64 个 CU（N=512 时只有 8 个 N 块），比 hipBLASLt 慢 1.3–2.3 倍。换成 64×32 后，这些点都比 hipBLASLt 快（最差 0.967）。
  - **72、80 行**：和 96、128 行用同一套规则；慢点只有上面列出的 Qwen3-14B 那几个。
  - **不用 2 级流水**：同一个 tile 下，2 级比 1 级平均慢 29%（几何平均 1.288，最多慢 2.7 倍）。每个点各挑最好的 2 级 tile，也比最好的 1 级平均慢 17%，66 个点里只有 1 个快 2% 以上。
- **129–256 行**：即使每档挑最好的规则，81 个点里也有 33 个比 hipBLASLt 慢，最多慢 15%。慢的主要是 N=5120 的形状（Qwen3-14B 的 o/down、Qwen3-32B TP=2 的 o/down、gpt-oss 的 qkv）和词表投影，因为 hipBLASLt 在 128 行以上换了 kernel（比如 Qwen3-14B down 从 606 µs 降到 417 µs）。所以正式实现只覆盖到 128 行。

**微基准（正式实现 vs main 的分派）**：

- 直接调 `rocm_unquantized_gemm_impl`，冷权重，HIP graph 回放。
- 36 个形状 × 18 个行数（1–256），共 648 个点，最大相对误差 1.8e-3。
- 下表是正式实现耗时 / main 耗时：

| 行数 | 点数 | 几何平均 | 最差 |
|---|---|---|---|
| 1–5（两边都走 `wvSplitK`） | 72 | 1.000 | 1.004 |
| 6–16 | 144 | 0.809 | 1.002 |
| 17–32 | 72 | 0.772 | 1.001 |
| 33–64 | 108 | 0.759 | 1.024 |
| 65–128 | 144 | 0.747 | 1.031 |
| 129 以上（两边都走 hipBLASLt） | 108 | 1.001 | 1.058 |

- 走 Triton 的点里，只有 Qwen3-14B o_proj 在 128 行慢了 3.1%，和扫描一致。两边同路径的点单点最多差 5.8%，这是测量噪声的量级。
- Qwen3-8B 和 DSpark 的形状（µs，main → 正式实现）：

| 形状 | 8 行 | 32 行 | 64 行 | 96 行 | 128 行 |
|---|---|---|---|---|---|
| qkv | 111 → 92 | 109 → 93 | 110 → 95 | 114 → 113 | 122 → 116 |
| o_proj | 76 → 62 | 78 → 61 | 83 → 64 | 98 → 76 | 166 → 77 |
| gate_up | 488 → 336 | 502 → 346 | 517 → 343 | 550 → 392 | 571 → 401 |
| down | 250 → 168 | 291 → 171 | 323 → 175 | 411 → 193 | 1115 → 201 |
| lm_head | 2316 → 1998 | 2384 → 2055 | 2491 → 2079 | 2593 → 2188 | 2673 → 2240 |
| DSpark fc | 360 → 274 | 440 → 281 | 548 → 289 | 673 → 313 | 1505 → 322 |
| DSpark ctx_kv | 175 → 151 | 178 → 155 | 188 → 151 | 200 → 190 | 207 → 193 |

128 行时 hipBLASLt 的 o_proj、down、fc 出现了异常慢的点，和第 2 条说的耗时不稳定一致。

**端到端（正式实现 vs main）**：

- main 组用 `git archive` 导出的 `852989f58` 原样代码，用 `PYTHONPATH` 指过去。
- 两组在同一张卡上先后跑，各用独立的编译缓存。不同卡之间 conc 1 的吞吐能差 12%，所以不跨卡比较。
- 每步耗时取 ITL 中位数。

| 场景 | 行数 | 每步耗时 | 吞吐（tok/s） |
|---|---|---|---|
| DSpark conc 1 | 8 | 44.0 → 34.2 ms | 90.5 → 115.9（+28.0%） |
| DSpark conc 4 | 32 | 47.4 → 36.4 ms | 313.3 → 403.0（+28.6%） |
| DSpark conc 16 | 128 | 65.4 → 45.1 ms | 627.0 → 811.4（+29.4%） |
| DSpark conc 32 | 256 | 71.6 → 71.6 ms | 1042.4 → 1146.9（+10.0%，见下） |
| DSpark conc 64 | 512 | 117.3 → 121.4 ms | 1268.3 → 1314.9（+3.7%，见下） |
| 基线 conc 1 / 4 | 1 / 4 | 不变 | +0.1% / +0.3% |
| 基线 conc 16 | 16 | 37.9 → 28.9 ms | 332.4 → 416.3（+25.2%） |
| 基线 conc 32 | 32 | 39.5 → 30.6 ms | 555.9 → 672.3（+20.9%） |
| 基线 conc 64 | 64 | 45.6 → 33.6 ms | 812.8 → 978.4（+20.4%） |

- 接受长度不变（conc 1 两边都是 4.172，conc 4 都是 4.255）。
- **DSpark conc 32/64 的提升不是稳态收益**：
  - 这两档的行数是 256/512，不在范围内，每步耗时没变（平均 TPOT 27.9 → 27.4 ms、43.5 → 43.5 ms）。
  - 吞吐的提升来自压测的收尾阶段：剩下的请求少于 16 个时，行数落回 128 以内。这轮压测只有约 20 s，收尾占比不小（conc 32 的总时长 20.8 → 18.9 s）。
  - 持续满载时，conc 32 预计只有 0–2% 的提升。原型放宽到 256 行时，conc 32 的每步耗时是 71.5 → 60.3 ms（−16%）。
- **小模型**：Qwen3-0.6B 不开投机。conc 1/4 是 −0.2% / −1.3%（两边同路径，属于噪声）；conc 16/32/64/128 是 +20.2% / +9.6% / +11.8% / +8.5%。它的 o_proj 和 down 的 N=1024，验证了窄权重那套 tile。

**启动时间**：清空 vLLM 和 Triton 缓存后冷启动（DSpark 配置）。正式实现 171 s 就绪，main 170 s；init engine 都是 119.5 s，torch.compile 47.4 vs 47.8 s。Qwen3-8B + DSpark 只编译出 4 个 kernel 变体，行数变化不触发重编译。

### 改法（分支 `gfx12-skinny-gemm`，commit `9166d37cc`，基于 `852989f58`，已提交为 #62799）

**分派**（`vllm/model_executor/layers/utils.py`）
- 位置：`rocm_unquantized_gemm_impl` 里 `LLMM1` 分支之后、AITER tgemm 和 `F.linear` 之前。其他架构上 `on_rdna4()` 为 False，行为不变。
- 条件：
  - 沿用 `skinny_gemm_enabled`：`VLLM_ROCM_USE_SKINNY_GEMM` 打开、fp16/bf16、K 是 8 的倍数；
  - `on_rdna4()`：只认 gfx1200 和 gfx1201；
  - 权重每行内连续，行 stride 可以有 padding（ROCm 加载权重时会为避免 cache 别名加 padding）；bias 连续；
  - `skinny_gemm_config` 不返回 None：6 ≤ 行数 ≤ 128、N ≥ 512、K 是 64 的倍数。
- 这个函数注册成了 custom op（`torch.ops.vllm.rocm_unquantized_gemm`），对 torch.compile 不透明，加分支不影响编译图。

**kernel**（新文件 `vllm/kernels/triton/skinny_gemm.py`）
- 每个 program 算一个 BM×BN 的输出块，沿 K 循环，fp32 累加，支持 bias。同一个 N 块的各个 M 块排在相邻的 program 上，权重只从显存读一次。
- 不做 split-K，tile 扫描里 split-K 一次都没赢过。
- 行数不参与特化（`do_not_specialize=["M"]`），行数变化不会触发重编译。
- tile 按行数分档，65–128 行再按 N 是否 ≥ 4096 分两种（见上面的规则表），不按形状查表。BK 取上限以内能整除 K 的最大值（256/128/64）。这些 tile 对 2 字节类型都放得进 RDNA4 每个 workgroup 64 KB 的 LDS。
- 只用 1 级流水（`num_stages=1`），2 级在 RDNA4 上更慢。

**顺带改一处**
- `vllm/model_executor/models/qwen3_dflash.py` 的 ctx_kv 融合投影直接调 `F.linear`，绕过了上面的分派，Qwen3 DSpark 也复用这段代码。
- 已改成 `dispatch_unquantized_gemm()(self, x, weight, bias)`，CUDA 上仍是 `F.linear`。单项收益很小（8 行时 175→151 µs），主要是让路径一致。

**和 RFC #61761 的分工**
- RFC #61761（DFlash2 在 MI355X 上的优化）里的 GEMM 调优针对 MI355X，走 AITER 的 GEMM 配置。本条只对 RDNA4 生效，两边的代码路径不重叠，是互补的。
- 原来列的备选方案"给 AITER 的 `gemm_a16w16` 补 gfx1201 配置，再放开 `use_aiter_triton_gemm`"和 RFC 的方向重叠，不再作为备选。
- 两边都会改 `rocm_unquantized_gemm_impl` 附近的代码。#62799 已提交，RFC 那边改到这里时要和 zhentaocc 协调。

**剩下的备选实现**
- 把 `wvSplitK` 模板扩到更多行（C++）。行数多了寄存器压力大，可能要换成基于 WMMA 的设计。

### 测试

都在 `tests/model_executor/layers/test_rocm_unquantized_gemm.py`：

- **更新现有用例**：原来的 `test_rocm_unquantized_gemm_gfx1x_n_gt_5_falls_back` 断言 gfx1x 超过 5 行就回退，现在只对 gfx11 成立，改名为 `..._gfx11_n_gt_5_falls_back`。
- **新增分派用例**（mock，不需要 GPU）：
  - RDNA4 上 6、64、128 行走 Triton；
  - 129 行、N < 512（用例里取 256）、K 不是 64 的倍数、fp32、权重转置、bias 不连续、gfx11，这几种情况都回退。
- **新增数值用例**（只在 RDNA4 上跑）：
  - fp16/bf16 × 有无 bias × 3 个形状（覆盖 N 不是 tile 整数倍、K=2880 不是 128 的倍数）× 7 个行数（每档都有）；
  - 3D 输入、转置输入、带 padding 的权重。
  - 和 fp32 参考比，容差 1e-2。
- **当前状态**：R9700 上 131 个通过、2 个跳过（原有的、只在 gfx11 上跑的 padding 用例）；ruff 检查和格式检查通过。
- **端到端复现时注意编译缓存**：vLLM 的 torch.compile 缓存不感知运行时替换，也不感知 `PYTHONPATH` 切换。做 A/B 时每组要用独立的 `VLLM_CACHE_ROOT`，否则会加载另一组的编译结果。

### 状态（2026-11-03）

- 10-21 推到 fork（`WhatGhost/vllm:gfx12-skinny-gemm`）并提交为 [#62799](https://github.com/vllm-project/vllm/pull/62799)，PR 描述是 `pr_desc_rdna4_skinny_gemm.md`。
- 还没有人工 review，只有机器人评论。
- 10-30 mergify 标了 `needs-rebase`，和 main 有冲突。
- CI 里失败的是 `pre-run-check` 和 readthedocs。`pre-run-check` 要求 PR 有 `verified`、`ready` 或 `ready-run-all-tests` 标签，或者作者已合入至少 4 个 PR（目前 0 个），只能等维护者加标签。readthedocs 在 #63015 上也同样失败，没细查。
- 换到 W7900 上复测的流程在 `w7900_test_guide.md`，打包为 `w7900_kit.tar.gz`；W7900 上的结果这里没有记录。

### 后续

- **129–256 行**：Qwen3-8B 的形状在 256 行时仍然是 Triton 快（gate_up 530 vs 646 µs，down 289 vs 350 µs，qkv 154 vs 214 µs），但别的形状会变慢。按形状放开到 256 行后，DSpark conc 32 的每步耗时能从 71.5 降到 60.3 ms（原型实测，−16%），持续满载时这部分收益可观；但需要按形状查表，先不放进第一版。
- **RDNA3（gfx11）**：没测过，先只对 RDNA4（gfx1200、gfx1201）启用。gfx1200（RX 9060 系列，32 个 CU）也没测过，tile 规则是在 64 个 CU 的 R9700 上调的。
- **384–512 行**（DSpark conc 64）：没有收益。TunableOp 显示库内 kernel 能快 10–20%（比如 qkv 512 行 317→258 µs），需要另想方案，作为后续。

---

## 2. 报告 hipBLASLt 在 gfx1201 小行数时选错 kernel（上游 issue）

这一条不是 vLLM PR，而是给 hipBLASLt 上游报 issue。修好后 PyTorch、vLLM、SGLang 都受益，但修复时间不由我们控制，所以和第 1 条并行推进。

**状态（2026-11-03）**：还没报。本机 gh 的 classic token 访问不了 ROCm 组织的仓库，要在网页上提。

- **现象**：gfx1201 BF16 TN GEMM，Qwen3-8B 的形状在 6–512 行都选 `MT128x128x32`。
- **库里其实有更快的 kernel**：用 PyTorch TunableOp 在 hipBLASLt / rocBLAS 里搜，6–64 行时：
  - gate_up、down 快 1.3–1.4 倍（473→335 µs，237→182 µs）；
  - qkv、o_proj 快约 1.1 倍。
  - 胜出的大多是 hipBLASLt 的 72674 号解和 rocBLAS 的 72675 号解。
- **耗时不稳定**：和指针对齐、调用历史有关。
  - o_proj 128 行：0 字节偏移时 150 µs，16 字节偏移时 80 µs。
  - down 128 行：测出过 250–1100 µs（这次正式实现的微基准里又测到 1115 µs）。
- **附件**（都在 `gemm/` 下）：
  - 复现脚本：`bench_skinny_gemm.py`
  - 默认结果：`results/gemm_r9700.jsonl`
  - TunableOp 结果：`results/gemm_tunableop_r9700.jsonl`，以及它选中的解 `results/tunableop_r97000.csv`

---

## 3. 用 R9700 数据支持 #61433（覆盖原第 3、4 条，不另开 PR）

**状态（2026-11-03）**：#61433 仍 open、未合入；本条末尾的评审评论草稿还没发。

### #61433 改了什么

- 在 `vllm/platforms/rocm.py` 的后端优先级里加了 `use_spec_decode`、`is_hybrid` 两个条件：开投机或者是混合模型时，ROCM_ATTN 不再排第一，而是排到 TRITON_ATTN 后面（`rocm_attn_first = use_rocm_attn and not (use_spec_decode or is_hybrid)`）。
- PR 里只有 MI355X 的数据（conc 1/16/64 吞吐 +11% / +31% / +54%），状态是等待评审。

这正好覆盖了原第 3 条（目标模型）和原第 4 条（草稿）想改的东西，我们不再另开 PR，改为给它补 R9700 的数据。

### R9700 实测

- 方法：把 #61433 对 `rocm.py` 的改动（PR head `761e95eecf`）打到 `852989f58` 的导出代码上，和原样的 `852989f58` 对比。启动时不传 `--attention-backend`，两组在同一张卡上先后跑。
- 后端选择和按代码推算的一致：

| 配置 | main（目标 / 草稿） | #61433（目标 / 草稿） |
|---|---|---|
| DSpark，默认 | ROCM_ATTN / ROCM_ATTN | TRITON_ATTN / TRITON_ATTN |
| DSpark，`VLLM_ROCM_USE_AITER=1` | AITER Unified / ROCM_ATTN | AITER Unified / TRITON_ATTN |
| Qwen3.5-9B（混合模型），不开投机 | ROCM_ATTN | TRITON_ATTN |

- 吞吐（tok/s）：

| 配置 | conc 1 | 4 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| DSpark 默认：main | 79.2 | 260.2 | 476.8 | 681.8 | 709.0 |
| DSpark 默认：#61433 | 90.8 | 306.7 | 664.9 | 1115.1 | 1274.6 |
| 提升 | +14.7% | +17.9% | +39.5% | +63.6% | +79.8% |
| DSpark 开 AITER：main | 88.0 | 294.2 | 628.6 | 1066.9 | 1251.4 |
| DSpark 开 AITER：#61433 | 88.0 | 298.9 | 656.8 | 1088.0 | 1308.1 |
| 提升 | −0.0% | +1.6% | +4.5% | +2.0% | +4.5% |
| Qwen3.5-9B：main | 35.1 | 124.8 | 280.7 | 444.6 | 629.1 |
| Qwen3.5-9B：#61433 | 36.3 | 131.4 | 293.6 | 470.3 | 672.4 |
| 提升 | +3.4% | +5.3% | +4.6% | +5.8% | +6.9% |

- 默认配置下，#61433 的结果和之前手动指定 TRITON_ATTN 的 c1 一致（差 −2.8% 到 +4.3%，在噪声范围内）；开 AITER 时的变化幅度和之前 c3 → c7 的结果一致。接受长度两边最多差约 3%（默认配置 conc 1：4.049 vs 4.172）。
- 默认配置的提升主要来自目标模型：投机验证时每个请求有 8 个 query token，ROCM_ATTN 会走 `chunked_prefill_paged_decode` 中 `max_query_len > 1` 的 prefix-prefill 分支。这是推测的变慢原因，还没单独 profile。
- 开 AITER 时只有草稿换了后端，提升在 5% 以内。

### #61433 没覆盖的部分

- **不开投机**：#61433 保持 ROCM_ATTN 排第一。R9700 上实测（Qwen3-8B，main，同卡），ROCM_ATTN 比 TRITON_ATTN 慢 0–2%（conc 1/4/16/32/64：−0.8% / −0.1% / −0.8% / −2.0% / −2.0%）。差距太小，不足以支撑改全局默认，在评审里提一句即可。
- **以 prefill 为主的负载**：没测。

### 评审评论草稿（未发，由你决定发不发、怎么发）

```markdown
Tested on AMD Radeon AI PRO R9700 (gfx1201, RDNA4), single GPU, with this PR's
`rocm.py` change applied on top of main 852989f58. Workload: Qwen3-8B + DSpark
draft (block size 7), ShareGPT, `--max-model-len 16384`, no `--attention-backend`.
Both arms ran on the same GPU.

Backend selection matches the description:
- Spec decode: ROCM_ATTN -> TRITON_ATTN for both target and draft.
- Spec decode with `VLLM_ROCM_USE_AITER=1`: target stays on ROCM_AITER_UNIFIED_ATTN,
  draft ROCM_ATTN -> TRITON_ATTN.
- Hybrid model (Qwen3.5-9B), no spec decode: ROCM_ATTN -> TRITON_ATTN.

Output throughput (tok/s) at concurrency 1 / 4 / 16 / 32 / 64:

| Config | main | this PR | Change |
|---|---|---|---|
| DSpark | 79.2 / 260.2 / 476.8 / 681.8 / 709.0 | 90.8 / 306.7 / 664.9 / 1115.1 / 1274.6 | +15% / +18% / +40% / +64% / +80% |
| DSpark, AITER=1 | 88.0 / 294.2 / 628.6 / 1066.9 / 1251.4 | 88.0 / 298.9 / 656.8 / 1088.0 / 1308.1 | 0% / +2% / +4% / +2% / +5% |
| Qwen3.5-9B, no spec | 35.1 / 124.8 / 280.7 / 444.6 / 629.1 | 36.3 / 131.4 / 293.6 / 470.3 / 672.4 | +3% / +5% / +5% / +6% / +7% |

Acceptance length differs by at most ~3% between the two arms. Without spec decode,
ROCM_ATTN is within 2% of TRITON_ATTN on R9700 (Qwen3-8B), so keeping ROCM_ATTN
first there looks fine.
```

---

## 4. AITER Unified 后端支持 non-causal（降级）

**状态（2026-11-03）**：R9700 上暂不做。MI355 侧在做（[`mi355_spec_optim_progress.md`](https://github.com/WhatGhost/testing_notes/blob/main/SpeculativeDecoding/mi355_spec_optim_progress.md) 的候选 4）：AITER 放开 gfx950 Gluon kernel 的非因果分支，vLLM 的 `ROCM_AITER_UNIFIED_ATTN` 声明支持非因果并把 `causal` 传给 AITER；MI355X 上 conc 16 还有 −8.5% 的回退在查。那边落地后，R9700 上要确认两点：

- R9700 没有 Gluon kernel。非因果时不能走到 AITER Triton kernel 里的 `assert causal`，要继续用 vLLM 的 Triton 回退（见下面"关键发现"）。
- DSpark 草稿的自动选择仍是 TRITON_ATTN，性能不回退（见"和 #61433 的联动"）。

### 问题

把草稿指定为 ROCM_AITER_UNIFIED_ATTN（c5），启动直接失败：

```
ValueError: Selected backend AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN is not valid for this configuration. Reason: ['non-causal attention not supported']
```

### 关键发现

vLLM 这边其实已经实现了 non-causal。

- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py` 第 327–356 行：`causal=False` 时会回退到 vLLM 自己的 Triton unified kernel。这个回退原本是给 Whisper 这类 encoder-decoder 模型用的。
- 但第 100 行的 `supports_non_causal()` 写死返回 `False`，所以 DSpark 草稿在选后端时就被过滤掉了。

### 和 #61433 的联动（降级原因）

- 现在 AITER Unified 不支持 non-causal，草稿选后端时会跳过它；#61433 之后草稿落到 TRITON_ATTN（R9700 开 AITER 时已实测）。
- 如果按本条让它支持 non-causal，那么在 AITER Unified 排在 TRITON_ATTN 前面的配置下（MI355X 默认、R9700 开 AITER 时），草稿的自动选择会从 TRITON_ATTN 变成 AITER Unified。
- AITER Unified 没有声明 `supports_draft_decode_metadata_update`，DSpark 每次回放草稿 graph 前都要在 CPU 上重建一遍元数据。按最小改法它跑的又是和 TRITON_ATTN 同一个 Triton kernel，所以对 DSpark 没有收益，只多了 CPU 开销。
- 因此要做的话，必须同时保证 DSpark 草稿仍优先 TRITON_ATTN（或者给 AITER Unified 补上原地更新），并先和 #61433 对齐。
- RFC #61761 计划给 AITER Unified 加 split-KV，可能会改同一个文件，也要协调。[#61760](https://github.com/vllm-project/vllm/pull/61760) 和本条是互补的，不冲突。

### 改法（最小，只改 vLLM）

让 `supports_non_causal()` 返回 `True`，同时处理一个例外：shuffled KV 布局。

- 这种布局只在 gfx1250 且开了 AITER shuffle 时启用（第 197 行），R9700 不受影响。
- 回退用的 Triton kernel 读不了这种布局。
- 第 201 行的 `_use_shuffle_kv` 只排除了 ENCODER_DECODER，DSpark 草稿这种 non-causal decoder 会被判成走 shuffled 路径。

处理办法二选一：要么在这种布局下 `supports_non_causal()` 仍返回 `False`，要么让 non-causal decoder 也不走 shuffled 路径。

### 改法（完整，可选，属于 AITER PR）

- 去掉 `aiter/ops/triton/attention/unified_attention.py:208` 的 `assert causal`。
- 给 `aiter/ops/triton/_triton_kernels/attention/unified_attention.py` 的 2D 和 3D kernel 加 `CAUSAL` 常量。现在 causal 是写死的：2D kernel 在第 254–290 行和第 473 行，3D kernel 在第 751–765 行和第 879 行。gfx950/gfx1250 的 Gluon kernel 已经有 `CAUSAL`。

只有 AITER kernel 在草稿形状上比 vLLM 的 Triton 快时才值得做，需要先测。

### 收益

主要是修复用户显式指定该后端时"无法运行"的问题。按最小改法，草稿走的是和 TRITON_ATTN 同一个 kernel，性能预计相同。

### 验证

重跑 c5 配置，检查接受长度和吞吐与 c7 一致；同时确认不指定草稿后端时，草稿仍选到 TRITON_ATTN。

---

## 5. R9700 的 fused_moe 调优配置（Qwen3.6-35B-A3B）

### 结论（2026-11-03）

- **单卡 int4**（GPTQ-Int4，`E=256,N=512,dtype=int4_w4a16`）：调好并验证完了。但上游 [#64045](https://github.com/vllm-project/vllm/pull/64045)（oleksandrdubyna88，分支 `perf/moe-config-r9700-int4-e256-n512`，10-27）加了同一个文件，选的 tile 基本相同。我们不另开 PR，11-03 在 #64045 下发了[评论](https://github.com/vllm-project/vllm/pull/64045#issuecomment-6595680514)（原文 `comment_pr64045.md`），附上独立验证数据，并建议补一条 `"512"`：M=512、768 时再快 13–14%。
- **TP=4 BF16**（`E=256,N=128`）：调好了。因为一般单卡部署，不提交。
- **还没调**：MTP 草稿层用的 BF16 配置 `E=256,N=512,device_name=AMD_Radeon_R9700.json`。单卡开 MTP 时，两组都对它提示用默认配置。
- 为 int4 准备的 PR（分支 `r9700-int4-moe-config`，工作树 `/target/spec_test/vllm-pr-r9700-int4-moe`，描述 `pr_desc_r9700_int4_moe_config.md`）没有 commit、没有推送，留作参考。

### 为什么要调

- 调之前 R9700 只有两个 FP8 配置（`E=128,N=512` 和 `E=64,N=768`，`block_shape=[128,128]`），没有 E=256 的。
- Qwen3.6-35B-A3B：256 个专家、top-8、`moe_intermediate_size` 512、hidden 2048、40 层。BF16 权重 67 GB，单卡 32 GB 只能用 4-bit。
- ROCm 上 4-bit GPTQ/AWQ 的 MoE 层用不了 Marlin（只支持 CUDA），会回退到 WNA16 的 Triton kernel（`fused_moe_kernel_gptq_awq`）。没有配置文件时，日志提示 `Using default MoE config. Performance might be sub-optimal!`。

### 调优方法

- 脚本 `moe_tune/tune_r9700.py`：在 `benchmark_moe.py` 的函数（`get_configs_compute_bound`、`prune_rocm_search_space`、`benchmark_config`、`save_configs`）外面套一层，每张卡一个进程，不用 Ray。
- 没用 `benchmark_moe.py --tune`：int4 kernel 在 1–2 个 warp、宽 tile 时每个要编译 20–65 s，第一次不剪枝的搜索预计超过 10 小时。剪枝方式：
  - `matrix_instr_nonkdim`、`kpack` 只影响 MFMA（CDNA），固定为 16 和 1；`waves_per_eu` 取 0–2（4 和 2 编译出的二进制相同）。
  - `BLOCK_SIZE_M` 上限是 max(32, 2 × next_pow2(⌈M × top_k / E⌉))。估算每 lane 超过 128 个 VGPR 的组合跳过，这些会 spill，编译也慢。
  - 候选超过 300 个时分两步：先固定 `GROUP_SIZE_M`=1、`waves_per_eu`=0，选出最好的 8 种 tile，再在它们附近搜 `GROUP_SIZE_M` × `waves_per_eu`。
  - 搜索阶段每个候选回放 max(2, min(20, 5120 // M)) 次。最后前 5 名和默认配置交替测 3 轮，每轮 100 次回放，取中位数。
- int4 一共测了 4,575 个候选，4 张卡用时 28 分钟。

### 单卡 int4 的结果（10-26、10-27）

kernel（单次 `fused_experts`，bf16 激活，group size 128，µs）：

| M | 1 | 4 | 16 | 64 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|
| 默认 | 80.0 | 265.8 | 655.1 | 1675.5 | 1974.7 | 2022.4 | 2068.9 | 3134.6 | 5370.6 |
| 调优 | 62.3 | 162.8 | 510.5 | 1101.4 | 1333.6 | 1662.4 | 2065.5 | 2977.0 | 4567.5 |

M=1–512 快 1.18–1.63 倍，M=2048–4096 快 1.05–1.35 倍，M=1024–1536 持平。18 个 M 的输出都和默认配置逐位一致。18 个 M 的完整数据在 `pr_desc_r9700_int4_moe_config.md`。

端到端（单卡，`palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`，不开投机，ShareGPT，输出 tok/s 均值）：

| conc | 默认环境：默认 → 调优 | 每组轮数 | `GPU_MAX_HW_QUEUES=1`：默认 → 调优 |
|---|---|---|---|
| 1 | 53.4 → 53.5（+0.2%） | 6 | 69.1 → 70.9（+2.5%） |
| 4 | 134.2 → 149.8（+11.7%） | 6 | 156.0 → 179.6（+15.1%） |
| 16 | 280.1 → 301.6（+7.6%） | 2 | 285.0 → 326.0（+14.4%） |
| 32 | 395.5 → 443.7（+12.2%） | 2 | 404.7 → 459.1（+13.4%） |

- 默认环境下，同一组各轮在 conc 1–4 最多差 25%，所以这两档各跑了 6 轮。`GPU_MAX_HW_QUEUES=1` 时每组两轮相差不到 1.5%（每组 2 轮）。两种环境是不同日期测的（10-26 和 10-27）。
- 开 MTP（3 个草稿 token，`GPU_MAX_HW_QUEUES=1`）：conc 1 +6.1%，conc 4 +10.3%，conc 16/32 +9–11%。接受长度两组都是 2.84–2.92。
  - conc 16/32 时 MTP 的 KV cache 只有 0.09–0.11 M token，请求会排队。
  - 每张卡第一次启动时编译缓存是冷的，KV cache 少约 1 GiB，所以这两档按 KV 大小配对比较。
- 贪心输出：conc 1 的 12 轮（每组 6 轮）全部一致。

### 和 #64045 的对比（11-03）

同一台机器，三份配置在同一进程里交替测 5 轮，取中位数（µs）：

| M | 默认 | #64045 | 我们的 |
|---|---|---|---|
| 1 | 82.9 | 64.0 | 64.1 |
| 64 | 1672.3 | 1116.3 | 1106.1 |
| 384 | 1932.7 | 1441.4 | 1425.3 |
| 512 | 1987.0 | 1886.5 | 1636.3 |
| 768 | 2042.6 | 2041.6 | 1764.2 |
| 2048 | 3070.9 | 2946.0 | 2916.1 |
| 4096 | 5289.9 | 4576.9 | 4507.0 |

- M ≤ 384 和 M ≥ 1024 时，两份文件相差不到 1.6%。
- #64045 只有 9 个 key，256 和 1024 之间没有。M=512 用的是 256 那条，M=768 用的是 1024 那条，比我们的 `"512"` 条目慢 13–14%。
- vLLM 按文件里的顺序取最近的 key，打平时取先出现的那个。所以 `"512"` 要放在 `"256"` 和 `"1024"` 之间，M=768 才会选到它。
- 他们是在 WSL2 + vLLM 0.29 镜像上用 `benchmark_moe.py --tune` 调的（约 11.5 小时），端到端只测了 conc 1 和 4。

### TP=4 BF16 的结果（10-23）

- kernel（`E=256,N=128`）：M=1–4096 快 1.01–1.16 倍。
- 端到端：4 卡 TP=4，Qwen3.6-35B-A3B BF16，不开投机，默认环境，每组 4 轮，输出 tok/s 均值：

  | conc | 默认 → 调优 |
  |---|---|
  | 1 | 70.4 → 75.4（+7%） |
  | 4 | 207.1 → 202.7（−2%） |
  | 16 | 411.5 → 450.5（+9%） |
  | 64 | 893.5 → 954.6（+7%） |

- 同一组各轮波动很大（比如 conc 4 的默认组在 153–230 之间），conc 1 和 4 的差别都在噪声内。

### 和 zhentaocc PR 的关系

- 这个模型是 GDN 混合模型。[#61434](https://github.com/vllm-project/vllm/pull/61434) 和 [#62381](https://github.com/vllm-project/vllm/pull/62381) 涉及 GDN，和本条不冲突，但会影响端到端结果，测试时要记下用的是哪个 vLLM 版本、是否包含这两个 PR。
- #61433 的 `is_hybrid` 条件会让这类混合模型在 ROCm 上默认用 TRITON_ATTN（R9700 上 Qwen3.5-9B 已实测，吞吐 +3–7%），端到端对比时两边的后端要一致。

---

## 6. ROCm 后端支持 fused multi-step draft decode（评估后不做）

### 背景

多步草稿 decode 时，ROCm 上只有 TRITON_ATTN 支持在 graph 里原地更新草稿元数据（下面叫"融合"）。ROCM_ATTN 和 ROCM_AITER_FA 每步都要在 graph 外重建一遍。设想是让 ROCm 后端也支持融合，所以先测这部分差距。你在 MI355X 上的结论是收益在噪声内（≤ 1.5%）。

### R9700 实测（10-22 到 10-23）

- 模型 Qwen3-8B + Qwen3-8B-eagle3，main `852989f58` 原样代码，K=3 和 K=7，conc 1/4/16。
- 4 组配置：
  - TRITON_ATTN 原生（融合）；
  - TRITON_ATTN 强制改走非融合；
  - ROCM_ATTN 原生（非融合）；
  - ROCM_ATTN 打补丁开启原地更新（融合）。补丁是测试用的 import hook（`fused_draft/patch/sitecustomize.py`），没有改 vLLM 代码；conc 1 打补丁前后接受长度一致（2.377）。
- 正序、反序各跑一轮完整 A/B，另跑一轮 profile。

K=3 吞吐（tok/s，第一轮 / 第二轮）：

| 配置 | conc 1 | conc 4 | conc 16 |
|---|---|---|---|
| TRITON 融合（原生） | 78.5 / 78.5 | 203.5 / 182.4 | 547.1 / 500.8 |
| TRITON 强制非融合 | 69.6 / 78.5 | 180.4 / 203.8 | 492.7 / 542.3 |
| ROCM_ATTN 非融合（原生） | 69.2 / 69.5 | 184.4 / 186.4 | 489.0 / 489.5 |
| ROCM_ATTN + 融合补丁 | 69.6 / 69.6 | 186.6 / 186.3 | 489.5 / 489.6 |

K=7 吞吐（tok/s，第一轮 / 第二轮）：

| 配置 | conc 1 | conc 4 | conc 16 |
|---|---|---|---|
| TRITON 融合（原生） | 56.7 / 52.1 | 182.2 / 187.7 | 410.4 / 402.0 |
| TRITON 强制非融合 | 51.9 / 52.6 | 186.5 / 200.6 | 405.0 / 401.9 |
| ROCM_ATTN 非融合（原生） | 52.1 / 51.8 | 201.5 / 184.0 | 444.6 / 442.0 |
| ROCM_ATTN + 融合补丁 | 52.2 / 52.3 | 184.0 / 180.0 | 403.0 / 403.8 |

### 结论

- 只看 K=3 的第一轮，TRITON 融合比非融合快 11–13%，很像收益；但反序重跑后结果倒了过来。差别来自第 7 条的快/慢状态：每次启动、每档并发会随机落进其中一种并保持住。
- K=3 按状态分开比较后，融合和非融合相差在 0.5% 以内。
- K=7 也没有融合变快的迹象：TRITON 两种方式在三档都没有一致的差别；ROCM_ATTN 在 conc 16 两轮都是非融合（原生）更快，打补丁开启融合后没有变快。
- 和 MI355X 的结论一致，不建议做。

---

## 7. 投机解码的快/慢两种状态和 `GPU_MAX_HW_QUEUES`（根因排查）

### 现象

- R9700 上开投机解码时，同一个服务进程里，每段压测（约 60 s）会随机落在两种状态之一，段内保持不变，段与段之间会切换。以 Qwen3-8B + EAGLE3（K=3）conc 1 为例：
  - 快态：ITL 29.4–29.8 ms，约 78.5 tok/s；
  - 慢态：ITL 32.6–33.3 ms，67–72 tok/s；
  - 两者相差约 12%。
- 两种状态下时钟基本相同，快态功耗更高（278 W 对 246 W），说明不是降频。
- 不开投机时（Qwen3-8B conc 1）没有这个现象，ITL 稳定在 25.1 ms。
- 影响：A/B 只跑一轮时，差别可能完全来自状态，第 6 条的第一轮就是这样。不同卡之间 conc 1 的吞吐也能差 12%。

### 排查

数据在 `fused_draft/regime/`。设置：Qwen3-8B + EAGLE3 K=3，conc 1，草稿用 TRITON_ATTN，每段 16 个请求，段间空闲 10 s。

| 设置 | 结果（每段 ITL 中位数） |
|---|---|
| 默认（r1、r3、r4、r6，共 6 次启动） | 快慢混合；r3 的两张卡全程慢态 |
| `HSA_ENABLE_INTERRUPT=0` | 全程慢态（33.1 ms） |
| 同卡另起进程，每 1 ms 发一个小 kernel（keepalive） | 快慢混合 |
| `--no-async-scheduling` | 快慢混合 |
| `HIP_FORCE_DEV_KERNARG=1`、`ROC_USE_FGS_KERNARG=0`、`DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` | 快慢混合 |
| `ROC_SYSTEM_SCOPE_SIGNAL=0` | 启动失败 |
| `GPU_MAX_HW_QUEUES=1`（2 张卡）、`GPU_MAX_HW_QUEUES=2` | 全程快态（29.5 ms） |

其他场景：

- EAGLE3 conc 16：默认 53.1–53.4 ms（487–499 tok/s），`GPU_MAX_HW_QUEUES=1` 是 47.5 ms（546–552 tok/s），快约 11%。两组不在同一张卡上。
- 不开投机 conc 1：默认和 `GPU_MAX_HW_QUEUES=1` 都是 25.1 ms。
- Qwen3.6 int4 单卡（第 5 条）：
  - `GPU_MAX_HW_QUEUES=1` 时两轮相差不到 1.5%；默认环境下同一组各轮最多差 25%，不开投机也是这样。
  - conc 1 的 69 对 53 tok/s 是跨日期、跨卡的对比（10-26 在 HIP 4/5，10-27 在 HIP 5/6），还没在同一时段对比过。

### 注意

- 10-23 机器死机，正好发生在 4 卡 TP=4 加 `GPU_MAX_HW_QUEUES=1` 的服务启动时。在那之前约 20 分钟里，单卡 hwq1 的多个实例同时跑都正常。因果没有确认，但之后多卡没再这样跑；要跑的话先确认。
- 单卡 `GPU_MAX_HW_QUEUES=1` 一直正常，最多同时跑过 3 个实例。

### 后续

- 在同一时段、同一张卡上对比默认环境和 `GPU_MAX_HW_QUEUES=1`，确认它对 MoE int4 模型的影响（包括不开投机的情况）。
- 确认后给 ROCm 报 issue，附上 `fused_draft/regime/` 的复现脚本。本机 gh 访问不了 ROCm 组织的仓库，要在网页上提。
- 根因还没找到：中断、kernarg、graph packet capture、async scheduling 都排除了，只确认了和 HW queue 数量有关。

---

## 已删除

- **原第 6 条：ROCM_ATTN 支持草稿元数据原地更新**。本来就只省 CPU 时间，预计收益很小；#61433 之后草稿默认用 TRITON_ATTN，这条基本没有价值了。

## 暂缓：需要 MI355X

MI355 侧已经查清的两点，见 [`mi355_spec_optim_progress.md`](https://github.com/WhatGhost/testing_notes/blob/main/SpeculativeDecoding/mi355_spec_optim_progress.md) 第 2 节：DSpark + AITER 的 hang 在 vLLM main 上不复现；ROCM_AITER_FA + DSpark 的接受长度塌陷是 ROCm Triton 3.7.x 绑定的 LLVM 的问题，Triton 3.8 正常。

- **DSpark + ROCM_AITER_FA 的 hang**：根因修复，对应 issue #9 的 MI355X P0。
- **ROCM_AITER_FA 的 non-causal 多 token decode 慢**：在 R9700 上做 kernel 级回放，它用的 AITER Triton 草稿 kernel 比 vLLM 的 Triton 慢 2–3 倍。可以改走更快的 kernel，但端到端要在 MI355X 上验证。
- **让 ROCM_AITER_FA 支持 RDNA4：不建议做。**
  - 它是 CK 和汇编 kernel，vLLM 限定只在 CDNA3 及以上启用，R9700 上直接报 `compute capability not supported`（c6）。
  - 移植要在 AITER/CK 层面做，而 RDNA4 上已经有两个 Triton 后端可用。

## 评估后暂不建议做

- **ROCm 后端支持 fused multi-step draft decode**：R9700 和 MI355X 上都没有收益，见第 6 条。
- **RMSNorm+QKV、residual+RMSNorm、gate+SiLU+mul 融合**（issue #9 Radeon P0）：在 R9700 上，norm 加激活函数在 DSpark conc 1 只占每步 1%（0.45 / 44.8 ms），conc 64 也只占 3.6%（4.0 / 110.6 ms）。第 1 条落地后占比会略有上升，到时再评估。RFC #61761 里有 QK-norm + RoPE + gate 的融合（面向 MI355X），到时先和 zhentaocc 协调。

## 和 zhentaocc 近期 PR 的关系

| PR / RFC | 和本清单的关系 |
|---|---|
| [#61433](https://github.com/vllm-project/vllm/pull/61433)：ROCm 开投机或混合模型时 TRITON_ATTN 优先 | 覆盖原第 3、4 条，改为提供 R9700 数据（第 3 条，已实测）；和第 4 条有联动 |
| [#61761](https://github.com/vllm-project/vllm/issues/61761)：DFlash2 在 MI355X 上的 RFC | GEMM 调优和第 1 条互补；AITER Unified split-KV 可能和第 4 条改同一文件；QK-norm + RoPE + gate 融合之后再协调 |
| [#61760](https://github.com/vllm-project/vllm/pull/61760) | 和第 4 条互补 |
| [#61434](https://github.com/vllm-project/vllm/pull/61434)、[#62381](https://github.com/vllm-project/vllm/pull/62381) | GDN 相关，不冲突；影响第 5 条的端到端测试 |
| [#61429](https://github.com/vllm-project/vllm/pull/61429) | 不冲突。和 [#48913](https://github.com/vllm-project/vllm/pull/48913) 改的不是同一个函数，DSpark 也不调用 `get_top_k_tokens` |
| [#62415](https://github.com/vllm-project/vllm/pull/62415)、[#53214](https://github.com/vllm-project/vllm/pull/53214) | 不冲突 |

## 环境和注意事项

- **GPU 编号**：8 张 R9700，HIP 编号和 rocm-smi 编号不一致，rocm-smi 的 [0]–[7] 依次是 HIP 2、3、1、0、6、7、5、4。
- **绑核**：HIP n 在 NUMA 节点 n^1 上。跑服务时用 `taskset` 绑到该节点的 CPU（`/sys/devices/system/node/node<n^1>/cpulist`），本机没有 numactl。
- **模型**（`/target/models/` 下）：Qwen3-8B、Qwen3-8B-eagle3、dspark_qwen3_8b_block7、Qwen3-0.6B、Qwen3.5-9B、Qwen3.6-35B-A3B（BF16）、Qwen3.6-35B-A3B-GPTQ-Int4。
- **编译缓存**：A/B 每组用独立的 `VLLM_CACHE_ROOT`，因为 torch.compile 缓存不感知 `PYTHONPATH` 切换。缓存是冷的那次启动 KV cache 会少约 1 GiB，开 MTP 时会影响排队。
- **测投机解码性能时**：注意第 7 条的快/慢状态。要么每组多跑几轮，要么单卡设 `GPU_MAX_HW_QUEUES=1`；多卡不要设。
- **gh**：本机 gh 的 classic token 被 ROCm 组织禁止，ROCm 仓库的 issue 和 PR 只能在网页上操作。
- **本地 vLLM 仓库**：
  - 它是 blob-less 的部分克隆，`git grep`、`git cat-file` 碰到本地没有的对象会去 origin 一个个拉，非常慢。
  - fork（`WhatGhost/vllm`）的 main 停在 10-19，往 fork 推新分支前先在网页上同步一下 fork。

## 数据和脚本

都在 R9700 机器的 `/target/spec_test/dspark_aiter_debug/` 下，没有上传到 GitHub（Python 环境和几份 vLLM 代码见最后一项）：

- **端到端结果**（`results/` 下）：
  - `c1_*` 到 `c7_*`：后端组合测试，支撑第 3、4 条。
  - `p0_*` 到 `p4_*`、`h0_*`、`h1_*`：#61433 实测和不开投机的后端对比，支撑第 3 条。
  - `g1_*` 到 `g6_*`：skinny GEMM 原型的端到端测试。
  - `f1_*` 到 `f4_*`、`q06_*`、`s1_*`、`s2_*`：正式实现的端到端 A/B（DSpark、基线、Qwen3-0.6B）和冷启动，支撑第 1 条。
- **端到端脚本**：`run_case.sh`，以及可以用 `MODEL` 换模型的副本 `run_case_model.sh`；`compare_cases.py` 对比两组结果（吞吐、ITL 中位数、接受长度、选中的后端、就绪时间）。
- **GEMM 微基准和 tile 扫描**（`gemm/` 下）：
  - `bench_skinny_gemm.py`、`triton_skinny_proto.py`：原型阶段的微基准和 tile 扫描，结果在 `results/`。
  - `sweep_formal.py`、`analyze_formal_sweep.py`、`search_rules.py`、`compare_bucket128.py`：正式 kernel 的 tile 扫描和规则搜索。第一轮结果在 `results/formal_sweep/`；补测（小 N、40/72/80 行、2 级流水）在 `results/formal_sweep2/`；最终规则的逐点评估在 `results/combined/analysis_final_le128.txt`。
  - `bench_dispatch_final.py`、`summarize_bench.py`：正式实现 vs main 的分派微基准，结果和单测日志在 `results/final/`。
  - `skinny_patch/`：原型阶段端到端测试用的运行时补丁，不是 PR 代码。
  - `prof/`、`analyze_trace.py`：serving profile 和每步耗时拆分。
- **fused multi-step draft decode**（第 6 条，`fused_draft/` 下）：
  - EAGLE3 端到端脚本 `run_case_eagle.sh`；ROCM_ATTN 的融合补丁 `patch/sitecustomize.py`。
  - 结果在 `results/k3_*`、`results/k7_*`，`_r2` 是反序那一轮。
  - profile 用 `run_profile_eagle.sh`，结果在 `prof/`；分析脚本 `analyze_draft.py`、`step_timeline.py`。
- **快/慢状态排查**（第 7 条，`fused_draft/regime/` 下）：
  - 分段压测 `run_segments.sh`（用 `RUN_ENV` 传环境变量）；`run_with_keepalive.sh` 加 `keepalive.py`；`chain_r8.sh`。
  - 结果在 `results/r1_*` 到 `r8_*`；时钟和线程监控脚本是 `monitor_amdsmi.py`、`monitor_threads.py`、`seg_clocks.py`。
- **fused_moe 调优**（第 5 条，`moe_tune/` 下）：
  - 调优脚本 `tune_r9700.py`。int4 结果在 `results/int4_n512/`，PR 格式的文件在其中的 `pr/`；BF16 TP=4 结果在 `results/n128/`；驱动日志是 `results/driver_*.log`。`results/int4_n512_aborted_vgpr192/` 和 `results/w4a16_n512/` 是中途停掉的两次。
  - 和 #64045 的对比：脚本 `compare_cfgs.py`，结果在 `results/cmp_pr64045/`。
  - 单卡 int4 端到端：`run_e2e_w4a16.sh`（环境变量 `HWQ`、`TUNED_DIR`、`VLLM_TREE`、`DETAILED` 等见文件头）和 `chain_e2e_w4a16_*.sh`，结果在 `e2e_w4a16/`。
  - TP=4 BF16 端到端：`run_e2e_moe.sh` 和 `chain_e2e_n128*.sh`，结果在 `e2e/`。
- **文档**：
  - `pr_desc_rdna4_skinny_gemm.md`：#62799 的描述。
  - `w7900_test_guide.md` 和 `w7900_kit/`：W7900 复测。
  - `pr_desc_r9700_int4_moe_config.md`：没用上的 int4 配置 PR 描述。
  - `comment_pr64045.md`：发到 #64045 的评论。
- **Python 环境和 vLLM 代码**：
  - `/target/spec_test/venv-main`：torch 2.13.0+git733fca1、Triton 3.8.0、ROCm 7.2.3。vLLM 以可编辑方式装在 `/target/spec_test/vllm`，这个仓库现在在 `gfx12-skinny-gemm` 分支上，所以不设 `PYTHONPATH` 时跑的是分支代码。
  - `/target/spec_test/vllm-base` 是用 `git archive` 导出的 `852989f58` 原样代码，`/target/spec_test/vllm-pr61433` 是在它上面打了 #61433 的 `rocm.py` 改动。编译产物软链接到 `/target/spec_test/vllm`，端到端时用 `PYTHONPATH` 指过去。
  - `/opt/venv`：最初装的 vLLM 0.32.0 正式包，带 AITER。
  - `/target/spec_test/vllm-pr-r9700-int4-moe`：int4 配置 PR 的工作树，基于 `5d35ce393`，只有一个未提交的配置文件。
