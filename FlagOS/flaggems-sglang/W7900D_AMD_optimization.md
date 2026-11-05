# W7900D 上 FlagGems-sglang 算子的优化空间

依据三类材料：RTX 4090 与 W7900D 两份报告中同样 484 个形状的 kernel 模式延迟；本机 Triton 缓存中 W7900D 测试时实际编译的 1257 个 gfx1100 kernel 版本（含 autotune 的全部候选；R9700 文档中 W7900D 的编译数字是把 TTIR 离线编译得到的 580 个版本，不含三个候选很多的 autotune kernel）；以及在 W7900D 上的两轮原型验证。第一轮基本沿用 R9700 文档的思路，改了 8 个算子，另外测了 5 个带 autotune 的算子实际选中的配置、跳过 AMD 后端 `optimize_epilogue` pass 的写回对照，并找到 R9700 文档没有提到的一个问题：寄存器溢出较多的 kernel 每次启动都要重新分配 scratch。第二轮不拘泥于 R9700 文档的改法，从耗时的构成、网格大小和编译出的指令入手，改写了 `context_attention`、`fused_moe_gemm`、`extend_attention` 和两个 decode 算子的算法或网格，查到了 `chunk_state_varlen` 慢的原因，并对全部 56 个算子扫了一遍 num_warps；又改了 10 个算子，`context_attention` 换成更快的写法。两轮共 18 个算子的改动套到同一个包副本（v2）上，跑了上游的完整测试、官方 benchmark，并对每个 benchmark 形状重复调用 100 次、检查结果逐位一致（见「第二轮：不按 R9700 文档的改法」）。v2 包让 100 个大形状的耗时从 4090 的 2.51 倍降到 1.67 倍（几何平均）。其余「可快多少」仍是按带宽或算力利用率推算的上限。

**落地情况（2026-11-05）**：这 18 个算子已分五个 PR 移植进上游的 rdna3 后端（`_amd/rdna3/ops/`，只在 gfx1100 上生效），都已合入：[#199](https://github.com/flagos-ai/FlagGems-sglang/pull/199)、[#205](https://github.com/flagos-ai/FlagGems-sglang/pull/205)、[#231](https://github.com/flagos-ai/FlagGems-sglang/pull/231)、[#256](https://github.com/flagos-ai/FlagGems-sglang/pull/256)，以及注意力一组（`decode_attention`、`decode_grouped_attention`、`extend_attention`、`log_scaling_tau`）的 [#271](https://github.com/flagos-ai/FlagGems-sglang/pull/271)（2026-11-02 合入）。第三轮查到的正确性问题另提了两个 PR，在审：通用实现的 [#289](https://github.com/flagos-ai/FlagGems-sglang/pull/289) 和 gfx1100 的 [#290](https://github.com/flagos-ai/FlagGems-sglang/pull/290)，见「正确性修复（2026-11-04）」。PR 中的实现与 v2 包不完全相同，各算子的落地版本和实际加速比见「第一批」的第一张表，各 PR 中的新改动和新发现见「第一个 PR（#199）」「MoE PR」「Mamba PR」「LoRA 与访存算子 PR」「注意力 PR」几节。差别最大的是 `context_attention`：合入的版本让 fp32 输入保持 fp32 计算，benchmark（只列 fp32）的大形状快 1.23–2.98 倍，而不是 v2 包把 fp32 输入转成 fp16 时的 4.64–18.53 倍；按合入状态，100 个大形状的耗时约为 4090 的 1.86 倍（只把 `context_attention` 换成 #199 的实测值、其余算子仍按 v2 包估算）。本文第一轮、第二轮和分析各节中的数字仍是原型阶段的，除非另行注明。

实验都在 PCI 地址 0000:e3:00.0 的 W7900D 上进行：脚本按 PCI 地址在 KFD 拓扑中找到这张卡，运行时得出它的 HIP 序号（本次为 7，换机器或重启后可能不同），确认没有其他进程占用它的显存，并用 numactl 绑定到它所在的 NUMA 节点 1（`opt_w7900d/run_proto.sh`），进入 torch 后再按 PCI 地址核对一次（`harness.check_device`）。上游代码 `FlagGems-sglang-b7a575a/` 没有修改：原型改的是模块或包的副本（`opt_w7900d/patched/`），或者用不同的启动参数调用上游 kernel；结束时上游目录 683 个文件的校验和与开始时相同，目录里也没有留下测试或 benchmark 的结果文件。

脚本和数据都在 `opt_w7900d/`：公共部分是 `harness.py`，各实验是 `exp_*.py`（依次运行用 `run_queue.sh`），编译统计是 `kstats.py`，报告数据的分析是 `analyze.py`；第一轮的表格和 v2 包的测试、benchmark、逐位一致性表由 `tables.py` 从 `results/` 生成（`results/table_v2_final.md` 是 v2 包逐形状的完整数据），第二轮各实验的表取自 `results/` 下对应的 jsonl。第一轮的两个包副本相对上游的改动见 `results/patched.diff`；第二轮的原型是 `patched/` 下的 `context_attention_v2.py`、`extend_attention_v2.py`、`decode_splitkv.py`、`fused_moe_gemm_grouped.py`、`per_group_transpose_v2.py`，v2 包是 `patched/FlagGems-sglang-w7900d-v2/`，相对上游的改动见 `results/patched_v2.diff`。W7900D 上该先做哪些，见「W7900D 上值得优化的算子：分批排序」；与 R9700 文档结论不同的地方，见「与 R9700 文档的对照」。

## 粗看：耗时对比与带宽、算力利用率

以 4090 为基准，只看 4090 上不少于 50 µs 的 100 个大形状（44 个算子），固定开销占比小。按峰值带宽之比（864 ÷ 1008 GB/s），访存型算子在 W7900D 上的耗时应是 4090 的 1.17 倍：

| GPU | 按峰值带宽预期的耗时比 | 大形状实测耗时比（几何平均） | 实测达到预期的比例 | 带宽归一化效率 < 0.5 / 0.5–0.8 / ≥ 0.8 的算子数 |
|---|---:|---:|---:|---:|
| W7900D | 1.17× | 2.51× | 46% | 15 / 28 / 1 |
| W7900D，v2 包（18 个算子改后） | 1.17× | 1.67× | 70% | 6 / 32 / 6 |
| R9700（R9700 文档） | 1.57× | 2.34× | 67% | 4 / 26 / 14 |

v2 包一行的大形状中，改过的 17 个算子（`sgemm_lora_b` 没有大形状）用 v2 包官方 benchmark 的值，其余算子仍用报告的值；带宽归一化效率 ≥ 0.8 的 6 个算子中，`context_attention`、`fused_moe_gemm`、`fused_moe_router_tensorcore` 是计算型，只看排序。这一行的 `context_attention` 用的是 v2 包把 fp32 输入转成 fp16 的写法，合入的 #199 没有采用：换成 #199 的 fp32 实测耗时后，`context_attention` 6 个大形状的耗时从 4090 的 0.87 倍变成 5.32 倍，100 个大形状合起来约 1.86 倍（实测达到预期的比例约 63%）；其余算子的 PR 版本大多与 v2 包持平或更快，没有重算。

按峰值带宽、算力折算的利用率（各取最大的形状；简单访存型指 `softcap_out`、`moe_sum_reduce`、`silu_and_mul`、`gelu_and_mul`、`fused_rmsnorm`；峰值取 `W7900D_report.md` 的规格表：864 GB/s、FP32 53 TFLOPS、FP16 / BF16 矩阵 106 TFLOPS、INT8 106 TOPS）。「实测改后」是 v2 包官方 benchmark 的值，没有放进 v2 包的取原型中最快的可用改法：

| 类别 | RTX 4090 | W7900D 现在 | W7900D 实测改后 |
|---|---:|---:|---:|
| 简单访存型算子的峰值带宽利用率 | 87%–91% | 45%–65% | 56%–69% |
| 同样数据量的拷贝（Triton 拷贝 kernel，读写共 256 MiB / 1 GiB） | - | 70% / 84% | - |
| 访存型中利用率最高的形状 | 93%（`softcap_inplace_logits`） | 87%（`softcap_inplace_logits`） | - |
| `context_attention`（fp32 输入） | 14.6 TFLOPS（TF32 峰值的 17.7%） | 2.3 TFLOPS（FP32 峰值的 4.4%） | 43.1 TFLOPS（v2：fp16 WMMA、K/V 先转 fp16、改网格，FP16 矩阵峰值的 41%） |
| `fused_moe_router_tensorcore`（bf16 输入） | 25.0 TFLOPS（TF32 峰值的 30.3%） | 2.3 TFLOPS（FP32 峰值的 4.3%） | 26.2 TFLOPS（v2：bf16 WMMA，BF16 矩阵峰值的 25%） |
| `fused_moe_gemm` | BF16 峰值的 3.0% | BF16 峰值的 3.6% | BF16 峰值的 63%（v2：按专家分组的 GEMM） |
| `w8a8_block_int8_matmul` | INT8 峰值的 42% | INT8 峰值的 47% | INT8 峰值的 64% |
| 延迟 ≤ 1.5 × 计时下限的形状 | 158 / 484（下限 3.07 µs） | 238 / 484（下限 8.80 µs） | - |

粗看的结论：W7900D 的空间比 R9700 大。大形状如果做到与 4090 相同的带宽利用率，平均还能快约 2.2 倍（R9700 约 1.5 倍）。不过这个上限对访存型算子偏乐观：同样数据量的拷贝在 W7900D 上只跑到峰值带宽的 70%（读写共 256 MiB）到 84%（1 GiB），而 4090 上简单访存型算子就有 87%–91%，实测中简单的逐元素算子调参数只快 1.0–1.4 倍。计算型算子空间最大，而且已经实测兑现：`fused_moe_router_tensorcore` 改用 bf16 WMMA 后最大形状 328.4 µs，与 4090（343.2 µs）持平；`context_attention` 第一轮改用 fp16 WMMA 快 2.6–3.6 倍，第二轮再去掉主机同步、改网格顺序、K/V 先整体转成 fp16，大形状快 4.6–18.5 倍，最大形状 50977 µs，比 4090（150785 µs）快 3.0 倍（合入的 #199 对 fp32 输入保持 fp32，这个形状 673.7 ms，比 4090 慢 4.5 倍；fp16、bf16 输入 40.5–40.9 ms）；`fused_moe_gemm` 改成按专家分组的 GEMM，T=4096 从 70562 µs 降到 4107 µs（4090 上同样是逐 token 的 GEMV，55385 µs）；`w8a8_block_int8_matmul` 消除溢出后到 INT8 峰值的 64%；`causal_conv1d_fn` 只改分块参数，最大形状从峰值带宽的 7% 升到 65%。其中 `context_attention` 第二轮的改动和 `fused_moe_gemm` 的改动是算法上的，与硬件无关，4090 上应同样有效（没有测）。另有几个形状异常慢（`context_attention` 只有几十个 token 的小形状约 280 µs，`fused_moe_router_cudacore` 的 [83, 512] × 32 个专家 261.6 µs），其中大部分时间是因为寄存器溢出较多时 ROCm 每次启动都要重新分配 scratch，设一个环境变量就能省掉这部分（见「scratch 门槛」），其余靠改参数消除溢出。小形状约一半在计时下限附近，kernel 本身能省的不多。

## 详细分析：结论

- 落地：第一批 18 个算子已分五个 PR 移植进上游的 rdna3 后端，都已合入（见「第一批」）。PR 阶段在原型之外又有几处改进和发现：`context_attention` 的 fp32 输入保持 fp32，并查到两种 fp32 分块会偶尔算错整块输出（「第一个 PR（#199）」）；WMMA 的累加改成每个 K 块单独求和（「MoE PR 中的两个发现」）；不改编译器、用 `tl.split`/`tl.join` 拿到 16 字节写回（「LoRA 与访存算子 PR」）；split-KV 按能否填满 96 个 CU 选 head 块、上游 `decode_attention` 的 fp32 输入在 gfx1100 上因 LDS 超限直接崩溃（「注意力 PR 中的改动和发现」）。下面各条记录的是原型阶段的结论和数字。
- AMD 后端原来没有任何专门实现，AMD 上的 56 个算子全部走通用实现，参数是按别的芯片选的（见 R9700 文档）。W7900D 上只改启动参数（分块、warp 数、流水级数），`causal_conv1d_fn`、`w8a8_block_int8_matmul`、`fused_moe_router_cudacore`、`silu_and_mul`、`fused_rmsnorm`、`per_group_transpose` 的大形状或个别慢形状就快 1.2–14 倍，其中 `causal_conv1d_fn`、`w8a8_block_int8_matmul`、`silu_and_mul`、`per_group_transpose` 的结果与上游逐位相同。第二轮对全部 56 个算子扫了一遍 num_warps（1–16），又有几个只改 warp 数就明显变快：LoRA 的 `chunked_sgmv_expand`、`gate_up_lora_b`、`sgemm_lora_b` 用 16 个 warp 快 1.59–1.88 倍，`bmm_chunk` 用 1 个 warp 快 1.22–1.24 倍，`log_scaling_tau` 网格大时用 16 个 warp 快 1.18 倍，结果都与上游逐位相同；`chunk_state_varlen` 只改分块（每块 1 个头、K 步长 16、1 级流水）快 1.51–2.42 倍，也逐位相同。
- 第二轮收益最大的是改算法和网格，这几处 R9700 文档没有提到，与硬件无关，4090 上应同样有效（没有测）：
  - `context_attention`：在 fp16 WMMA 的基础上去掉主机同步，让同一 (序列, 头) 的 query 块在网格中相邻、共享缓存中的 K/V，head_dim > 64 时把 K、V 先整体转成 fp16（在 kernel 里转换 fp32 的 K/V，head_dim > 64 的版本每条 lane 溢出 1052–1204 字节，先转之后 100–216 字节）。大形状快 4.64–18.53 倍、小形状 6.08–15.96 倍，比第一轮的 fp16 写法又快 1.8–6.4 倍；最大形状到 FP16 矩阵峰值的 41%，比 4090 快 3.0 倍。合入的 #199 只在 fp16、bf16 输入上这样做（大形状快 8.25–20.43 倍），fp32 输入保持 fp32 计算，只保留去掉同步、不重复读 K/V 等改动，benchmark（只列 fp32）的大形状快 1.23–2.98 倍、小形状 6.60–15.92 倍。
  - `fused_moe_gemm`：上游每个 (token, slot) 对单独做 GEMV、把所属专家的权重重读一遍，三张卡都只到矩阵峰值的 3%–4%。改成先按专家排序、再用 `tl.dot` 的分组 GEMM 后，T ≥ 8 时快 2.00–17.18 倍，T=4096 到 BF16 峰值的 63%，比 4090 快 13 倍。
  - `extend_attention`：上游先用单独的 kernel 把前缀 K/V 拷成连续的缓冲区（只跑到峰值带宽的 4%–21%，占总耗时的 8%–34%），每个 query 头又各自读一遍所属 KV 头的 K/V。在 attention kernel 里经 `kv_indices` 直接读前缀、把共享一个 KV 头的 query 头放进同一个 tile 后，快 1.28–2.20 倍。
  - `decode_attention`、`decode_grouped_attention`：batch 1 时上游只有 8 个程序（W7900D 有 96 个 CU）。按序列切分（split-KV）后 batch 1 快 3.23、5.49 倍，batch 8 快 1.09、1.99 倍；网格中 KV 头必须是最快的一维，否则 batch 512、4096 时反而比上游慢 5.8–6.3 倍。batch ≥ 64 时收益主要来自 warp 数（8 个改成 4 个），`decode_attention` 快 1.08–1.20 倍，`decode_grouped_attention` 0.98–1.11 倍。
- 空间最大的是 fp32 矩阵乘，但 `context_attention` 在 W7900D 上的改法与 R9700 不同。R9700 上用 bf16x3 快 2.23–3.11 倍；W7900D 上 bf16x3 反而比上游慢（大形状 0.44–0.69 倍），原因不是峰值之比：上游在长序列上只用到 FP32 向量峰值的 4%–5%，主要受寄存器溢出限制，bf16x3 的溢出并没有降下来（见「第一轮」的 `context_attention` 一节）。W7900D 上可用的是两处 `tl.dot` 的输入转成 fp16、只做一次 WMMA（fp32 累加）：大形状快 2.55–3.59 倍，容差占比不超过 0.18，尾数精度与 4090 默认用的 TF32 相同（都是 10 位），但指数范围小，绝对值超过 65504 的输入会溢出；转成 bf16 则有 3 / 6 个大形状超出容差。不想降低精度时，只把 tile 从 32×32 改成 32×16，大形状快 1.10–2.21 倍。以上是第一轮的结果，第二轮在 fp16 写法上的改进见上一条。合入时选了不降低精度：fp32 输入仍按 fp32 计算（见上一条末尾）。
- `fused_moe_router_tensorcore` 改用 bf16 WMMA、按 K 分块求和，大形状快 7.76–11.01 倍（v2 包官方 benchmark），[4096, 4096] × 256 个专家从报告中的 3780.5 µs 降到 328.4 µs。W7900D 上 M=4096 的测试用例通过（容差占比 0.88，全部 127 个候选配置都通过；R9700 上这个用例不通过），bf16x3 的候选也全部通过，但只快 3.15–4.29 倍。
- W7900D 特有：每条 lane 溢出超过约 1.5 KB 的 kernel，每次启动多花约 150–200 µs。ROCm 运行时给一次 dispatch 准备的 scratch 超过 `HSA_SCRATCH_SINGLE_LIMIT`（默认 146800640 字节，约 140 MiB）时，改为每次 dispatch 临时分配、用完释放；W7900D 有 96 个 CU、每个 CU 32 个 wave 槽、wave32，门槛折合每条 lane 1493 字节。测试和 benchmark 实际运行的版本中，有 4 个 kernel 超过它（autotune 候选中还有更多，但没有被选中）：`context_attention` head_dim > 64 的版本（2.6 KB）、`fused_moe_router_cudacore` 的 32×512 tile（2056 B）、`chunked_embedding_lora_a`（1764 B）、`w8a8_block_int8_matmul` 中 K=512 的版本（1564 B）。把门槛设成 4 GiB 后这些形状快 1.2–6.0 倍，输出逐位相同，其余形状不变；改参数消除溢出的效果更好（v2 包中 cudacore 14.50 倍、w8a8 12.88 倍）。R9700 只有 64 个 CU，如果每个 CU 也是 32 个槽，门槛折合约 2.2 KB，cudacore 这个版本在 R9700 上溢出 2068 B，不超过，这可能就是 [83, 512] × 32 个专家在 R9700 上 67.88 µs、W7900D 上 261.6 µs 的原因（没有在 R9700 上验证）。
- 访存型算子：同样数据量的拷贝只到峰值带宽的 70%–84%，按 4090 的利用率推算的空间要打折扣。`causal_conv1d_fn` 用 R9700 文档的规则（BLOCK_D=1、BLOCK_T 按最大序列长度）快 4.69–9.19 倍，收益比 R9700（2.24–4.07 倍）更大；`silu_and_mul`、`fused_rmsnorm`、`per_group_transpose` 改参数快 1.21–1.40 倍；`gelu_and_mul`、`softcap_out`、`moe_sum_reduce` 分别搜索了 15、16、45 种配置，大形状最多快 1.03–1.16 倍，已接近同样数据量的拷贝。
- WMMA 写回：R9700 文档指出 W7900D 上 WMMA kernel 的写回被拆成 2 或 4 字节的标量写。实测跳过 `optimize_epilogue`（写回前先经 LDS 转成普通布局）后，大多数 kernel 改成 16 字节向量写；LoRA 的 `chunked_sgmv_expand`、`gate_up_lora_b`、`sgemm_lora_b` 快 1.45–2.04 倍，`chunk_state`、`bmm_chunk` 快 1.03–1.11 倍，`extend_attention` 反而慢 4%–10%，输出都逐位相同。不能全局关掉，要按 kernel 选。第二轮发现这三个 LoRA 算子只把 num_warps 从 4 改成 16 就快 1.59–1.88 倍，与跳过 `optimize_epilogue` 相近，不用改编译器。LoRA PR 中又找到不改编译器也能拿到宽写回的办法（把累加器做一次 `tl.split`/`tl.join` 往返），合入版本大形状快 1.31–2.82 倍，见「LoRA 与访存算子 PR」。
- 主机同步：`context_attention` 的封装每次调用都对 GPU 张量取值。W7900D 上去掉它，最小的形状从 75.69 µs 降到 22.96 µs（v2 包中 12.48 µs）。
- 第一轮没有查明原因的几个算子，第二轮查到了：`chunk_state_varlen`（带宽归一化效率 0.26，R9700 0.64）的两个 dot 都沿 token 方向求和，而 token 在 x 和 B 中都是最慢的一维，两个操作数都要经 LDS 转置，gfx1100 上编译成逐个元素的 2 字节 LDS 读（每条 WMMA 指令对应 8–32 条 `ds_load_u16`）；`chunk_state` 有同样的问题，但改分块没有收益。`extend_attention`（0.43 对 0.87）慢在前缀 gather 和每个 query 头重复读 K/V，decode 两个算子慢在小 batch 的网格太小和大 batch 的 warp 数（见上）。`chain_speculative_sampling`（0.32 对 0.49）没有再查。
- 只看第一次调用的结果不够：第二轮有两个配置（`context_attention` 不预先转换 K/V 的 32×16 tile、`extend_attention` 合并头的 kernel 用 2 个 warp）前几次调用在容差内，之后结果开始变化，最后非法访存；串行执行 kernel、改 scratch 设置都不影响，原因没有查明，v2 包没有用这两个配置。v2 包的 18 个算子都在全部 benchmark 形状上重复调用 100 次，结果逐位一致。
- 空间小的部分：小形状。238 / 484 个形状的延迟不到本机计时下限（8.80 µs）的 1.5 倍，kernel 本身能省的不多，要靠减少 kernel 数和主机同步、在推理框架里用 HIP graph。

## W7900D 上值得优化的算子：分批排序

下面这五批是前两轮的结论，只覆盖最早的 56 个算子。上游后来又新增了 85 个，全部 141 个算子已在 2026-11-03 用另一套口径重新分批，见「第三轮」；那一轮不依赖 4090 的数据，并给每个有空间的算子定了病因，有几个算子的归属因此变了（如 `chunked_sgmv_shrink` 从「只有小形状」变成能快 2.5 倍）。

按大形状（4090 上不少于 50 µs）上能快多少，把有大形状的 44 个算子分成五批，另列只有小形状的 12 个算子。「大形状耗时是 4090 的倍数」是各算子大形状的几何平均，按峰值带宽之比，访存型算子在 W7900D 上应是 4090 的 1.17 倍；「推算可快」对访存型算子按做到与 4090 相同的带宽利用率推算，即这个倍数 ÷ 1.17，也就是带宽归一化效率的倒数；`fused_moe_gemm` 按算力利用率推算。W7900D 上这个推算偏乐观：同样数据量的拷贝只到峰值带宽的 70%–84%；实测调参数的 7 个访存型算子中，只有问题明确的 `causal_conv1d_fn` 接近推算值（推算 8.57 倍，实测 4.69–9.19 倍），其余 6 个只快 1.0–1.4 倍（推算 1.6–3.1 倍），如 `gelu_and_mul` 推算 1.61 倍、实测 1.00–1.03 倍，`silu_and_mul` 推算 2.02 倍、实测 1.21–1.36 倍。没有发现具体问题的算子，调参数的收益大概也在这个范围。同一批内按能快的倍数排序。

「用途」取自上游 `docs/tasks/` 的分类（与 R9700 文档相同），可以按要跑的模型取舍：不用 LoRA 就跳过 LoRA 类，不跑 Mamba 或线性注意力混合模型就跳过 Mamba、FLA 类。

### 第一批：已移植进上游 rdna3 后端（18 个，五个 PR）

这 18 个算子先在包副本（v2 包）上做原型、跑完完整测试，再分五个 PR 移植进上游的 rdna3 后端。PR 中的实现在验证时又改过，所以先列落地情况：加速比取自各 PR 的官方 benchmark（与上游交替各跑 3 次、绑 NUMA、取中位数；bf16，`context_attention` 是 benchmark 唯一列出的 fp32），大形状的范围；fp16、fp32 和逐形状的数字见各 PR 的描述（`opt_w7900d/pr{1,3,4,5,6}/PR_BODY.md`）。

| PR | 状态 | 算子 | PR 版本的加速比 | 与 v2 包的差别 |
|---|---|---|---|---|
| [#199](https://github.com/flagos-ai/FlagGems-sglang/pull/199) | 2026-10-21 合入 | `context_attention` | fp32：大形状 1.23–2.98×，小形状 6.60–15.92×；fp16、bf16 输入：8.25–20.43× | fp32 输入保持 fp32 计算，不再转 fp16，长序列用 32×32 分块（64×32 和 32×16 会偶尔算错整块输出）；fp16、bf16 输入用输入类型的 WMMA、fp32 累加；去掉主机同步、不重复读 K/V 这两项保留 |
|  |  | `w8a8_block_int8_matmul` | 1.18–2.41×，[64, 512] × [1024, 512] 13.36× | 同 v2（BLOCK_M ≤ 32，消除溢出）；bf16、fp32 输出逐位一致，fp16 输出 450 万个元素中有 10 个差一次舍入 |
| [#205](https://github.com/flagos-ai/FlagGems-sglang/pull/205) | 2026-10-22 合入 | `fused_moe_gemm` | T ≥ 8：2.24–16.28×，T=1 1.11× | v2 把各 K 块串在 WMMA 累加器里，误差偏向 0；PR 改成每个 K 块单独求和（tile-sum） |
|  |  | `fused_moe_router_tensorcore` | bs 64–4096：4.71–13.54×（fp32 1.06–2.30×） | 分块求和同 v2；GEMM 的启动参数按形状选，不再 autotune |
|  |  | `fused_moe_router_cudacore` | 大形状 1.06–1.49×（fp16 最多 2.07×）；(83, 512, 32, 3) 带 bias 14.16× | v2 只改了专家数 ≤ 32 的路径，PR 中又改写了专家数 > 32 的 logits kernel（v2 时大形状不变） |
| [#231](https://github.com/flagos-ai/FlagGems-sglang/pull/231) | 2026-10-28 合入 | `causal_conv1d_fn` | 4.58–9.09× | 同 v2，只换启动参数，逐位一致 |
|  |  | `chunk_state_varlen` | 1.53–2.42× | 分块同 v2（K 步 16、1 级流水），只在 dstate ≥ 128 时用，更小的状态保持上游的启动参数；逐位一致 |
|  |  | `bmm_chunk` | 1.22–1.23×（fp32 3.34–3.37×） | v2 一律 1 个 warp；PR 只在更快的形状上减少 warp，fp32 的 K 步改成 16，单个 warp 不再溢出；逐位一致 |
| [#256](https://github.com/flagos-ai/FlagGems-sglang/pull/256) | 2026-11-02 合入 | `chunked_sgmv_expand` | 2.28–2.82×（fp32 5.77–20.59×） | v2 只改 warp 数；PR 用 `tl.split`/`tl.join` 换 16 字节读写、不再先 clone 输出，fp32 改成小 tile 的 FMA；逐位一致 |
|  |  | `gate_up_lora_b` | 1.31–1.76×（fp32 最多 16.85×） | 同上 |
|  |  | `sgemm_lora_b` | 较大的两个形状 1.46×、2.05× | 同上 |
|  |  | `per_group_transpose` | 最大形状 1.29× | 启动参数按形状选；逐位一致 |
|  |  | `silu_and_mul` | 最多 1.40× | 同上 |
|  |  | `fused_rmsnorm` | 最多 1.23× | 同上；一行内求和顺序变了，不逐位一致 |
| [#271](https://github.com/flagos-ai/FlagGems-sglang/pull/271) | 2026-11-02 合入 | `decode_attention` | 1.09–3.44×；fp32 上游崩溃、本次能跑 | split-KV 的配置规则重写（按能否填满 96 个 CU 选 head 块，K/V 分块统一 32 个 token）；fp32 用不分块的 kernel |
|  |  | `decode_grouped_attention` | 1.01–5.43×（fp16 最多 62.74×，fp32 最多 15.00×） | 同上；fp32 也走 split-KV |
|  |  | `extend_attention` | 1.36–2.67×（fp32 1.69–5.79×） | 输出不再先清零；fp32 也走合并头的 kernel（tile 减半、warp 加倍） |
|  |  | `log_scaling_tau` | 最大形状 1.51×（fp16 1.33×） | 改写成一个 kernel，一行切成 512 个元素的段；逐位一致 |

下表是 v2 包的原型结果（落地前），保留作对照。第一轮的 8 项改动（`context_attention` 换成第二轮的写法）加上第二轮的 10 项，都套到同一个包副本（`patched/FlagGems-sglang-w7900d-v2/`）上，用上游的测试文件和 benchmark 跑过，并在全部 benchmark 形状上重复调用 100 次、检查结果逐位一致（见「v2 包的完整正确性测试、逐位一致性和官方 benchmark」）。加速比是官方 benchmark 中相对同一批运行的上游实现的值，与正文中原型实验的数字（如 `causal_conv1d_fn` 4.69–9.19 倍）略有出入，差别在测量波动内；「逐位相同」指与上游的输出逐位相同。

| 算子 | 用途 | 大形状耗时是 4090 的倍数（现在 → v2） | 实测加速比 | 改法和状态 |
|---|---|---:|---:|---|
| `context_attention` | 注意力 | 8.67 → 0.87 | 大形状 4.64–18.53×，其余 6.08–15.96× | 第二轮：fp16 WMMA（fp32 累加）；head_dim > 64 时 K、V 先整体转成 fp16；网格中 query 块最快；按 `max_input_len` 定网格、去掉主机同步；exp2；序列短时 32×32 tile。容差占比 ≤ 0.18，fp16 的范围限制见「需要实测确认的事项」 |
| `fused_moe_gemm` | MoE | 1.57 → 0.40 | T ≥ 8：2.00–17.18×，T=1 不变 | 第二轮：先按专家排序，再用 `tl.dot` 的分组 GEMM；T=4096 到 BF16 峰值的 63%。测试的 3 个用例矩阵太小，仍走原来的 GEMV，分组路径只在原型中按测试的容差核对过 |
| `fused_moe_router_tensorcore` | MoE | 9.97 → 1.06 | 大形状 7.76–11.01×，M=64 4.83× | ROCm 上 bf16 输入不再提升到 fp32，直接用 bf16 WMMA，K 方向按分块求和；8 个用例全部通过（上游 M=4096 的用例不通过） |
| `causal_conv1d_fn` | Mamba | 10.00 → 1.52 | 4.54–9.35× | 与 R9700 文档相同的规则：BLOCK_D=1，BLOCK_T 按最大序列长度；逐位相同 |
| `decode_grouped_attention` | 注意力 | 1.71 → 1.03 | batch 1、8：5.49×、1.99×；batch 64–4096：0.98–1.11×；小形状 1.35–1.63×；head_dim 512：1.70× | 第二轮：split-KV（batch 小时 16 个 query 头一块、按序列切分，batch 大时 64 个头一块、不切分）；head_dim 512 仍走上游 kernel，改用 1 个 warp |
| `decode_attention` | 注意力 | 1.77 → 1.28 | batch 1：3.23×，batch 8–4096：1.08–1.20×；head_dim 512：1.12× | 第二轮：split-KV、4 个 warp；head_dim 512 仍走上游 kernel，改用 2 个 warp；head_dim 80 的小形状慢 2%–4% |
| `extend_attention` | 注意力 | 2.73 → 1.73 | 大形状 1.28–2.20×，其余 1.14–2.05× | 第二轮：在 attention kernel 里经 `kv_indices` 直接读前缀，去掉单独的 gather；共享一个 KV 头的 query 头放进同一个 tile；容差占比与上游相同（0.31） |
| `chunk_state_varlen` | Mamba | 4.54 → 3.01 | 1.51–2.42× | 第二轮：每块 1 个头、BLOCK_K 16、1 级流水；逐位相同 |
| `w8a8_block_int8_matmul` | 量化 | 6.03 → 4.23 | 大形状 1.35–1.50×，[64, 512] × [1024, 512] 12.88× | BLOCK_M 不超过 32、num_stages=2，消除 BLOCK_M=64 版本的溢出；最大形状到 INT8 峰值的 64%；逐位相同 |
| `chunked_sgmv_expand` | LoRA | 2.87 → 1.60 | 1.79–1.88×，小形状 1.16–1.21× | 第二轮：16 个 warp；逐位相同 |
| `gate_up_lora_b` | LoRA | 3.80 → 2.41 | 1.59×（8 段 × 64 个 token 1.10×） | 第二轮：16 个 warp；逐位相同 |
| `sgemm_lora_b` | LoRA | 只有小形状 | 两个较大的形状 1.73×、1.13× | 第二轮：16 个 warp；逐位相同 |
| `per_group_transpose` | 量化 | 3.62 → 2.59 | 1.42×（最大形状） | 数据量不少于 2^24 个元素时用 8 个 warp，benchmark 中只有最大的形状用到；逐位相同；写回仍是 2 字节标量写，向量化写回的原型只再快 4% |
| `bmm_chunk` | Mamba | 2.52 → 2.07 | 1.22–1.24× | 第二轮：1 个 warp；逐位相同 |
| `fused_rmsnorm` | 激活、归一化 | 2.44 → 1.99 | 1.21–1.22× | 按行宽选 num_warps（BLOCK ≥ 4096 用 16 个、≥ 1024 用 4 个，否则 2 个）；归约顺序变了，结果不逐位相同，容差占比与上游相同（0.36） |
| `silu_and_mul` | 激活、归一化 | 2.35 → 1.82 | 1.21–1.39× | 都用二维 kernel（一个程序算一行中的 1024 列）、16 个 warp，去掉按行循环的分支；逐位相同 |
| `log_scaling_tau` | 注意力 | 2.35 → 1.99 | 1.18× | 第二轮：程序数不少于 4096 时用 16 个 warp，benchmark 中只有最大的形状用到；逐位相同 |
| `fused_moe_router_cudacore` | MoE | 1.84 → 1.86（大形状不变） | [83, 512] × 32 个专家 14.50×，[37, 256] × 16 个专家 1.75× | block_h > 128 时用 4 个 warp，消除 32×512 tile 每条 lane 2056 B 的溢出；[83, 512] 从 4090 的 19.2 倍降到 1.31 倍 |

上游的 benchmark 在同一批中重跑了一次，各形状是报告的 0.90–1.08 倍。几处 1 µs 左右的差别在小形状上，单次运行中看起来像回退：`sgemm_lora_b` 一个形状 0.90 倍，`gate_up_lora_b` 一个 0.93 倍，`bmm_chunk` 三个 0.95–0.97 倍。另外各交替跑了 3 次上游和 v2，四次的中位数比值是 0.97–1.03，属于测量波动。四次都慢的只有两处：`decode_attention` head_dim 80 的小形状（慢 2%–4%，约 0.4 µs）和 `decode_grouped_attention` batch 512（慢 2%）；注意力 PR 中改了配置规则后分别是 1.04 倍和 1.01 倍。

### 不改代码：调高 scratch 门槛

把 `HSA_SCRATCH_SINGLE_LIMIT` 设成 4 GiB（4294967296）后，官方 benchmark 中受影响的形状如下（耗时单位为 µs）。输出与默认设置逐位相同，这 4 个算子的其余形状为 0.95–1.09 倍（测量波动）：

| 算子 | 形状 | RTX 4090 | 默认 | 4 GiB | 改代码消除溢出后（v2 包） |
|---|---|---:|---:|---:|---:|
| `context_attention` | 4 头、head_dim 80–128、1–3 条序列（每条 8–30 个 token）的 3 个形状 | 34.8 | 282.1–285.0 | 85.7–91.4（3.12–3.29×） | 17.64–18.28 |
| `context_attention` | 1 × 128，32 × 128 | 50.18 | 432.0 | 240.4（1.80×） | 48.60 |
| `context_attention` | 8 × 128，32 × 128 | 122.9 | 1180 | 999.3（1.18×） | 255.1 |
| `fused_moe_router_cudacore` | [83, 512] × 32 个专家 | 13.66 | 258.5 | 102.0（2.53×） | 17.92 |
| `w8a8_block_int8_matmul` | [64, 512] × [1024, 512] | 6.14 | 187.0 | 31.2（5.99×） | 14.52 |
| `chunked_embedding_lora_a` | 4 段 × 2048 个 token、2 个 LoRA、rank 64、词表 128256 | 34.78 | 223.6 | 68.4（3.27×） | - |

- 改代码消除溢出的效果都比调门槛好（`context_attention` 的 v2 写法同时去掉了主机同步、改了网格，不只是消除溢出）；调门槛适合还没改代码的算子（`chunked_embedding_lora_a` 唯一的版本溢出 1764 字节），也可以作为兜底。
- 代价是 scratch 一直挂在队列上：按每条 lane 2648 字节算约 260 MB 显存。可以在启动推理服务前设置这个环境变量；ROCm 文档说明 7.0 起也可以用 HIP 的 `hipExtLimitScratchCurrent` 在程序里修改，`HSA_NO_SCRATCH_RECLAIM=1` 也能让 scratch 常驻，这两种都没有测。

### 第二批：写回经 LDS 转换布局（1 个）

第一轮把 LoRA 的三个算子和 `bmm_chunk` 也放在这一批。第二轮发现它们只改 num_warps 就能拿到相近的收益，已移到第一批；LoRA PR 中又找到不改编译器的宽写回办法（`tl.split`/`tl.join` 往返，见「LoRA 与访存算子 PR」），所以这一批原来写的「要改编译器」不再是唯一的路。下表是原型阶段的对照：

| 算子 | 跳过 `optimize_epilogue` | 只改 num_warps（v2 包） | 写回指令（正常 → 跳过） |
|---|---:|---:|---|
| `chunked_sgmv_expand` | 1.89×（大形状），2.04×（8 段 × 64 个 token、输出 2 × 4096 列） | 1.79×、1.88×（16 个 warp） | 2 B × 192 → 不变 |
| `gate_up_lora_b` | 1.72×（大形状），1.32×（8 段 × 64 个 token） | 1.59×、1.10×（16 个 warp） | 2 B × 384 → 16 B × 48 |
| `sgemm_lora_b`（只有小形状） | 两个较大的形状 1.45–1.73× | 1.13–1.73×（16 个 warp） | 2 B × 448 → 16 B × 48 |
| `bmm_chunk` | 1.08–1.10× | 1.22–1.24×（1 个 warp） | 2 B × 56 → 8 B × 5 |

两种改法能否叠加没有测。剩下的是 `chunk_state`：

| 算子 | 用途 | 大形状耗时是 4090 的倍数 | 推算可快 | 跳过 `optimize_epilogue` 的实测加速比 | 写回指令（正常 → 跳过） |
|---|---|---:|---:|---:|---|
| `chunk_state` | Mamba | 1.71 | 1.47× | 1.03–1.11× | 4 B × 160 → 16 B × 37、4 B × 2 |

`chunk_state` 和 `chunk_state_varlen` 一样，两个 dot 操作数都要经 LDS 转置（每条 WMMA 指令对应 12–32 条 2 字节 LDS 读），但第二轮试的 13 种分块、warp 数和流水级数的组合在两个大形状上都没有明显比上游快（最好的快 1.3%，在测量波动内），只改 num_warps 时上游的 4 个 warp 就是最快的，见「`chunk_state_varlen`、`chunk_state`：两个操作数都经 LDS 转置」。Triton 的 AMD 后端在所有架构上都运行这个 pass，没有开关；`extend_attention` 跳过后慢 4%–10%，所以要加一个按 kernel 选择的选项（收益大的都是 bf16 输出、写回指令多的 kernel），或者用 Gluon 显式转换布局。详见「WMMA 写回：跳过 `optimize_epilogue`」。现在也可以在单个 kernel 里用 `tl.split`/`tl.join` 往返累加器达到同样效果（LoRA PR 的做法），`chunk_state` 上没有试；它写回能拿到的只有 1.03–1.11 倍，主要差距在 LDS 转置。

### 第三批：要改写 kernel，推算可快 1.7 倍以上（12 个）

第一轮放在这一批的 `fused_moe_gemm`、`chunk_state_varlen`、`extend_attention`、`log_scaling_tau` 已在第二轮改好，移到了第一批。「只改 num_warps」一栏取自第二轮对全部算子的 warp 数扫描（见「num_warps 扫描」）：每个形状先筛选，快 3% 以上的再复测；表中是各大形状复测后的加速比（筛选时快不到 3% 的记为 1.00），都不到 1.05 倍的记为没有收益。这些没有放进 v2 包，收益小或者不逐位相同，可以和改写 kernel 一起做。

| 算子 | 用途 | 大形状耗时是 4090 的倍数 | 推算可快 | 只改 num_warps | W7900D 上已知的问题和改法 |
|---|---|---:|---:|---:|---|
| `chain_speculative_sampling` | 采样、语法约束 | 3.62 | 3.10× | 1.00–1.17×（全部 kernel 8 个 warp） | 一次调用要执行多个 kernel（缓存中 11 种），含按顺序认领、发布的步骤；合并 kernel。第二轮没有再查 |
| `merge_state` | 注意力 | 2.58 | 2.21× | 1.06× |  |
| `fla_layernorm_gated` | 线性注意力（FLA） | 2.30 | 1.97× | 1.13×（不逐位相同） |  |
| `mrope_fused` | RoPE | 2.29 | 1.96× | 1.12× |  |
| `ernie45_rope_fused` | RoPE | 2.27 | 1.94× | 1.11× | 有 2 字节标量读 |
| `silu_and_mul_masked` | MoE | 2.19 | 1.88× | 带 autotune，没有扫 | autotune 选中的版本不溢出；12 / 171 个行 kernel 候选溢出 2.6–10.4 KB（都超过 scratch 门槛），没有被选中 |
| `per_token_quant_int8` | 量化 | 2.16 | 1.85× | 1.07–1.08× |  |
| `qkv_lora_b` | LoRA | 2.15 | 1.84× | 没有收益 | 封装里有主机同步；2 字节标量读写，经 LDS 写回只快 1.05–1.07 倍；另外三个 LoRA 算子用 16 个 warp 的规则对它无效 |
| `state_passing` | Mamba | 2.12 | 1.82× | 带 autotune，没有扫 | autotune 选中的版本不溢出 |
| `mamba_layernorm_gated` | Mamba | 2.08 | 1.78× | 没有收益 | 有 2 字节标量读写 |
| `per_token_group_quant_int8` | 量化 | 2.07 | 1.78× | 1.07–1.10× |  |
| `embedding_lora_a` | LoRA | 2.03 | 1.74× | 没有收益 | 按行 gather，有 2 字节标量读 |

### 第四批：推算可快 1.2–1.7 倍（10 个）

第一轮放在这一批的 `decode_attention`、`decode_grouped_attention` 已在第二轮改好（split-KV），移到了第一批。这一批大多没有发现溢出、标量访存这类具体问题，差距应来自分块大小、num_warps 等参数，需要在卡上搜索；按上文访存型算子的实测，调参数大概只能快 1.0–1.4 倍，第二轮的 warp 数扫描也只在 4 个算子上快 5% 以上（「只改 num_warps」一栏的含义同第三批）。

| 算子 | 用途 | 大形状耗时是 4090 的倍数 | 推算可快 | 只改 num_warps | W7900D 上已知的问题和改法 |
|---|---|---:|---:|---:|---|
| `chunk_local_cumsum_vector` | 线性注意力（FLA） | 1.98 | 1.69× | 1.17×（1 个 warp） |  |
| `fused_norm_rope_stacked` | 投机解码 | 1.93 | 1.65× | 没有收益 |  |
| `fused_dual_residual_rmsnorm` | 激活、归一化 | 1.92 | 1.65× | 没有收益 |  |
| `moe_fused_mul_sum` | MoE | 1.88 | 1.61× | 1.00–1.06×（1 个 warp） |  |
| `rotary_embedding` | 扩散模型 | 1.83 | 1.57× | 1.04–1.09×（16 个 warp） |  |
| `apply_token_bitmask` | 采样、语法约束 | 1.82 | 1.56× | 1.00–1.08×（1 个 warp） |  |
| `chunk_scaled_dot_kkt` | 线性注意力（FLA） | 1.74 | 1.49× | 没有收益 | storeopt kernel 写回是 4 字节标量写，跳过 `optimize_epilogue` 后变成向量写，但耗时不变 |
| `selective_state_update` | Mamba | 1.65 | 1.41× | 没有收益 | 两个小调优键选中了溢出 1164 B 的版本（低于门槛），与不溢出的候选一样快 |
| `hc_head` | 激活、归一化 | 1.60 | 1.37× | 没有收益 |  |
| `draft_topk1` | 投机解码 | 1.60 | 1.37× | 没有收益（1.03–1.04×） |  |

### 第五批：已接近上限，或实测没有收益（4 个）

| 算子 | 用途 | 大形状耗时是 4090 的倍数 | 推算可快 | W7900D 上已知的问题和改法 |
|---|---|---:|---:|---|
| `softcap_out` | 激活、归一化 | 1.85 | 1.59×（实测 ≤ 1.16×） | 16 种分块中最快的只在 [512, 128256] 上快 1.16 倍（60% → 69%），其余两个大形状 1.03–1.06 倍；第二轮只改 warp 数（16 个），这两个形状快 1.10–1.12 倍，[512, 128256] 没有收益 |
| `moe_sum_reduce` | MoE | 1.94 | 1.67×（实测 ≤ 1.09×） | 45 种分块，大形状 1.03–1.09 倍；最大形状 65% → 67% |
| `gelu_and_mul` | 激活、归一化 | 1.87 | 1.61×（实测 ≤ 1.03×） | 15 种分块，大形状 1.00–1.03 倍；最大形状 61%，与 R9700 的结论相同 |
| `softcap_inplace_logits` | 激活、归一化 | 1.34 | 1.15× | 最大形状已达到峰值带宽的 87% |

### 只有小形状的算子（11 个）

激活、归一化：`l2norm`；线性注意力（FLA）：`chunk_local_cumsum_scalar`、`fused_gdn_gating`；LoRA：`chunked_embedding_lora_a`、`chunked_sgmv_shrink`、`sgemm_lora_a`；Mamba：`causal_conv1d_update`、`chunk_cumsum`；MoE：`moe_fused_gate`、`sigmoid_gate_topk_renorm`；RoPE：`interleaved_rope`。它们在 4090 上都不到 50 µs，W7900D 上的耗时是 4090 的 2.01–3.02 倍（各算子全部形状的几何平均），除 `chunked_embedding_lora_a` 外不超过两者计时下限之比（2.87 倍），以 kernel 启动和计时的固定开销为主。`chunked_embedding_lora_a`（3.02 倍）唯一的版本溢出 1764 字节、超过 scratch 门槛，最大的形状调高门槛后快 3.27 倍。只改 num_warps，`causal_conv1d_update` 用 8 个 warp 快 1.09–1.14 倍，其余不超过 1.12 倍、大多在测量波动内。同样只有小形状的 `sgemm_lora_b`（2.67 倍）已放进 v2 包（16 个 warp，两个较大的形状快 1.13–1.73 倍）。其余要靠合并 kernel、去掉主机同步、在 SGLang 里用 HIP graph（见「小形状：以固定开销为主」）。

### 另外

- 海光后端有 32 个同样基于 HIP 的专门实现，可以作为参考，但它们是按 wave64 的 DCU 调的，RDNA 是 wave32，参数需要重新测。
- 第一批里有的改法与 R9700 不同，合入时要按 gfx 版本区分：`context_attention` 在 R9700 上用 bf16x3，在 W7900D 上用 fp16 单次或改分块；router 的分块求和、`causal_conv1d_fn` 的规则两张卡相同；`w8a8_block_int8_matmul` 的 BLOCK_M ≤ 32 和 cudacore 的 4 个 warp 是针对 gfx1100 溢出的，R9700 上要复测。落地时的做法：#199 把 `ARCH_MAP` 改成按精确的 GPU 目标匹配，`_amd` 只把 gfx1100 映射到 rdna3 后端，其他 AMD 卡（gfx1101、gfx1151、gfx1201 等）仍走通用实现，所以这些 gfx1100 的规则不会用到 R9700 上；`context_attention` 合入的是 fp32 输入保持 fp32、fp16 和 bf16 输入用 WMMA 的写法。
- 第二轮改算法和网格的几处（`context_attention` 的网格顺序和 K/V 先转换、`fused_moe_gemm` 的分组 GEMM、`extend_attention` 去掉 gather 和合并头、decode 的 split-KV）与硬件无关，R9700 和 4090 上应同样有效，但没有测；fp16 的写法和各算子的 num_warps 规则（LoRA 16 个、`bmm_chunk` 1 个、`chunk_state_varlen` 的分块等）是按 gfx1100 测出来的，换卡要复测。

## 第三轮：141 个算子在同一口径下重新分批（2026-11-03）

上面的分批只覆盖最早的 56 个算子，用的是「W7900D 耗时是 4090 的倍数」这个相对指标。上游后来又新增了 85 个算子，而且那个指标依赖另一张卡，换算成「能快多少」要靠假设。这一轮把全部 141 个算子（已移植 18 + 旧剩余 38 + 新增 85）放在同一把尺子下重测，并为每个有空间的算子找出具体病因。脚本和数据在 `opt_w7900d/survey/`，`results/batches.md` 是 140 行的完整排名表。

### 口径

主指标是**耗时 ÷ 同字节数拷贝的耗时**，两者都用 `do_bench` 刷缓存后取中位数，所以它直接回答「离这张卡在这个尺寸上真能跑到的搬运速度还有多远」，不需要另一张卡做参照，也不用再为 70%–84% 的拷贝上限打折扣。只统计 ≥ 30 µs 的形状，更小的被约 9 µs 的启动开销主导。**这个比值只对访存型算子成立**：注意力、GEMM、MoE router 这 12 个算子的成本是算力而不是搬运，表里单列为「算力型」。

另外三个诊断指标，每个都能直接指向改法：

- **主机侧占比**：连续调用 20 次的墙上时间，减去 profiler 报的 GPU 时间。封装里对 GPU 张量取值会让 GPU 空转，Python 封装太重也会。注意不能用 `do_bench` 的数字来减：它每次迭代都刷缓存，而 profiler 里的循环是背靠背跑的、数据还在缓存里，两者相减会把「缓存命中」算成「主机开销」（`ltx2_split_rotary` 刷缓存 399 µs、背靠背 82 µs，差了 4.9 倍）。
- **编译产物**：把 Triton 缓存里 1796 个 kernel 实例的汇编全部解析，取寄存器数、溢出字节、LDS 和访存指令的宽度分布。不用上卡。访存宽度是其中最有用的一项：同样的数据量，用 16 字节指令搬运只需要 2 字节指令的八分之一条数。
- **真实字节**：原来按整张量计数，而且预分配的输出缓冲既算成读、又算成写，所以 KV cache 这类算子会算出 2263% 的带宽。改成按调用前后实际改变的元素数统计写入，读取保留整张量作为上限。做这件事时要注意改写检测必须放在预热之前，否则缓冲区里已经是写过的值，会检测不到任何变化。

rocprof 这条路走不通：这台机器上 GL2C（L2）计数器全部返回 0，`FETCH_SIZE`、`WRITE_SIZE` 也是 0，只有 SQ 系列可用，所以拿不到 HBM 流量。而且一次采集两个计数器会报「超出硬件采集能力」。

### 正确性和可用性问题

| 算子 | 现象 | 根因 | 处理（2026-11-04） |
|---|---|---|---|
| `residual_gate_add` | 单测 12 个用例失败 11 个 | 编译器把 `r + (u * g)` 映射成 `v_dot2_bf16_bf16`。这条指令内部按 fp32 算、只在最后舍入一次，跳过了参考实现要求的中间舍入（参考是 `(r.float() + (u.float() * g.float()).to(dtype).float()).to(dtype)`）。复测：四分之一的 bf16 输出差 1 个 ulp，平时在容差内，bf16 下出现大幅抵消、整个结果就等于那 1 个 ulp（0.015625，绝对容差 0.015）时才失败。全仓只有这个算子和 `rmsnorm_scale` 编译出了这条指令；后者后来由 batch9 补了参考实现，单测在 W7900D 上全部通过 | gfx1100 版本改用 fp32 计算、显式舍入乘积，与参考逐位一致：[#290](https://github.com/flagos-ai/FlagGems-sglang/pull/290) |
| `chain_speculative_sampling` | 单测 9 个用例失败 2 个（B=512 和 B=4096、V=32000） | HIP 上 fp32 输入走的是 AMD 专属分支 `new__roc_plain`（`_uses_roc` 恒为真），NVIDIA 走 `base_plain`。接受链逐位一致（`accept_index`、`accept_token_num` 完全相同），只有最终采样的 token 差 ±1 个位置，1 / 2048 和 3 / 16384 行，都落在判定边界的 0–1 个 float32 步之内。复测：这条分支的行和与参考逐位一致，但它复刻的是 wave64 的扫描顺序，CDF 每一行都和 gfx1100 上的 `torch.cumsum` 不同；而参考用的 `torch.cumsum` 在 ROCm 上本身不确定（rocPRIM 的 decoupled look-back，同一行跑 3000 次，V=32000 有 3 种结果、V=151936 有 2487 种），任何 kernel 都无法保证与它逐位一致 | 未修，要先决定单测怎么比较 |
| `rmsnorm_scale_shift` | hidden 不是 2 的幂时编译失败 | `_rmsnorm_scale_shift_kernel` 把 hidden 直接当 `tl.arange` 的长度，3072、5120、1536 这些常见维度全不能用。测的 6 个 3072 形状全部报 `arange's range must be a power of 2` | 通用实现的块大小改为下一个 2 的幂、平方和的最后一块加掩码：[#289](https://github.com/flagos-ai/FlagGems-sglang/pull/289) |
| `qk_head_layernorm` | q、k 头数不同时非法访存 | `n_rows = q.numel() // dh`，然后用网格的第二维选 q 还是 k，按 q 的行数去处理 k。GQA 形状（32 个 q 头、8 个 kv 头）越界；越界落在已分配的显存里时不报错，单测发现不了 | q、k 各用自己的行数：[#289](https://github.com/flagos-ai/FlagGems-sglang/pull/289) |
| `silu_and_mul_masked` | 单测跑不完 | autotune 在这张卡上超过 30 分钟，1800 秒超时被杀（复测：(T, H) = (256, 4096) 一个 key 就要 1835 秒）。这个算子也是全仓溢出最多的（5952 B/lane，11 个候选版本溢出）。复测时又查到通用实现的两个问题：按列 kernel 的 `FUSED` 配置只在 half = 2048 时正确，其他宽度被 autotune 选中就会算错（最大误差约 20）；half 不是 2 的幂时按行 kernel 编译失败 | gfx1100 版本去掉 autotune：[#290](https://github.com/flagos-ai/FlagGems-sglang/pull/290)；通用实现的两个问题：[#289](https://github.com/flagos-ai/FlagGems-sglang/pull/289) |
| `ep_scatter` | 一个形状输出带 NaN | 填充行没有初始化，看代码像是有意为之 | 未处理 |
| 10 个 benchmark | 调不通自己的算子 | 7 个生成的参数个数不对，3 个多传了算子不接受的关键字参数（`block_size`、`page_size`、`scaling`）。详见新增算子文档 | 上游 [#272](https://github.com/flagos-ai/FlagGems-sglang/pull/272) 已修 |

后两类里 `rmsnorm_scale_shift`、`qk_head_layernorm` 和 10 个坏 benchmark 是通用实现和测试本身的问题、与硬件无关，适合给上游提；`residual_gate_add` 和 `chain_speculative_sampling` 是 AMD 路径独有的。

### 新的分批

140 个有数据的算子分成：超过 2 倍的 6 个、1.5–2 倍的 3 个、1.15–1.5 倍的 15 个、已接近拷贝速度（≤ 1.15）的 74 个、只有小形状的 30 个、算力型的 12 个。只有 `silu_and_mul_masked` 没有测到（autotune 超时）。

**第一批 · 先修正确性**：上表的 4 个算子加 3 项可用性问题。2026-11-04 起，除 `chain_speculative_sampling` 和 `ep_scatter` 外都已处理，见上表最后一列和「正确性修复（2026-11-04）」。

**第二批 · 只改封装，不碰 kernel**：这几个算子的 kernel 本身没问题，时间花在主机侧。

| 算子 | 连续调用 | 其中 GPU | 主机侧 | 病因 |
|---|---:|---:|---:|---|
| `create_chunked_prefix_cache_kv_indices` | 28.3 µs | 5.3 µs | 81% | Python 循环里逐个 `int(chunk_cu_seq_lens[i])`，每轮一次同步 |
| `chunked_embedding_lora_a` | 230.8 µs | 56.2 µs | 76% | 另外唯一的版本溢出 1764 B、超过 scratch 门槛 |
| `get_target_cache_loc` | 70.4 µs | 19.7 µs | 72% | `.tolist()` + `.numpy()` + `int(arr.sum())` |
| `qkv_lora_b` | 151.3 µs | 98.0 µs | 35% | `int(seg_lens.max())`；一次 5 个 kernel；88% 的访存不超过 4 字节，带宽只有 32% |

全仓用 AST 扫了一遍，真正会阻塞主机的只有这 4 处加 `context_attention`（rdna3 版本已去掉）和 `chunked_sgmv_expand`——后者虽然已经移植进 rdna3，但 rdna3 版本第 335 行仍保留着 `int((seg_indptr[1:] - seg_indptr[:-1]).max().item())`。

**第三批 · 病因明确、收益大**：

| 算子 | 耗时÷拷贝 | 病因与已验证的收益 |
|---|---:|---|
| `tiny_n_gemm` | 3.64 | n=1536 时 tile 数为 96，`tiles < _num_sms()` 判成假，于是走单遍 dot kernel，而那个 kernel 溢出 2156 B、超过 scratch 门槛。`_num_sms()` 取的是 torch 的 `multi_processor_count`，gfx1100 上是 WGP 数 48，不是 96 个 CU，所以实际比较的是 `96 < 48`，tile 数是门槛的两倍，不是刚好卡在边界上。**强制走 split-K 实测 4.74–5.09 倍**（490 → 96–105 µs，split_k=28），n=512 的形状也快 1.10–1.12 倍。拆开看：把 `HSA_SCRATCH_SINGLE_LIMIT` 调到 4 GiB 只能到 341 µs（1.44 倍），剩下的 3.4 倍是单遍 kernel 本身不合适。只把判断改成 `<=` 不会生效（实测仍走单遍 kernel），判断条件和 split_k 都要按实测重定：现有公式进入分支后按 48 算 split_k 是 2、按 96 算是 3，都只快 2.3–3.0 倍；复核时 split_k=10 最快，n=1536 快 4.99–5.19 倍、n=512 快 1.19–1.29 倍，见「`tiny_n_gemm` 复核」 |
| `top_p_renorm` | 3.15 | 网格是 `(batch,)`，一行一个程序，batch=1 时 96 个 CU 只有 1 个在干活（V=131072 要 118 µs，同字节数的拷贝只要 11 µs）；而且按浮点位模式二分，每轮重读整行。解码期采样的 batch 通常就是 1–64，正好是最差的区间 |
| `chunked_sgmv_shrink` | 2.50 | 带宽 29%，主机侧还占 42%。上一轮按 4090 的口径把它归在「只有小形状」里，换口径才看出来 |
| `recompute_w_u` | 2.00 | `_packed_dot` 有 64 条 2 字节访存对 12 条 16 字节，带宽 26% |

`chain_speculative_sampling`（4.34）和 `get_target_cache_loc`、`qkv_lora_b`、`chunked_embedding_lora_a` 虽然比值也在这一档或更高，但前者的主要工作是修正确性加合并 18 个 kernel，后三个归在第二批。

**第四批 · 1.15–1.5 倍（15 个）**：`add3`、`fused_softmax_pool`、`embedding_lora_a`、`fused_qk_gemma_rmsnorm_rope_gate`、`ltx2_split_rotary`、`per_group_transpose`、`dsv3_fused_a_gemm`、`residual_gate_add`、`state_passing`、`hc_head`、`moe_sum_reduce`、`dcp_lse_combine`、`moe_fused_mul_sum`、`apply_token_bitmask`、`sgemm_lora_b`。大多没有发现溢出或窄访存这类具体问题，差距应来自分块和 warp 数，按前两轮的实测单项大概 1.0–1.4 倍。

**不建议投入**：74 个已接近拷贝速度的，30 个只有小形状的。后者要靠合并 kernel、去掉主机同步和 HIP graph，不是单个 kernel 的事。

### 溢出的全貌

1796 个编译实例里只有 12 个算子溢出，按每条 lane 的字节数排：`silu_and_mul_masked` 5952、`context_attention` 2792、`group_norm_silu` 2256、`tiny_n_gemm` 2156、`chunked_embedding_lora_a` 1764、`selective_state_update` 1164、`post_reorder_deepgemm` 592、`sgemm_lora_a` 300、`vocab_parallel_embedding` 和 `extend_attention` 各 132、`dsv3_fused_a_gemm` 68、`state_passing` 28。门槛折合每条 lane 1493 字节，超过的有前 5 个；其中 `silu_and_mul_masked`、`group_norm_silu`、`post_reorder_deepgemm` 带 autotune，溢出的是候选而不一定是选中的版本，`context_attention` 已在 #199 中解决，所以真正受影响的是 `tiny_n_gemm` 和 `chunked_embedding_lora_a`，都已列入上面的批次。

### 已移植的 18 个复查

访存型的都已落在拷贝速度附近：`log_scaling_tau` 0.88、`causal_conv1d_fn` 0.91、`fused_rmsnorm` 0.94、`gate_up_lora_b` 1.11、`chunked_sgmv_expand` 1.13、`sgemm_lora_b` 1.16、`per_group_transpose` 1.31。算力型的另算：`context_attention` 最大形状（64 条序列 × 2048 个 token × 32 头 × 128）688 ms，按因果掩码折半约 64 TFLOPS，是 bf16 WMMA 峰值的一半左右，对注意力来说正常。剩下 `per_group_transpose` 1.31 和 `sgemm_lora_b` 1.16 还有一点空间，已列进第四批。

### `tiny_n_gemm` 复核：`_num_sms()` 是 WGP 数（2026-11-03）

第三批对 `tiny_n_gemm` 的病因判断成立，但最初对判断条件的解释（tile 数「正好等于 96 个 CU」）和改法（「把判断改成 `<=` 即可」）不对，第三批表中已改正。

复核在另一台 W7900D 上进行，环境如下：

- 硬件和代码：PCI 设备号 `1002:744b`，用的是 0000:03:00.0 这张卡；软件与本文其余实验相同（ROCm 7.2.3、PyTorch 2.10.0、Triton 3.6.0）；上游为 fed2d06。
- 计时：与官方 benchmark 相同（`do_bench`，warmup 1000、rep 100，取中位数）；bf16，k=7168，m 取 2、4、8、16。
- 绑核：容器里没有 numactl，只用 taskset 把 CPU 绑在 GPU 所在的 NUMA 节点 0，内存没有绑。
- 卡的占用：测量期间每秒按 KFD 的 `vram_<gpu_id>` 检查一次，卡上没有其他进程。第一次运行时没有做这项监控，其中一行整体偏高 100–200 µs，该次运行没有采用。
- 数据：下文是另外两次运行的范围，其中一次把各写法倒序测。

结果：

- `_num_sms()` 取的是 torch 的 `multi_processor_count`。这个值在 gfx1100 上是 WGP 数 48，KFD 拓扑中则是 96 个 CU（`simd_count` 192 ÷ `simd_per_cu` 2）。
  - m ≤ 16、n=1536 时 tile 数为 96，实际比较的是 `96 < 48`，tile 数是门槛的两倍。
  - n=512 时 tile 数 32 小于 48，走 split-K，按现有公式 split_k = 4。
- 只把判断改成 `<=` 不会生效：在内存里替换源码、拦截 kernel 启动后，n=1536 的四个形状仍然走单遍 kernel。原先「按现有公式 tiles=96 只会得到 3」是按 96 算的；按 48 算，进入分支后 split_k 是 2。
- 两个 kernel 都用满 256 个 VGPR。单遍 kernel 每条 lane 溢出 2156 B，与上文一致；split-K kernel 也溢出 468 B，但低于 1493 B 的门槛。
- scratch 门槛调到 4 GiB 后，n=1536 从 485.0–486.5 µs 降到 337.1–342.0 µs（1.43 倍），与上文的 341 µs 一致。n=512 本来就走 split-K，所以不受影响（0.98 倍，在测量波动内）。
- 强制走 split-K 时，耗时随 split_k 的变化并不单调，必须逐个实测。下表是各 split_k 的耗时（µs，取四个 m、两次运行的范围），K 步数指每个程序要走几个 256 元素的 K 步：

| split_k | 每个程序的 K 步数 | n=512 | n=1536 |
|---|---:|---:|---:|
| 上游（n=512 为 split_k=4，n=1536 为单遍） | - | 55.0–60.3 | 485.0–486.5 |
| 2 | 14 | 70.7–92.8 | 164.3–208.5 |
| 3 | 10 | 66.8–74.5 | 165.2–179.3 |
| 4 | 7 | 54.6–60.0 | 119.9–134.1 |
| 5 | 6 | 70.7–75.9 | 155.1–171.0 |
| 6 | 5 | 61.6–68.0 | 122.8–129.6 |
| 7 | 4 | 99.4–104.2 | 163.6–178.3 |
| 10 | 3 | 45.2–47.1 | 93.4–97.3 |
| 14 | 2 | 46.9–49.5 | 94.4–101.2 |
| 28 | 1 | 49.1–52.1 | 95.3–108.4 |

- 与上游比较的加速比：
  - split_k=10 最快：n=1536 快 4.99–5.19 倍，n=512 快 1.19–1.29 倍。
  - 原先用的 28：n=1536 快 4.49–5.09 倍（m=16 最低），n=512 快 1.10–1.16 倍，与原先的结果一致。
  - 现有公式会给出的 2 和 3：n=1536 上只快 2.33–2.96 倍。
- 所有写法都在单测容差内（容差占比不超过 0.51）。每个形状连续调用 20 次，结果逐位一致。
- 改法：在 rdna3 版本里重写走 split-K 的条件，可以按 CU 数比较（2 × `multi_processor_count`），也可以直接按形状判断；split_k 按实测选，k=7168 时取 10。NVIDIA 上 `multi_processor_count` 就是 SM 数，通用实现的门槛本身没错，所以不宜为了 gfx1100 去改通用实现。

### 正确性修复（2026-11-04）

第三轮的正确性问题分两个 PR 提给上游，都在审：通用实现的 [#289](https://github.com/flagos-ai/FlagGems-sglang/pull/289)（3 个算子，新增 9 个单测用例）和 gfx1100 的 [#290](https://github.com/flagos-ai/FlagGems-sglang/pull/290)（2 个 rdna3 版本）。`chain_speculative_sampling` 和 `ep_scatter` 没有处理，原因见上表。都在 W7900D 上验证：

- #289：
  - `rmsnorm_scale_shift`：块大小取下一个 2 的幂，平方和最后一个不满 128 的块加掩码。2 的幂宽度编译出同一个 kernel，与 master 逐位一致；其他宽度在容差内但不逐位一致（kernel 复刻的 FlashInfer 归约顺序只对 2048、4096 有定义），新用例的容差占比不超过 0.51。
  - `qk_head_layernorm`：q、k 各用自己的行数，超出的程序直接返回，head_dim 不一致时报 `ValueError`；头数相同时与 master 逐位一致。
  - `silu_and_mul_masked`：剪掉 BLOCK ≠ half 的 `FUSED` 配置；half 不是 2 的幂时改走按列 kernel。
  - 9 个新用例在 master 上失败 6 个：`rmsnorm_scale_shift` 的 4 个宽度和 half=768 编译失败，K 比 Q 宽时留下 NaN；K 比 Q 窄的两个用例和 half=1024 的用例可能碰巧通过。benchmark 形状编译出的 kernel 与 master 相同，0.97–1.02×。
- #290：
  - `residual_gate_add`：fp32 计算、显式舍入乘积，bf16、fp16、fp32 都与参考逐位一致，汇编里不再有 dot2 指令。启动参数与通用实现相同，0.97–1.02×，最大形状 730 → 740 µs。
  - `silu_and_mul_masked`：去掉 autotune，每个形状只编译一个 kernel，单测从空缓存 4 秒跑完。每个程序处理一行、最多 512 列（每 lane 2 个元素），列是网格最快的一维，超出 `masked_m` 的行直接退出，列带掩码、任意宽度可用。启动规则在 22 个形状上扫描定下：行放在网格第 0 维时大形状慢 2–4 倍；512 列在每个形状上都在最快配置的 1.08 倍以内；256 列在 `masked_m` 稀疏的输入上慢 24–32%，每 lane 2 个元素时在 half=768 上慢一倍（原因没有查）。有效行与通用实现的按列 kernel 逐位一致；官方 benchmark 相对 master（autotune 之后）0.98–1.07×。
- 所有改动都在 NaN 投毒的输出缓冲上连续调用 100 次，结果逐位相同。官方 benchmark 与 master 交替各跑 3 次、取中位数；这台机器的容器里没有 numactl，只用 taskset 绑了 CPU。

## 怎么判断

- 带宽归一化效率 = (4090 耗时 ÷ W7900D 耗时) ÷ (W7900D 峰值带宽 ÷ 4090 峰值带宽)，只看 4090 上不少于 50 µs 的形状，固定开销占比小。等于 1 表示 W7900D 的带宽利用率与 4090 相同。对计算型算子它只能用来排序。数据与 R9700 文档 W7900D 一列相同（`results/analysis.json` 的 `check_vs_r9700_doc` 逐个核对过）。
- 绝对带宽利用率按理论最小访存量计算，由 `analyze.py` 重新计算，算法与 R9700 文档相同（只有 `decode_attention` 多算了 q 和输出，见「访存型算子：带宽利用率」）。为了知道 W7900D 上能跑到多少，另外测了同样数据量的拷贝（`exp_mem.py` 的 copy 一项）。
- 编译统计取自本机测试时的 Triton 缓存（`/target/fgs-test/triton_cache`，1257 个版本，`results/kstats.json`）：寄存器（VGPR，每个 wave 最多 256 个）、溢出（scratch，每条 lane 的字节数）、占用率（每个 SIMD 同时驻留的 wave 数，最多 16）、WMMA 和 FMA 指令数、全局访存指令的宽度。带 autotune 的 kernel 列出全部候选版本，实际选中的版本另行记录（见「autotune 的选择」）；其余 kernel 是固定配置，缓存里的版本就是测试实际运行的版本。
- scratch 门槛：溢出字节数 × 32 条 lane × 每个 CU 的 wave 槽数（KFD 拓扑的 `max_slots_scratch_cu`，W7900D 为 32）× 96 个 CU，与 `HSA_SCRATCH_SINGLE_LIMIT` 比较。
- 原型的计时与报告相同（`triton.testing.do_bench`，warmup 1000、rep 100，取中位数）；配置多的实验先用 warmup 100 筛选，再把最快的几个按 1000 复测。同一进程中未改动的上游版本，大形状的耗时大多与报告相差 5% 以内，例外是 `silu_and_mul` 有一次两个宽形状慢 28%–36%（之后的官方 benchmark 中与报告一致），以及 `extend_attention` 的 [256, 256, 32, 8, 128] 两次都比报告慢 19%（4793–4795 对 4023 µs）；小形状在计时下限附近，相差可达 10%–15%。加速比都相对同一进程中的上游计算。官方 benchmark 重跑中，上游在第一轮的 8 个算子上是报告的 0.90–1.10 倍（各算子几何平均 0.98–1.01），在 v2 包的 18 个算子上是 0.90–1.08 倍。
- 小形状的测量波动：同一配置在不同进程中可差 ±7%。v2 包官方 benchmark 中看起来变慢的小形状，另外交替跑了 3 次上游和 v2（`results/fullbench_rep_up{1,2,3}/`、`fullbench_rep_v2_{1,2,3}/`），按四次的结果判断。
- 逐位一致性（`exp_determinism.py`）：每个 benchmark 形状用同一组输入连续调用 100 次，每次的输出与第一次逐字节比较。`bmm_chunk` 的 causal 版本只计算 i ≤ j 的块，输出用 `torch.empty` 分配，i > j 的位置是未初始化的内容，测试比较前也把两边这些位置清零（`tests/test_bmm_chunk.py`）；这里同样只比较有定义的部分。
- 正确性按测试的容差比较：`context_attention` 的 atol、rtol 均为 1e-2；`fused_moe_router_tensorcore` 要求专家 id 完全一致，权重的 atol、rtol 均为 1e-4；其余按各自测试文件。「容差占比」是 max(|实际 − 参考| ÷ (atol + rtol × |参考|))，不超过 1 为通过。只改启动参数的，另外比较输出是否与上游逐位相同。

## 第一轮：沿用 R9700 文档的思路

这一节是第一轮的原型，基本按 R9700 文档列出的问题和改法在 W7900D 上逐个验证；第二轮见下一节。每个实验单独一个进程，经 `run_proto.sh` 按 PCI 地址选卡、确认空闲并绑核；依次运行各实验的是 `run_queue.sh`，日志在 `results/<实验>.log`。`context_attention`、`fused_moe_router_tensorcore` 改的是模块副本（`patched/context_attention_proto.py`、`patched/router_tensorcore_proto.py`），其余算子用不同的启动参数调用上游 kernel；scratch 门槛和写回两项实验不改代码，只改环境变量或编译器 pass（`exp_ops.py`）。形状为 benchmark comprehensive 级别的全部形状（`fused_moe_router_tensorcore` 不含 M ≤ 8 的 3 个，它们走另一个 FMA kernel，几种改法都不涉及）。

| 算子 | 改法 | 大形状加速比 | 其余形状 | 精度 | 结论 |
|---|---|---:|---:|---|---|
| `context_attention` | 两处 `tl.dot` 的输入转成 fp16（fp32 累加），head_dim > 64 时 64×32 tile | 2.56–3.48× | 1.18–3.59× | 容差占比 ≤ 0.18 | 可用，完整测试通过 |
|  | head_dim > 64 时 tile 改成 32×16，仍按 fp32 计算 | 1.08–2.28× | 1.18–3.75× | 与上游相同（≤ 0.0002） | 可用，完整测试通过 |
|  | 两处 `tl.dot` 用 `bf16x3` | 0.44–0.69× | 0.85–0.92× | 容差占比 ≤ 0.003 | 比上游慢 |
|  | 两处 `tl.dot` 的输入转成 bf16 | 1.52–2.34× | 1.15–3.07× | 3 / 6 个大形状超出容差 | 不可用 |
| `fused_moe_router_tensorcore` | 直接用 bf16 WMMA，按 K 分块求和 | 7.81–10.81× | 1.35–4.82× | M=4096 时容差占比 0.88，127 个候选全部通过 | 可用，完整测试通过 |
|  | `tl.dot` 用 `bf16x3` | 3.15–4.29× | 1.08–2.07× | 与分块求和相同，119 个候选全部通过 | 可用，但慢一倍多 |
|  | 直接用 bf16 WMMA，一条链求和 | 7.48–10.48× | 1.34–5.16× | M ≥ 64 时超出容差（1.43–2.22） | 不可用 |
| `causal_conv1d_fn` | BLOCK_D=1，BLOCK_T 按最大序列长度 | 4.68–9.16× | 1.02–1.03× | 与上游逐位相同 | 可用 |
| `w8a8_block_int8_matmul` | BLOCK_M ≤ 32、num_stages=2 | 1.34–1.50× | 0.99–12.39× | 与上游逐位相同 | 可用 |
| `fused_moe_router_cudacore` | block_h > 128 时 4 个 warp | 不涉及 | 1.44×、14.16× | 专家 id 一致，容差占比 ≤ 0.04 | 可用 |
| `silu_and_mul` | 二维 kernel，BLOCK 1024、16 个 warp | 1.21–1.36× | 0.97–1.43× | 与上游逐位相同 | 可用 |
| `fused_rmsnorm` | 按行宽选 num_warps | 1.21× | 0.96–1.15× | 容差占比 ≤ 0.36（与上游相同） | 可用 |
| `per_group_transpose` | 数据量大时 8 个 warp | 1.40× | 不变 | 与上游逐位相同 | 可用，还有空间 |
| `gelu_and_mul`、`softcap_out`、`moe_sum_reduce` | 15、16、45 种分块，各形状取最快的 | 1.00–1.14× | 0.95–1.14× | 与上游逐位相同 | 收益小 |
| 不改代码 | `HSA_SCRATCH_SINGLE_LIMIT` 设成 4 GiB | 受影响的 2 个 1.18–1.80× | 受影响的 6 个 2.53–5.99× | 与默认设置逐位相同 | 可用 |
| 不改代码 | 跳过 `optimize_epilogue` | 0.90–1.89× | 0.88–2.04× | 与正常编译逐位相同 | 按 kernel 选 |

加速比是相对同一进程中上游的值；`silu_and_mul`、`fused_rmsnorm`、`per_group_transpose` 三行和 scratch 门槛一行是官方 benchmark 的值（参数搜索中每个形状只复测了最快的几个变体），调高 scratch 门槛时没有受影响的形状不变。这 8 项改动的官方 benchmark 结果见本节最后。

### `context_attention`

耗时单位为 µs，括号内为相对上游（本次）的加速比，fp16 一列还有容差占比；上游是 fp32 输入按 ieee 计算（FMA）。bf16x3 一列来自另一次运行，加速比相对那次运行中的上游（与本列的上游相差 0.94–1.01 倍）。

| 形状（条数 × token，头数 × head_dim） | RTX 4090 | 上游（报告） | 上游（本次） | bf16x3 | fp16 单次，64×32 | fp32 FMA，32×16 | 去掉主机同步 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 × 128，32 × 128 | 50.18 | 429.2 | 433.4 | 593.9（0.68×） | 132.0（3.28×，0.10） | 189.9（2.28×） | 338.2（1.28×） |
| 1 × 2048，32 × 128 | 730.2 | 14150 | 13718 | 31274（0.44×） | 3946（3.48×，0.09） | 12112（1.13×） | 13596（1.01×） |
| 8 × 128，32 × 128 | 122.9 | 1191 | 1188 | 1723（0.69×） | 463.1（2.56×，0.14） | 780.8（1.52×） | 1114（1.07×） |
| 8 × 2048，32 × 128 | 18613 | 118591 | 119139 | 235412（0.49×） | 35008（3.40×，0.12） | 99811（1.19×） | 118946（1.00×） |
| 64 × 128，32 × 128 | 1068 | 7090 | 7009 | 9737（0.68×） | 2579（2.72×，0.14） | 4878（1.44×） | 6967（1.01×） |
| 64 × 2048，32 × 128 | 150785 | 942395 | 938144 | 2016484（0.47×） | 327715（2.86×，0.18） | 865253（1.08×） | 936082（1.00×） |
| 2 × 8，4 × 128 | 34.85 | 282.1 | 280.2 | 329.8（0.85×） | 78.07（3.59×，0.07） | 74.81（3.75×） | 187.4（1.50×） |
| 3 × 30，4 × 96 | 34.82 | 284.7 | 287.0 | 326.8（0.89×） | 93.50（3.07×，0.12） | 83.85（3.42×） | 189.7（1.51×） |
| 1 × 20，4 × 80 | 34.82 | 280.3 | 281.8 | 321.4（0.88×） | 80.01（3.52×，0.08） | 89.05（3.16×） | 186.0（1.51×） |
| 1 × 9，4 × 13 | 34.78 | 76.13 | 75.69 | 81.83（0.92×） | 64.02（1.18×，0.05） | 64.14（1.18×） | 22.96（3.30×） |

- fp16 单次：两处 `tl.dot` 的输入转成 fp16、fp32 累加，head_dim > 64 时 tile 改成 64×32（保持上游 32×32 时大形状 1.77–2.95 倍，64×64 为 1.17–2.69 倍）。大形状快 2.56–3.48 倍，容差占比最多 0.18。最大形状从 942395 µs 降到 327715 µs（2.3 → 6.7 TFLOPS），仍是 4090 的 2.2 倍。fp16 与 4090 默认的 TF32 一样是 10 位尾数，但只有 5 位指数，绝对值超过 65504 的输入会溢出；测试和 benchmark 的输入是标准正态分布，没有覆盖这种情况。
- 其他组合：两处都转成 bf16 大形状快 1.52–2.34 倍，但 3 / 6 个大形状超出容差（1.14–1.60），不能用；qk 用 fp16、pv 用 bf16 容差占比 0.99，太接近上限；qk 仍用 ieee、只把 pv 改成 bf16 或 fp16 反而慢（0.81–1.04 倍）。
- `bf16x3` 比上游慢：大形状 0.44–0.69 倍，小形状 0.85–0.92 倍；换成 64×32、64×64 的 tile 或 8 个 warp，大形状也都不超过 0.68 倍；`bf16x6` 大形状只有 0.22–0.48 倍。慢的原因不是 BF16 矩阵峰值只有 FP32 向量峰值的 2 倍（106 对 53 TFLOPS，R9700 约 4 倍）：上游在长序列上只跑到 2.3–2.5 TFLOPS，是 FP32 向量峰值的 4%–5%，瓶颈在寄存器溢出（每条 lane 2648 字节）；bf16x3 仍溢出 2312 字节，每个块还要多做两次 WMMA 和高低位拆分，具体慢在哪一步没有拆开测。
- 只改 FMA 的分块（仍按 fp32 计算，误差与上游相同）：32×16 的 tile 大形状快 1.08–2.28 倍，4 头的小形状快 3.16–3.75 倍；16×32 的 tile 或 8 个 warp 在小形状上与它相近、大形状上略差（0.88–2.29 倍），64×16、16×64、2 个 warp 更慢。小形状的大部分收益来自溢出降到 scratch 门槛以下：上游 32×32 每条 lane 溢出 2628–2648 字节，32×16 为 932–948 字节。
- 溢出量与小形状耗时的关系很清楚：每条 lane 溢出不超过约 950 字节的版本，4 头、head_dim 80–128 的小形状为 74–97 µs；不少于 1888 字节的版本为 283–515 µs（如 2 个 warp 的 4076–4124 字节、bf16x3 64×64 的 7772–7872 字节）。
- 去掉封装里的主机同步（`int(b_seq_len.max().item())`，改为信任已有的 `max_input_len` 参数）：最小的形状 75.69 → 22.96 µs；4 头、head_dim 80–128 的小形状约 283 → 188 µs；大形状中 1 × 128 快 1.28 倍，其余 1.00–1.07 倍。第一轮的两个包副本都保留了这个同步，第二轮的写法去掉了。
- 上游 kernel 对任何输入都先把 q、k、v 转成 fp32，bf16 输入同样走 FMA；bf16 输入直接用 bf16 WMMA 时乘积是精确的，但 benchmark 没有 bf16 形状，没有测。

### `fused_moe_router_tensorcore`

耗时单位为 µs，括号内为相对上游的加速比和权重的容差占比。四种算法在同一个进程中测量（`exp_router.py`），改法与 R9700 文档相同：上游先把 bf16 输入转成 fp32、按 ieee 计算；bf16x3 把一次 `tl.dot` 拆成三次 WMMA；「一条链」不提升到 fp32、直接 `acc += tl.dot(a, b)`；「分块求和」改成 `acc = tl.fma(tl.dot(a, b), ones, acc)`，每个 K 分块单独从 0 求和，再加到 fp32 累加器上。

| 形状 | RTX 4090 | PyTorch 参考 | 上游（报告） | 上游（本次） | bf16x3 | bf16 WMMA，一条链 | bf16 WMMA，分块求和 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [64, 4096] × 256 个专家 | 26.62 | 218.3 | 208.5 | 207.4（0.20） | 100.0（2.07×，0.26） | 40.16（5.16×，1.54，超出） | 43.00（4.82×，0.29） |
| [512, 4096] × 256 个专家 | 59.39 | 624.9 | 536.4 | 535.0（0.85） | 169.6（3.15×，0.67） | 71.50（7.48×，1.43，超出） | 68.48（7.81×，0.77） |
| [4096, 4096] × 256 个专家 | 343.2 | 2871 | 3780 | 3665（1.04，超出） | 854.4（4.29×，0.88） | 349.5（10.48×，2.22，超出） | 339.1（10.81×，0.88） |
| [37, 256] × 16 个专家，softcap 30 | 9.22 | 79.98 | 24.56 | 24.36（0.01） | 22.60（1.08×，0.01） | 18.24（1.34×，0.01） | 18.04（1.35×，0.01） |
| [83, 512] × 32 个专家，带 bias | 11.26 | 81.12 | 32.52 | 32.12（0.03） | 25.80（1.24×，0.04） | 19.16（1.68×，0.04） | 19.36（1.66×，0.03） |

M=4096 时各算法在全部候选配置上的范围。logits 误差是相对同样输入的 fp64 结果的最大绝对误差；两列容差占比都用测试的容差，参考值分别是测试用的 fp32 结果和 fp64 结果：

|  | 求和方式 | logits 最大误差 | 容差占比：对 fp32 参考 | 容差占比：对 fp64 | 通过测试的候选 | 不能运行的候选 |
|---|---|---:|---:|---:|---:|---:|
| 上游（fp32，ieee） | 一条链 | 1.3e-4–6.1e-4 | 0.75–1.04 | 0.19–0.93 | 8 / 126 | 12 |
| bf16x3 | 按 K 分块 | 5.9e-5–1.1e-4 | 0.88–0.96 | 0.06–0.17 | 119 / 119 | 19 |
| bf16 WMMA，一条链 | 一条链 | 4.7e-4–3.5e-3 | 0.93–2.22 | 0.43–2.05 | 4 / 127 | 11 |
| bf16 WMMA，分块求和 | 按 K 分块 | 5.9e-5–1.1e-4 | 0.88–0.96 | 0.06–0.17 | 127 / 127 | 11 |

- 所有形状、所有候选配置上，四种算法选出的专家 id 都与参考一致，差别只在权重。
- 精度由求和方式决定，与 R9700 文档的结论相同：分块求和与 bf16x3 的误差范围完全相同，一条链的 bf16 WMMA 在 M ≥ 64 的三个形状上都超出容差。
- W7900D 上 M=4096 的 fp32 参考（PyTorch 的 fp32 GEMM）相对 fp64 用掉 93% 的容差（R9700 上 97%），这可能是分块求和、bf16x3 在这里全部通过，而 R9700 上有一部分候选不通过的原因。上游（FMA，一条链）对 fp32 参考为 0.75–1.04，126 个候选只有 8 个通过，相对 fp64 全部通过。余量仍然很小（最高 0.96），测试的参考值改用 fp64 更稳妥。
- 速度：分块求和与一条链一样快，大形状比 bf16x3 快 2.5 倍；最大形状 339.1 µs（第一轮官方 benchmark 中 321.5 µs，v2 包中 328.4 µs），4090 343.2 µs，PyTorch 参考 2871 µs。M > 128 时分块求和选中的都不拆分 K。不能运行的候选需要的 LDS 超过 64 KiB。

### scratch 门槛：`HSA_SCRATCH_SINGLE_LIMIT`

- 现象：上游 `context_attention` 4 头、head_dim 80–128 的三个小形状（1–3 条序列、共 16–90 个 token）都要约 283 µs，head_dim 13 的形状只要 76 µs；`fused_moe_router_cudacore` 的 [83, 512] × 32 个专家要 261.6 µs，比 [512, 4096] × 256 个专家（164.0 µs）还慢；`w8a8_block_int8_matmul` 的 [64, 512] × [1024, 512] 要 187.6 µs（4090 6.14 µs）。原型中把溢出降到约 950 字节以下就恢复正常。
- 原因：ROCm 运行时为一次 dispatch 准备 scratch 时，如果所需大小不超过 `HSA_SCRATCH_SINGLE_LIMIT`（默认 146800640 字节），分配一次后一直挂在队列上；超过时改为这次 dispatch 临时分配、用完释放（[ROCr 环境变量文档](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/environment_variables.html)）。所需大小按整张卡能同时驻留的 wave 数算：每条 lane 的溢出字节 × 32 条 lane × 每个 CU 32 个 wave 槽 × 96 个 CU，门槛折合每条 lane 1493 字节。支持异步回收 scratch 的 GPU 改用 3 GB 的 `HSA_SCRATCH_SINGLE_LIMIT_ASYNC`，W7900D 上实测仍是 140 MiB 的门槛：溢出 1368 B 的版本（`fused_moe_router_cudacore` block_h 256、1 个 warp）和 1164 B 的版本不受影响，1564 B 的版本受影响，与 1493 B 的门槛一致。
- 验证：上游代码不变，默认设置和门槛设成 4 GiB 各跑一遍 9 个算子的全部 benchmark 形状（`exp_ops.py`，`results/ops_scratch_default.jsonl`、`ops_scratch_4g.jsonl`）。72 个形状中变快的 8 个正好是溢出超过 1493 字节的版本所在的形状（见上文「不改代码」的表），其余 64 个形状 0.90–1.13 倍（测量波动）；对照组（`extend_attention` 溢出 340 B、`chunk_state_varlen` 376 B、`selective_state_update` 1164 B、`silu_and_mul_masked`）都不变。输出与默认设置逐位相同，只有 router M=4096 的一次 autotune 选了不同的配置。第一轮的官方 benchmark 用同样的设置又跑了 4 个算子（`results/fullbench_upstream_4g/`），结果相同。

### `causal_conv1d_fn`

耗时单位为 µs，括号内为相对上游的加速比；规则与 R9700 文档相同：BLOCK_D=1；BLOCK_T 取最大序列长度向上取到 2 的幂，限制在 128–2048；num_warps = BLOCK_T ÷ 256，限制在 4–8。

| 形状 | RTX 4090 | 上游（报告） | 上游（本次） | 按规则 | 规则选的配置 | 峰值带宽占比（上游 → 按规则） |
|---|---:|---:|---:|---:|---|---:|
| 8 条 × 2048 token，dim 4096，宽度 4 | 325.6 | 4380 | 4379 | 478.3（9.16×） | BLOCK_D 1、BLOCK_T 2048、8 warps | 7% → 65% |
| 32 条 × 512 token，dim 2048，宽度 4 | 166.9 | 1240 | 1242 | 265.5（4.68×） | BLOCK_D 1、BLOCK_T 512、4 warps | 13% → 59% |
| 1 条 × 7 token，dim 16，宽度 4 | 4.10 | 10.12 | 10.16 | 9.84（1.03×） | BLOCK_D 1、BLOCK_T 128、4 warps | - |
| 4 条 × 1 token，dim 8，宽度 3 | 4.10 | 9.92 | 10.12 | 9.88（1.02×） | BLOCK_D 1、BLOCK_T 128、4 warps | - |

- 全部 54 种配置（含上游）的结果都与上游逐位相同；规则选的配置在两个大形状上正好是最快的。
- 耗时主要随 BLOCK_T 下降：最大形状上 BLOCK_D=1、4 个 warp 时，BLOCK_T 从 128 到 4096 依次为 3642.9 → 1876.1 → 950.6 → 541.2 → 560.0 → 800.6 µs，BLOCK_T 2048 配 8 个 warp 为 478.3 µs。BLOCK_D 取 2、4 时最快分别为 559.6、613.2 µs。
- 改后 W7900D 比 R9700 略快（最大形状 478.3 对 501.6 µs），但按峰值带宽之比（864 对 640 GB/s）应快 1.35 倍：R9700 达到峰值带宽的 84%，W7900D 只有 65%。上游时 W7900D 却慢一倍多（4380 对 2042.6 µs），小 tile 的上游实现在 W7900D 上更吃亏。上游版本的读写是 2 字节标量访存（如 32 条 2 字节读、4 条 2 字节写），改后仍是，收益来自分块。

### `w8a8_block_int8_matmul`

耗时单位为 µs。上游的分块：BLOCK_M 取不超过 M 的最大 2 的幂（限制在 16–64），BLOCK_N 64，4 个 warp，3 级流水。配置名为 BLOCK_M_BLOCK_N_warp 数_流水级数；每个形状搜索了 BLOCK_M 16–128、BLOCK_N 64 或 128、4 或 8 个 warp、2 或 3 级流水共 32 种。

| 形状 [M, K] × [N, K] | RTX 4090 | 上游（报告） | 上游（本次，配置，每条 lane 溢出） | 按规则（BLOCK_M ≤ 32、2 级流水） | 最快的配置 | INT8 峰值占比（按规则） |
|---|---:|---:|---:|---:|---:|---:|
| [1, 4096] × [1024, 4096] | 16.67 | 52.40 | 54.00（16_64_w4_s3，0 B） | 54.18（1.00×） | 16_64_w4_s3：53.88 | - |
| [1, 4096] × [4096, 4096] | 30.72 | 114.8 | 106.2（16_64_w4_s3，0 B） | 101.4（1.05×） | 同规则 | - |
| [1, 4096] × [7168, 4096] | 48.13 | 207.5 | 210.5（16_64_w4_s3，0 B） | 157.9（1.33×） | 同规则 | - |
| [8, 4096] × [1024, 4096] | 16.48 | 54.64 | 59.48（16_64_w4_s3，0 B） | 55.04（1.08×） | 同规则 | - |
| [8, 4096] × [4096, 4096] | 29.70 | 115.7 | 106.6（16_64_w4_s3，0 B） | 102.6（1.04×） | 16_128_w8_s2：101.9 | - |
| [8, 4096] × [7168, 4096] | 49.15 | 201.5 | 206.7（16_64_w4_s3，0 B） | 179.2（1.15×） | 16_128_w8_s2：148.0 | - |
| [64, 4096] × [1024, 4096] | 20.48 | 104.1 | 112.9（64_64_w4_s3，480 B） | 60.46（1.87×） | 同规则 | - |
| [64, 4096] × [4096, 4096] | 33.79 | 195.7 | 198.1（64_64_w4_s3，480 B） | 124.3（1.59×） | 同规则 | - |
| [64, 4096] × [7168, 4096] | 51.20 | 305.7 | 304.8（64_64_w4_s3，480 B） | 207.8（1.47×） | 64_64_w8_s3：201.6 | 17% |
| [512, 4096] × [1024, 4096] | 32.77 | 176.5 | 167.2（64_64_w4_s3，480 B） | 112.9（1.48×） | 同规则 | - |
| [512, 4096] × [4096, 4096] | 72.70 | 470.8 | 461.4（64_64_w4_s3，480 B） | 319.4（1.44×） | 64_64_w8_s2：312.4 | 51% |
| [512, 4096] × [7168, 4096] | 123.9 | 739.5 | 734.2（64_64_w4_s3，480 B） | 503.5（1.46×） | 同规则 | 56% |
| [4096, 4096] × [1024, 4096] | 134.1 | 853.8 | 842.2（64_64_w4_s3，480 B） | 561.0（1.50×） | 同规则 | 58% |
| [4096, 4096] × [4096, 4096] | 483.3 | 2833 | 2838（64_64_w4_s3，480 B） | 2060（1.38×） | 同规则 | 63% |
| [4096, 4096] × [7168, 4096] | 867.8 | 4854 | 4818（64_64_w4_s3，480 B） | 3585（1.34×） | 同规则 | 63% |
| [7, 512] × [256, 512] | 5.12 | 13.40 | 13.16（16_64_w4_s3，212 B） | 13.28（0.99×） | 16_64_w4_s3：12.56 | - |
| [64, 512] × [1024, 512] | 6.14 | 187.6 | 188.8（64_64_w4_s3，1564 B） | 15.24（12.39×） | 同规则 | - |
| [256, 4096] × [1024, 4096] | 27.65 | 125.7 | 126.0（64_64_w4_s3，480 B） | 89.72（1.40×） | 32_64_w8_s2：81.00 | - |

- gfx1100 上 BLOCK_M=64 的版本每条 lane 溢出 480 字节，K=512 时 1564 字节（超过 scratch 门槛，所以 [64, 512] × [1024, 512] 要 187.6 µs）。按规则只把 BLOCK_M 限制在 32、流水改成 2 级，除 [7, 512] 的版本（228 B）外都不再溢出；大形状快 1.34–1.50 倍，最大形状 4818 → 3585 µs（INT8 峰值的 47% → 63%，官方 benchmark 中 3563 µs、64%）。
- 规则在 11 个形状上就是 32 种中最快的，其余形状上最快的配置比规则快 1%–21%（差得最多的是 [8, 4096] × [7168, 4096]，16×128、8 个 warp）。18 个形状上全部配置的结果都与上游逐位相同；128×128、4 个 warp、3 级流水需要 66048 字节 LDS，超过 64 KiB，不能运行。

### `fused_moe_router_cudacore`

| 形状 | RTX 4090 | 上游（报告） | 上游（本次） | 按规则（block_h > 128 时 4 个 warp） | 全部配置中最快 |
|---|---:|---:|---:|---:|---:|
| [1, 64] × 8 个专家 | 3.90 | 10.04 | 9.88（block_h 64、1 个 warp，不溢出） | 不变 | 9.76（同上游） |
| [37, 256] × 16 个专家 | 6.14 | 26.24 | 22.48（block_h 256、1 个 warp，不溢出） | 15.64（1.44×，不溢出） | 12.48（block_h 128、1 个 warp） |
| [83, 512] × 32 个专家 | 13.66 | 261.6 | 258.3（block_h 512、1 个 warp，溢出 2056 B） | 18.24（14.16×，不溢出） | 同规则 |

- 专家数 ≤ 32 时 block_h = 32768 ÷ block_e（超过行宽时取行宽向上取到 2 的幂）、num_warps 固定为 1，[83, 512] × 32 个专家时一个 wave 要放下 32×512 的 tile。block_h 512 时 2 个 warp 溢出 548 字节、61.4 µs，4 个 warp 不溢出、18.2 µs；只把 block_h 减到 256、仍用 1 个 warp 时溢出 1368 字节（低于门槛）、57.0 µs，溢出本身也很慢。
- [37, 256] 上 block_h 128、1 个 warp 更快（12.5 µs），但要同时改 block_h，规则没有用。大形状（专家数 256）走另一个 kernel，不受影响。专家 id 全部与参考一致，权重容差占比最多 0.04。

### 访存型算子的参数

先标定 W7900D 上同样数据量能跑到多少（`exp_mem.py`，bf16 拷贝，读写各一半）：

| 读写总量 | PyTorch `copy_` | Triton 拷贝，最快的配置 | Triton 拷贝，最慢的配置 |
|---|---:|---:|---:|
| 256 MiB | 554 GB/s（64%） | 604 GB/s（70%，BLOCK 1024、8 warps） | 469 GB/s（54%，BLOCK 8192、4 warps） |
| 1 GiB | 660 GB/s（76%） | 729 GB/s（84%，BLOCK 1024、8 warps） | 638 GB/s（74%，BLOCK 8192、4 warps） |

只列 4090 上不少于 50 µs 的形状，耗时单位为 µs，括号内为相对报告的加速比和峰值带宽占比。`gelu_and_mul`、`silu_and_mul` 的形状是 [bs, d]（输入 [bs, 2d]），`per_group_transpose` 是 [M, K]（8 个专家）：

| 算子（形状） | RTX 4090 | 上游（报告） | 峰值带宽占比 | 参数搜索中最快的配置 | 主副本（官方 benchmark） |
|---|---:|---:|---:|---|---:|
| `gelu_and_mul` [4096, 4096] | 112.6 | 228.1 | 51% | BLOCK 256、8 warps：222.1（1.03×，52%） | - |
| `gelu_and_mul` [4096, 8192] | 219.1 | 380.2 | 61% | BLOCK 512、8 warps：380.5（1.00×，61%） | - |
| `silu_and_mul` [4096, 4096] | 120.8 | 322.6 | 36% | 二维 kernel，BLOCK 1024、16 warps：224.7（1.44×，52%） | 237.3（1.36×，49%） |
| `silu_and_mul` [4096, 8192] | 229.4 | 476.5 | 49% | 同上：397.2（1.20×，59%） | 392.7（1.21×，59%） |
| `fused_rmsnorm` [4096, 4096] | 76.80 | 198.0 | 39% | 16 warps：165.7（1.20×，47%） | 164.7（1.20×，47%） |
| `fused_rmsnorm` [4096, 8192] | 148.5 | 342.0 | 45% | 16 warps：282.1（1.21×，55%） | 279.9（1.22×，55%） |
| `per_group_transpose` [8192, 4096] | 181.2 | 655.4 | 24% | BLOCK_M 32、BLOCK_K 32、8 warps：470.7（1.39×，33%） | 471.7（1.39×，33%） |
| `softcap_out` [64, 128256] | 57.34 | 110.8 | 51% | BLOCK 512、8 warps：104.8（1.06×，54%） | - |
| `softcap_out` [512, 32000] | 110.6 | 209.9 | 54% | BLOCK 512、8 warps：203.5（1.03×，56%） | - |
| `softcap_out` [512, 128256] | 442.4 | 765.8 | 60% | BLOCK 2048、8 warps：659.0（1.16×，69%） | - |
| `moe_sum_reduce` [512, 8, 7168] | 90.24 | 218.2 | 35% | 每个程序 4 行、BLOCK_N 4096、8 warps：199.3（1.09×，38%） | - |
| `moe_sum_reduce` [4096, 8, 7168] | 598.0 | 935.1 | 65% | 每个程序 2 行、BLOCK_N 4096、8 warps：911.4（1.03×，67%） | - |

- 数据量越小越难跑满：同样的 Triton 拷贝读写共 256 MiB 时 70%、1 GiB 时 84%，表中 50–100 MB 的形状上游只有 35%–54%，4090 在这些形状上有 73%–89%，所以按 4090 推算的空间在 W7900D 上要打折扣。
- `silu_and_mul`：上游按总 tile 数选 kernel，超过 2048 个时改走「每行一个程序、循环处理整行」的 kernel，网格最多 512 个程序，两个大形状都走这条路（[4096, 4096] 上只有峰值带宽的 36%）。都改用二维 kernel（一个程序算一行中的 1024 列）、16 个 warp 后，大形状快 1.20–1.44 倍，在 24 种变体中最快，官方 benchmark 中其余形状 0.97–1.43 倍；读写本来就是 16 字节向量，结果逐位相同。
- `fused_rmsnorm`：每行一个程序、固定 2 个 warp，行宽 8192 时寄存器 212 个、占用率 7（gfx1100 与 gfx1201 相同）。行宽 ≥ 4096 时用 16 个 warp，大形状快 1.20–1.22 倍；归约顺序变了，结果不逐位相同，完整测试中的容差占比与上游相同（0.36）。
- `per_group_transpose`：读取是 16 字节向量，转置后的写回是 2 字节标量写（每个版本 8 条）。18 种分块中没有统一最快的：8 个 warp 只在最大的形状上快 1.39 倍，在 K=4096、行数少的两个形状（[128, 4096]、[1024, 4096]）上慢约 10%；换 BLOCK_M、BLOCK_K 在中等形状上能快 1.16–1.44 倍，但各形状最快的分块不同。规则只在数据量不少于 2^24 个元素时用 8 个 warp。最大形状改后仍只有峰值带宽的 33%，经 LDS 转一次布局、让写回也向量化还有空间（没有测）。
- `gelu_and_mul`、`softcap_out`、`moe_sum_reduce`：分别试了 15、16、45 种分块，结果都与上游逐位相同。大形状最多快 1.03、1.16、1.09 倍，最大形状已到峰值带宽的 60%–65%，同样数据量（200–530 MB）的拷贝约 70%–80%。`gelu_and_mul` 在 R9700 上也没有收益。

### WMMA 写回：跳过 `optimize_epilogue`

（后续：LoRA PR 中发现把累加器做一次 `tl.split` / `tl.join` 往返就能让这个 pass 不再生效，不必改 Triton，见「LoRA 与访存算子 PR」。）

R9700 文档指出：gfx1100 的 WMMA 结果布局里每条 lane 的元素隔列分布，Triton 直接从这个布局写回，只能逐个元素写。直接写回是 AMD 后端的 `optimize_epilogue` pass 去掉了写回前的布局转换造成的；这个 pass 在所有架构上都运行，没有开关。这里在 W7900D 上把这个 pass 换成空操作（`PROTO_NO_EPILOGUE=1`，用单独的 Triton 缓存 `triton_cache_noepi`），写回前就会经 LDS 转成普通的 blocked 布局。比较了 14 个用 WMMA 的算子的全部 benchmark 形状（`exp_ops.py`，正常编译和跳过各跑一遍），下表只列大形状，耗时单位为 µs：

| 算子 | 形状 | RTX 4090 | 报告 | 正常编译 | 跳过 `optimize_epilogue` |
|---|---|---:|---:|---:|---:|
| `chunked_sgmv_expand` | 4 段 × 256 个 token，rank 64，输出 4096 + 1024 + 1024 列 | 68.61 | 197.2 | 196.8 | 104.2（1.89×） |
| `gate_up_lora_b` | 4 段 × 256 个 token，rank 64，输出 4096 | 50.35 | 191.3 | 194.8 | 113.4（1.72×） |
| `bmm_chunk` | [8, 16, 256, 8, 64] | 180.5 | 450.4 | 470.1 | 427.4（1.10×） |
| `bmm_chunk` | [32, 4, 256, 8, 64] | 181.2 | 458.8 | 457.9 | 423.0（1.08×） |
| `chunk_state` | [8, 16, 256, 32, 8, 64, 128] | 379.9 | 673.3 | 673.6 | 653.5（1.03×） |
| `chunk_state` | [32, 4, 256, 64, 8, 64, 128] | 671.7 | 1110 | 1092 | 982.7（1.11×） |
| `qkv_lora_b` | 8 段 × 64 个 token / 4 段 × 256 个 token | 60.32 / 92.16 | 130.7 / 196.3 | 131.0 / 198.3 | 124.5 / 185.0（1.05× / 1.07×） |
| `w8a8_block_int8_matmul` | 6 个大形状 | - | - | - | 1.00–1.06× |
| `decode_attention` | 5 个大形状 | - | - | - | 0.97–1.06× |
| `decode_grouped_attention` | 6 个大形状 | - | - | - | 0.99–1.03× |
| `chunk_scaled_dot_kkt`、`chunk_state_varlen` | 3 个大形状 | - | - | - | 1.00× |
| `extend_attention` | [1, 2048, 32, 8, 128] | 69.63 | 170.0 | 170.3 | 177.8（0.96×） |
| `extend_attention` | [8, 2048, 32, 8, 128] | 830.5 | 1937 | 1932 | 2139（0.90×） |
| `extend_attention` | [64, 512, 32, 8, 128] / [256, 256, 32, 8, 128] | 714.8 / 1147 | 2038 / 4023 | 2116 / 4795 | 2252 / 5104（0.94× / 0.94×） |

各 kernel 全部版本的写回指令数之和：

| kernel（算子） | 正常编译 | 跳过 `optimize_epilogue` | LDS 最大 |
|---|---|---|---:|
| `_bmm_chunk_kernel`（`bmm_chunk`） | 2 B × 56 | 8 B × 5 | 2 → 2 KiB |
| `_chunk_scaled_dot_kkt_gstore_kernel`（`chunk_scaled_dot_kkt`） | 16 B × 64 | 16 B × 64 | 16 → 16 KiB |
| `_chunk_scaled_dot_kkt_storeopt_kernel`（`chunk_scaled_dot_kkt`） | 4 B × 40 | 16 B × 4、8 B × 3 | 4 → 4 KiB |
| `_chunk_state_fwd_kernel`（`chunk_state`） | 4 B × 160 | 16 B × 37、4 B × 2 | 12 → 16 KiB |
| `_chunk_state_varlen_grouped_kernel`（`chunk_state_varlen`） | 16 B × 38、8 B × 1 | 不变 | 32 → 32 KiB |
| `_chunked_sgmv_expand_kernel`（`chunked_sgmv_expand`） | 2 B × 192 | 2 B × 192 | 24 → 24 KiB |
| `_decode_attention_dot_kernel`（`decode_attention`） | 4 B × 8 | 16 B × 2 | 32 → 32 KiB |
| `_extend_attention_kernel`（`extend_attention`） | 4 B × 288 | 16 B × 72 | 16 → 32 KiB |
| `_flash_kernel`（`decode_grouped_attention`） | 4 B × 80 | 16 B × 20 | 32 → 64 KiB |
| `_gate_up_lora_b_kernel`（`gate_up_lora_b`） | 2 B × 384 | 16 B × 48 | 16 → 16 KiB |
| `_qkv_lora_b_kernel`（`qkv_lora_b`） | 2 B × 96 | 2 B × 80 | 16 → 16 KiB |
| `_sgemm_lora_a_split_kernel`（`chunked_sgmv_shrink`） | 4 B × 112 | 16 B × 20 | 8 → 8 KiB |
| `_sgemm_lora_a_stage1`（`sgemm_lora_a`） | 4 B × 120 | 16 B × 28 | 64 → 64 KiB |
| `_sgemm_lora_b_kernel`（`sgemm_lora_b`） | 2 B × 448 | 16 B × 48 | 16 → 16 KiB |
| `_w8a8_int8_dot`（`w8a8_block_int8_matmul`） | 2 B × 408 | 16 B × 51 | 32 → 32 KiB |

- 写回大多变成 16 字节向量写，指令数降到 1/4–1/8，与 R9700（gfx1201）上的写回相同；有的 kernel LDS 用量翻倍（`_extend_attention_kernel` 16 → 32 KiB、`_flash_kernel` 32 → 64 KiB）。
- 收益最大的是 bf16 输出的 LoRA kernel：`chunked_sgmv_expand` 1.89–2.04 倍、`gate_up_lora_b` 1.32–1.72 倍、`sgemm_lora_b` 1.45–1.73 倍（后两个各有一个形状不到 50 µs）。`chunked_sgmv_expand` 的读写宽度都没有变（输出先读出再累加，读写都是 2 字节），指令总数少了 9%–13%，加速的原因没有查明。`extend_attention` 慢 4%–10%，其余在 0.88–1.11 倍之间。
- 输出除 `gate_up_lora_b`、`sgemm_lora_a` 各一个 shuffled 形状外全部逐位相同。这两个形状的行排列由未设种子的 `torch.randperm` 生成（`reference/_lora_batch_utils.py` 第 64 行），两个进程的输入本来就不同。
- 结论：对写回占比高的 kernel，经 LDS 转换布局后写回值得做，但不能对所有 kernel 一刀切。R9700 文档里「LDS 转换本身也有开销，是否划算需要实测」的问题，在 W7900D 上的答案是因 kernel 而异。

### autotune 的选择

5 个带 autotune 的算子，benchmark 的全部形状在新进程中调优一次（`exp_autotune.py`，`results/autotune.jsonl`），记录每个调优键选中的配置、全部候选的计时，并与对应版本的编译统计对应。

| 算子 | kernel | 调优键 | 候选数 | 不能运行的候选 | 选中的版本溢出的键 | 不溢出的候选中最快的比选中的慢 |
|---|---|---:|---:|---:|---|---:|
| `fused_moe_gemm` | `_fused_moe_gemm_single_token_kernel` | 3 | 6 | 0 | - | - |
| `fused_moe_router_tensorcore` | `_router_gemm_kernel` | 5 | 130–138 | 12 | 2 个，380–448 B | 1.52–1.61× |
| `fused_moe_router_tensorcore` | `_router_topk_kernel` | 8 | 6 | 0 | - | - |
| `selective_state_update` | `_selective_state_update_batchgroup_kernel` | 1 | 25 | 0 | - | - |
| `selective_state_update` | `_selective_state_update_fulltile_kernel` | 5 | 10 | 0 | 2 个，1164 B | 1.00–1.03× |
| `silu_and_mul_masked` | `_silu_and_mul_masked_col_kernel` | 2 | 123 | 0 | - | - |
| `silu_and_mul_masked` | `_silu_and_mul_masked_row_kernel` | 3 | 57 | 0 | - | - |
| `state_passing` | `_state_passing_kernel` | 5 | 9 | 0 | - | - |

- 共 32 个调优键，与 R9700 文档相同。选中的版本中，`_router_gemm_kernel` 在 M=512（380 B）、M=4096（448 B）时溢出，不溢出的候选最快也要 807.6、5608 µs，比选中的慢 1.52–1.61 倍：这个 kernel 慢在 FMA，而不是溢出（与 R9700 相同）。`_selective_state_update_fulltile_kernel` 在两个小调优键上选中了溢出 1164 B 的版本（R9700 上选中的都不溢出），低于 scratch 门槛，耗时与不溢出的候选相当（10.2 对 10.2 µs、9.4 对 9.7 µs）。其余 kernel 选中的都不溢出，编译统计里那些超过 scratch 门槛的候选（`_router_gemm_kernel` 的 144 个、`_silu_and_mul_masked_row_kernel` 的 12 个、`_silu_and_mul_masked_col_kernel` 的 6 个）都没有被选中。
- `_router_gemm_kernel` 每个调优键有 12 个候选因为 LDS 超过 64 KiB 不能运行，白白占用编译时间。
- 选择在进程间会变：上游 router 在 M=4096 上前后四次运行选了 64×32×64（3 级流水，两次）、32×64×64（2 级流水）、64×32×64（2 级流水），耗时在 3490–3784 µs 之间，选中的配置不同时输出也不逐位相同（scratch 实验中唯一输出不同的形状就是这个原因）。需要可复现时可以设置 `TRITON_CACHE_AUTOTUNING=1`，把调优结果缓存到磁盘，之后的进程沿用。

### 第一轮 8 项改动的完整正确性测试和官方 benchmark

把第一轮的 8 项改动套到包的副本上（`patched/FlagGems-sglang-w7900d/`），另做一个只把 `context_attention` 的 tile 改成 32×16 的副本（`patched/FlagGems-sglang-w7900d-ctx-fma/`），改动见 `results/patched.diff`。用上游的测试文件跑这 8 个算子的全部用例（`exp_fulltest.py test`，括号内为各用例中最大的容差占比，由 pytest 插件 `tolplugin.py` 记录），再用上游的 benchmark 跑一遍（comprehensive 级别、kernel 模式、warmup 1000、iter 100，与报告相同）。上游实现在同一批中重跑，作为对照；测试和 benchmark 的结果文件都写到 `results/fulltest_*`、`results/fullbench_*`，上游目录里没有留下文件。

| 算子 | 用例数 | 上游实现 | 主副本（8 项改动） | 只改 `context_attention` 分块 |
|---|---:|---:|---:|---:|
| `context_attention` | 11 | 11 通过（0.0002） | 11 通过（0.18） | 11 通过（0.0002） |
| `fused_moe_router_tensorcore` | 8 | 7 通过，1 不通过（1.04） | 8 通过（0.88） | - |
| `fused_moe_router_cudacore` | 8 | 8 通过（0.94） | 8 通过（0.94） | - |
| `causal_conv1d_fn` | 3 | 3 通过（0） | 3 通过（0） | - |
| `w8a8_block_int8_matmul` | 18 | 18 通过（0.002） | 18 通过（0.002） | - |
| `silu_and_mul` | 4 | 4 通过（0） | 4 通过（0） | - |
| `fused_rmsnorm` | 19 | 19 通过（0.36） | 19 通过（0.36） | - |
| `per_group_transpose` | 3 | 3 通过（0） | 3 通过（0） | - |

官方 benchmark 中相对同一批运行的上游实现的加速比：

| 算子 | 大形状 | 其余形状 | 大形状耗时是 4090 的倍数（现在 → 改后） | 本次上游 ÷ 报告 |
|---|---:|---:|---:|---:|
| `fused_moe_router_tensorcore` | 7.78–11.12× | 1.00–4.74× | 9.97 → 1.04 | 0.95–1.03 |
| `causal_conv1d_fn` | 4.69–9.19× | 1.02–1.03× | 10.00 → 1.53 | 0.98–1.03 |
| `context_attention`（fp16） | 2.55–3.59× | 1.18–3.65× | 8.67 → 2.84 | 0.95–1.01 |
| `context_attention`（只改分块） | 1.10–2.21× | 1.02–3.89× | 8.67 → 6.18 | 同上 |
| `w8a8_block_int8_matmul` | 1.35–1.50× | 0.99–12.34× | 6.03 → 4.17 | 0.90–1.05 |
| `per_group_transpose` | 1.40× | 0.93–1.04×（代码不变） | 3.62 → 2.60 | 0.98–1.04 |
| `fused_rmsnorm` | 1.21× | 0.96–1.15× | 2.44 → 2.01 | 0.92–1.04 |
| `silu_and_mul` | 1.21–1.36× | 0.97–1.43× | 2.35 → 1.83 | 0.95–1.10 |
| `fused_moe_router_cudacore` | 0.99–1.00× | 0.96–14.33× | 1.84 → 1.86 | 0.93–1.02 |

- 上游 `fused_moe_router_tensorcore` 的 M=4096 用例不通过（容差占比 1.04），与之前的正确性测试一致；改用分块求和后通过（0.88）。
- `per_group_transpose` 第一次用的规则（K ≥ 1024 时 8 个 warp）在官方 benchmark 中让 K=4096、行数少的两个形状慢了 11%–12%，改成按数据量后重跑了测试和 benchmark（`results/fulltest_patched_pgt*`、`fullbench_patched_pgt/`），表中是重跑的结果。

## 第二轮：不按 R9700 文档的改法

第二轮不限于 R9700 文档列出的问题：先把耗时最多的几个算子拆开看（各个 kernel 分别的耗时、网格中的程序数、编译出的指令），按找到的原因改算法、网格或数据流，另外对全部 56 个算子扫了一遍 num_warps。运行方式与第一轮相同（`run_proto.sh` 按 PCI 地址选卡、确认空闲并绑核）。要改写的算子先做成模块副本（`patched/context_attention_v2.py`、`fused_moe_gemm_grouped.py`、`extend_attention_v2.py`、`decode_splitkv.py`、`per_group_transpose_v2.py`），每处改动做成开关逐项测；只改参数的用不同的启动参数调用上游 kernel。各表的加速比相对同一进程中的上游，最终的数字以本节最后 v2 包的官方 benchmark 为准。

### `context_attention`：去掉主机同步、改网格顺序、K/V 先转成 fp16

第一轮的 fp16 写法之后还有三处问题：

- 封装每次调用都执行 `b_seq_len.max().item()` 来定网格，主机要等 GPU 上之前的工作做完，小形状上这比 kernel 本身还慢；
- 网格是 (序列, 头, query 块)，query 块是变化最慢的一维，同一 (序列, 头) 的各个 query 块不在同一时间运行，读的是同样的 K/V，却用不上缓存；
- head_dim > 64 时在 kernel 里把 fp32 的 K/V 转成 fp16，每条 lane 溢出 1.0–1.2 KB。

`patched/context_attention_v2.py` 按传入的 `max_input_len` 定网格，每个程序依次处理第 pid_m、pid_m + grid_m、…… 个 query 块，不读序列长度，更长的序列也能算完；网格改成 (query 块, 头, 序列)；完全在对角线以下的 key 块不加 mask；softmax 用 exp2；K、V 在 kernel 之外先整体转成 fp16。逐项叠加的结果（`exp_ctx.py`，`results/ctx3.jsonl`，6 个大形状、4 个小形状）：

| 写法 | 大形状几何平均 | 大形状 | 小形状 | 每条 lane 溢出 |
|---|---:|---:|---:|---:|
| 第一轮的 fp16 写法（64×32 tile） | 3.01× | 2.56–3.55× | 1.18–3.63× | - |
| v2 kernel，各项都不开 | 2.83× | 2.32–3.21× | 1.17–3.23× | 0–1172 B |
| + 去掉主机同步 | 3.17× | 2.62–4.23× | 6.83–9.85× | 同上 |
| + 网格中 query 块最快 | 3.44× | 2.71–4.16× | 6.58–9.56× | 同上 |
| + 对角线以下的块不加 mask | 4.80× | 2.72–7.26× | 6.67–9.46× | 0–1236 B |
| + exp2 | 4.95× | 2.82–7.32× | 6.47–10.22× | 0–1204 B |
| + K、V 先用 torch 转成 fp16 | 9.32× | 3.88–19.14× | 3.88–11.73× | 0–216 B |

另外试了从最长的 causal 块开始处理，没有影响（3.43 倍），v2 包保留了这个顺序。K、V 先转换的收益最大，大形状的几何平均几乎翻倍；小形状多了两次 torch 的转换，下限从 6.47 倍降到 3.88 倍。在完整版上每次去掉一项（`results/ctx4.jsonl`）：

| 去掉的一项 | 大形状几何平均 | 大形状 | 小形状 |
|---|---:|---:|---:|
| 无（完整版） | 9.11× | 3.80–18.77× | 3.76–11.76× |
| 恢复主机同步 | 7.71× | 3.17–18.87× | 0.90–3.30× |
| 网格恢复原来的顺序 | 6.85× | 2.74–13.47× | 3.81–11.55× |
| 对角线以下也加 mask | 8.93× | 3.64–18.85× | 3.78–11.31× |
| 不用 exp2 | 8.93× | 3.77–18.41× | 3.81–11.61× |

K、V 改用一个 Triton kernel 一次转完（代替 torch 的两次转换），大形状不变（9.14 倍），小形状少一次启动，到 4.82–13.71 倍。在完整版上又试了 13 种 tile、warp 数和流水级数的组合，大形状几何平均在 2.80–8.93 倍之间，都不如 64×32：多一级流水时溢出升到 1084–1488 B，只有 5.91 倍；32×32 tile 不溢出，在每条序列不超过 128 个 token 的形状上比 64×32 快 6%–22%，在每条 2048 个 token 的形状上慢 1.34–1.45 倍。

最终的规则（`results/ctx5b.jsonl` 的 auto 一项，也是 v2 包的写法）：head_dim ≤ 64 时用 64×64 tile，不预先转换；head_dim > 64 时用一个 Triton kernel 把 K、V 转成 fp16，`max_input_len` 不超过 128 用 32×32 tile，否则用 64×32。原型中大形状几何平均 9.87 倍（4.66–18.63 倍），小形状 6.45–16.53 倍，每条 lane 溢出 0–216 B。bf16 输入仍按 fp32 计算（fp16 放不下 bf16 的范围）。v2 包官方 benchmark 中的结果（耗时单位为 µs）：

| 形状（序列数 × 每条 token 数，头数 × head_dim） | RTX 4090 | W7900D 上游 | v2 | 加速比 | v2 是 4090 的倍数 |
|---|---:|---:|---:|---:|---:|
| 1 × 128，32 × 128 | 50.18 | 430.5 | 48.60 | 8.86× | 0.97 |
| 1 × 2048，32 × 128 | 730.2 | 14126 | 1042 | 13.56× | 1.43 |
| 8 × 128，32 × 128 | 122.9 | 1185 | 255.1 | 4.64× | 2.08 |
| 8 × 2048，32 × 128 | 18613 | 118678 | 6612 | 17.95× | 0.36 |
| 64 × 128，32 × 128 | 1068 | 7002 | 1383 | 5.06× | 1.29 |
| 64 × 2048，32 × 128 | 150785 | 944662 | 50977 | 18.53× | 0.34 |
| 2 × 8，4 × 128 | 34.85 | 283.1 | 17.84 | 15.87× | 0.51 |
| 3 × 30，4 × 96 | 34.82 | 284.5 | 18.28 | 15.56× | 0.52 |
| 1 × 20，4 × 80 | 34.82 | 281.6 | 17.64 | 15.96× | 0.51 |
| 1 × 9，4 × 13 | 34.78 | 75.92 | 12.48 | 6.08× | 0.36 |

- 每条 2048 个 token 的形状上 v2 已比 4090 上的上游快；每条 128 个 token、序列多的两个形状仍比 4090 慢 1.3–2.1 倍。
- 代价：K、V 的 fp16 副本要额外的显存，最大的形状（131072 个 token × 32 头 × 128）两份共 2.1 GB；fp16 的范围比 fp32 小，绝对值超过 65504 的输入会溢出，第一轮的 fp16 写法也有这个限制。容差占比 ≤ 0.18。

### `fused_moe_gemm`：按专家分组的 GEMM

上游每个 (token, slot) 对单独做一次 GEMV（`_fused_moe_gemm_single_token_kernel`，网格为 (T, top_k, N ÷ BLOCK_N)），各自把所属专家的整块权重读一遍：T 个 token、top_k = 2、E 个专家时，每个专家的权重平均要读 2T ÷ E 遍，乘加也在 FMA 单元上。v2 包改成两步：

- `_moe_align_kernel`（一个程序）把 T × top_k 个 (token, slot) 对按专家稳定排序，每个专家补齐到 BLOCK_M 的整数倍；
- `_fused_moe_grouped_gemm_kernel` 每个程序算同一个专家的 BLOCK_M 个对 × BLOCK_N 列，用 `tl.dot`（WMMA，fp32 累加），乘上路由权重后按 (token, slot) 写回。

对数不少于 16、N × K 不少于 2^20、输入为 bf16 或 fp16 时走分组路径，否则仍用上游的 kernel。分块按每个专家平均分到的对数选（17 种分块的结果见 `exp_moe.py`、`results/moe2.jsonl`）：不超过 16 个用 16×32×128、2 个 warp、3 级流水；不超过 256 个用 128×64×64、4 个 warp、2 级流水；更多用 128×128×64、8 个 warp、2 级流水。v2 包官方 benchmark 中 5 个大形状的结果（8 个专家，N = K = 4096，top_k = 2）：

| T | RTX 4090 | W7900D 上游 | v2 | 加速比 | v2 是 4090 的倍数 | 说明 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 132.1 | 127.3 | 129.5 | 0.98× | 0.98 | 只有 2 个对，仍走上游的 kernel |
| 8 | 290.8 | 1026 | 513.5 | 2.00× | 1.77 | 每个专家平均 2 个对，矩阵单元大部分空着 |
| 64 | 911.4 | 1394 | 549.8 | 2.54× | 0.60 | 8 个专家的权重（256 MiB）各读一遍，峰值带宽的 57% |
| 512 | 6964 | 9191 | 894.7 | 10.27× | 0.13 | BF16 峰值的 36% |
| 4096 | 55385 | 70562 | 4107 | 17.18× | 0.074 | BF16 峰值的 63% |

- 分组路径的累加顺序与上游不同，输出不逐位相同。按测试的容差，T ≥ 8 的 4 个形状容差占比 0.53–0.56（上游 0.27–0.56），与上游输出的最大绝对差为 1.0。
- 测试文件的 3 个用例矩阵都小于 2^20，走的仍是上游的 kernel，完整测试通过不覆盖分组路径；分组路径只在原型中按测试的容差核对过（见「需要实测确认的事项」）。
- 这里的分组 GEMM 把各 K 块串在 WMMA 累加器里累加，fp16 输出的精度比上游差；MoE PR 中改成了 tile-sum，扫配置时串联累加的一个 128×256 tile 还会非法访存，见「MoE PR 中的两个发现」。

### `extend_attention`：去掉前缀 gather，合并共享 KV 头的 query 头

上游先用一个单独的 kernel（`_extend_attention_gather_kernel`）按 `kv_indices` 把各序列的前缀 K/V 拷到连续的缓冲区，再由 attention kernel 读；attention kernel 每个 query 头一个程序，共享一个 KV 头的几个 query 头各读一遍 K/V。分开计时（`exp_extdiag.py`，`results/extdiag.jsonl`；形状为 [batch, n_ctx, query 头, KV 头, head_dim]，前缀和新 token 的长度各在 1 到 n_ctx ÷ 2 之间随机取；耗时单位为 µs）：

| 形状 | 总耗时 | gather | gather 占比 | gather 的带宽（峰值的比例） | attention kernel | attention 的算力 |
|---|---:|---:|---:|---:|---:|---:|
| [1, 2048, 32, 8, 128] | 170.7 | 13.7 | 8% | 184 GB/s（21%） | 151.3 | 19.2 TFLOPS |
| [8, 2048, 32, 8, 128] | 1959 | 213.8 | 11% | 172 GB/s（20%） | 1669 | 40.5 TFLOPS |
| [64, 512, 32, 8, 128] | 2028 | 586.4 | 29% | 115 GB/s（13%） | 1399 | 21.4 TFLOPS |
| [256, 256, 32, 8, 128] | 4031 | 1262 | 31% | 108 GB/s（13%） | 2368 | 11.9 TFLOPS |
| [4, 256, 12, 4, 128] | 76.6 | 11.0 | 14% | 115 GB/s（13%） | 67.1 | 2.4 TFLOPS |
| [4, 256, 12, 4, 80] | 72.6 | 10.4 | 14% | 76 GB/s（9%） | 62.4 | 1.6 TFLOPS |
| [2, 128, 8, 8, 64] | 28.4 | 9.7 | 34% | 32 GB/s（4%） | 19.1 | 0.5 TFLOPS |

`patched/extend_attention_v2.py` 在 attention kernel 里经 `kv_indices` 直接读前缀，不再 gather；共享一个 KV 头的 G 个 query 头放进同一个 tile（一个 64 行的 tile 放 64 ÷ G 个 token 的 G 个头），每个 K/V tile 只读一次；网格为 (query 块, KV 头, 序列)，从最长的 causal 块开始。各变体（`exp_ext.py`，`results/ext.jsonl`，相对同一进程中的上游）：

| 变体 | 每条 lane 溢出 | [1, 2048, 32, 8, 128] | [8, 2048, 32, 8, 128] | [64, 512, 32, 8, 128] | [256, 256, 32, 8, 128] | [4, 256, 12, 4, 128] | [4, 256, 12, 4, 80] | [2, 128, 8, 8, 64] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 不合并头（其余同下一行） | 492 B | 0.99× | 1.14× | 1.14× | 1.32× | 1.04× | 1.12× | 1.30× |
| 合并头，64×64 tile，不需要 mask 的块单独循环 | 360 B | 1.16× | 1.25× | 1.45× | 1.88× | 1.20× | 1.19× | 1.32× |
| 合并头，64×64 tile，都加 mask | 276 B | 1.27× | 1.26× | 1.54× | 1.98× | 1.47× | 1.67× | 1.35× |
| 合并头，64×32 tile，不需要 mask 的块单独循环 | 184 B | 1.27× | 1.28× | 1.66× | 2.19× | 1.83× | 1.98× | 1.21× |
| 合并头，64×32 tile，都加 mask（v2 包） | 124 B | 1.29× | 1.29× | 1.71× | 2.27× | 1.94× | 2.07× | 1.21× |

- 溢出是 head_dim 128、每个 KV 头 4 个 query 头的版本；上游每条 lane 溢出 340 B。v2 包的写法在 head_dim 80、64 时不溢出。
- 在 `context_attention` 上有效的「不需要 mask 的块单独循环」在这里反而慢（BLOCK_N 64 时慢 0%–40%，32 时慢 0%–6%）：多一个循环，溢出更多。
- 2 级流水溢出 1096 B，0.83–1.20 倍；64 行、2 个 warp 的 tile 溢出 2092 B，最慢的形状只有 0.36 倍，之后报了非法访存（见下文「两个会导致非法访存的配置」）。
- v2 包的条件：bf16 或 fp16、最后一维连续、K 与 V 的布局相同、head_dim ≤ 128、每个 KV 头的 query 头不超过 8 个，否则走上游的写法。官方 benchmark 中大形状快 1.28–2.20 倍，其余 1.14–2.05 倍，容差占比与上游相同（0.31）。
- PR 中又去掉了输出的清零，并给 fp32 也用上这个 kernel（tile 减半、warp 加倍），见「注意力 PR 中的改动和发现」。

### `decode_attention`、`decode_grouped_attention`：split-KV

上游 `decode_attention` 每个 (请求, KV 头) 一个程序（`_decode_attention_dot_kernel`），从头到尾读完一条 KV 序列：batch 1、8 个 KV 头时整张卡只有 8 个程序，而 W7900D 有 96 个 CU。v2 包的 `ops/_decode_splitkv.py`（原型 `patched/decode_splitkv.py`）两个算子共用：

- 第一步的每个程序处理一个 KV 头的 BLOCK_H 个 query 头、一段 KV 序列，用 `tl.dot`（fp32 累加）做 online softmax，写出这一段的输出和 log-sum-exp；第二步合并各段。分段数按程序数凑到约 512 个、每段不少于 128 个 token 来定，从 `kv_indices` 的长度在主机上算出，不需要同步。
- 网格为 (KV 头 × query 头块, 分段, 请求)，KV 头是最快的一维，同一个请求的各个 KV 头同时读，而同一个 token 的各个 KV 头在内存中相邻。早一版把 KV 头放在后面，batch 512、4096 时比上游慢 6.3 倍、5.8 倍（`results/decode.jsonl`）。
- 配置按每个 KV 头的 query 头数（G）和网格大小选：G ≤ 16 时（`decode_attention` 的 G = 4），每块 16 个头（WMMA 至少要 16 行）、BLOCK_N 64，程序数不到 64 才分段；G > 16 时（`decode_grouped_attention`，G = 128），batch × KV 头数不到 64 时每块 16 个头、BLOCK_N 32、分段，否则每块 64 个头、BLOCK_N 64、不分段；都用 4 个 warp。
- 只支持 bf16 或 fp16、head_dim ≤ 256、K/V 最后一维连续，其余情况仍走上游的 kernel（head_dim 512 的 tile 放不进 LDS）；head_dim 512 时上游 kernel 改用 2 个（`decode_attention`）或 1 个（`decode_grouped_attention`）warp。

v2 包官方 benchmark 中的结果（形状为 batch，query 头 ÷ KV 头，head_dim，每条 KV 序列的长度；耗时单位为 µs）：

| 算子 | 形状 | RTX 4090 | W7900D 上游 | v2 | 加速比 |
|---|---|---:|---:|---:|---:|
| `decode_attention` | 1，32 ÷ 8，128，2048 | 62.54 | 128.6 | 39.76 | 3.23× |
|  | 8，32 ÷ 8，128，2048 | 109.6 | 201.0 | 184.6 | 1.09× |
|  | 64，32 ÷ 8，128，512 | 197.4 | 349.4 | 322.8 | 1.08× |
|  | 512，32 ÷ 8，128，128 | 349.2 | 581.2 | 520.0 | 1.12× |
|  | 4096，32 ÷ 8，128，128 | 2414 | 3756 | 3128 | 1.20× |
|  | 2，16 ÷ 1，512，128（小形状，上游 kernel、2 个 warp） | 33.92 | 29.24 | 26.04 | 1.12× |
| `decode_grouped_attention` | 1，128 ÷ 1，128，2048 | 84.99 | 144.3 | 26.30 | 5.49× |
|  | 8，128 ÷ 1，128，2048 | 91.14 | 149.4 | 74.96 | 1.99× |
|  | 64，128 ÷ 1，128，512 | 51.20 | 99.04 | 89.00 | 1.11× |
|  | 512，128 ÷ 1，128，128 | 101.4 | 268.0 | 274.0 | 0.98× |
|  | 4096，128 ÷ 1，128，128 | 720.9 | 1232 | 1181 | 1.04× |
|  | 2，128 ÷ 8，512，128（上游 kernel、1 个 warp） | 108.5 | 113.5 | 66.64 | 1.70× |

- 收益的来源（`exp_decode.py`，`results/decode_v2.jsonl`）：batch 1、8 来自分段；batch ≥ 64 时 `decode_attention` 不分段，用 4 个 warp 快 1.09–1.21 倍，用上游的 8 个 warp 只有 0.96–1.00 倍，收益来自 warp 数。
- `decode_grouped_attention` batch 512 慢 2%，原型中没有一种配置在这个形状上明显快于上游（最多 1.02 倍）；`decode_attention` head_dim 80 的小形状慢 2%–4%（约 0.4 µs），四次运行都慢。小形状的 3 个 `decode_grouped_attention` 形状快 1.35–1.63 倍。PR 中改了配置规则，这两处分别变成 1.01 倍和 1.04–1.09 倍，并加上了 fp32 路径，见「注意力 PR 中的改动和发现」。
- 最大的形状上，v2 达到峰值带宽的 82%（`decode_attention`）和 53%（`decode_grouped_attention`），见「访存型算子：带宽利用率」。

### `chunk_state_varlen`、`chunk_state`：两个操作数都经 LDS 转置

这两个 kernel 的 dot 都沿 token 方向求和（状态 = Σ_t B[t]ᵀ x[t]），而 token 在 x（[token, 头, P]）和 B（[token, 组, N]）中都是最慢的一维。WMMA 的每条 lane 要拿到沿求和方向连续的元素，编译器先把两个操作数都写进 LDS，再用 2 字节的 `ds_load_u16` 逐个元素读回来；操作数不需要转置的 WMMA kernel 读 LDS 用的是 16 字节的 `ds_load_b128`。按测试时 Triton 缓存中编译出的汇编统计（静态指令数）：

| kernel（算子） | 版本数 | 每条 WMMA 指令对应的 2 字节 LDS 读 |
|---|---:|---:|
| `_chunk_state_varlen_grouped_kernel`（`chunk_state_varlen`） | 5 | 8–32 |
| `_chunk_state_fwd_kernel`（`chunk_state`） | 5 | 12–32 |
| `_extend_attention_kernel`（`extend_attention`） | 5 | 8 |
| `_flash_kernel`（`decode_grouped_attention`） | 2 | 3.2–8 |
| `_decode_attention_dot_kernel`（`decode_attention`） | 1 | 1.8 |
| `bmm_chunk`、`chunk_scaled_dot_kkt` 的两个 kernel、`gate_up_lora_b`、`chunked_sgmv_expand`、`sgemm_lora_b`、`w8a8_block_int8_matmul` | 2–18 | 0 |

改数据布局要动调用方，这里只改分块：每块 1 个头（上游 2 个）、BLOCK_K 16（上游 32）、1 级流水（上游 2 级），tile 更小、数量更多，更能掩盖 LDS 的延迟（`exp_tiles.py`，`results/tiles.jsonl`；BLOCK_P 64、BLOCK_N 128 不变）：

| 每块头数，BLOCK_K，warp 数，流水级数 | 2048 个 token（8 条 × 256）、32 个头 | 8192 个 token（32 条 × 256）、64 个头 |
|---|---:|---:|
| 上游：2，32，4，2 | 133.4 µs | 710.2 µs |
| 1，32，4，2 | 2.02× | 1.27× |
| 1，16，4，2 | 2.37× | 1.53× |
| 1，32，4，1 | 2.41× | 1.44× |
| 2，32，8，2 | 2.28× | 1.41× |
| 1，16，4，1（v2 包，另一次运行） | 2.39× | 1.54× |

- BLOCK_K 128 只有 0.41 倍。结果与上游逐位相同；v2 包官方 benchmark 中两个形状快 2.42 倍、1.51 倍，小形状 1.00–1.04 倍。
- `chunk_state` 的两个 dot 操作数也要转置（每条 WMMA 对应 12–32 条 2 字节读），但 13 种分块、warp 数和流水级数的组合中，最好的只在一个大形状上快 1.3%，另一个大形状上全部比上游慢。要从数据布局入手（让 token 方向连续，或用 Gluon 显式指定布局）。
- R9700 文档中 `chunk_state_varlen` 的带宽归一化效率是 0.64，W7900D 是 0.26，可能 gfx1201 上编译出的 LDS 读不同，没有核对。

### num_warps 扫描

`exp_warps.py` 包装 Triton 的 `JITFunction.run`，让被测算子的每次 Triton 启动都用指定的 num_warps（1、2、4、8、16），带 autotune 的 kernel 不受影响，上游代码不动。每个 benchmark 形状先跑上游，再依次跑 5 种 warp 数，输出与上游逐位比较；先用短计时筛选，最快的比上游快 3% 以上时，再与上游一起按 warmup 1000、rep 100 复测（`results/warps_all.jsonl`，56 个算子、475 个形状）。各算子最快的 warp 数从 1 到 16 都有，上游大多固定用 4 个，没有一个对所有算子都合适的值。

大形状上快 5% 以上、已放进 v2 包或被其他改法取代的：

| 算子 | 上游 → 最快的 warp 数 | 大形状加速比 | 与上游逐位相同 | v2 包 |
|---|---|---:|---|---|
| `chunked_sgmv_expand` | 4 → 8–16 | 1.79× | 是 | 16 个 warp |
| `gate_up_lora_b` | 4 → 16 | 1.63× | 是 | 16 个 warp |
| `sgemm_lora_b`（只有小形状） | 4 → 16 | 两个较大的形状 1.20×、1.81× | 是 | 16 个 warp |
| `bmm_chunk` | 8 → 1 | 1.25× | 是 | 1 个 warp |
| `log_scaling_tau` | 2 → 16 | 1.25× | 是 | 程序数不少于 4096 时 16 个 warp |
| `decode_grouped_attention`，head_dim 512 | 8 → 1 | 1.65× | 否 | 1 个 warp |
| `per_group_transpose` | 4 → 16 | 1.47× | 是 | 沿用第一轮的 8 个 warp（1.42×），同一次扫描中 16 个只比 8 个快 3% |
| `chunk_state_varlen` | 4 → 8 | 1.40× | 是 | 改分块（1.51×） |
| `w8a8_block_int8_matmul` | 4 → 8 | 1.06–1.52× | 是 | 沿用第一轮的 BLOCK_M 规则（1.35–1.50×），两者都是消除溢出 |
| `context_attention` | 4 → 8 | 1.00–2.28× | 是 | 改写 |
| `causal_conv1d_fn` | 4 → 16 | 1.21–1.28× | 是 | 沿用第一轮的分块规则（4.54–9.35×） |
| `fused_rmsnorm` | 2 → 16 | 1.22× | 否 | 第一轮按行宽选 warp 数的规则 |
| `decode_attention` | 8 → 4 | 1.07–1.21× | 是 | split-KV，4 个 warp |
| `silu_and_mul` | 4 → 16 | 1.04–1.15× | 是 | 沿用第一轮的二维 kernel（1.21–1.39×） |

小形状上，`fused_moe_router_cudacore` 的 [83, 512] × 32 个专家用 4 个 warp 快 14.2 倍（不逐位相同），即第一轮的改法。没有放进 v2 包的算子，扫描结果列在第三、四、五批和「只有小形状的算子」中。

### 没有收益或不能用的改法

- `per_group_transpose` 向量化写回（`patched/per_group_transpose_v2.py`：凑成 16 字节写回，每个程序处理多个 tile；`exp_pgt.py`，`results/pgt.jsonl`）：最大形状最好的 460.2 µs，比 8 个 warp 的规则（同一进程中 480.8 µs）只快 4.5%，也只到峰值带宽的 34%，没有放进 v2 包。瓶颈不只在写回，没有再查。
- `chunk_state` 改分块、`extend_attention` 拆分 mask 和 2 级流水、`context_attention` 多一级流水或更大的 tile，见上文。
- 第一轮的 `context_attention` bf16x3（比上游慢）和 bf16 输入（超出容差），见第一轮。

### 两个会导致非法访存的配置

原型中有两个配置会让 GPU 出错：`context_attention` v2 kernel 用 32×16 tile、在 kernel 里转换 fp32 的 K/V（8 × 128，32 × 128 的形状，每条 lane 溢出 364 B），报 "unspecified launch failure" 或 "illegal memory access"；`extend_attention` v2 kernel 用 64 行、2 个 warp 的 tile（[8, 2048, 32, 8, 128]，溢出 2092 B），报 "illegal memory access"。`exp_crash.py` 每次调用后同步、与第一次的输出逐位比较：前几次输出正常，之后开始与第一次不同（最大绝对差 3–4，`context_attention` 从第 1–7 次起，`extend_attention` 约第 97 次起），随后报错，内核日志中有 GPU 页错误；只调用 3 次时不报错，但其中一次的第 3 次输出已与第一次不同（最大绝对差 0.65）。默认设置、`AMD_SERIALIZE_KERNEL=3`、`HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0`、`HSA_SCRATCH_SINGLE_LIMIT` 设成 4 GiB 四种设置下都会出错（`results/crash_matrix.log`、`crash_loop.log`）。溢出量本身不是原因：对照组 `extend_attention` v2 默认配置每条 lane 也溢出 360 B，四种设置下都正常，连续 2000 次调用输出都相同。原因没有查明，v2 包没有用这两个配置。

第一轮通过完整测试的 `context_attention` 32×16 tile 是按 fp32 计算的另一个写法，不是这里的配置；但 #199 的新 kernel 用 fp32 计算、`BLOCK_D` 128 时，32×16 约 3% 的调用会算错整块输出，64×32 配 8 个 warp 约 8%，见「第一个 PR（#199）」。MoE PR 中串联累加的分组 GEMM 又有一个配置出现同样的症状，见「MoE PR 中的两个发现」。注意力 PR 中扫 `extend_attention` 的 tile 时，128 个 token 的 K/V 分块也出现同样的症状（溢出只有 163 B），见「注意力 PR 中的改动和发现」。

另外一处误判：逐位一致性检查一度显示 `bmm_chunk` 用 1 个 warp 时连续调用的输出会变，v2 包因此改回过上游的 8 个 warp；之后发现上游的 8 个 warp 在 100 次调用中也有 99 次与第一次不同，变的只是 causal 版本不计算、`torch.empty` 分配后留下的 i > j 部分。只比较 i ≤ j 的部分后，1–16 个 warp 都逐位一致，并与上游相同（`exp_bmm.py`，`results/bmm_masked*.jsonl`），v2 包恢复用 1 个 warp。

### v2 包的完整正确性测试、逐位一致性和官方 benchmark

v2 包（`patched/FlagGems-sglang-w7900d-v2/`）改了 18 个算子的文件，另加两个 decode 算子共用的 `ops/_decode_splitkv.py`，改动见 `results/patched_v2.diff`。第一轮的 8 项中 `context_attention` 换成第二轮的写法，其余 7 项不变。测试和 benchmark 的做法与第一轮相同（`exp_fulltest.py`，括号内为各用例中最大的容差占比），上游实现在同一批中重跑作为对照，结果文件在 `results/fulltest_v2*`、`fulltest_upstream18*`、`fullbench_v2/`、`fullbench_upstream18/`。

| 算子 | 用例数 | 上游实现 | v2 包 |
|---|---:|---:|---:|
| `context_attention` | 11 | 11 通过（0.0002） | 11 通过（0.18） |
| `fused_moe_gemm` | 3 | 3 通过（0.030） | 3 通过（0.030） |
| `fused_moe_router_tensorcore` | 8 | 7 通过，1 不通过（1.04） | 8 通过（0.88） |
| `causal_conv1d_fn` | 3 | 3 通过（0） | 3 通过（0） |
| `decode_grouped_attention` | 9 | 9 通过（0.047） | 9 通过（0.072） |
| `decode_attention` | 9 | 9 通过（0.048） | 9 通过（0.061） |
| `extend_attention` | 7 | 7 通过（0.31） | 7 通过（0.31） |
| `chunk_state_varlen` | 5 | 5 通过（0.23） | 5 通过（0.23） |
| `w8a8_block_int8_matmul` | 18 | 18 通过（0.002） | 18 通过（0.002） |
| `chunked_sgmv_expand` | 5 | 5 通过（0.45） | 5 通过（0.45） |
| `gate_up_lora_b` | 5 | 5 通过（0.42） | 5 通过（0.42） |
| `sgemm_lora_b` | 5 | 5 通过（0.39） | 5 通过（0.39） |
| `per_group_transpose` | 3 | 3 通过（0） | 3 通过（0） |
| `bmm_chunk` | 5 | 5 通过（0.48） | 5 通过（0.48） |
| `fused_rmsnorm` | 19 | 19 通过（0.36） | 19 通过（0.36） |
| `silu_and_mul` | 4 | 4 通过（0） | 4 通过（0） |
| `log_scaling_tau` | 7 | 7 通过（0.070） | 7 通过（0.070） |
| `fused_moe_router_cudacore` | 8 | 8 通过（0.94） | 8 通过（0.94） |

逐位一致性（`exp_determinism.py`，`results/determinism_v2*.jsonl`）：18 个算子的全部 155 个 benchmark 形状，各用同一组输入连续调用 100 次，输出都与第一次逐位相同。

官方 benchmark 中相对同一批运行的上游实现的加速比：

| 算子 | 大形状 | 其余形状 | 大形状耗时是 4090 的倍数（现在 → v2） | 本次上游 ÷ 报告 |
|---|---:|---:|---:|---:|
| `context_attention` | 4.64–18.53× | 6.08–15.96× | 8.67 → 0.87 | 0.99–1.00 |
| `fused_moe_gemm` | 0.98–17.18× | 0.96–1.00× | 1.57 → 0.40 | 0.96–1.03 |
| `fused_moe_router_tensorcore` | 7.76–11.01× | 1.00–4.83× | 9.97 → 1.06 | 0.96–1.03 |
| `causal_conv1d_fn` | 4.54–9.35× | 1.00–1.02× | 10.00 → 1.52 | 0.97–1.02 |
| `decode_grouped_attention` | 0.98–5.49× | 1.35–1.63× | 1.71 → 1.03 | 0.99–1.01 |
| `decode_attention` | 1.08–3.23× | 0.95–1.12× | 1.77 → 1.28 | 0.98–1.08 |
| `extend_attention` | 1.28–2.20× | 1.14–2.05× | 2.73 → 1.73 | 0.99–1.01 |
| `chunk_state_varlen` | 1.51× | 1.00–2.42× | 4.54 → 3.01 | 1.00–1.05 |
| `w8a8_block_int8_matmul` | 1.35–1.50× | 0.96–12.88× | 6.03 → 4.23 | 0.91–1.06 |
| `chunked_sgmv_expand` | 1.79× | 1.16–1.88× | 2.87 → 1.60 | 0.99–1.02 |
| `gate_up_lora_b` | 1.59× | 0.93–1.10× | 3.80 → 2.41 | 0.95–1.03 |
| `sgemm_lora_b` | - | 0.90–1.73× | - | 0.98–1.07 |
| `per_group_transpose` | 1.42× | 0.96–1.03× | 3.62 → 2.59 | 0.99–1.03 |
| `bmm_chunk` | 1.22–1.24× | 0.95–0.97× | 2.52 → 2.07 | 0.94–1.02 |
| `fused_rmsnorm` | 1.21–1.22× | 0.95–1.17× | 2.44 → 1.99 | 0.94–1.05 |
| `silu_and_mul` | 1.21–1.39× | 1.00–1.33× | 2.35 → 1.82 | 0.90–1.08 |
| `log_scaling_tau` | 1.18× | 0.98–1.04× | 2.35 → 1.99 | 0.99–1.04 |
| `fused_moe_router_cudacore` | 0.99–1.00× | 0.99–14.50× | 1.84 → 1.86 | 0.99–1.01 |

- 100 个大形状合起来，W7900D 的耗时从 4090 的 2.51 倍降到 1.67 倍（几何平均），见「粗看」的第一张表。
- 小于 1 的几处：大形状中 `decode_grouped_attention` batch 512（0.98 倍）四次运行都慢；`fused_moe_gemm` T=1（0.98 倍）和 `fused_moe_router_cudacore` 的大形状（0.99 倍）走的代码没有变，属于测量波动；其余在小形状上，见第一批表格下的说明。逐形状的耗时见 `results/table_v2_final.md`。

## 第一个 PR（#199）：`context_attention` 的 fp32 保持 fp32，两种分块会算错整块输出

第一个 PR 把 `w8a8_block_int8_matmul` 和 `context_attention` 移植进上游，同时建立了 rdna3 后端（分支 `feat/amd-gfx1100-w8a8-context-attention`，2026-10-21 合入）。脚本和结果在 `opt_w7900d/pr1/`。

- **后端只给 gfx1100**：上游的 `get_arch()` 只按主版本号查 `ARCH_MAP`，AMD 上从 RDNA3 独显到 RDNA3.5 APU 的所有 gfx11 目标都落进同一个桶。PR 改成先按 `gcnArchName` 的精确目标（去掉 `:sramecc+` 之类的后缀）、再按 `major.minor`、最后按主版本号匹配，`_amd` 只把 gfx1100 映射到 rdna3；gfx1101、gfx1102、gfx1151、gfx1201、gfx1250、gfx942 都不进这个后端，NVIDIA 的映射不变。后面四个 PR 的规则都只在 gfx1100 上生效，靠的就是这一条。
- **`context_attention` 没有采用 v2 包的 fp16 写法**：v2 包把 fp32 输入也转成 fp16 走 WMMA，容差占比 0.181；PR 让 fp32 输入保持 fp32（FMA），容差占比 0.0002，与上游相同，只有 fp16、bf16 输入走输入类型的 WMMA。代价在长序列上：benchmark（只列 fp32）全部形状 1.23–15.92 倍、几何平均 3.41 倍（fp16 写法时 4.55–18.34 倍），其中长序列只有 1.23–1.39 倍，短序列的收益主要来自去掉主机同步。fp16、bf16 输入 6.50–21.45 倍。
- **两种分块会把一整块输出算错**：fp32 计算、`BLOCK_D` 128 时，64×32 配 8 个 warp（最初给长序列选的配置）在 (64, 2048, 32, 128) 上约 8% 的调用出错，两张卡都能复现；32×16 约 3% 的调用出错。改 HSA 的 scratch 设置没有用，看起来是这两个配置的编译问题。长序列因此改用 32×32 配 4 个 warp，代码里加了注释防止以后调参换回去。最终规则在 12 个形状上压测（覆盖每个分支、因果与非因果、head 数和长度能否被 16 整除），每个形状 100 次调用逐位一致，容差占比不超过 0.0003。注意 32×16 正是第一轮「不降低精度、只改分块」的那个 tile：第一轮是在上游 kernel 里改的，完整测试通过；放到 PR 的新 kernel 里就会偶尔出错，所以这条退路不能直接搬。
- **`w8a8_block_int8_matmul`**：沿用 v2 的规则（BLOCK_M ≤ 32，2 级流水），溢出从每条 lane 212–1564 B 降到 0；bf16、fp32 输出与上游逐位一致，fp16 输出在 232 个用例的 450 万个元素中有 10 个差一次舍入。`context_attention` 的 fp16、bf16 路径溢出从 972–2492 B 降到 0–180 B。

## MoE PR 中的两个发现

把 `fused_moe_gemm`、`fused_moe_router_tensorcore`、`fused_moe_router_cudacore` 移植到上游的 rdna3 后端时（上游 b908cb3，分支 `feat/amd-gfx1100-moe`，#205，2026-10-22 合入），又查到两件与 gfx1100 的 WMMA 有关的事。脚本和结果在 `opt_w7900d/pr3/` 和 `opt_w7900d/pr3/results/`；实验分在这台机器的几张空闲卡上，都按 PCI 地址确认没有其他进程占用，并绑定到所在的 NUMA 节点（`pr3/run.sh`）。

### WMMA 往 fp32 累加器里加数时偏向 0

bf16 或 fp16 输入的 `tl.dot(a, b, acc)` 编译成 `v_wmma`，把乘积加进 fp32 累加器时不按 IEEE 就近舍入。`wmma_rounding.py` 用整数算出精确值作对照：A 的每行只放一个非零元素，每个输出就是 C 加上一个精确的乘积，fp32 FMA 的 `tl.dot`（ieee）65536 个结果全部是正确舍入的值，WMMA 有 17%（fp16 输入）和 12%（bf16 输入）不是，其中 70% 和 57% 偏向 0；在本该向远离 0 进位的输出中，约 40% 被直接截断。

把各 K 块串在 WMMA 累加器里（`acc = tl.dot(a, b, acc)`）时，K=4096 要连续累加 256 次，误差朝同一个方向累积。改成每个 K 块的 dot 从 0 开始、块之间用 fp32 FMA 相加后（下称 tile-sum：`acc = tl.fma(tl.dot(a, b), 1, acc)`，用 `tl.fma` 而不用加号，是为了不让 Triton 把加法折回 dot 的累加器），偏差基本消失。K=4096、64 个 16×16 输出 tile 的结果如下；「舍入不一致」指转成 fp16 / bf16 后与精确值正确舍入的结果不同，输入正负随机；平均偏差用全正的输入，单位为结果的 fp32 ulp：

| 累加方式 | fp16 舍入不一致 | bf16 舍入不一致 | 全正输入的平均偏差（fp16 / bf16） |
|---|---:|---:|---:|
| 串联 WMMA | 3.92% | 0.27% | −113 / −37 |
| tile-sum，BLOCK_K 64–256 | 0.13%–0.26% | 0.01%–0.02% | −0.5 至 −4.4 / 0 至 −0.3 |
| fp32 FMA 的 `tl.dot`（上游 router 的写法） | 0.42% | 0.05% | 0.0 / −0.1 |
| fp32 乘积按块 `tl.sum` 后相加（上游 `fused_moe_gemm` 的写法） | 0.07%–0.08% | 0%–0.02% | 0.0 / 0.0 |

- fp16 受影响更大，不是因为范围或精度低：两个 fp16 的乘积有 22 位有效位（bf16 只有 16 位），加进 fp32 累加器时要舍掉的更多，单步测试中要向远离 0 进位的输出是 bf16 的 3.0 倍，串联累加的偏差也是 3.1 倍；fp16 输出的刻度又比 bf16 细 8 倍，同样的 fp32 误差更容易改变最后一位。
- 对算子的影响：`fused_moe_gemm` 的分组路径串联累加时（v2 包就是这样写的），20 个 benchmark 和模型形状上 fp16 有 0.36%–6.54% 的输出与 fp64 正确舍入的结果不同（上游 0.05%–0.15%，tile-sum 0.13%–0.54%），bf16 0.01%–0.46%（上游不超过 0.02%，tile-sum 0.01%–0.06%），测试仍通过。`fused_moe_router_tensorcore` 的 GEMM 串联累加时，测试中 bf16 的 M=64、512、4096 三个用例都超出容差（容差占比 1.54、1.43、2.22，对 fp64 1.59–2.05）；tile-sum 是 0.29、0.77、0.88（对 fp64 0.07–0.11，上游 0.38–0.93；`router_chained_check.py`）。v2 包 router 的「按 K 分块求和」就是 tile-sum。
- 代价：在同一张卡、同一进程中交替测 120 个形状，tile-sum 的耗时几何平均是串联累加的 0.996（bf16）、0.9955（fp16），个别大形状慢 2%–6%。
- 上游不用 WMMA，fp32 的每一步都正确舍入，误差没有方向；router 的上游写法沿 K 顺序做 4096 次 FMA，随机误差比 tile-sum 大，所以 tile-sum 的 router 比上游更接近 fp64。

### 串联累加的分组 GEMM 有一个配置会非法访存

扫 `fused_moe_gemm` 分组路径的配置时（`exp_grouped.py`，当时还是串联累加），b 组（qwen3_30b）两次、d 组（dsv2_lite_down 等）一次报 HIP 700「illegal memory access」，都在 T=4096 的形状上；b 组加了调试输出的那次，最后启动的是 128_256_32_w4_s3（BLOCK_M 128、BLOCK_N 256、BLOCK_K 32、4 个 warp、3 级流水，每条 lane 溢出 1736 B），形状为 qwen3_30b_down（E 128、N 2048、K 768、top_k 8）。当时逐个配置单独启动、每次同步，没能复现；扫描改为不启动溢出的配置后不再出现。之后单独复现（`chained_wrong.py`、`chained_corrupt.py`）：

- 串联累加的 kernel 反复启动这个配置，3 个进程都报 HIP 700，有的第 1 次启动就报错，有的先正常跑 2–3 次，内核日志中有 GPU 页错误；没有报错时，有的输出正确，有一次运行中 8240 个输出算错、14080 个输出没有写，三次启动的结果各不相同，坏的输出都在没填满的行块里。完整扫描中排在它后面、共用同一份输入和排序结果的 5 个配置输出全错（最大误差 105，输出最大 134），这 5 个单独跑都正确，可能是它写坏了共用的数据。
- 同一个配置换成 tile-sum 的 kernel，连续 40 次输出都正确，输入张量没有被改动；串联累加的 kernel 用同一个配置在 a、c 组的另外 6 个形状上输出也正确，只在部分形状上出错。
- 两个 kernel 的寻址、mask 和写回完全相同，只差累加的一行，换成后来重写的排序 kernel 照样出错，所以不是源码里的索引问题，而是这个配置编译出的代码有问题，具体原因没有查。溢出量本身不是原因：tile-sum 的 kernel 在同一配置下溢出 3536 B（也超过 scratch 门槛），一直正常。症状与上文「两个会导致非法访存的配置」相同。
- PR 中的代码用 tile-sum，配置规则中 BLOCK_M 不超过 64，只用不溢出的配置。用 PR 的代码重跑 b、d 两组并启动全部配置（共 4014 个，其中 629 个溢出），都没有出错，输出的最大误差不超过 0.5。改动累加方式或配置规则后，要在实际形状上连续启动多次检查。

## MoE PR 中的 `fused_moe_router_cudacore`：专家数 > 32 的路径

v2 包和 PR 的第一版只改了专家数 ≤ 32 的单 kernel 路径，专家数 > 32 时仍是上游的 logits kernel（每个程序 2 个 token × 8 个专家，沿 hidden 每块先 `tl.sum` 归约，4 个 warp、2 级流水），大形状与上游一样快。PR 中改写了这个 logits kernel 和它的启动参数，仍只用 FMA；脚本和结果在 `opt_w7900d/pr3/`。

### 启动参数

每个程序算 1、2 或 4 个 token，共用读进来的一块权重。「延后」归约指把乘积按元素累加到与分块同样大的 fp32 累加器里，hidden 循环结束后只归约一次；「每块」指每块先归约。都用 1 级流水：

| 输入 | batch size | 每个程序的 token | 归约 | 分块（专家 × hidden） | warp |
|---|---|---:|---|---|---:|
| bf16、fp16 | ≤ 8 | 1 | 每块 | 4 × 2048 | 8 |
| bf16、fp16 | 9–64 | 2 | 延后 | 8 × 1024 | 4 |
| bf16、fp16 | 65–768 | 2 | 延后 | 8 × 512 | 2 |
| bf16、fp16 | > 768 | 2 | 延后 | 32 × 256 | 4 |
| fp32 | ≤ 8 | 1 | 延后 | 4 × 2048 | 8 |
| fp32 | 9–256 | 2 | 延后 | 8 × 512 | 4 |
| fp32 | > 256 | 4 | 延后 | 32 × 128 | 8 |

hidden 比分块小时，hidden 方向的分块缩到 hidden 向上取 2 的幂，但不小于 32 条 lane × 16 字节（16 位输入 256 个元素，fp32 128 个），warp 数也不超过分块中这样的行数，使每个 warp 只覆盖一行专家，原因见下文「上游在小 hidden 加 softcap 时算错」。

### 官方 benchmark

与上游交替各 3 次，取中位数，形状 (bs, hidden, E, topk)：

| 形状 | bf16 | fp16 | fp32 |
|---|---:|---:|---:|
| (1, 4096, 256, 8) | 1.11 倍 | 1.07 倍 | 1.03 倍 |
| (8, 4096, 256, 8) | 1.06 倍 | 1.09 倍 | 1.03 倍 |
| (64, 4096, 256, 8) | 1.10 倍 | 1.17 倍 | 1.10 倍 |
| (512, 4096, 256, 8) | 1.22 倍 | 1.65 倍 | 1.55 倍 |
| (4096, 4096, 256, 8) | 1.49 倍 | 2.07 倍 | 1.78 倍 |

bf16 最大形状从 1350.2 µs 降到 903.5 µs；另外 3 个 benchmark 形状走专家数 ≤ 32 的路径，与之前相同。

### 上游在小 hidden 加 softcap 时算错

专家数 > 32 时，上游在 hidden 较小、带 softcap 的部分形状上算错 logits，每次结果不同，选出的专家也随之出错（`cudacore_small_hidden.py`、`check_cudacore_module.py`）。各运行 10 次：

| 形状 (bs, hidden, E, topk)，softcap 30 | 输入 | 上游每次选错专家的行 | PR |
|---|---|---|---:|
| (769, 128, 128, 8)，带 bias | bf16 | 48–68 行（共 769 行） | 0 |
| (1536, 128, 160, 4) | bf16 | 25–44 行（共 1536 行） | 0 |
| (500, 128, 40, 4) | bf16 | 2–12 行（共 500 行） | 0 |
| (7, 128, 512, 6) | bf16 | 0–1 行（共 7 行） | 0 |
| (256, 16, 60, 2) | fp16 | 0–2 行（共 256 行） | 0 |

- 去掉 softcap，这些形状都正确，10 次结果也相同；fp32 的 (1536, 64, 160, 4) 带 softcap 也正确。
- 错误只出现在一个 warp 跨多行专家的布局上（TTGIR 中 `threadsPerWarp` 沿专家方向大于 1，如 [2, 16]、[4, 8]）。上游的启动在 hidden 小于 32 条 lane × 16 字节时就是这种布局，例如 bf16 hidden 128：分块 8 × 128，每条 lane 读 8 个元素，一个 warp 覆盖 2 行；PR 的 logits kernel 用上游的启动参数时，bf16 (1536, 128, 160, 4) 也出错。上游错的 logits 都在一个程序两个 token 中的同一个（bf16 (1536, 128, 160, 4) 全在奇数行，fp16 (256, 16, 60, 2) 全在偶数行）。
- 变体扫描中这种布局的 193 个版本里，bf16 有 4 个、fp16 有 1 个在 benchmark 形状上（不带 softcap）出错，fp32 的 100 个没有出错；每个 warp 只覆盖一行的 340–433 个版本没有一次出错。原因没有查明；上游在其他芯片上是否也这样没有测。

## Mamba PR：只改启动参数，输出逐位一致

把 `causal_conv1d_fn`、`chunk_state_varlen`、`bmm_chunk` 移植到 rdna3 后端时（分支 `feat/amd-gfx1100-mamba`，#231，2026-10-28 合入），三个算子都沿用上游的 kernel，只换启动参数，所以输出与上游逐位相同。脚本和结果在 `opt_w7900d/pr4/`。

- `causal_conv1d_fn`：上游固定 128 个 token × 4 个通道一块。改成时间方向的分块跟着最长的序列走，每个程序读连续的通道行。
- `chunk_state_varlen`：dstate ≥ 128 时改用更小的 tile、K 方向每步 16 个 token、1 级流水；更小的状态仍用上游的启动参数。
- `bmm_chunk`：32×32 tile 在更少的 warp 下更快；fp32 的 K 方向每步 16 个元素，这样单个 warp 不溢出。

官方 benchmark（做法同其他 PR，交替各 3 次、绑 NUMA、取中位数；运行间波动最大 1.11 倍）：

| 算子 | 大形状 bf16 | fp16 | fp32 | 小形状 |
|---|---|---|---|---|
| `causal_conv1d_fn` | 4.58–9.09×（4340 → 478 µs，1218 → 266 µs） | 4.81–9.51× | 4.51–4.59× | 0.92–1.03× |
| `chunk_state_varlen` | 1.53–2.42× | 1.71–2.59× | 2.43–2.85× | 0.98–1.02× |
| `bmm_chunk` | 1.22–1.23× | 1.22–1.23× | 3.34–3.37× | 0.95–1.07× |

- 逐位一致性在三种 dtype、99 / 137 / 130 个形状上核对（benchmark 形状加上围绕每条规则阈值的形状；causal 的 `bmm_chunk` 只比较它保证的 i ≤ j 部分，原因见「两个会导致非法访存的配置」末尾那段误判）；benchmark 形状和每个分支各连续调用 100 次结果相同。
- 这三个算子在 v2 原型轮就已经发现「只调参数就够」（见「num_warps 扫描」和「`chunk_state_varlen`、`chunk_state`：两个操作数都经 LDS 转置」）；PR 阶段只是把规则按形状阈值写死并补齐边界形状的验证，没有新的改法。`chunk_state_varlen` 剩下的差距仍是两个 dot 操作数都要经 LDS 转置，没有解决。

## LoRA 与访存算子 PR：用 `tl.split`/`tl.join` 换宽写回，再用 `tl.fma` 换回逐位一致

把 `chunked_sgmv_expand`、`gate_up_lora_b`、`sgemm_lora_b`、`per_group_transpose`、`silu_and_mul`、`fused_rmsnorm` 移植到 rdna3 后端时（上游 d2529b3，分支 `feat/amd-gfx1100-lora-memory`，#256，2026-11-02 合入），前三个 LoRA 算子找到一个不用改 Triton 就能拿到宽写回的办法，后三个访存算子只换启动参数。脚本和结果在 `opt_w7900d/pr5/`。

### 不改编译器也能避开 2 字节写回

第一轮查到 AMD 后端的 `optimize_epilogue` 会去掉 WMMA 写回前的布局转换，导致逐元素（2 字节）读写，而这个 pass 没有开关（见「WMMA 写回：跳过 `optimize_epilogue`」）。PR 里换了个办法：**把累加器做一次 `tl.split` / `tl.join` 往返**，这个 pass 就不再认得这个写回模式，于是先把累加器转成普通 blocked 布局，`base_output` 的读和结果的写都变成 16 字节访问。值本身不变。

```
low, high = tl.split(tl.reshape(accumulator, (block_m, block_n // 2, 2)))
accumulator = tl.reshape(tl.join(low, high), (block_m, block_n))
```

代价是**改变了舍入**：原来 `base + accumulator * scaling` 由编译器收缩成一条 FMA，往返之后不再收缩，结果与上游不同。把尾声显式写成 `tl.fma(accumulator, scaling, base)` 就回到逐位一致。按变体逐位比较 41 组输入（`pr5/expand_epilogue.py`，`results/expand_epilogue.log`，列为尾声写法 × `enable_fp_fusion`）：

| 输入 dtype | 加号 + 收缩 | `tl.fma` + 收缩 | 分块 fp32 写法 + 不收缩 | `tl.fma` + 不收缩 |
|---|---:|---:|---:|---:|
| bf16 | 0 / 41 不同 | 0 / 41 | 8 / 41 | 0 / 41 |
| fp16 | 1 / 41 | 0 / 41 | 28 / 41 | 0 / 41 |
| fp32 | 2 / 41 | 26 / 41 | 0 / 41 | 26 / 41 |

所以三条路径的尾声写法并不相同：16 位走 WMMA 时用 `tl.fma`；fp32 不走 WMMA，秩循环本身用 `tl.fma` 累加、尾声用乘加并关掉 `enable_fp_fusion`；`gate_up_lora_b` 只在 `block_k ≥ 16`（即真的走 WMMA）时才显式 `tl.fma`，秩 ≤ 8 时通用版的 dot 不是 WMMA、也不会被收缩，显式写反而不一致。

### 输出宽度不是 16 的倍数时一度不一致

上面的规则定好之后，逐位比较在**输出宽度为奇数**的形状上仍然失败（`pr5/odd_width.py`，`results/odd_width_before.log`，24 组形状）：

| 算子 | bf16 | fp16 | fp32 |
|---|---:|---:|---:|
| `sgemm_lora_b` | 2 / 24 不同 | 17 / 24 | 0 / 24 |
| `gate_up_lora_b` | 3 / 24 | 17 / 24 | 0 / 24 |
| `chunked_sgmv_expand` | 1 / 24 | 12 / 24 | 15 / 24 |

原因是 **Triton 会对能被 16 整除（或等于 1）的整型参数做特化**：输出宽度变化会让 stride 的整除性变化，同一个 kernel 因此编译成不同的版本，收缩与否也跟着变。按上面每条路径分别写死尾声之后，三个算子在这 24 组形状（含秩 8）上全部逐位一致（`results/odd_width_after2.log`）。

### 访存算子：先看离拷贝上限还有多远，再决定改不改

`per_group_transpose`、`silu_and_mul`、`fused_rmsnorm` 只换 tile 和 warp 数，收益有限。改之前先用同样字节数的 `torch.clone` 作参照（`pr5/mbu.py`，`results/mbu.log`），bf16：

| 算子 / 形状 | 搬运字节 | 上游 | 本次 | 本次带宽占峰值 | 拷贝 | 本次 ÷ 拷贝 |
|---|---:|---:|---:|---:|---:|---:|
| `per_group_transpose` (4096, 1024) | 128 MiB | 661.6 µs | 512.5 µs | 30% | 253.6 µs | 0.49 |
| `per_group_transpose` (4096, 128) | 16 MiB | 52.8 µs | 47.5 µs | 41% | 40.6 µs | 0.86 |
| `silu_and_mul` (4096, 4096) | 96 MiB | 323.1 µs | 231.5 µs | 50% | 208.1 µs | 0.90 |
| `silu_and_mul` (512, 4096) | 12 MiB | 38.1 µs | 38.8 µs | 38% | 31.8 µs | 0.82 |

`silu_and_mul` 大形状已到拷贝的 90%，调参数只能再拿 10%；`per_group_transpose` 最大形状只有拷贝的一半，差距在转置本身的访问模式上，换 tile 和 warp 数拿不到（向量化写回的原型也只快 4.5%，见「没有收益或不能用的改法」）。这张表是决定「哪几个算子值得继续调」的依据。

官方 benchmark 的几何平均（三种 dtype 各交替 3 次，运行间波动最大 1.14 倍）：

| 算子 | bf16 | fp16 | fp32 |
|---|---|---|---|
| `chunked_sgmv_expand` | 1.99×（1.63–2.82） | 1.96× | 12.34×（5.77–20.59） |
| `gate_up_lora_b` | 1.23×（1.04–1.76） | 1.17× | 8.36×（1.37–16.85） |
| `sgemm_lora_b` | 1.28×（1.02–2.05） | 1.21× | 1.61× |
| `per_group_transpose` | 1.06×（0.98–1.29） | 1.06× | 1.07× |
| `silu_and_mul` | 1.13×（0.98–1.40） | 1.08× | 1.10× |
| `fused_rmsnorm` | 1.07×（0.95–1.23） | 1.07× | 1.10× |

- fp32 的倍数大，是因为通用版的 fp32 路径在秩循环上溢出寄存器；改成小 tile 的 FMA 之后溢出消失。
- benchmark 以外每个算子 20–44 个形状的几何平均为 1.94× / 1.42× / 1.49× / 1.17× / 1.10× / 1.09×；最低 0.93×，是 `per_group_transpose` 在 8 个大小不等的专家、k = 2048 时——要避开它得先把每个专家的行数从设备读回主机，那次同步比省下的时间还贵。
- 六个算子里只有 `fused_rmsnorm` 不是逐位一致（它改了一行内求和的顺序），其余五个在 benchmark 形状加边界形状（每个算子 63–180 组，含秩 0 的适配器、打乱的置换、非 16 倍数的输出宽度）上与上游逐位相同。

## 注意力 PR 中的改动和发现

把 `decode_attention`、`decode_grouped_attention`、`extend_attention`、`log_scaling_tau` 移植到上游的 rdna3 后端时（基于上游 d2529b3 开发，提交前变基到 7e9943f，分支 `feat/amd-gfx1100-attention`，#271，2026-11-02 合入），v2 包的四个原型都做了改动，另外查到一处上游在 gfx1100 上的崩溃。脚本和结果在 `opt_w7900d/pr6/` 和 `opt_w7900d/pr6/results/`；每次实验前按 KFD 的 `vram_<gpu_id>` 确认整张卡没有其他进程占用，一次只用一张卡，并绑定到所在的 NUMA 节点（`pr6/run.sh`）。

### split-KV 的启动规则：从能否填满 96 个 CU 反推 head 块

v2 按 `batch × KV 头数` 的阈值写死每块的 head 数，这次改成：**先取最大的 head 块（最多 64 行），算上分段后程序数不到 96 就把块减半**，直到 16 行。理由是一块里的 query 头共用该程序读进来的 K/V 分块，块越大 KV 缓存被重复读的次数越少（重复次数 = ⌈G ÷ 块大小⌉），只有在填不满卡的时候才牺牲它去换并行度。扫描见 `pr6/decode_explore.py`、`pr6/splitkv_explore.py`（`results/explore_*`、`results/skv_*`），耗时单位 µs，倍数相对 v2：

| 形状（batch，G，head_dim，序列长度） | v2 的配置与耗时 | 本次的配置与耗时 | 倍数 |
|---|---|---|---:|
| grouped 8，128，128，2048 | 16 行、32 token、16 段：75.6 | 64 行、32 token、8 段：58.4 | 1.30× |
| grouped 64，128，128，512 | 64 行、64 token、不分段：89.7 | 64 行、32 token、不分段：84.5 | 1.06× |
| grouped 512，128，128，128 | 64 行、64 token、不分段：268.7 | 64 行、32 token、不分段：257.7 | 1.04× |
| decode 1，4，128，2048 | 16 行、64 token、16 段：40.0 | 16 行、32 token、16 段：37.6 | 1.07× |
| decode 2，1，80，10 | 16 行、64 token、不分段：12.2 | 16 行、32 token、不分段：11.0 | 1.11× |

- 分段数的上限从「凑到约 512 个程序」降到 192：grouped batch 8 用 8 段比 16 段快 4%（58.4 对 60.8），decode batch 8（64 个程序）切 2 段与不切相差 0.5%。段数仍按 2 的幂取整、每段不少于 128 个 token。
- K/V 分块统一成 32 个 token：64 个只在个别形状上快 0.3%–0.5%（在运行间波动内），但 4 个 warp 下必然溢出——BLOCK_H 16/32/64 配 64 个 token 分别溢出 93/108/142 B，配 32 个 token 时全部为 0。按新规则可达的 bf16 配置现在没有一个溢出。
- 流水级数按程序数定：不超过 64 个程序用 2 级（小形状快 8%），否则 1 级。曾经想按 head 宽度再加一道门限（宽 tile 预取占寄存器），实测 head_dim 256 的形状 2 级反而快（35.2 对 37.6），门限去掉了——这类「看起来合理」的规则必须逐条实测。
- 效果：v2 中 `decode_attention` head_dim 80 的小形状 0.95 倍变成 1.04–1.09 倍，`decode_grouped_attention` batch 512 的 0.98 倍变成 1.01 倍。

### 把合并步骤融进第一步：不值得

分段后第二步只做合并，一次启动约 9–13 µs，batch 1 时占总耗时的 1/4 到 1/2（decode 第一步 33.1 + 合并 10.5；grouped 第一步 19.6 + 合并 12.7；空 kernel 8.3–9.1 µs）。试了常见的「最后到达的那个程序顺手做合并」：每组一个原子计数器，`tl.atomic_add(..., sem="release", scope="gpu")` 之后拿到 `NUM_SPLITS - 1` 的程序用 `volatile` 读回各段结果做合并，计数器由它清零以便复用（`pr6/splitkv.py` 的 `FUSE`）。结果只有两个形状受益——grouped batch 8 从 58.4 到 54.9（+6%）、batch 1 +1%–4%——而 batch 512、4096 反而是 0.92–0.96 倍，因为合并的工作被串到了最后一个程序的关键路径上。收益不抵跨 workgroup 内存序的假设和一块常驻计数器缓冲，没有采用。

### float32：上游的 `decode_attention` 在 gfx1100 上直接崩

上游 `decode_attention` 按形状而不是按 dtype 分流，`(H_Q 32, H_KV 8, head_dim 128)` 一律走 `_launch_dot`；fp32 下这个 kernel 要 69632 B LDS，超过 gfx1100 的 65536 B，抛 `OutOfResources`。单测和 benchmark 都只列 bf16，所以上游没有暴露：官方 benchmark 的 fp32 基线三次运行全部失败，九个形状一个都没跑出来。rdna3 版本不用这个 kernel，fp32 全部九个形状都能跑（133.8–6260.8 µs），误差 2e-7–2e-6。

split-KV 自己的 fp32 路径（两次 `tl.dot` 都加 `input_precision="ieee"`，走 FMA）只在 query 组够大时划算（`pr6/fp32_decode.py`，`results/fp32_*`）：

| 算子 | G | fp32 split-KV 相对不分块 kernel |
|---|---:|---|
| `decode_grouped_attention` | 128 | 3.15–15.0 倍（benchmark 形状） |
| `decode_attention` | 4 | 0.34–0.77 倍（除 batch 1 的 1.07 倍） |

G = 4 时 16 行的 tile 里 12 行是填充，FMA 把这些填充也算了一遍。所以规则是 **fp32 只在 G ≥ 16 时用 split-KV**，否则走不分块的 kernel——这同时绕开了上面的崩溃。fp32 下 BLOCK_H 一律取 16，更大的块只会带来更多填充行。另外 v2 给不分块 kernel 加的「head_dim ≥ 512 用 2 个 warp」只对 16 位成立：fp32 下每条 lane 的数据量翻倍，同一形状只有 0.68 倍，加上 dtype 判断后回到 1.02 倍。

### `extend_attention`：输出不先清零，float32 把 tile 减半

- 上游用 `torch.zeros_like(q_extend, dtype=torch.float32)` 分配输出，而 kernel 随后会写满它。fp32 输出在 benchmark 规模下是 16–270 MB，这一次清零实测 16.5 µs（最小形状）到 96 µs（batch 8），相当于再过一遍 KV 缓存那么大的内存。packed kernel 覆盖每一个 (token, head)，改成 `torch.empty_like` 后净快 2%–21%（回退路径仍用 `zeros_like`）。
- fp32 也走 packed kernel（`input_precision="ieee"`），但最优 tile 与 16 位不同：16 位走矩阵单元，64 行的 tile 能喂饱；fp32 走 FMA，寄存器成了瓶颈，要把 tile 减半、warp 加倍（`pr6/extend_fp32.py`，`results/extend_fp32.log`，每个配置单独起一个进程）：

| block_m × block_n，warp 数 | fp32 溢出 | bf16 溢出 | fp32 (256, 256, 32, 8, 128) | fp32 (2, 128, 8, 8, 64) |
|---|---:|---:|---:|---:|
| 64 × 32，4（16 位用的配置） | 1025 B | 31 B | 1.81× | 0.89× |
| 64 × 32，8 | 723 B | 0 B | 2.15× | 4.70× |
| 32 × 32，4 | 733 B | 0 B | 2.20× | 5.75× |
| **32 × 32，8（采用）** | **216 B** | 0 B | **2.43×** | 5.68× |
| 64 × 64，8 | 985 B | 32 B | 1.46× | 1.05× |

### `log_scaling_tau`：一行切成 512 个元素，每条 lane 一个 32 位字

v2 只改了 warp 数（grid ≥ 4096 时 16 个）。这次把上游的两个 kernel 合成一个：每个程序取一行里 512 个元素的一段，warp 数按「每条 lane 恰好搬一个 32 位字」定（bf16/fp16 8 个，fp32 16 个），相邻程序取同一行的相邻段，读取带 `.cg` 绕过 CU 私有缓存。收益只出现在工作集超出缓存之后（`pr6/tau_explore.py`，`results/tau_*`，fp16）：

| 形状 | 元素数 | 搬运字节 | 上游 | 本次 | 上游带宽 | 本次带宽 |
|---|---:|---:|---:|---:|---:|---:|
| (1, 1, 128) 等四个小形状 | ≤ 66K | ≤ 264 KB | 9.2–9.4 µs | 9.2–9.7 µs | — | — |
| (128, 16, 128) | 0.26M | 1.0 MB | 10.7 µs | 10.6 µs | 98 GB/s | 99 GB/s |
| (2048, 16, 128) | 4.2M | 16.8 MB | 43.7 µs | 41.2 µs | 384 GB/s | 407 GB/s |
| (4096, 16, 128) | 8.4M | 33.6 MB | 91.8 µs | 72.2 µs | 365 GB/s | 464 GB/s |
| (8192, 16, 128) | 16.8M | 67.1 MB | 210.5 µs | 143.2 µs | 319 GB/s | 469 GB/s |
| (16384, 16, 128) | 33.6M | 134 MB | 362.6 µs | 252.5 µs | 370 GB/s | 532 GB/s |

四个小形状总共只搬 0.5 KB–264 KB，按 460 GB/s 算需要 0.001–0.6 µs，整个耗时都是 kernel 启动，写法无关。8M 元素以上上游只到峰值带宽的 70%–75%，这正是收益的来源。本次实现与上游逐位一致。

(2048, 16, 128) 是唯一一个两种计时给出相反结论的形状：这张表的冷计时（每次清掉缓存）下本次快 6%，而官方 benchmark 的热计时下 fp16 慢 3%（39.7 对 41.2 µs）、bf16 快 1.20 倍（通用实现的 bf16 要 49.4 µs）——两版在这个规模都已贴着拷贝带宽，缓存里留着上一次的数据时上游的宽 tile 略占便宜。试过按数据量分流回宽 tile，但同样元素数的 (4096, 8, 128) 反而偏好窄片（快 8%），没有自洽的规则，保持单一实现。

### 方法上的两条

- **先只编译、不启动，筛掉危险配置。** `kernel.warmup(..., grid=(1,))` 加 `_init_handles()` 就能拿到 `n_regs`、`n_spills` 和 LDS，不需要真的启动（`pr6/spills.py`、`pr6/final_spills.py`）。扫 `extend_attention` 的 tile 时，`block_m 32 / block_n 128` 报非法访存并污染整个上下文——之后同一进程里连 `torch.cuda.empty_cache()` 都抛错，后面几十个配置的结果全是垃圾，只能一个进程测一个配置。事后回查：出事的维度是 128 个 token 的 K/V 分块，不是溢出量本身（那个配置 bf16 只溢出 163 B，而最终采用的 fp32 配置溢出 216 B，10200 次调用没有问题），与「两个会导致非法访存的配置」一节的结论一致。
- **输出用 `torch.empty` 的算子，验证时要给缓冲投毒。** 改成不清零之后，某一行没被写到本应立刻暴露，但 PyTorch 的缓存分配器很可能把上一次调用刚释放的同尺寸块原样发回来，里面正好是上次的正确结果，检查就看不出来。`pr6/validate.py` 在每次调用前先分配一个同尺寸的 NaN 张量再释放，让 `empty_like` 拿到被污染的块。

### 正确性和官方 benchmark

单测 `tests/test_{decode_attention,decode_grouped_attention,extend_attention,log_scaling_tau}.py` 共 32 个用例，rdna3 与 `ARCH=none` 都通过。benchmark 形状加上每个算子 10–14 个围绕各条启动规则阈值的形状（单个 token、短于一个 K/V 分块的序列、不整除的切分、非 2 的幂的 head 数和宽度、每个 CU 一个序列及再多一个、2–9 个 head 的 query 组、分块支持的最宽 head 及再宽一档），三种 dtype 下最大容差占比 0.31；每个用例连续调用 60 次（`extend_attention` 200 次，含 fp32，合计 10200 次）结果逐位相同。`log_scaling_tau` 与上游逐位一致。

官方 benchmark（`pr6/bench.sh`，上游 `ARCH=none` 与 rdna3 交替各 3 次、绑 NUMA、取中位数；fp16/fp32 用本地扩展 dtype 列表的 benchmark 副本）：

| 算子 | bf16 | fp16 | fp32 |
|---|---|---|---|
| `decode_attention` | 1.04–3.44×（几何平均 1.24×） | 1.01–3.28×（1.23×） | 上游崩溃，本次 11.1–6260.8 µs |
| `decode_grouped_attention` | 1.01–5.43×（1.69×） | 1.53–62.74×（10.41×） | 1.01–15.00×（2.90×） |
| `extend_attention` | 1.36–2.67×（1.84×） | 1.34–2.55×（1.76×） | 1.69–5.79×（2.74×） |
| `log_scaling_tau` | 0.97–1.51×（1.09×） | 0.95–1.33×（1.03×） | 0.96–1.35×（1.04×） |

- `decode_grouped_attention` 的 fp16 倍数大，是因为上游的分块路径只认 bfloat16，fp16 和 fp32 都落到逐 token 的 kernel（batch 4096 的 fp16 上游要 73.2 ms，本次 1.17 ms）。
- 低于 1.00 的行都是 `log_scaling_tau`：9–10 µs 的启动开销，以及上面说的 (2048, 16, 128) 的 fp16、fp32（bf16 下这个形状快 1.20 倍，因为通用实现的 bf16 在这里要 49.4 µs，比它自己的 fp16 慢）。
- `log_scaling_tau` 的 benchmark 只声明了 fp16，第一次跑官方 benchmark 时没加 `--dtypes`，标成 bf16 的那一列其实是 fp16；表中的 bf16 是后来用扩展 dtype 列表的副本、交替 5 次重测的（见下面的稳定性复测）。
- benchmark 以外的形状（每个算子 10–14 个，bf16）几何平均 1.66×、3.25×、2.33×（fp32 为 4.23×）、1.00×；最低 0.81×，是 `decode_grouped_attention` 在 4 个 head 一组、每序列只有一个 token 时——tile 大部分是填充，而上游 kernel 是一个 token 一个程序。

### 稳定性复测

提交前在同一张卡（GPU 4）上把官方 benchmark 重跑了一遍：上游与 rdna3 交替 5 轮、三种 dtype 全部用扩展 dtype 列表的副本加 `--dtypes` 指定，共 120 次 pytest；被测代码的校验和记在 `pr6/results/stab_code_sha256.txt`，每轮的结果配对比较（`pr6/stability.py`，`results/stab_summary.txt`）。下表的加速比是各形状 5 轮配对比值的中位数再取几何平均，与上面官方表（中位数耗时之比）的算法略有不同，`log_scaling_tau` bf16 因此是 1.10× 对 1.09×。

| 算子 | bf16 | fp16 | fp32 | 5 轮 × 全部形状中最低的单轮 | rdna3 耗时的变异系数 | 各形状中位数与 10-30 那次相比 |
|---|---:|---:|---:|---:|---:|---:|
| `decode_attention` | 1.24× | 1.23× | 上游崩溃 | 0.97× | ≤ 3.2% | ≤ 2.8% |
| `decode_grouped_attention` | 1.70× | 10.37× | 2.91× | 0.97× | ≤ 1.9% | ≤ 2.4% |
| `extend_attention` | 1.84× | 1.76× | 2.71× | 1.32× | ≤ 1.6% | ≤ 2.4% |
| `log_scaling_tau` | 1.10× | 1.03× | 1.04× | 0.88× | ≤ 5.5% | ≤ 10.1% |

- 三个注意力算子的大形状每一轮都领先，且范围很窄（bf16 下 decode batch 1 为 3.40–3.46×，grouped batch 1 为 5.37–5.47×，extend [256, 256, 32, 8, 128] 为 2.62–2.66×），rdna3 耗时的变异系数不超过 0.8%，没有任何一轮跌破 1.0（最低的是 grouped batch 512，1.01×）。单轮低于 1.0 的只有三个 11–28 µs 的小形状，5 轮中位数都是 1.02×。
- `log_scaling_tau` 的真实收益同样稳定：(8192, 16, 128) 为 1.51× / 1.33× / 1.34×（bf16 / fp16 / fp32），变异系数 0.1%–0.2%；(2048, 16, 128) 在 fp16、fp32 下的 0.96×、0.94× 每轮都一样，是稳定的小退步而不是噪声。9 µs 的小形状 5 轮在 0.88–1.08× 之间跳：其中 (1, 1, 128) 新旧两版的启动配置完全相同（1 个程序、2 个 warp），照样在 0.90–1.07× 之间跳，所以这些波动是启动开销本身的噪声，不是代码造成的；判断 10 µs 以下的形状要靠多轮中位数，单轮的比值不可信。

## 小形状：以固定开销为主

| GPU | 计时下限（最短的算子延迟） | 延迟 ≤ 1.5 × 下限的形状 | 延迟 ≤ 2 × 下限的形状 |
|---|---:|---:|---:|
| RTX 4090 | 3.07 µs | 158 | 240 |
| W7900D | 8.80 µs | 238 | 274 |

这部分时间主要是 kernel 启动和计时的固定开销，与 kernel 写得好坏关系不大。能做的是：合并一次调用中的多个 kernel（如 `chain_speculative_sampling` 在缓存中有 11 种 kernel，最小的形状 W7900D 上也要 75.12 µs，4090 26.62 µs）；去掉主机同步（见下文）；在 SGLang 里用 HIP graph 减少 CPU 端的启动开销。另外，溢出超过 scratch 门槛的 kernel 即使形状很小，每次也要多花 150–200 µs，这部分不是固定开销，见上文。`context_attention` 的 v2 写法同时去掉了主机同步和超过门槛的溢出，4 个小形状从 76–285 µs 降到 12.48–18.28 µs，是 4090 的 0.36–0.52 倍。

## 访存型算子：带宽利用率

各算子取最大的形状，按理论最小访存量估算达到峰值带宽的比例；「改后」是 v2 包官方 benchmark 的值，标「参数搜索」的是原型中最快的分块（收益小，没有放进 v2 包）：

| 算子（形状） | 最小访存量 | RTX 4090 | W7900D 现在 | W7900D 改后 |
|---|---|---:|---:|---:|
| `softcap_inplace_logits`（[512, 128256]） | 525 MB，原地读、写 fp32 | 93% | 87% | - |
| `decode_attention`（batch 4096，KV 524288 个 token） | 2215 MB，读 q 和 K、V 缓存、写输出 | 91% | 69% | 82%（v2 包） |
| `moe_sum_reduce`（[4096, 8, 7168]） | 528 MB，读 [T, k, H]、写 [T, H] | 88% | 65% | 67%（参数搜索） |
| `gelu_and_mul`（[4096, 16384]） | 201 MB，读 [bs, 2d]、写 [bs, d] | 91% | 61% | 61%（参数搜索） |
| `softcap_out`（[512, 128256]） | 394 MB，读 bf16、写 fp32 | 88% | 60% | 69%（参数搜索） |
| `decode_grouped_attention`（batch 4096，KV 524288 个 token） | 537 MB，读 q 和 K、V 缓存、写输出 | 74% | 50% | 53%（v2 包） |
| `silu_and_mul`（[4096, 16384]） | 201 MB，读 [bs, 2d]、写 [bs, d] | 87% | 49% | 59%（v2 包） |
| `fused_rmsnorm`（[4096, 8192]） | 134 MB，读、写 [bs, h] | 90% | 45% | 56%（v2 包） |
| `per_group_transpose`（[8192, 4096]） | 134 MB，读 [M, K]、写转置 | 73% | 24% | 33%（v2 包） |
| `causal_conv1d_fn`（dim 4096，16384 个 token） | 268 MB，读 x、写 out | 82% | 7% | 65%（v2 包） |

- 能跑满的例子说明问题不全在硬件：同样是 Triton 生成的代码，`softcap_inplace_logits` 在 W7900D 上达到 87%，`decode_attention` 改后到 82%。但同样数据量的拷贝只有 70%（读写共 256 MiB）到 84%（1 GiB），表中 134–528 MB 的算子改后大致也就在 55%–70%，再往上的空间不大；`per_group_transpose`（33%）和 `decode_grouped_attention`（53%）是例外，还有明显空间。`per_group_transpose` 向量化写回只再快 4.5%（见「第二轮」），瓶颈不只在写回。
- `causal_conv1d_fn` 改后从 7% 升到 65%，读写仍是 2 字节标量访存，收益不来自访存宽度，与 R9700 文档的结论相同。
- `decode_attention` 上游 69%、`decode_grouped_attention` 上游 50%，比 4090（91%、74%）低得多；R9700 文档中两者的带宽归一化效率是 0.95 和 1.17（W7900D 0.66、0.68）。写回不是原因（改成 16 字节向量后没有变化）。batch 小时是网格太小（batch 1 时 `decode_attention` 只有 8 个程序），batch 大时上游的 8 个 warp 不如 4 个；v2 包改成 split-KV、4 个 warp，带宽归一化效率到 0.92 和 1.13（见「第二轮」）。`decode_grouped_attention` 最大形状只到 53%，剩下的差距没有再查。这里的最小访存量包括 q 和输出；R9700 文档的 `decode_attention` 只算 K、V 缓存（2147 MB），按那个口径 4090 88%、W7900D 上游 66%、v2 包 79%。

## 矩阵乘类算子

| 算子（形状） | 运算量 | RTX 4090 | W7900D 现在 | W7900D 改后 |
|---|---|---:|---:|---:|
| `context_attention`（64 条 × 2048 token，fp32） | 2199 GFLOP | 14.6 TFLOPS（TF32 峰值的 18%） | 2.3 TFLOPS（FP32 峰值的 4.4%） | 43.1 TFLOPS（v2 包，FP16 矩阵峰值的 41%）；第一轮 6.7 TFLOPS（fp16 WMMA）、2.6 TFLOPS（只改分块） |
| `fused_moe_router_tensorcore`（[4096, 4096] × 256 个专家） | 8.6 GFLOP | 343.2 µs（tf32x3） | 3780.5 µs，参考实现 2871.5 µs | 328.4 µs（v2 包，bf16 WMMA，分块求和） |
| `fused_moe_gemm`（T 4096、topk 2、N 4096、K 4096） | 275 GFLOP | BF16 峰值的 3.0% | BF16 峰值的 3.6% | BF16 峰值的 63%（v2 包，4107 µs） |
| `w8a8_block_int8_matmul`（M 4096、N 7168、K 4096） | 241 GOP | INT8 峰值的 42% | INT8 峰值的 47% | INT8 峰值的 64% |
| `fused_moe_router_cudacore`（[83, 512] × 32 个专家） | - | 13.66 µs | 261.6 µs | 17.92 µs |

- `context_attention`：benchmark 的 10 个形状都是 fp32 输入。`tl.dot(q, k)` 没有指定精度，4090 上默认用 TF32 张量核，AMD 上只能 ieee，编译成 FMA，head_dim > 64 时 32×32 的 tile 配 4 个 warp，寄存器到上限 256 还每条 lane 溢出 2.6 KB，超过 scratch 门槛。第一轮改用 fp16 WMMA 后最大形状仍是 4090 的 2.2 倍（只到 FP16 矩阵峰值的 6.4%）；第二轮把 K/V 的转换移出 kernel、改网格顺序、对角线以下的块不加 mask 后到 41%，比 4090 快 3.0 倍。
- `fused_moe_router_tensorcore`：参考实现 `x.float() @ W.float().t()` 在 W7900D 上同样慢（2871.5 µs），改调 PyTorch 矩阵乘没有用。改用 bf16 WMMA、分块求和后与 4090 持平，测试也通过；gfx1100 上 268 个 autotune 候选有 228 个溢出（最多 39776 字节），24 个需要的 LDS 超过 64 KiB。
- `fused_moe_gemm`：每个 (token, slot, N 块) 一个程序，用 `tl.sum(a[:, None] * b)` 而不是 `tl.dot`，每个 token 都要把所属专家的整块权重重读一遍。第二轮改成按专家分组的 GEMM（先按专家排序，再用 `tl.dot`），T=4096 快 17.18 倍，比 4090 上的上游快 13.5 倍。
- `w8a8_block_int8_matmul`：WMMA int8 指令用上了，改分块消除溢出后到 INT8 峰值的 63%–64%，比 4090（42%）还高。
- `fused_moe_router_cudacore`：见上文，改用 4 个 warp 后接近 4090。

## 编译结果中的寄存器溢出

每格为本机测试缓存中的版本数；「超过 scratch 门槛」指每条 lane 溢出超过 1493 字节、每次启动都要重新分配 scratch 的版本：

| kernel（算子） | 溢出的版本 | 每条 lane 溢出 | 寄存器最多 | 占用率最低 | 超过 scratch 门槛 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| `_router_gemm_kernel`（`fused_moe_router_tensorcore`） | 228 / 268 | 168–39776 B | 256 | 5 | 144 | autotune 候选，提升到 fp32 后的 GEMM 编译成 FMA；24 个候选需要的 LDS 超过 64 KiB；实测选中的版本在 2 个调优键上溢出 380–448 B |
| `_silu_and_mul_masked_row_kernel`（`silu_and_mul_masked`） | 12 / 171 | 2616–10408 B | 256 | 5 | 12 | autotune 候选；实测选中的版本都不溢出 |
| `_context_attention_fwd_kernel`（`context_attention`） | 7 / 7 | 1032–2648 B | 256 | 5 | 6 | fp32 `tl.dot` 编译成 FMA，没有 WMMA；head_dim 13 的形状不受门槛影响（调高门槛后 1.01 倍） |
| `_silu_and_mul_masked_col_kernel`（`silu_and_mul_masked`） | 6 / 246 | 2620 B | 256 | 5 | 6 | autotune 候选；实测选中的版本都不溢出 |
| `_router_kernel`（`fused_moe_router_cudacore`） | 1 / 3 | 2056 B | 256 | 5 | 1 | 专家数 ≤ 32 的单 kernel 路径，固定 num_warps=1；溢出的是 32×512 tile（[83, 512] × 32 个专家） |
| `_validated_gather_rank_loop_words`（`chunked_embedding_lora_a`） | 1 / 1 | 1764 B | 256 | 5 | 1 | 唯一的版本就溢出 |
| `_w8a8_int8_dot`（`w8a8_block_int8_matmul`） | 12 / 18 | 212–1564 B | 256 | 5 | 1 | 溢出的是 BLOCK_M=64 的版本，即 M ≥ 64 的形状；超过门槛的是 K=512 的版本 |
| `_selective_state_update_fulltile_kernel`（`selective_state_update`） | 2 / 50 | 1164 B | 256 | 5 | 0 | autotune 候选；两个小调优键选中了它 |
| `_chunk_state_varlen_grouped_kernel`（`chunk_state_varlen`） | 2 / 5 | 372–376 B | 256 | 5 | 0 | WMMA 版本 |
| `_extend_attention_kernel`（`extend_attention`） | 4 / 5 | 340 B | 256 | 5 | 0 | WMMA 版本 |
| `_sgemm_lora_a_stage1`（`sgemm_lora_a`） | 1 / 5 | 300 B | 256 | 5 | 0 | WMMA 版本 |
| `_gate_up_lora_b_kernel`（`gate_up_lora_b`） | 1 / 5 | 68 B | 256 | 5 | 0 | WMMA 版本 |
| `_sgemm_lora_b_kernel`（`sgemm_lora_b`） | 1 / 6 | 36 B | 256 | 5 | 0 | WMMA 版本 |
| `_state_passing_kernel`（`state_passing`） | 4 / 45 | 8–28 B | 256 | 5 | 0 | autotune 候选；实测选中的版本都不溢出 |

与 R9700 文档 W7900D 一列（离线编译）的数字一致；R9700 文档中未编译的三个 autotune kernel 在这里有了实机数据。

## 主机同步

| 算子 | 位置 | RTX 4090 最小形状 | W7900D 最小形状 | W7900D 去掉同步后 |
|---|---|---:|---:|---:|
| `context_attention` | `context_attention.py` 第 231 行 `int(b_seq_len.max().item())` | 34.78 µs | 76.13 µs | 22.96 µs（第一轮原型），12.48 µs（v2 包，同时改了网格和 tile） |
| `qkv_lora_b` | `qkv_lora_b.py` 第 149 行 `int(seg_lens.max())` | 41.98 µs | 78.97 µs | 没有测 |

这些值只是用来定网格大小。`context_attention` 已经有 `max_input_len` 参数，同步取真实最大值只是防止调用方传入偏小的值，可以直接信任这个参数；也可以按上界启动、在 kernel 里判断越界。v2 包按 `max_input_len` 定网格，每个程序循环处理多个 query 块，传入的值偏小也能算完。W7900D 上这一次同步大约 50 µs，比 4090 的整个调用还长。

## 逐算子：大形状的带宽归一化效率

只列 4090 上有不少于 50 µs 形状的 44 个算子，按 W7900D 的值从低到高排序；计算型算子（`context_attention`、`fused_moe_router_tensorcore`、`w8a8_block_int8_matmul`、`fused_moe_gemm` 等）只看排序。「改后」是 v2 包官方 benchmark 的值；`qkv_lora_b`、`chunk_state` 两行是写回实验（跳过 `optimize_epilogue`）的值，没有放进 v2 包。

| 算子 | 大形状数 | W7900D | R9700（R9700 文档） | W7900D 改后 | 主要发现 |
|---|---:|---:|---:|---:|---|
| `causal_conv1d_fn` | 2 | 0.12 | 0.33 | 0.77 | 4×128 的小 tile，错位读取编译成 2 字节标量访存；加大 BLOCK_T 快 4.54–9.35 倍 |
| `fused_moe_router_tensorcore` | 2 | 0.12 | 0.20 | 1.11 | bf16 输入提升成 fp32 后按 ieee 计算、编译成 FMA；改用 bf16 WMMA、分块求和快 7.76–11.01 倍 |
| `context_attention` | 6 | 0.13 | 0.15 | 1.34 | fp32 `tl.dot` 编译成 FMA，head_dim > 64 的版本溢出 2.6 KB、超过 scratch 门槛；封装里有主机同步；网格中 query 块最慢。fp16 WMMA、K/V 先转成 fp16、改网格、去掉同步后快 4.64–18.53 倍；bf16x3 更慢 |
| `w8a8_block_int8_matmul` | 6 | 0.19 | 0.66 | 0.28 | BLOCK_M=64 的版本溢出 480–1564 B；BLOCK_M ≤ 32、2 级流水快 1.35–1.50 倍，到 INT8 峰值的 64% |
| `chunk_state_varlen` | 1 | 0.26 | 0.64 | 0.39 | 两个 dot 操作数都经 LDS 转置，每条 WMMA 指令对应 8–32 条 2 字节 LDS 读；每块 1 个头、BLOCK_K 16、1 级流水快 1.51 倍（2048 个 token 的形状 2.42 倍） |
| `gate_up_lora_b` | 1 | 0.31 | 0.68 | 0.49 | 写回是 2 字节标量写；16 个 warp 快 1.59 倍（经 LDS 写回 1.72 倍） |
| `chain_speculative_sampling` | 3 | 0.32 | 0.49 |  | 一次调用要执行多个 kernel，含按顺序认领、发布的步骤 |
| `per_group_transpose` | 1 | 0.32 | 0.57 | 0.45 | 转置写回是 2 字节标量写；8 个 warp 快 1.42 倍，向量化写回只再快 4.5% |
| `chunked_sgmv_expand` | 1 | 0.41 | 0.70 | 0.73 | 2 字节标量读写；16 个 warp 快 1.79 倍（跳过 `optimize_epilogue` 1.89 倍） |
| `extend_attention` | 4 | 0.43 | 0.87 | 0.67 | 前缀先由单独的 kernel gather（只到峰值带宽的 4%–21%），每个 query 头各读一遍 K/V；去掉 gather、合并共享 KV 头的 query 头后快 1.28–2.20 倍。写回是 4 字节标量写，经 LDS 转换后反而更慢 |
| `merge_state` | 1 | 0.45 | 0.70 |  |  |
| `bmm_chunk` | 2 | 0.46 | 0.67 | 0.56 | 写回是 2 字节标量写；1 个 warp 快 1.22–1.24 倍（经 LDS 写回 1.08–1.10 倍） |
| `fused_rmsnorm` | 2 | 0.48 | 0.69 | 0.59 | 每行一个程序、num_warps=2，大行宽时寄存器 212 个、占用率 7；16 个 warp 快 1.21–1.22 倍 |
| `silu_and_mul` | 2 | 0.50 | 0.66 | 0.64 | 大形状走按行循环的 kernel，网格最多 512 个程序；改用二维 kernel、16 个 warp 快 1.21–1.39 倍 |
| `log_scaling_tau` | 1 | 0.50 | 0.67 | 0.59 | 程序数多时 16 个 warp 快 1.18 倍 |
| `fla_layernorm_gated` | 1 | 0.51 | 0.67 |  |  |
| `mrope_fused` | 1 | 0.51 | 0.74 |  |  |
| `ernie45_rope_fused` | 1 | 0.51 | 0.75 |  | 有 2 字节标量读 |
| `silu_and_mul_masked` | 1 | 0.53 | 0.69 |  | autotune 候选中有溢出 2.6–10.4 KB 的版本，实测选中的版本不溢出 |
| `per_token_quant_int8` | 2 | 0.54 | 0.72 |  |  |
| `qkv_lora_b` | 2 | 0.54 | 0.77 | 0.57 | 封装里有主机同步；2 字节标量读写，经 LDS 写回只快 1.05–1.07 倍 |
| `state_passing` | 2 | 0.55 | 0.82 |  | autotune 候选中有少量溢出版本，实测选中的版本不溢出 |
| `mamba_layernorm_gated` | 1 | 0.56 | 0.70 |  | 有 2 字节标量读写 |
| `per_token_group_quant_int8` | 2 | 0.56 | 0.70 |  |  |
| `embedding_lora_a` | 1 | 0.57 | 0.53 |  | 按行 gather，有 2 字节标量读 |
| `chunk_local_cumsum_vector` | 2 | 0.59 | 0.80 |  |  |
| `moe_sum_reduce` | 2 | 0.60 | 0.68 |  | 45 种分块最多快 1.09 倍 |
| `fused_norm_rope_stacked` | 2 | 0.61 | 0.81 |  |  |
| `fused_dual_residual_rmsnorm` | 2 | 0.61 | 0.84 |  |  |
| `moe_fused_mul_sum` | 2 | 0.62 | 0.74 |  |  |
| `gelu_and_mul` | 2 | 0.62 | 0.74 |  | 每个程序只算 256 个元素，读写只有 4 字节；15 种分块最多快 1.03 倍 |
| `softcap_out` | 3 | 0.63 | 0.87 |  | 16 种分块只在最大形状上快 1.16 倍 |
| `fused_moe_router_cudacore` | 2 | 0.63 | 0.85 | 0.63 | 大形状没有问题；MoE PR 中改写专家数 > 32 的 logits kernel 后，这两个形状快 1.22–2.07 倍（三种输入，见「MoE PR 中的 `fused_moe_router_cudacore`：专家数 > 32 的路径」）；单 warp 处理 32×512 tile 的版本溢出 2056 B，[83, 512] 改用 4 个 warp 快 14.50 倍 |
| `rotary_embedding` | 2 | 0.64 | 0.79 |  |  |
| `apply_token_bitmask` | 3 | 0.64 | 0.79 |  |  |
| `decode_attention` | 5 | 0.66 | 0.95 | 0.92 | 写回改成 16 字节向量后没有变化；batch 1 时只有 8 个程序，batch 大时 8 个 warp 不如 4 个；split-KV、4 个 warp 快 1.08–3.23 倍 |
| `chunk_scaled_dot_kkt` | 2 | 0.67 | 0.85 |  | storeopt kernel 写回是 4 字节标量写 |
| `decode_grouped_attention` | 6 | 0.68 | 1.17 | 1.13 | 写回改成 16 字节向量后没有变化；split-KV 在 batch 1、8 上快 5.49、1.99 倍，batch 512 慢 2%；head_dim 512 用 1 个 warp 快 1.70 倍 |
| `chunk_state` | 2 | 0.68 | 0.87 | 0.74 | 写回是 4 字节标量写；经 LDS 写回快 1.03–1.11 倍 |
| `selective_state_update` | 2 | 0.71 | 0.88 |  | 两个小调优键选中溢出 1164 B 的版本（低于门槛） |
| `hc_head` | 2 | 0.73 | 0.96 |  |  |
| `draft_topk1` | 2 | 0.73 | 0.81 |  |  |
| `fused_moe_gemm` | 5 | 0.74 | 0.70 | 2.93 | 逐 token 的 GEMV，不用矩阵单元（三张卡都慢）；按专家分组的 GEMM 在 T ≥ 8 时快 2.00–17.18 倍 |
| `softcap_inplace_logits` | 3 | 0.87 | 0.84 |  | 最大形状达到峰值带宽的 87% |

## 与 R9700 文档的对照

- 一致的部分：R9700 文档中 W7900D 一列的带宽归一化效率、溢出字节数（`context_attention` 1032–2648 B、`_router_kernel` 2056 B、`chunked_embedding_lora_a` 1764 B、`w8a8_block_int8_matmul` 212–1564 B 等）、WMMA 写回宽度，与本机实测和实机编译的结果相同。
- `context_attention`：R9700 上的改法 bf16x3 在 W7900D 上反而慢（大形状 0.44–0.69 倍）；W7900D 上要用 fp16 单次 WMMA（2.55–3.59 倍）或只改 FMA 的分块（1.10–2.21 倍）。合入的 #199 对 fp32 输入保持 fp32（benchmark 大形状 1.23–2.98 倍），用的是 32×32 或 16×16 的 FMA 分块——第一轮的 32×16 在新 kernel 里偶尔会算错整块输出，见「第一个 PR（#199）」；fp16、bf16 输入用 WMMA。这不能用两代 WMMA 的峰值差别解释（W7900D 的 BF16 矩阵峰值是 R9700 的 0.55 倍）：W7900D 上上游和 bf16x3 都远没到峰值，每条 lane 都溢出 2.3–2.6 KB 寄存器。第二轮的改动（去掉主机同步、网格中 query 块最快、对角线以下的块不加 mask、K/V 在 kernel 之外先转换）要在溢出降下来之后才见效：PR 早先版本的 bf16 输入路径保留了前三项，但 dot 仍按 fp32 FMA 算、每条 lane 溢出 2616 字节，大形状只快 0.98–1.33 倍。所以这些改动在 bf16x3 的写法上未必有效，R9700 上没有测。
- `fused_moe_gemm`：R9700 文档建议改成按专家分组的 GEMM，按 INT8 GEMM 达到的峰值比例推算 R9700 上最大形状可快约 22 倍；W7900D 上实测最大形状快 17.18 倍，到 BF16 峰值的 63%。
- `decode_attention`、`decode_grouped_attention`、`extend_attention`、`chunk_state_varlen`：R9700 文档中这几个的带宽归一化效率是 0.64–1.17，W7900D 只有 0.26–0.68，R9700 文档只列了溢出和写回宽度。W7900D 上查到的主要原因不在这两点：decode 是网格太小和 warp 数，`extend_attention` 是单独的前缀 gather 和每个 query 头各读一遍 K/V，`chunk_state_varlen` 是两个 dot 操作数都经 LDS 转置、逐个元素读回。改后分别到 0.92、1.13、0.67、0.39。其中 decode 和 `extend_attention` 的改法是算法上的，R9700 上应同样有效；LDS 转置是否也出现在 gfx1201 上没有核对。
- LoRA 的三个算子和 `bmm_chunk`：R9700 文档把 gfx1100 上的差距归于 WMMA 结果布局导致的 2 字节标量写回；W7900D 上跳过 `optimize_epilogue` 确实快 1.08–2.04 倍，但只改 num_warps（LoRA 16 个、`bmm_chunk` 1 个）也能快 1.10–1.88 倍，不用改编译器。
- `fused_moe_router_tensorcore`：分块求和在 W7900D 上通过 M=4096 的用例，全部 127 个候选都在容差内（R9700 上 119 个中 42 个超出，那次选中的候选为 1.01）；bf16x3 的候选也全部通过（R9700 上 111 个中 39 个超出）。可能的原因是 W7900D 上的 fp32 参考离 fp64 更近（用掉 93% 的容差，R9700 97%）。R9700 文档建议的「参考值改用 fp64」仍然有必要，余量只有 4%。
- `causal_conv1d_fn`：同一个规则，W7900D 上收益更大（4.69–9.19 倍，R9700 2.24–4.07 倍），在两张卡上都正好是 54 种配置中最快的。
- `w8a8_block_int8_matmul`、`fused_moe_router_cudacore`、`fused_rmsnorm`：R9700 文档给出的方向（gfx1100 上减小 BLOCK_M、用 2 级流水；4 个 warp；按行宽加大 num_warps）在 W7900D 上实测有效，v2 包官方 benchmark 中分别快 1.35–1.50 倍（K=512 的形状 12.88 倍）、14.50 倍、1.21–1.22 倍。cudacore 只减小 block_h 不够（256 时仍溢出 1368 B，57.0 µs）。
- `silu_and_mul`：R9700 文档认为读取已是 16 字节向量，差距应来自分块、num_warps、每个程序处理的行数这类参数；W7900D 上实测如此：大形状走的是按行循环的 kernel（网格最多 512 个程序），改用二维 kernel、16 个 warp 快 1.21–1.36 倍。
- `per_group_transpose`：R9700 文档建议经 LDS 转置后向量化写回（没有测）；W7900D 上只调 warp 数，最大形状快 1.40 倍，其他形状用 8 个 warp 反而变慢。第二轮试了向量化写回（凑成 16 字节），最大形状只比 8 个 warp 再快 4.5%，也只到峰值带宽的 34%。
- `gelu_and_mul`：两张卡上调分块都没有收益。
- WMMA 写回：R9700 文档留下的「经 LDS 转换布局是否划算需要实测」，W7900D 上的答案是因 kernel 而异（0.90–2.04 倍）。
- scratch 门槛：R9700 文档没有提到。按 R9700 的 CU 数推算门槛约 2.2 KB，`context_attention` 上游版本的 2648 B 在 R9700 上也超过；R9700 上 4 头小形状上游 219.1 µs，改成溢出 ≤ 536 B 的 bf16x3 后 76.65 µs，其中可能有一部分就是这个原因，可以在 R9700 上用 `HSA_SCRATCH_SINGLE_LIMIT` 验证。
- autotune：W7900D 上 `_selective_state_update_fulltile_kernel` 在两个小调优键上选中了溢出 1164 B 的版本（R9700 上都不溢出），对耗时没有影响。

## 需要实测确认的事项

- 第二到第五批的「推算可快」是按带宽或算力利用率推算的上限；W7900D 上实测调参数的访存型算子除 `causal_conv1d_fn` 外只快 1.0–1.4 倍，对全部算子扫 num_warps 也只在少数算子上快 5% 以上，推算值偏乐观。`chain_speculative_sampling`（推算可快 3.10 倍）没有查。
- `context_attention` 的 fp16 改法（只针对 v2 包）：v2 包把 fp32 输入转成 fp16 计算，fp16 的指数范围小，q、k、v 中绝对值超过 65504 的元素会溢出成 inf，测试和 benchmark 的输入（标准正态分布）没有覆盖；K/V 预先转换还要额外的显存（最大的形状 2.1 GB）。合入的 #199 没有采用这个写法，fp32 输入仍按 fp32 计算，这一条对上游代码已不成立；fp16、bf16 输入本来就是半精度，走输入类型的 WMMA 没有这个问题。
- `fused_moe_gemm` 的分组路径：仓库里测试文件的 3 个用例矩阵太小，走不到这个路径，CI 覆盖不到。#205 中在 benchmark 形状（T ≥ 8 时走分组路径）、bf16 和 fp16 下按测试的容差核对过（容差占比与上游相同，0.562 和 0.070），另有 300 个随机用例和 20 个模型形状与 fp64 正确舍入的结果比较；仓库的测试没有补用例。
- `fused_moe_router_tensorcore` 的 M=4096 用例：W7900D 上分块求和的全部候选通过，但最高 0.96，余量很小，不同 ROCm 版本的 rocBLAS / hipBLASLt 舍入可能不同，建议测试的参考值改用 fp64。
- v2 包的规则都只在 benchmark 的形状上比较过耗时：`per_group_transpose` 的 2^24 门槛和 `log_scaling_tau` 的 4096 个程序各只在最大的一个形状上用到；`context_attention` 按 `max_input_len` 选 tile、`fused_moe_gemm` 按每个专家的对数选分块、split-KV 按网格大小选配置，换成实际模型的形状要复测；`causal_conv1d_fn` 的规则只比较过 benchmark 的 4 个形状（同 R9700 文档）。PR 阶段补了 benchmark 以外的形状：Mamba PR 在每个算子 99–137 个形状上核对逐位一致，LoRA PR 在每个算子 20–44 个形状、注意力 PR 在每个算子 10–14 个形状上比较了耗时（几何平均和最慢的形状见各节）；`log_scaling_tau` 已改写，不再有 4096 个程序的门槛。规则仍是在有限的形状上定的，换成实际模型的形状最好复测。
- `decode_grouped_attention` batch 512 慢 2%、`decode_attention` head_dim 80 的小形状慢 2%–4%：PR 中改了配置规则后分别是 1.01 倍和 1.04–1.09 倍（见「注意力 PR 中的改动和发现」）。剩下的差距没有再查：`decode_grouped_attention` 最大形状只到峰值带宽的 53%；query 组只有 4 个头、每序列只有一个 token 时 tile 大部分是填充，比上游逐 token 的 kernel 慢 19%。
- `chunk_state`（以及 `chunk_state_varlen` 剩下的差距）：两个 dot 操作数都经 LDS 转置，改分块没有用，要改数据布局（让 token 方向连续）或用 Gluon 显式指定布局，没有做。
- 两个会导致非法访存的配置（见「两个会导致非法访存的配置」）、#199 中 `context_attention` 两种会算错整块输出的 fp32 分块（见「第一个 PR（#199）」）、MoE PR 中串联累加的分组 GEMM 的一个配置（见「MoE PR 中的两个发现」）和 `extend_attention` 128 个 token 的 K/V 分块（见「注意力 PR 中的改动和发现」）原因都没有查明。改参数时要连续调用多次检查结果，只看第一次调用不够；扫配置时一个进程只测一个配置，否则出错后同一进程里后续配置的结果都不可信。
- scratch 门槛：调高后 scratch 常驻显存，多进程、多队列时每个队列都占一份；R9700 上是否同样受影响需要在 R9700 上测。
- WMMA 写回：按 kernel 关闭 `optimize_epilogue` 需要改 Triton；LoRA PR 中改用在 kernel 里 `tl.split`/`tl.join` 往返累加器，不改 Triton 也拿到了 16 字节读写，但这依赖 pass 目前的匹配方式，Triton 升级后要复查写回宽度（以及为保持逐位一致而显式写的 `tl.fma` 尾声）。原型中 `chunked_sgmv_expand` 写回宽度没变也快了近一倍，原因没有查明。
- autotune 的选择在不同进程间会变（router M=4096 的四次运行相差最多约 8%）；需要可复现时设置 `TRITON_CACHE_AUTOTUNING=1`。
