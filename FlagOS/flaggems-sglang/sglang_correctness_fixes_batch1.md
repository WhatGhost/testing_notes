# SGLang 接入正确性修复（4 个算子，已完成、未提 PR）

- **时间**：2026-11-11 至 11-12
- **背景**：在 W7900（gfx1100）上用 FlagGems-sglang 接 SGLang 0.5.11 跑 Qwen3.6-27B / 35B-A3B（TP=4）时，发现 4 个算子存在正确性或接口问题。问题清单和复现脚本在 `/target/flag-plugin-test/logs/flaggems_sglang/issues/`。
- **状态**：4 项都已修复并验证。代码在 fork 分支 `WhatGhost:fix/sglang-integration-batch1`（commit `6ed7e26`，基于上游 `5c2ed0f`，14 个文件，+864 / −206）。还没有开 PR，等与 FlagOS 讨论放置位置和单测尺度（见最后一节）。

## 各算子的工作

### 1. `chunk_gated_delta_rule_fwd`（GDN prefill）

- **问题**：实现里的分段近似在真实模型的门控下不成立，用 Qwen3.6-27B 真实权重时输出误差高达 78%；不支持 value head 多于 key head，会静默输出错误的头数；初始状态和最终状态共用一套步长。
- **工作**：
  - 把 kernel 重写为精确的逐步递推，支持 HV 为 H 的整数倍；
  - 补全参考实现（HV>H、`cu_seqlens`），新增 3 个回归用例；
  - 针对 gfx1100 调优（程序数、寄存器排布、8 步预取）。
- **结果**：
  - 精度：真实权重下误差从 0.78 降到 1.7e-3，比 SGLang FLA 的约 5e-3 还准。
  - 性能：按 SGLang 的调用方式，比 SGLang FLA 快 1.00–2.59 倍。官方 benchmark 上，部分形状比原来那个结果错误的版本慢（0.62–0.98 倍），`(4, 4096)` 快 1.34 倍。
- **顺带发现**：gfx1100 上 Triton 3.6 开软件流水时编译出的结果是错的（varlen、BLOCK_V=8），已绕开。

### 2. `moe_align_block_size`

- **问题**：不写调用方传入的输出缓冲区，而是另分配张量返回；SGLang 不看返回值，于是读到未初始化的内存。
- **工作**：改为写入调用方的缓冲区；参考实现同步改成原地写，与 sgl_kernel 一致；单测加上缓冲区检查。
- **结果**：性能不变（1.00–1.02 倍）。

### 3. `fused_gate_sigmoid_mul_add`

- **问题**：门控权重为 `[1, H]`（SGLang 中 `Linear(H, 1)` 的权重形状）时，按错误的步长读取，GPU 越界访存导致进程崩溃。
- **工作**：改为按最后一维的步长读取，并加形状检查；单测新增 `[1, H]` 权重的用例。
- **结果**：原有的一维用法编译出完全相同的 kernel，性能不变（0.98–1.01 倍）。

### 4. `fused_qkvzba_split_reshape_cat`

- **问题**：只支持 Qwen3-Next 的交错排布，用在 Qwen3.5 / 3.6 的连续排布上结果错误。
- **工作**：原算子不动，新增算子 `fused_qkvzba_split_reshape_cat_contiguous`。它的签名与 SGLang 一致，任意头数比都能用，包括 SGLang 自己的 kernel 编译失败的比例 3；参考实现、单测、benchmark、baseline 和 `operators.yaml` 都已配齐。
- **结果**：比 PyTorch 参考实现快 1.35–3.42 倍。

## 验证

- 4 个复现脚本修复后全部 PASS。
- 5 个算子的单测在 rdna3 和 `ARCH=none` 两种模式下都是 24 passed；新加的回归用例在原实现上会失败，修复后通过。
- 官方 benchmark 与上游 master 交替跑 3 轮对比，结果见上面各算子。

## 待决事项（为什么还没提 PR）

- 加强后的单测是所有厂商共用的，会让其他厂商的现有实现失败：`moe_align_block_size` 6 家全部失败；`chunk_gated_delta_rule_fwd` 的 HV>H 用例 4 家失败；`fused_gate_sigmoid_mul_add` 的 `[1, H]` 用例有 hygon、enflame、kunlunxin、metax 4 家失败。
- 需要与 FlagOS 确认三件事：修复放在通用实现还是只放 `_amd/rdna3`；单测加强到什么程度；一个 PR 提交还是拆开。

更详细的记录见 `W7900D_AMD_optimization.md` 的「SGLang 接入正确性（2026-11-10）」一节。
