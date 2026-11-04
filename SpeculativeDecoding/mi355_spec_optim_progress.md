# ROCm 投机解码工作进度

背景：路线图 https://github.com/AMD-AIM/speculative-bench/issues/9 。机器只有 MI355X（gfx950），Radeon 条目无法验证。

最后更新：2026-11-03

## 1. 进行中的 PR

| PR | 内容 | 分支 | 状态 |
|---|---|---|---|
| [AITER #6755](https://github.com/ROCm/aiter/pull/6755) | gfx950 Gluon unified attention：序列结尾之外的 V 行清零，防止残留 NaN 经 `0×NaN` 污染输出 | `WhatGhost/aiter:fix/gfx950-gluon-ua-mask-oob-v` | open，10-26 已回复性能问题，等 cagrikymk |
| [vLLM #63015](https://github.com/vllm-project/vllm/pull/63015) | wave64 上 `wvSplitK` N=5 固定 YTILE 2，避免 VGPR spill | `WhatGhost/vllm:perf/rocm-wvsplitk-5row` | open，无人 review，需维护者 `/amd-ci run` |

### AITER #6755 经过
1. 初版同时改 Triton 和 Gluon。
2. 维护者指出 Triton 部分与 [#6475](https://github.com/ROCm/aiter/pull/6475) 重复 → 收窄为只改 Gluon；同时修了 Copilot 3 条测试意见（gfx950 only、加 decode-only batch、FP8 用 FP8 query、断言走 Gluon、加 prefill 2D batch）。
3. 维护者测出 prefill + 滑动窗口最多 +17% → 两个原因、两个 commit：
   - `fd1886e48`：清零上界从 `max_seq_prefix_len` 改为真实序列长度，prefill 只有每条序列最后一个 query block 清零。
   - `c052d5de7`：`configs/gfx950/gluon/attention/unified_attention/DEFAULT.json` 的 `attn_2d.D_LEQ_64.SW` 设 `waves_per_eu=3`，VGPR 172→168，保住 3 wave/SIMD。
4. 结果（block 64，对比 main）：h64 SW128 bf16 +12.8% → −0.2%，SW256 bf16 +14.4% → +1.4%；剩 decode_short 约 +1 us（必须执行的清零）。
5. 测试：新测试 72/72（main 上 49 个失败），全量 `test_unified_attention.py` 2570 passed / 0 failed，强制 1 buffer、2/4 warp 均通过。

### vLLM #63015 要点
- 现象：Qwen3-32B EAGLE3、Qwen3.8-27B MTP 都在 N=4 + conc=1 塌陷（每步约 36 ms，相邻档 17–19 ms）。
- 根因：验证 5 行，`wvSplitK` N=5 对宽 M 选 YTILE 3/4，超过 16-wave block 的 128 VGPR 上限，spill 最多 111 VGPR/lane。
- 同版本 A/B（只换 `_rocm_C.abi3.so`，3+3 轮）：Qwen3.8-27B MTP n=4 conc=1 吞吐 92.4 → 191.3 tok/s（2.07×），AL 不变；宽投影 GEMM 快约 4×。
- 单测补了 n=5 形状，2444 passed / 0 failed。

## 2. 根因结论（不提 PR，供参考）

- **DSpark + AITER hang**：vLLM main 上不复现。
- **ROCM_AITER_FA + DSpark 接受长度塌陷**：ROCm Triton 3.7.x（`f0b55c07`）开了 `amdgpu-use-amdgpu-trackers`，但绑定的 LLVM 缺 #179305（`global_load_lds` 误计 lgkmcnt，waitcnt 偏弱）。AITER Gluon 在 KV 每侧 > 2 GiB 时走 `global_load_lds`，暴露该问题。
  - Triton 3.8 正常；`/opt/venv` 已换 3.8。
  - 只在 AITER 0.1.25 + 14B 形状 + 短上下文触发；AITER main 不触发。vLLM/AITER 测试都抓不到，已放弃加测试。
  - 材料：`/target/spec_test/triton_bisect/`、`scratch/aiter_fa_correctness/`。

## 3. PR 候选评估

| 候选 | 结论 | 数据 |
|---|---|---|
| 1b：`wvSplitK` N=4 YTILE 3 spill（M 不被 3 整除，如 10240、11008） | 暂缓，等 #63015 有反馈后追加。kernel 级快 21–33%，端到端未测（需 32B/70B 权重） | `scratch/wvsplitk_5row/n4/` |
| 2：ROCm 后端支持 fused draft decode | 不做。上游无人做，但 MI355X 上收益在噪声内（≤1.5%） | `scratch/fused_draft/out/` |
| 3：投机场景后端选择可见化 | 不做。main 已打印选择结果、回退原因和不支持原因 | `scratch/backend_log/out/` |
| **4：AITER Unified 支持非因果（DSpark/DFlash 草稿头）** | **要做，进行中** | 见下 |

### 候选 4 进度
- AITER gfx950 Gluon kernel 已有 `CAUSAL=False` 分支，只被 wrapper `assert causal` 挡住。[AITER #6108](https://github.com/ROCm/aiter/pull/6108)（作者已暂停）只做 Triton，Gluon gfx950/gfx1250 留空。
- 实际要做两件：AITER 放开 Gluon 非因果（补测试、双向滑动窗口）；vLLM `ROCM_AITER_UNIFIED_ATTN` 声明支持非因果并传 `causal`。
- kernel 级（q_len 8，非因果，40/8 头）：Gluon 比 vLLM Triton 快 1.1–3.2×，正确。
- 端到端（Qwen3-14B DSpark n=7，target AITER UA，2 轮）草稿头用 UA 对比：

  | conc | vs ROCM_ATTN（默认） | vs TRITON_ATTN |
  |---|---|---|
  | 1 | +1.7% | +0.8% |
  | 16 | −1.6% | **−8.5%** |
  | 64 | +4.2% | +0.9% |

- **待办：查清 conc=16 回退**（GPU kernel 时间不多，墙钟多，疑似主机端开销；graph 录制结构相同，已排除）。查清前不提 PR。
- 草稿头 attention 占 GPU kernel 时间：batch 1 约 1.3%，batch 16 约 3.8%，端到端上限是几个百分点。
- 临时补丁：`scratch/aiter_ua_noncausal/tmp_aiter_no_assert.patch`、`tmp_vllm_ua_noncausal.patch`（两个仓库已恢复干净）。

## 4. 环境

### 4.1 新机器（2026-11-03 起，当前工作机）

容器 `rocm724_gyy_spectest`（镜像 `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`），宿主 `/home/yuygao/spectest` 挂到 `/target/spectest`。

| 组件 | 新机器 | 旧机器 |
|---|---|---|
| ROCm | 7.2.4 | 7.2.3 |
| torch | 2.14.0+rocm7.2（download.pytorch.org/whl/rocm7.2，对齐 vLLM main 的 `torch == 2.14.0`） | 2.12.0+git6bbd260 |
| Triton | 3.8.0+amd.rocm7.2.0.git7cb7b059 | 同 |
| AITER（venv） | 0.1.27 wheel（vLLM `Dockerfile.rocm_base` 推荐版本） | 0.1.25 |
| vLLM | main `ddbe9beb75`（11-03），editable，**不含 #63015** | main，含 #63015 |

- 目录：源码 `/target/spectest/{vllm,aiter,speculative-bench,testing_notes}`（`origin` + `fork`），模型 `/target/models`，缓存 `/target/spectest/cache/{hf,vllm,triton}`，环境变量 `/target/spectest/env.sh`，交接材料 `/target/spectest/handoff/`。
- 依赖锁定：`/target/spectest/core-constraints.txt`（torch/vision/audio/triton）。交接包里的旧 `core-constraints.txt` 锁的是 Triton 3.7.1，不要用。
- 安装注意：
  - torch 2.14+rocm7.2 依赖 `triton-rocm`，与 AMD Triton 共用 `triton/` 目录。pip 装包后运行 `scripts/fix_triton.sh` 恢复 3.8.0。
  - AITER 编译要设 `AITER_USE_SYSTEM_TRITON=1`（否则重装 Triton），且需装 `setuptools_scm`（否则版本号为 0.0.0）。
  - vLLM 识别 ROCm 需要 `amdsmi`（不在 vLLM 依赖里），从 `/opt/rocm/share/amd_smi` 安装。
- 选卡：HIP → rocm-smi 编号为 0→3、1→0、2→2、3→1、4→7、5→4、6→6、7→5。**卡与其他容器共用**，测性能前看 rocm-smi 的 VRAM% 和 SCLK。
- 冒烟：Qwen3-14B + DSpark n=7（target AITER UA）正常，接受长度约 5.05。

### 4.2 旧机器（已停用）

- `/opt/venv`：vLLM main（editable，`/target/spec_test/vllm`，含 #63015 修复）+ Triton 3.8.0 + AITER 0.1.25（site-packages）。注意：脚本把 `/target/spec_test/aiter` 加进 `sys.path` 会换成仓库版 AITER。
- `/opt/venv-vllm0.28`：旧 0.28.0，正式基准数据出自这里。
- 缓存：`HF_HOME=/target/.cache/huggingface`，`VLLM_CACHE_ROOT=/target/.cache/vllm`（`~/.bashrc` 开头）。
- 选卡必须用 `speculative-bench/profile/scripts/pick_free_gpu.py`（HIP 编号与 rocm-smi 不一致）。
- 本机 gh 的 classic token 被 ROCm 组织禁止，AITER 的 PR 只能读公开 API，改标题/发评论需在网页操作。

## 5. 下一步

1. 候选 4：查 conc=16 回退的主机端开销。
2. 跟进 AITER #6755、vLLM #63015 的 review。
3. 1b 等 #63015 有反馈后再评估。

## 6. 文件索引

| 文件 | 内容 |
|---|---|
| `scratch/aiter_pr_description_short.md` | AITER #6755 当前 PR 描述 |
| `scratch/aiter_pr_replies.md`、`scratch/aiter_pr_reply_perf.md` | AITER #6755 已发出的回复 |
| `scratch/vllm_wvsplitk_pr_description.md` | vLLM #63015 PR 描述 |
| `scratch/aiter_gluon_pr/` | #6755 性能对比（`review/`、`review_bs64/`、`review_w3/`）和测试脚本 |
| `scratch/wvsplitk_5row/` | #63015 A/B 数据（`ab/` 正式，`ab_run1_noisy/` 作废）、两个版本的 .so |
| `scratch/aiter_ua_noncausal/`、`scratch/draft_attn_share/` | 候选 4 的 kernel、端到端和 profile 数据 |
