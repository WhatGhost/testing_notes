# ROCm 投机解码工作进度

背景：路线图 https://github.com/AMD-AIM/speculative-bench/issues/9 。机器只有 MI355X（gfx950），Radeon 条目无法验证。

最后更新：2026-11-12

## 1. 进行中的 PR

| PR | 内容 | 分支 | 状态 |
|---|---|---|---|
| [AITER #6755](https://github.com/ROCm/aiter/pull/6755) | gfx950 Gluon unified attention：序列结尾之外的 V 行清零，防止残留 NaN 经 `0×NaN` 污染输出 | `WhatGhost/aiter:fix/gfx950-gluon-ua-mask-oob-v` | open，10-26 已回复性能问题；11-11 再次请 cagrikymk review |
| [vLLM #63015](https://github.com/vllm-project/vllm/pull/63015) | wave64 上 `wvSplitK` N=5 固定 YTILE 2，避免 VGPR spill | `WhatGhost/vllm:perf/rocm-wvsplitk-5row` | open，已分配 reviewer；**`needs-rebase`**：与 main 在 `csrc/rocm/skinny_gemms.cuh` 冲突（11-04 起） |
| [AITER #7817](https://github.com/ROCm/aiter/pull/7817) | 候选 4（AITER 侧）：`unified_attention(causal=False)` 在 gfx950 Gluon kernel 上运行 | `WhatGhost/aiter:feat/gfx950-gluon-ua-noncausal` | open，reviewer Boss2002n。Copilot 5 条意见已修复并回复；bot 只剩 Aiter Test Gate（runner 掉线），11-12 已请 Boss2002n 重跑 |
| [vLLM #67013](https://github.com/vllm-project/vllm/pull/67013) | 候选 4（vLLM 侧）：`ROCM_AITER_UNIFIED_ATTN` 支持非因果 | `WhatGhost/vllm:rocm-aiter-ua-non-causal` | open，**依赖 AITER #7817**，reviewer tjtanaa、AndreasKaratzas |

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

### AITER #7817 / vLLM #67013 要点（候选 4）
- AITER：kernel 不改。wrapper 去掉 `assert causal`，`causal=False` 只在 gfx950 Gluon 上运行；滑动窗口（左右任一边界有限，含 decode 批次）、非 gfx950（含 `backend="gluon"`）、Gluon 不支持的配置（含 V 与 K dtype 不同）都报 `ValueError`；decode 批次按因果运行（结果相同）。新增 `unified_attention_supports_non_causal()` 供调用方查询。新测试 225 个，全量 `test_unified_attention.py` 2959 passed / 0 failed；benchmark 加 `-non_causal`。
- vLLM：`supports_non_causal()` 返回 True。decoder 层在 AITER 支持且条件满足时走 Gluon，其余（含 Whisper 等 encoder 层、旧 AITER）走原有的 vLLM Triton 退回路径，所以 AITER 未升级时也正确（速度等于 TRITON_ATTN）。
- 自动选择时草稿头仍会选 ROCM_ATTN（ROCm 优先级列表第一）；要用 UA 需在 `--speculative-config` 里写 `"attention_backend": "ROCM_AITER_UNIFIED_ATTN"`。
- 相关的上游工作：AITER #7244（yuychang，给 Triton kernel 加非因果，Gluon 保持只支持因果），与 #7817 的关系留到 review 时讨论。

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
| **4：AITER Unified 支持非因果（DSpark/DFlash 草稿头）** | **已提 PR：AITER #7817、vLLM #67013** | 见下 |

### 候选 4 结论（新机器，2026-11-04 至 11-11）

**旧机器上「conc=16 比 TRITON_ATTN 慢 8.5%」在新机器上复现不出来。**第一批数据里看到的回退，原因是 socket 0 的 CPU 慢（见 4.1），主机进程落到慢 socket 上会被拖慢。控制住 CPU 落点后，短上下文下 UA 和 TRITON 持平。

排查中否定的假设：
- 草稿 metadata 每步在主机端重建（只有 TRITON 能在 graph 里原地更新）：用开关做了双向 A/B，影响在 ±1% 内。重建和 GPU 执行是重叠的（`--async-scheduling`）。
- graph 录制时 `max_seqlen_k=max_model_len`（40960）导致 kernel 配置不合适：kernel 级实测 40960 和真实长度耗时相同。
- batch 16 profile：GPU 利用率 99%，不是主机端瓶颈；GEMM 占 GPU 时间约 74%，草稿 attention 只占约 0.4%–4%。

端到端（Qwen3-14B + DSpark n=7，target UA，草稿头分别用四种后端；绑核、门控、2 轮交替，72 项全部有效，两轮差异 ≤3%；批次 `work/cand4/d4g_1111_0342/`），输出吞吐 tok/s：

| 上下文 | conc | ROCM_ATTN（默认） | TRITON_ATTN | ROCM_AITER_FA | **UA** | UA vs TRITON | UA vs ROCM_ATTN |
|---|---:|---:|---:|---:|---:|---:|---:|
| ShareGPT | 1  | 353.0  | 360.1  | 348.5  | 355.3  | −1.3% | +0.7% |
| ShareGPT | 16 | 3315.6 | 3445.8 | 3242.7 | 3453.9 | +0.2% | +4.2% |
| ShareGPT | 64 | 5821.9 | 6064.9 | 5545.8 | 6101.8 | +0.6% | +4.8% |
| 8K  | 1 | 206.9 | 244.2 | 230.2 | 261.4 | +7.0%  | +26.3% |
| 16K | 1 | 151.7 | 193.0 | 185.3 | 219.3 | +13.6% | +44.6% |
| 32K | 1 | 95.6  | 130.6 | 124.2 | 160.0 | **+22.5%** | **+67.4%** |
| 8K  | 8 | 855.8 | 946.9 | 684.3 | 995.4 | +5.1%  | +16.3% |
| 16K | 8 | 488.0 | 556.0 | 379.5 | 592.4 | +6.5%  | +21.4% |
| 32K | 8 | 239.8 | 270.6 | 206.7 | 289.5 | +7.0%  | +20.7% |

- 长上下文数据集：从 ShareGPT 拼成 8K/16K/32K 的自然文本提示词（`work/longctx/`）；随机 token 会让接受率失真，不能用。
- 各后端接受长度一致（长上下文 3.0–3.2，ShareGPT 3.8–3.9）。
- ROCM_AITER_FA 能当草稿后端，但长上下文、conc=8 时比 ROCM_ATTN 还慢 14%–22%，不能替代 UA。
- 两批未加门控的数据（`d4_1110_0746`、`d4_1110_0953`）受外部负载干扰，但结论与上表一致。

kernel 级（gfx950，DSpark 校验形状：40/8 头、head 128、每请求 8 个 query、page 64、bf16，非因果）：
- Gluon 非因果 / 因果耗时比 0.98–1.00，没有额外开销。
- Gluon 比 vLLM Triton 快：batch 1 时 3.5× / 11.4× / 21.9×（kv 1.5K/8K/32K），batch 16–64 时 1.7×–2.3×。

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
- **CPU socket 0 很慢**：2 路 EPYC 9965，socket 0（CPU 0–191）单核比 socket 1（192–383）慢约 6 倍（同一段 Python 循环 2.0 s vs 0.34 s；`scaling_cur_freq` 读数不可信）。主机进程落到 socket 0 时吞吐会掉一半。测性能必须：用 NUMA1 的卡（rocm-smi 4–7 = HIP 5/7/6/4），vllm serve 和 bench 都用 `taskset -c 192-383`，开跑前做 CPU 自检。rocm-smi 0–3 在 NUMA0。
- **门控测试**：机器与其他容器共用，外部负载会污染数据。脚本 `work/cand4/ab_draft4_gated.sh`：每项测量前等其他卡 GPU% ≤5% 且 1 分钟负载 <20（连续 3 次采样），测完复核测量窗口，有干扰就判无效并重测。
- **长批次要脱离会话运行**：`setsid nohup <脚本> > log 2>&1 < /dev/null & disown`，否则 Claude 会话结束时会被一起杀掉。
- **profiler**：pytorch.org 的 torch 2.14+rocm7.2 在 `torch/lib` 里自带一套 ROCm 运行时，与系统 ROCm 7.2.4 的 rocprofiler-sdk 冲突，rocprofv3 和 rocprof-sys 在 import torch 时崩溃。办法（不动 `/opt/venv`）：副本 `/target/spectest/torch_sysrocm/torch`，其中 11 个运行时库软链接到 `/opt/rocm/lib`；使用时 `PYTHONPATH=/target/spectest/torch_sysrocm`，rocprof-sys 还要 `LD_LIBRARY_PATH=/target/spectest/torch_sysrocm/torch/lib`。
- **测试 AITER main**：main 要求 flydsl 0.5.0（venv 是 0.4.0），装在 `tools/flydsl_main`，用 `PYTHONPATH=tools/flydsl_main:aiter-pr`。vLLM 测试依赖（tblib）在 `tools/vllm_testdeps`。格式工具：`tools/lintenv`（AITER：black + ruff 0.16.0）、`tools/lintenv-vllm`（ruff 0.17.2）。
- **GitHub 访问**：ROCm 组织禁止 classic token（连读取都 403）；本机 gh 已改为 OAuth 登录（`gh auth login --web`），可读写 ROCm/aiter 的 PR。未登录的公共 API 每 IP 每小时 60 次，出口 IP 共用，经常被用光。ROCm/aiter 的 CI 重跑需要 admin 权限，要请维护者操作。
- **当前仓库状态**：PR worktree 在 `/target/spectest/{aiter-pr,vllm-pr}`；`/target/spectest/vllm`（editable 安装）现在打着 #67013 的改动，早期的临时补丁存在 `work/cand4/vllm_tmp_all_1111.patch`。

### 4.2 旧机器（已停用）

- `/opt/venv`：vLLM main（editable，`/target/spec_test/vllm`，含 #63015 修复）+ Triton 3.8.0 + AITER 0.1.25（site-packages）。注意：脚本把 `/target/spec_test/aiter` 加进 `sys.path` 会换成仓库版 AITER。
- `/opt/venv-vllm0.28`：旧 0.28.0，正式基准数据出自这里。
- 缓存：`HF_HOME=/target/.cache/huggingface`，`VLLM_CACHE_ROOT=/target/.cache/vllm`（`~/.bashrc` 开头）。
- 选卡必须用 `speculative-bench/profile/scripts/pick_free_gpu.py`（HIP 编号与 rocm-smi 不一致）。
- 旧机器 gh 用的是 classic token，被 ROCm 组织禁止，AITER 的 PR 只能读公开 API，改标题/发评论需在网页操作。

## 5. 下一步

1. AITER #7817：等维护者重跑失败的 Standard Tests（runner 掉线），然后等 Boss2002n review。
2. vLLM #67013：等 review；#7817 合入、vLLM 升级 aiter 锁定版本后才能合。
3. vLLM #63015：rebase 解决 `csrc/rocm/skinny_gemms.cuh` 的冲突。
4. AITER #6755：等 cagrikymk 回复。
5. 1b 等 #63015 有反馈后再评估。

## 6. 文件索引

| 文件 | 内容 |
|---|---|
| `scratch/aiter_pr_description_short.md` | AITER #6755 当前 PR 描述 |
| `scratch/aiter_pr_replies.md`、`scratch/aiter_pr_reply_perf.md` | AITER #6755 已发出的回复 |
| `scratch/vllm_wvsplitk_pr_description.md` | vLLM #63015 PR 描述 |
| `scratch/aiter_gluon_pr/` | #6755 性能对比（`review/`、`review_bs64/`、`review_w3/`）和测试脚本 |
| `scratch/wvsplitk_5row/` | #63015 A/B 数据（`ab/` 正式，`ab_run1_noisy/` 作废）、两个版本的 .so |
| `scratch/aiter_ua_noncausal/`、`scratch/draft_attn_share/` | 候选 4 在旧机器上的 kernel、端到端和 profile 数据 |
| `/target/spectest/work/cand4/` | 新机器候选 4 数据与脚本：`d4g_1111_0342/`（门控批次，正式）、`out_1104_0444_pin/`、`var_1104_0734/`（metadata 开关 A/B）、`summarize.py`、`ab_draft4_gated.sh` |
| `/target/spectest/work/cand4/profile/`、`kernel/` | rocprofv3 profile（batch 16）、kernel 级 max_seqlen_k 对比 |
| `/target/spectest/work/longctx/` | 8K/16K/32K 长上下文数据集及生成脚本 |
| `/target/spectest/work/pr_tests/` | #7817/#67013 的测试日志、kernel benchmark、最终代码端到端验证 |
| `/target/spectest/work/pr_drafts/` | #7817/#67013 的 PR 描述、Self-Review、Copilot 回复、重跑请求 |
