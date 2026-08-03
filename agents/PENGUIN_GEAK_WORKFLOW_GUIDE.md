# 使用 PenguinHarness 优化 GEAK：可行性、系统设计与用户操作指南

> 文档状态：总体设计与操作方案  
> 当前状态：尚未完成代码集成，本文中标记为“计划新增”的文件和命令不能直接使用  
> 推荐首个目标：使用 Penguin 优化 GEAK single-kernel 工作流的 `kernel_workflow/roles/tech_lead.md`

## 1. 这件事要解决什么问题

GEAK 能够组织多个 Agent 优化 GPU kernel，但 GEAK 自己的角色提示词、规划策略和知识组织方式仍然是人工编写的。我们希望在 GEAK 外面再增加一层 PenguinHarness，让 Penguin 根据 GEAK 的历史运行证据提出改进，然后用真实 Benchmark 验证这些改进是否有效。

这里存在两层优化：

```text
外层：Penguin 优化 GEAK 的 Agent 行为
内层：GEAK 优化具体 GPU kernel
```

例如：

```text
Penguin 修改 GEAK 的 tech_lead.md
              │
              ▼
GEAK 使用新 TechLead Prompt 优化 KNN、Attention 等 kernel
              │
              ▼
比较修改前后 GEAK 在同一批 kernel 上的正确率和最终加速比
```

因此，这不是让 Penguin 直接编写更快的 kernel，而是让 Penguin 提高 GEAK 选择优化方向、分配预算、利用历史证据和停止搜索的能力。

## 2. 可行性结论

### 2.1 结论

该方案可行，但需要在 Penguin 和 GEAK 之间增加一个确定性的集成层。不能只给 Penguin 一句话“请优化 GEAK”，然后允许它直接修改整个仓库。

推荐架构为：

- Penguin：分析证据、提出假设、生成受限 Candidate diff。
- Claude Code：作为 GEAK Workflow JS 的运行宿主。
- GEAK：执行实际的 kernel 优化、正确性测试、GPU 测速和 Director 验证。
- Meta-Optimizer Controller：隔离候选、调度评测、计算统计结果、决定接受或回滚。

### 2.2 为什么可行

GEAK 已经具备以下基础：

- `kernel_workflow.js` 中有确定性的预算循环、并行调度和验证流程；
- TechLead、Engineer、Verifier、Director 等角色分别存放在 Markdown 文件中，适合作为可控的优化对象；
- 最终返回包含 `final_speedup`、`final_geomean`、`validation_status`、`budget_used` 等结构化字段；
- Engineer 的性能声明会由独立 Verifier 和最终 Director 复测；
- e2e 工作流已经有通过 Claude Agent SDK/Claude CLI 调用 Workflow 的外部适配器范例。

Penguin 已经具备以下基础：

- 可编辑的 Agent State、`AGENTS.md` 和 Skill；
- `exec_command` 本地命令执行及后台进程管理；
- `run_subagent` 子 Agent 调度；
- Benchmark、scoreboard、Trace 和 Agent 自优化方法；
- “证据 → 假设 → 候选 → 评测 → 接受或回滚”的现有设计思想。

### 2.3 为什么不能直接运行

当前仍有三个缺口：

1. Penguin 不能原生解释 Claude Code Workflow JS。
2. GEAK single-kernel 工作流缺少稳定的、面向外部调用者的非交互 Runner。
3. Penguin 现有 `agent-optimization` Skill 默认只允许修改 Penguin Agent State，不应直接放宽到任意外部 Git 仓库。

所以正确的调用关系是：

```text
Penguin
  └── 启动 GEAK Runner
        └── 启动 Claude Code
              └── 调用 Workflow
                    └── 执行 kernel_workflow.js
```

而不是：

```text
Penguin 直接执行 kernel_workflow.js   # 不可行
node kernel_workflow.js               # 不可行
```

## 3. 当前能力与计划新增能力

| 能力 | 当前状态 | 需要做什么 |
|---|---|---|
| GEAK single-kernel Workflow | 已有 | 保留 |
| GEAK e2e 外部 Runner | 已有 `interface/run_e2e.py` | 抽取公共 Workflow Driver |
| GEAK single-kernel 外部 Runner | 没有 | 新增 `interface/run_kernel.py` |
| Penguin 本地命令执行 | 已有 | 用来调用 Runner |
| Penguin Agent 自优化 | 已有，但对象是 Agent State | 新增 GEAK 专用 Skill |
| 外部 Git Candidate 隔离 | 没有专用实现 | 新增 worktree Controller |
| GEAK Meta-Benchmark | 没有 | 建立 Canary/Dev/Holdout 数据集 |
| 确定性性能接受规则 | 没有跨项目实现 | 新增统计、完整性和接受模块 |
| 自动提交与回滚 | 没有跨项目实现 | 只在 evolution 分支执行 |

## 4. 系统总体架构

```text
┌──────────────────────────────────────────────────────────┐
│                     用户 / Operator                      │
│  选择目标、Benchmark、GPU、轮数、预算和审批策略          │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                 Penguin Meta-Optimizer                   │
│  读取公开证据 → 形成假设 → 生成受限 Candidate diff       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│             Deterministic Controller                     │
│  Schema/allowlist → Git worktree → GPU queue → 统计判定  │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  GEAK run_kernel.py                      │
│  参数归一化、超时、Claude SDK/CLI、结果恢复、result.json │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  Claude Code Workflow                    │
│  kernel_workflow.js → TechLead/Engineer/Verifier/Director│
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                   Frozen Benchmark                       │
│  correctness oracle + GPU benchmark + workload weights   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
             accept → evolution branch
             reject → 删除 worktree，保留失败证据
```

## 5. 第一阶段优化什么

### 5.1 唯一允许修改的文件

```text
GEAK/kernel_workflow/roles/tech_lead.md
```

选择它的原因：

- TechLead 决定每轮探索哪些方向；
- 它影响 specialist 的组合、预算分配和停止条件；
- 它的改动不会直接修改正确性 oracle 或计时方法；
- 单文件改动更容易归因和回滚。

### 5.2 第一阶段禁止修改的文件

至少保护：

```text
GEAK/kernel_workflow/kernel_workflow.js
GEAK/kernel_workflow/roles/director.md
GEAK/kernel_workflow/roles/benchmark_engineer.md
GEAK/kernel_workflow/roles/verify_engineer.md
GEAK/kernel_workflow/scripts/**
所有 Benchmark、unittest、reference_io.pt 和 workload 数据
Meta-Optimizer 的评分器与接受逻辑
```

这样可以防止候选通过放宽正确性、篡改计时器或者伪造性能结果来获得“提升”。

### 5.3 后续扩展顺序

在 TechLead MVP 被证明有效后，依次考虑：

1. `roles/engineer.md`；
2. `roles/deep_engineer.md`；
3. 只读引用型知识文件；
4. e2e 工作流角色；
5. 最后才考虑工作流 JS 控制逻辑。

每个阶段仍应一次只开放一个优化单元，并重新建立 Reference。

## 6. 需要给 GEAK 增加的内容

以下均为计划新增。

### 6.1 公共 Claude Workflow Driver

建议新增：

```text
GEAK/interface/workflow_driver.py
```

从现有 `interface/run_e2e.py` 中抽取：

- Claude Agent SDK 调用；
- `claude -p` 回退；
- `enableWorkflows`/`ultracode` 设置；
- Workflow 同步与后台完成等待；
- timeout 和进程清理；
- 结果 JSON 解析；
- 磁盘产物恢复；
- 稳定错误分类。

### 6.2 Single-kernel Runner

建议新增：

```text
GEAK/interface/run_kernel.py
GEAK/interface/run_kernel.md
GEAK/interface/test_run_kernel.py
```

计划调用形式：

```bash
python3 GEAK/interface/run_kernel.py \
  --kernel-path /path/to/case \
  --workflow-dir /path/to/candidate/GEAK/kernel_workflow \
  --budget 6 \
  --gpu-ids 0 \
  --eval-dir /path/to/evaluation \
  --result-json /path/to/result.json
```

Runner 无论成功、失败还是超时，都应写出一个合法 `result.json`。

### 6.3 版本化结果协议

建议结果至少包含：

```json
{
  "schema_version": "geak-kernel-result-v1",
  "run_id": "run-0001",
  "status": "succeeded",
  "kernel_id": "knn",
  "reference_geak_sha": "...",
  "candidate_geak_sha": "...",
  "verification": {
    "correctness_passed": true,
    "validation_status": "accepted",
    "final_speedup": 1.23,
    "final_geomean": 1.23,
    "final_weighted": null
  },
  "resources": {
    "budget_used": 6,
    "wall_time_seconds": 1800,
    "input_tokens": null,
    "output_tokens": null
  },
  "environment": {
    "gpu": "...",
    "rocm": "...",
    "driver": "...",
    "claude_version": "...",
    "model_id": "..."
  },
  "artifacts": {
    "eval_dir": "...",
    "report_path": "...",
    "director_validation": "...",
    "final_patch": "..."
  },
  "error": null
}
```

### 6.4 可观测性与完整性数据

GEAK 还应输出：

- 每轮 TechLead 选择的方向；
- 各方向 claimed/verified speedup；
- Verifier 驳回原因；
- 停止原因；
- 每个角色的耗时和 Token（宿主能够提供时）；
- 环境指纹；
- 工作流、角色文件、Benchmark 和 oracle 哈希。

## 7. 需要给 Penguin 增加的内容

以下均为计划新增。

### 7.1 GEAK Meta-Optimization Skill

```text
Penguin-harness/packages/skills/skills/geak-meta-optimization/SKILL.md
```

职责：

- 读取 Reference 和历史评测证据；
- 每轮只提出一个可证伪假设；
- 只修改 allowlist 中的文件；
- 生成 Candidate manifest 和 diff；
- 把运行和评分交给 Controller；
- 根据接受/拒绝结果更新 hypothesis ledger；
- 不直接修改 Benchmark 或评分器。

### 7.2 GEAK Evaluation Skill

```text
Penguin-harness/packages/skills/skills/geak-workflow-evaluation/SKILL.md
```

职责：

- 处理一个 Candidate × Case × Run 评测请求；
- 调用确定性 Controller；
- 返回严格的机器协议；
- 不生成候选、不修改 GEAK、不自己决定得分；
- 不向优化 Agent 泄露 holdout 内容。

### 7.3 确定性 Controller

建议新增：

```text
Penguin-harness/packages/geak-optimizer/
├── src/cli.ts
├── src/config.ts
├── src/protocol.ts
├── src/worktree.ts
├── src/integrity.ts
├── src/scheduler.ts
├── src/evaluator.ts
├── src/statistics.ts
├── src/scoreboard.ts
└── test/
```

它负责所有不能依赖 LLM 自觉遵守的操作：

- Candidate schema 校验；
- 路径 allowlist 和 protected hash；
- 临时 Git worktree；
- GPU 独占和超时；
- Reference/Candidate 配对运行；
- 统计置信度；
- 接受、提交和回滚；
- scoreboard 原子更新。

### 7.4 计划提供的命令

```text
penguin geak baseline --config target.yaml
penguin geak evaluate --config target.yaml --candidate manifest.yaml
penguin geak optimize --config target.yaml --rounds 5
penguin geak status --config target.yaml
penguin geak rollback --config target.yaml --to <evaluation_id>
```

这些命令目前不存在，需要在集成阶段实现。

## 8. Workload/Workflow 项目目录

建议建立一个独立工作目录，不把运行状态直接写入两个源码仓库：

```text
/home/yuygao/agents/geak-penguin-workflow/
├── README.md
├── OBJECTIVE.md
├── target.yaml
├── benchmark.yaml
├── agent_state/
│   ├── system_config.yaml
│   ├── AGENTS.md
│   └── skills/
├── references/
├── candidates/
├── evaluations/
├── traces/
├── worktrees/
├── hypothesis_ledger.yaml
└── scoreboard.yaml
```

其中：

- 源代码版本使用 Git SHA 和 diff 管理；
- 大型 GEAK 运行产物只记录路径和哈希；
- Candidate worktree 是临时目录；
- Reference checkout 不允许被候选直接修改；
- accepted Candidate 只提交到 `penguin-evolution` 分支。

## 9. 配置示例

### 9.1 `target.yaml`

```yaml
schema_version: geak-meta-target-v1
target_id: geak-tech-lead-mi300x

repositories:
  geak: /home/yuygao/agents/GEAK
  penguin: /home/yuygao/agents/Penguin-harness

workflow:
  type: single-kernel
  runner: /home/yuygao/agents/GEAK/interface/run_kernel.py
  budget: 6
  gpu_ids: [0]
  timeout_seconds: 7200

optimization:
  unit: kernel_workflow/roles/tech_lead.md
  allowed_paths:
    - kernel_workflow/roles/tech_lead.md
  protected_paths:
    - kernel_workflow/kernel_workflow.js
    - kernel_workflow/roles/director.md
    - kernel_workflow/roles/benchmark_engineer.md
    - kernel_workflow/roles/verify_engineer.md
    - kernel_workflow/scripts/**
  max_changed_files: 1
  max_diff_lines: 200
  max_rounds: 5

acceptance:
  required_validation_status: accepted
  require_correctness: true
  dev_relative_gain_min: 0.02
  holdout_relative_gain_min: 0.01
  per_kernel_regression_limit: 0.03
  confidence: 0.95
  human_approval_before_commit: true
```

### 9.2 `benchmark.yaml`

```yaml
schema_version: geak-meta-benchmark-v1
benchmark_id: single-kernel-v1
runs_per_case: 3

canary:
  - id: knn
    kernel_path: /home/yuygao/agents/GEAK/examples/tasks/knn

dev:
  - id: knn
    kernel_path: /home/yuygao/agents/GEAK/examples/tasks/knn

holdout:
  manifest: /secure/path/geak-holdout.yaml
  expose_case_details_to_optimizer: false
```

真实使用时 Dev 不能只有一个任务。建议至少选择 5–10 个覆盖不同瓶颈、shape、dtype 和实现方式的 kernel。

## 10. 完整的自动优化工作流

### Step 1：冻结环境

记录：

- GEAK Git SHA；
- Penguin Git SHA；
- Claude Code 版本；
- Penguin 使用的 provider/model/thinking level；
- GEAK 内部 Claude 模型；
- GPU、驱动、ROCm、Python 与依赖；
- Benchmark 版本和内容哈希。

只要其中关键项改变，就应重新生成 Reference，不能与旧结果直接比较。

### Step 2：生成 Reference Baseline

使用未经修改的 GEAK 在完整 Dev Benchmark 上重复运行。每个 case 至少 3 次，用于估计：

- 正确运行率；
- `final_speedup` 分布；
- suite geomean；
- 墙钟和 Token 成本；
- Reference 自身噪声范围。

Reference 不完整或运行不稳定时，不开始优化。

### Step 3：Penguin 分析证据

Penguin 可以读取：

- `analysis.json`；
- TechLead 每轮计划；
- Engineer 和 Verifier 结果；
- `tech_lead_report.md`；
- `director_validation.json`；
- 公开的聚合 Benchmark 结果；
- hypothesis ledger。

Penguin 不可以读取隐藏 holdout 内容或评分器内部实现。

### Step 4：生成一个 Candidate

每个 Candidate 必须包含：

- `candidate_id`；
- 父 Reference SHA；
- 一个可证伪假设；
- 预期影响；
- 风险；
- 只涉及 allowlist 的 diff；
- Penguin Session/Trace ID。

例如：

```yaml
candidate_id: cand-0001
parent_sha: <reference-sha>
hypothesis: >-
  强制 TechLead 为每个并行方向声明不同的 bottleneck 和 focus_files，
  可以降低重复探索并改善预算利用率。
changed_paths:
  - kernel_workflow/roles/tech_lead.md
expected_effects:
  - lower_duplicate_direction_rate
  - higher_verified_suite_score
risks:
  - excessive_diversity_may_skip_the_best_strategy_family
```

### Step 5：静态门禁

Controller 在使用 GPU 前检查：

- Candidate 父 SHA 正确；
- 只修改允许文件；
- protected paths 哈希不变；
- diff 大小未超限；
- Prompt 没有要求跳过正确性或伪造性能；
- JSON/YAML schema 合法；
- 相关单元测试通过。

失败则立即拒绝。

### Step 6：建立临时 worktree

```text
Reference SHA
  └── worktrees/cand-0001/
        └── apply candidate.diff
```

不在用户当前 GEAK checkout 上直接修改。

### Step 7：Canary 评测

先运行 1–2 个快速 case，发现：

- Workflow 无法启动；
- JSON 输出格式损坏；
- correctness 失败；
- 明显性能回归；
- 超时或超预算。

Canary 失败立即拒绝，不消耗完整 Dev 评测成本。

### Step 8：Dev 评测

对 Reference 和 Candidate 做同环境配对评测。建议随机或交错执行：

```text
Reference → Candidate → Candidate → Reference
```

最终性能只读取 Director verified 的 `final_speedup`，不能使用 Engineer 或 TechLead 自报数字。

### Step 9：隐藏 Holdout

只有 Dev 通过的 Candidate 才进入 Holdout。优化 Agent只得到聚合结果，例如：

```json
{
  "passed": false,
  "suite_relative_gain": -0.004,
  "reason": "holdout_regression"
}
```

不向它返回隐藏任务内容和逐题调 Prompt 线索。

### Step 10：接受或拒绝

必须全部通过：

- correctness；
- Director `validation_status == accepted`；
- 完整性检查；
- Dev 最小收益；
- Holdout 最小收益；
- 单 kernel 最大回归限制；
- 统计置信度；
- 成本和墙钟上限。

接受后提交 `penguin-evolution` 分支；拒绝后删除临时 worktree，但保留 diff、假设和失败摘要。

### Step 11：进入下一轮

接受的 Candidate 成为新 Reference。Penguin 根据新证据提出下一个单一假设，直到：

- 达到目标；
- 达到最大轮数；
- 用完 Token/GPU 时间预算；
- 连续多轮没有可信改进；
- 用户停止。

## 11. 评测与接受标准

### 11.1 主指标

每个 kernel 使用 GEAK 返回的：

```text
final_speedup
```

如果启用了 workload alignment，它对应时间加权性能；否则对应无权重 geomean。

跨 kernel 指标建议为：

```text
suite_score = exp(mean(log(final_speedup_i)))
```

### 11.2 硬门禁与统计门槛

初始建议：

```yaml
all_correct: true
validation_status: accepted
integrity_checks_passed: true
valid_run_rate_not_lower_than_reference: true
dev_suite_relative_gain_min: 0.02
holdout_suite_relative_gain_min: 0.01
per_kernel_regression_limit: 0.03
confidence: 0.95
wall_time_regression_limit: 0.20
```

这些数字必须根据 Reference 自身的噪声重新校准。如果 Reference 的 95% 波动已经达到 ±2%，就不能把 Candidate 的 1% 提升认定为真实改善。

### 11.3 辅助指标

- 正确运行率；
- Candidate win rate；
- 重复方向率；
- Verifier 驳回率；
- 每单位 GEAK budget 的收益；
- 墙钟时间；
- Penguin 和 Claude Token；
- 估算 API/GPU 成本。

正确性失败不能被其他高分抵消。

## 12. 用户的完整操作工作流

本节描述系统实现完成后，你作为用户实际需要做什么。

### 12.1 一次性准备

1. 安装并构建 PenguinHarness。
2. 在 Penguin 中配置一个可用模型。
3. 安装并登录 Claude Code，或者配置 Claude Agent SDK 所需凭据。
4. 确认 Claude Code 支持 Workflow，并启用 `enableWorkflows`/`ultracode`。
5. 准备符合 GEAK 要求的 GPU、ROCm 和 Python 环境。
6. 确认 `run_kernel.py` 的 smoke test 通过。

这套系统有两套模型调用：

```text
Penguin 模型：生成 GEAK Prompt 候选
Claude 模型：执行 GEAK 内部 Workflow Agents
```

两边都要有有效凭据和独立的成本预算。

### 12.2 创建优化项目

计划命令：

```bash
penguin geak init \
  --dir /home/yuygao/agents/geak-penguin-workflow \
  --geak-repo /home/yuygao/agents/GEAK \
  --target kernel_workflow/roles/tech_lead.md
```

随后编辑：

- `target.yaml`：GPU、预算、允许路径和接受规则；
- `benchmark.yaml`：Canary/Dev/Holdout；
- `OBJECTIVE.md`：优化目标和约束。

### 12.3 检查环境

计划命令：

```bash
penguin geak doctor --config target.yaml
```

它应检查：

- 两个仓库路径和 Git 状态；
- Claude CLI/SDK 与登录状态；
- Workflow capability；
- GPU/ROCm；
- Python 依赖；
- Benchmark case；
- 输出目录权限；
- 模型配置和预算。

### 12.4 建立基线

```bash
penguin geak baseline --config target.yaml
```

用户确认：

- 所有 Reference run 是否正确完成；
- GPU 测量噪声是否可接受；
- Benchmark 是否足够有代表性；
- 预计一次 Candidate 的时间和费用；
- 接受阈值是否高于噪声。

### 12.5 先评测一个人工 Candidate

在开启自动进化前，建议先验证评测系统：

```bash
penguin geak evaluate \
  --config target.yaml \
  --candidate candidates/manual-smoke/manifest.yaml
```

至少测试：

- 一个无行为变化 Candidate，系统应判为“不确定/拒绝”；
- 一个故意破坏 Prompt 的 Candidate，系统应识别回归；
- 一个试图修改 protected path 的 Candidate，应在 Tier 0 被拒绝。

### 12.6 启动 Penguin 自动优化

```bash
penguin geak optimize \
  --config target.yaml \
  --rounds 5
```

也可以用 Penguin Goal 模式启动上层 Agent：

```bash
penguin run \
  --agent-id geak-optimizer \
  --workspace /home/yuygao/agents/geak-penguin-workflow \
  --approve allow-all \
  --goal 500k \
  -m "按照 OBJECTIVE.md 和 target.yaml，完成最多 5 轮 GEAK TechLead 元优化。"
```

更推荐第一条确定性命令；它内部再调用 Penguin SDK。Goal 模式适合交互探索，不应替代代码级完整性和接受逻辑。

### 12.7 观察进度

```bash
penguin geak status --config target.yaml
```

用户应看到：

- 当前 Reference SHA；
- 当前轮次和 Candidate；
- 所处评测层级；
- 已用 GPU 时间、Token 和费用；
- 每个 case 的完成状态；
- 已接受/拒绝候选；
- 当前最佳 suite score；
- 是否等待人工审批。

### 12.8 审批 Candidate

MVP 默认不自动合并主分支。用户 review：

1. Candidate diff 是否只修改预期文件；
2. 假设是否与证据一致；
3. Dev/Holdout 是否都通过；
4. 是否出现重要单任务回归；
5. 成本是否可以接受；
6. protected hashes 是否全部通过。

批准后 Candidate 进入 `penguin-evolution` 分支。之后再由用户决定是否合并 GEAK 主分支。

### 12.9 查看结果

主要结果位于：

```text
geak-penguin-workflow/scoreboard.yaml
geak-penguin-workflow/hypothesis_ledger.yaml
geak-penguin-workflow/evaluations/<evaluation_id>/result.json
geak-penguin-workflow/candidates/<candidate_id>/candidate.diff
```

最终报告应回答：

- Penguin 修改了什么；
- 为什么提出这个修改；
- 哪些 case 改善或退化；
- Dev 和 Holdout 的统计收益；
- 正确性和完整性是否通过；
- 使用了多少 Token、GPU 时间和费用；
- 接受或拒绝的确定性原因。

### 12.10 回滚

```bash
penguin geak rollback \
  --config target.yaml \
  --to <evaluation_id>
```

回滚通过 Git SHA 完成。Benchmark、scoreboard 和历史 Evidence 不删除，以便审计和避免未来重复尝试同一个失败方向。

## 13. 用户需要提供的输入

首次运行前，用户至少需要确定：

| 输入 | 示例 |
|---|---|
| GEAK 路径 | `/home/yuygao/agents/GEAK` |
| Penguin 路径 | `/home/yuygao/agents/Penguin-harness` |
| 优化单元 | `kernel_workflow/roles/tech_lead.md` |
| GPU | `0` |
| GEAK 单次 budget | `6` |
| 最大进化轮数 | `5` |
| Canary cases | 1–2 个 |
| Dev cases | 5–10 个 |
| Holdout cases | 与 Dev 不重合 |
| 每 case 重复次数 | 至少 3 |
| Penguin 模型 | provider + model ID |
| Claude 模型 | `GEAK_CLAUDE_MODEL` |
| Token/GPU 时间上限 | 用户预算决定 |
| 是否提交前人工审批 | MVP 建议 `true` |

## 14. 用户最终得到的输出

一次完整运行会产生：

- 一个或多个 Candidate diff；
- 每个候选的假设、证据和风险；
- 每个 case 的 Reference/Candidate 结果；
- correctness 和 Director 验证结果；
- Dev/Holdout 聚合性能及置信区间；
- Token、时间和费用报告；
- accepted/rejected 决策；
- 可复现的环境和内容哈希；
- accepted GEAK Git commit；
- 可回滚的版本历史。

## 15. 安全与防作弊原则

1. Candidate 不能修改评分器。
2. Benchmark 和 correctness oracle 在评测中只读。
3. protected paths 运行前后都校验哈希。
4. 生成 Candidate 的 Agent 不给自己的 Candidate 打分。
5. 只信 Director verified 性能。
6. 每个候选使用独立 Git worktree。
7. 同一 GPU 不并发跑性能 Benchmark。
8. Holdout 对候选生成 Agent 隐藏。
9. 接受逻辑使用确定性代码，不使用 LLM 自由判断。
10. MVP 只提交 evolution 分支，不自动合并主分支。

## 16. 成本与时间预期

总成本大致为：

```text
总成本 ≈ 进化轮数
       × 晋级到各层的候选数
       × Benchmark case 数
       × 每 case 重复次数
       × 单次 GEAK Workflow 成本
```

GEAK 一次运行内部还会调用多个 Claude Agent 并执行 GPU Benchmark，因此成本可能很高。控制成本的主要手段是：

- 先过静态门禁；
- 使用短 Canary；
- 只有 Canary 通过才跑 Dev；
- 只有 Dev 通过才跑 Holdout；
- 限制每轮一个 Candidate；
- 限制 GEAK budget；
- 设置 Token、墙钟和 GPU 小时硬上限；
- 连续无改进时提前停止。

## 17. 实施顺序

### Phase 0：可复现基线

- 选择 Benchmark；
- 固定环境；
- 重复运行当前 GEAK；
- 测量噪声和失败率。

### Phase 1：GEAK Runner

- 抽取 `workflow_driver.py`；
- 实现 `run_kernel.py`；
- 固定结果协议；
- 完成 timeout、后台 Workflow 和产物恢复测试。

### Phase 2：只评测、不自动优化

- 实现 Controller；
- 实现 worktree、完整性、GPU queue 和统计；
- 人工提供 Candidate，验证接受/拒绝链路。

### Phase 3：接入 Penguin

- 增加两个 GEAK 专用 Skill；
- 创建 `geak-optimizer` Agent；
- 让 Penguin 生成单文件候选；
- 完成多轮 hypothesis ledger。

### Phase 4：Holdout 与生产化

- 隐藏评测；
- 成本配额；
- 中断恢复；
- 人工审批；
- evolution 分支和 rollback。

## 18. MVP 成功标准

第一版不以“自动修改了 Prompt”作为成功，而以以下条件作为成功：

- Penguin 只能修改 `tech_lead.md`；
- 系统能可靠拒绝越权、错误和回归 Candidate；
- Reference/Candidate 使用相同环境和配置；
- 所有分数来自 Director verified 结果；
- 不污染用户当前 GEAK checkout；
- 每次接受或拒绝都有可审计证据；
- 至少一个 Candidate 在 Dev 上超过噪声带；
- 该 Candidate 在独立 Holdout 上仍然改善；
- 用户可以完整回滚。

## 19. `workload_spec` 与本文 Workflow 的区别

GEAK 中还存在一个叫 `workload_spec` 的概念，它描述真实模型运行时某个 kernel 的 shape、dtype、调用次数和性能权重。例如可以从 Torch Trace 生成：

```bash
python3 GEAK/e2e_workflow/scripts/parse_profile.py \
  --torch-trace trace.json \
  --target <kernel_name> \
  --workload-out workload.json
```

这个 `workload.json` 用来告诉 GEAK“哪些 case 对真实端到端性能更重要”。它可以作为 Meta-Benchmark 的输入，但它不是 Penguin 优化 GEAK 的外层工作流。

本文设计的是：

```text
Penguin → 优化 GEAK
```

GEAK `workload_spec` 描述的是：

```text
GEAK → 应该针对什么真实 kernel 负载进行优化
```

二者可以结合，但不能互相替代。

## 20. 最终建议

推荐按下面的最短路径开始：

```text
先选 1 个 Canary + 5 个 Dev kernel
        ↓
实现 GEAK run_kernel.py
        ↓
实现只评测模式，不接 Penguin
        ↓
验证无变化、回归、越权三类 Candidate
        ↓
接入 Penguin，只允许修改 tech_lead.md
        ↓
完成 3–5 轮自动进化
        ↓
最后增加隐藏 Holdout 和自动提交
```

最重要的是先让评测和回滚可信，再让 Penguin 自动产生改动。只有 Candidate 在冻结的 Dev 和隐藏 Holdout 上都通过正确性与真实性能门槛，才能说明 Penguin 确实优化了 GEAK。
