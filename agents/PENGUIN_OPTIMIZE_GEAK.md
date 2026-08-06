# 用 Penguin 优化 GEAK：落地手册

> 文档目的：让人或 agent 照着本文，从零搭起「Penguin 自动优化 GEAK 提示词资产」的闭环。
> 文档状态：设计 + 步骤已就绪；标注为「待实现」的文件尚不存在，需按本文创建。
> 事实标注：带 `文件:行号` 的结论均已在代码中核对（汇总见附录 B）；未标注者为设计决策。
> 环境基线（已核对）：**8× AMD Instinct MI355X**、ROCm 7.2、torch 2.9.1+rocm7.2、Python 3.12、Node 24.19、pnpm 11.18、Claude Code 2.1.220。

**开工前必读**：§2.3 指出的 runner 缺口是整个项目的第一个阻塞项，§6 指出的 case 荒是第二个。
这两件事不解决，后面所有步骤都无法执行。不要跳过。

---

## 1. 这件事在做什么

两层嵌套的优化：

```
外层：Penguin 修改 GEAK 的提示词资产（roles/*.md、knowledge/*.md）
          ↓
内层：GEAK 用新提示词去优化真实 GPU kernel
          ↓
      拿真实加速比回来，判断这次修改是不是真的让 GEAK 变强了
```

**不是**让 Penguin 直接写更快的 kernel，而是提升 GEAK「选方向、分预算、用历史证据、何时停止」的能力。

### 三个系统的角色

| 系统 | 位置 | 角色 |
|---|---|---|
| GEAK | `/target/agents/GEAK` | **被优化对象**。JS 编排的多 agent kernel 优化流水线，跑在 Claude Code 里 |
| Penguin | `/target/agents/Penguin-harness` | **优化器**。TypeScript agent 框架，自带 benchmark/进化闭环 |
| Hyperloom | `/target/agents/Hyperloom` | 本阶段不涉及，见 §13 |

---

## 2. 三个关键前提

### 2.1 大部分能力不需要改 Penguin 代码

> **「被优化对象必须是 Penguin Agent」是 Skill 提示词里的策略约束，不是引擎的能力边界。**

证据：

- 内置 `exec_command` 支持 `workdir` 并以该目录 spawn 子进程（`exec-command.ts:68-85`）——
  Evaluator 本来就能在任意目录起任意命令
- 内置 `edit_file` / `write_file` / `diff` —— Optimizer 本来就能改任意文件
- `BenchmarkService` 全 381 行只有只读方法（`benchmark-service.ts`），**不执行任何评测**
- 整个进化循环由两个 markdown 驱动

约束的实际出处只有两行提示词：`agent-evaluation/SKILL.md:71`（规定用 `penguin run` 启动）、
`agent-optimization/SKILL.md:65`（规定只能改 `AGENTS.md`/Skills/`system_config.yaml`）。

### 2.2 但「不靠 LLM 自觉」的部分必须是确定性代码

这是本方案的第二条铁律，与 2.1 同等重要。

凡是**候选方自己不该有裁量权**的环节 —— 越权检查、哈希校验、统计判定、接受/拒绝 ——
必须由脚本执行，不能写进 Skill 指望 LLM 遵守。Skill 只负责它擅长的：读证据、提假设、生成候选。

本文的做法是轻量折中：**用几个小脚本承担确定性职责**（§7.2 的 `gate.sh` / `decide.py`），
而不是建一整套 TypeScript 包。职责边界清楚即可，不必重投入。

### 2.3 ⚠ GEAK 缺少 single-kernel 非交互 runner —— 第一阻塞项

**已核对**：`GEAK/interface/` 下只有 `run_e2e.md`、`run_e2e.py`、`test_run_e2e_alignment.py`、
`test_run_e2e_recovery.py`。**没有 `run_kernel.py`，也没有 `workflow_driver.py`。**

也就是说，e2e 工作流有外部调用入口，**single-kernel 工作流没有**。而本方案要反复自动化调用的
恰恰是 single-kernel 工作流。

正确的调用链是：

```
Penguin (exec_command)
  └── GEAK run_kernel.py          ← 不存在，必须新建
        └── Claude Code
              └── Workflow
                    └── kernel_workflow.js
```

不能指望 `node kernel_workflow.js` 直接跑 —— 它是 Claude Code Workflow 的编排脚本，不是独立程序。

**工期影响**：建这个 runner（含从 `run_e2e.py` 抽取公共 driver）是**数天**的活，不是几小时。
这是阶段 1 的全部内容，必须先完成。

---

## 3. GEAK 的资产结构（优化面在哪）

### 3.1 加载机制

`kernel_workflow/kernel_workflow.js:393-394` 给每个 agent 的固定开场白：

```
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
```

**一个 role 文件内部按 `## PHASE=xxx` 分段，同一文件在流水线不同阶段被反复调用。**

| 流水线阶段 | role 文件 | PHASE |
|---|---|---|
| 建隔离环境 | `director.md` | `setup` |
| 分析 + 路线图 | `tech_lead.md` | `analyze` |
| 建测量契约 | `benchmark_engineer.md` | — |
| 基线 profile | `profile_engineer.md` | `baseline` |
| 每轮：派方向 | `tech_lead.md` | `plan_round` |
| 每轮：优化（并行 ×4） | `engineer.md` / `deep_engineer.md` | — |
| 每轮：独立验证 | `verify_engineer.md` | — |
| 每轮：合并 | `integrator.md` | — |
| 每轮：再 profile | `profile_engineer.md` | `reprofile` |
| 每轮：更新记忆 | `tech_lead.md` | `update_memory` |
| 最终仲裁 | `director.md` | `validate` |
| 最终报告 | `tech_lead.md` | `report` |

### 3.2 三层资产

| 层 | 位置 | 文件数 | 性质 |
|---|---|---|---|
| L1 决策指令 | `kernel_workflow/roles/` | 9 + fragment | 角色行为，直接注入 |
| | `e2e_workflow/roles/` | 8 + fragment | 同上 |
| L2 专题知识 | `kernel_workflow/knowledge/` | 8 | 按 specialty 选读 |
| | `e2e_workflow/knowledge/` | 36 | 含 `learned/`、`gemm_tuning/` |
| L3 参考库 | `perf_knowledge/` | **642** | REFERENCE ONLY |

`perf_knowledge/` 分布：`operators/` 441、`languages/` 49、`backends/` 35、`hardware/` 27、`index/` 16、其余各约 10。

**L3 不适合做优化对象**。`kernel_workflow.js:68` 注释：

> knowledge base — REFERENCE ONLY (facts/how-to, **never decisions**; the author always measures regardless)

它是信息供给而非决策，且 642 个文件无法把评测信号归因到单个卡片。

### 3.3 specialty → knowledge 映射（`engineer.md:24-31`）

| specialty | 加载的 knowledge |
|---|---|
| algorithm | `hip_optimization`(P0/P1) 或 `triton_optimization` + `geomean_levers` |
| memory | `hip_optimization`(P1/P2) 或 `triton_optimization` + `amd_instinct` |
| compute | `hip_optimization`(P3/P4) + `amd_instinct` |
| host_runtime | `wrapper_optimization` + `geomean_levers` |
| **全部** | **`self_monitoring`（Always also read）** |

---

## 4. 可改与禁改清单

### 4.1 禁改（protected）—— 安全底线

**本节是整个方案的安全底线，实施时不得放宽。而且不能只靠声明，必须靠 §7.2 的哈希校验强制。**

| 文件 | 禁改原因 |
|---|---|
| `kernel_workflow/kernel_workflow.js` | 确定性编排。README 强调 budget loop / parallelism / verification 刻意放在 JS 而非提示词里 |
| `kernel_workflow/roles/verify_engineer.md` | **评分数据来源**。「信任锚点」，把候选 patch 应用到干净副本独立重测 |
| `kernel_workflow/roles/benchmark_engineer.md` | **测量契约**。建立所有 agent 必须使用的不可变测量基础设施 |
| `kernel_workflow/roles/director.md` | `PHASE=validate` 对着 TRUE original baseline 独立仲裁 |
| `kernel_workflow/scripts/**` | 测量与校验脚本 |
| `perf_knowledge/index/capability_index.yaml` | 文件头标注 AUTO-GENERATED，同目录有 `_gen_sources.py`，手改会被覆盖 |
| `perf_knowledge/operators/**` | REFERENCE ONLY，441 个文件无法归因 |
| benchmark case / oracle / reference 数据 | 评分依据本身 |
| `gate.sh` / `decide.py` / `score.sh` | 评分器与接受逻辑 |

> **为什么这条最重要**：分数来自 verify_engineer 产出的 verified 数字和 benchmark_engineer 建立的测量契约。
> 若把这些交给 Optimizer，它可以通过**放宽验证标准**提分 —— 那不是让 GEAK 变强，是让尺子变松。

`director.md` 的 setup 与 validate 在同一文件，本阶段整体冻结；后续若要优化 setup，需先按 PHASE 拆分。

### 4.2 可改（editable）—— 分阶段解锁

```toml
# 阶段一（仅一个文件，先验证链路）
"kernel_workflow/knowledge/self_monitoring.md"

# 阶段二
"kernel_workflow/knowledge/geomean_levers.md"
"kernel_workflow/knowledge/optimization_strategies.md"
"kernel_workflow/knowledge/hip_optimization.md"
"kernel_workflow/knowledge/triton_optimization.md"
"kernel_workflow/knowledge/profiling_guide.md"

# 阶段三
"kernel_workflow/roles/tech_lead.md"
"kernel_workflow/roles/engineer.md"
"kernel_workflow/roles/deep_engineer.md"
"perf_knowledge/index/decision_trees.md"
```

**注意**：`roles/_fragments/expert_skills.md` 默认无效 —— 文件开头写明仅当 `use_expert_skills` 为 true 时注入，
默认 OFF 时「nothing is injected and behavior is byte-identical」。不显式开启则优化它毫无效果。

### 4.3 为什么阶段一选 `self_monitoring.md`

- **影响面最大**：`engineer.md:31` 写着 "Always also read"，每个 engineer 每次都加载
- **内容是纯数值阈值**，改动因果最容易归因：
  - 停滞：`steps_since_improvement >= 8` 警告 → `>=12` 强制提交 → `>=15` 立即提交
  - 死胡同：同类策略连续失败 3 次 → 换类别
  - 天花板：最近 3 次 benchmark 相差 1% 以内 → 停止调参
  - 崩溃循环：连续 3 次相同错误 → 回原始源码重来
  - 多样性：连续 3 次同类改动 → 强制换类别
- 单文件 92 行，Optimizer 每轮只提一个 bounded candidate，收敛快

> 另一个合理选择是 `tech_lead.md`（决定方向与预算，杠杆更大）。本文不选它，是因为它 329 行、
> 承担 analyze/plan_round/update_memory/report 四个 PHASE，因果链更长、更难归因。
> 等阶段一验证链路可信后再解锁它。

**方法论原则：editable 集合的大小必须匹配评测信号的分辨率。**

---

## 5. 架构设计

### 5.1 Holder Agent 模式

`benchmarksDir` 硬编码为 `<agentDir>/benchmarks`（`paths.ts:142-144`），因此保留一个 Penguin Agent 作为
**holder**，白嫖它的 scoreboard 存储和 Web UI 只读视图；真正被改的是 GEAK 仓库里的原文件。

Penguin agent 目录布局（`paths.ts` 核对）：

```
<PENGUIN_HOME>/<projectId>/
├── .project_config.toml
└── agents/<agentId>/
    ├── agent_state/
    │   ├── system_config.yaml
    │   ├── AGENTS.md
    │   ├── skills/<name>/SKILL.md
    │   ├── tools/  memory/  schedule/
    │   └── .vault.toml
    ├── benchmarks/<benchmark_id>/     ← benchmark 挂这里
    ├── workspaces/                    ← Evaluator 建隔离工作区
    ├── traces/
    └── scratchpad/<sessionId>/
```

### 5.2 数据流

```
Optimizer Session（LLM 负责：读证据、提假设、生成候选）
  │
  ├─ 1. 读 adapter.toml + 历史 scoreboard + hypothesis ledger
  ├─ 2. 提出一个可证伪假设，改 editable 文件，产出 candidate diff
  │
  ├─ 3. gate.sh 静态门禁（确定性，不动 GPU）  ← 越权/哈希/diff 上限
  │      失败 → 立即拒绝，零 GPU 成本
  │
  ├─ 4. Canary（1–2 case × 1 run）           ← 快速崩坏筛查
  │      失败 → 拒绝，不进 Dev
  │
  ├─ 5. Dev（5–10 case × 3 runs，配对交错）  ← 优化驱动信号
  │      每个 Evaluator：
  │        gpu_lease.sh 取独占 GPU
  │        → run_kernel.py 跑 GEAK
  │        → verify_engineer 产出 verified 加速比
  │        → score.sh 换算 0–100
  │        → 返回协议 YAML
  │
  ├─ 6. Holdout（Optimizer 不可见内容，只回聚合结果）  ← 泛化裁判
  │
  ├─ 7. decide.py 统计判定（确定性）         ← 接受/拒绝
  └─ 8. 接受 → 提交 evolution 分支（人工审批后）
         拒绝 → 删 worktree，保留 diff + 假设 + 失败摘要
```

### 5.3 相比 Penguin 原生流程的差异

| 环节 | 原生 | 本方案 |
|---|---|---|
| 启动被测对象 | `penguin run --agent-id ...` | `exec_command` → `run_kernel.py` |
| 可编辑状态 | `AGENTS.md` / Skills | adapter 声明的 glob + **哈希强制** |
| 隔离 | 复制 workspace | `git worktree` |
| 快照回滚 | `snapshots/v<N>.tar.gz` | git commit / branch |
| 评分 | LLM 判 rubric | **`score.sh` 客观算分** |
| 评测层级 | 单层 | **Canary / Dev / Holdout 三层** |
| 接受判定 | LLM 比较分数 | **`decide.py` 统计检验** |
| 其余（candidate 循环、污染防护、scoreboard 格式） | — | **完全沿用** |

---

## 6. 评测设计（本方案的可信度基础）

### 6.1 三层的职责分离

它们不是「测三遍」，而是分工：

| 层 | 规模 | Optimizer 可见性 | 回答什么问题 |
|---|---|---|---|
| **Canary** | 1–2 个快 case × 1 run | 可见 | 有没有明显崩坏？（起不来/JSON 坏/正确性挂/明显回归/超时） |
| **Dev** | 5–10 case × 3 runs | **逐 case 详细可见** | 该往哪个方向改？ |
| **Holdout** | 与 Dev 不重合，5+ | **仅聚合结果** | 提升是真本事，还是只记住了 Dev 的答案？ |

**关键在 Dev 与 Holdout 的分工。** Optimizer 必须看到 Dev 的详细结果，否则无从改进；
但这个「看到」就是信息泄露通道。所以设计成 **Dev 给信息，Holdout 当裁判**。

Holdout 只返回类似：

```json
{"passed": false, "suite_relative_gain": -0.004, "reason": "holdout_regression"}
```

**不得**返回隐藏 case 的内容、逐题得分或任何可用于反推的线索。一旦泄露，Holdout 退化成第二个 Dev，失去意义。

### 6.2 为什么单层不行（具体例子）

`self_monitoring.md` 里有 `steps_since_improvement >= 8 → 换方向`。

假设 Optimizer 改成 `>= 15`。在 softmax/layernorm 这类**小 kernel** 上，搜索空间小、每步便宜，
多熬几步更容易撞到好结果 —— 分数涨了，单层方案判定「接受」。

但换成大 GEMM，15 步不改进还硬撑就是白烧预算。**这个改动在 Dev 上是提升，在真实场景是退化。**
没有 Holdout，这个结论永远不会被推翻，还会被 commit 进 GEAK。

跑 8 轮后得到的将是「一份针对这几个小 kernel 特调的 self_monitoring.md」，而非「更好的 self_monitoring.md」。

### 6.3 ⚠ Case 荒 —— 第二阻塞项

**已核对**：`GEAK/examples/tasks/` 下只有 **1 个** case（`knn`）。

三层需要 **12–17 个互不重合的 kernel**（Canary 1–2 + Dev 5–10 + Holdout 5+）。缺口巨大。

Case 来源候选（需自行评估）：
- 从 `/target/aiter`、`/target/vllm` 提取真实 kernel
- GEAK 依赖 `datasets`，可考虑接现成 kernel 数据集
- 手工构造（softmax / layernorm / rmsnorm / 小 GEMM / attention 片段）

**选 case 的标准**：编译 + benchmark 快（单 case 20–40 分钟内），覆盖不同瓶颈、shape、dtype、实现方式。
Dev 与 Holdout 必须在瓶颈类型上有可比性，否则 Holdout 的否决说明不了问题。

**这是一块独立工作量，必须计入排期。**

### 6.4 配对交错评测

Reference 和 Candidate 必须在**同一时段、同环境**下交错执行：

```
Reference → Candidate → Candidate → Reference
```

理由：GPU 温度、系统负载、后台任务都会随时间漂移。若先跑完所有 Reference 再跑 Candidate，
漂移会被误读为改动效果。交错可抵消一阶漂移。

### 6.5 统计门槛（不能用固定阈值）

**核心原则：接受门槛必须由 Reference 自身噪声决定，不能拍脑袋。**

> 如果 Reference 的 95% 波动已经达到 ±2%，就不能把 Candidate 的 1% 提升认定为真实改善。

初始建议（**必须按实测噪声重新校准**）：

```yaml
all_correct: true                        # 正确性，无条件
validation_status: accepted              # GEAK Director 仲裁通过
integrity_checks_passed: true            # 哈希门禁
valid_run_rate_not_lower_than_reference: true
dev_suite_relative_gain_min: 0.02
holdout_suite_relative_gain_min: 0.01
per_kernel_regression_limit: 0.03        # 防止均值涨、单点崩
confidence: 0.95
wall_time_regression_limit: 0.20
```

跨 kernel 汇总用几何平均：`suite_score = exp(mean(log(speedup_i)))`

`per_kernel_regression_limit` 尤其重要：没有它，一个把 A 提升 20%、把 B 拖垮 30% 的候选也可能因均值为正而通过。

### 6.6 环境指纹

每次评测记录，关键项变化时**必须重建 Reference**，不能与旧结果比较：

- GEAK git SHA / Penguin git SHA
- Claude Code 版本、GEAK 内部使用的模型 ID
- Penguin 的 provider / model / thinking_level
- GPU 型号、驱动、ROCm 版本
- Python 及关键依赖版本
- benchmark case 内容哈希、protected paths 哈希

---

## 7. 需要新增的文件

### 7.1 GEAK 侧

| 文件 | 说明 | 工期 |
|---|---|---|
| `interface/workflow_driver.py` | 从 `run_e2e.py` 抽取：Claude SDK 调用、`claude -p` 回退、Workflow 等待、timeout、进程清理、结果 JSON 解析、产物恢复、错误分类 | 数天 |
| `interface/run_kernel.py` | single-kernel 非交互 runner（§2.3） | 同上 |
| `interface/test_run_kernel.py` | runner 的测试 | — |
| `.penguin/gpu_lease.sh` | GPU 独占租约 | 1h |
| `.penguin/score.sh` | verified 加速比 → 0–100 | 1h |

`run_kernel.py` 计划调用形式：

```bash
python3 interface/run_kernel.py \
  --kernel-path <case> --workflow-dir <candidate>/kernel_workflow \
  --budget 6 --gpu-ids 0 --eval-dir <out> --result-json <out>/result.json
```

**无论成功、失败还是超时，都必须写出一个合法 `result.json`**，否则上层无法区分「跑挂了」和「没跑」。

结果协议建议至少包含：`schema_version`、`status`、`kernel_id`、`reference_geak_sha`、
`candidate_geak_sha`、`verification{correctness_passed, validation_status, final_speedup, final_geomean}`、
`resources{budget_used, wall_time_seconds, tokens}`、`environment{gpu, rocm, claude_version, model_id}`、
`artifacts{eval_dir, report_path, director_validation, final_patch}`、`error`。

### 7.2 确定性脚本（不能靠 LLM 自觉的部分）

| 文件 | 职责 |
|---|---|
| `gate.sh` | **静态门禁**：父 SHA 正确、只改 editable、protected paths 哈希不变、diff 行数上限、schema 合法、单测通过、候选内容未要求跳过正确性或伪造性能。**在动 GPU 之前执行，失败即拒** |
| `decide.py` | **统计判定**：配对比较、置信区间、per-kernel 回归检查、按 §6.5 门槛输出 accept/reject |

这两个脚本本身列入 protected（§4.1）。

### 7.3 Penguin 侧

| 文件 | 说明 |
|---|---|
| `external-agent-evaluation/SKILL.md` | 改写 `agent-evaluation`：启动改为 `exec_command` 调 `run_kernel.py`（外套 `gpu_lease.sh`）；隔离改为 `git worktree`；评分改为跑 `score.sh` |
| `external-agent-optimization/SKILL.md` | 改写 `agent-optimization`：可编辑集改为 adapter 的 `editable`；快照改为 git；**接受判定交给 `decide.py`，Skill 不自行裁决** |
| `adapter.toml` | 声明 target（见 7.4） |

安装到 `<agentDir>/agent_state/skills/<name>/SKILL.md`。

### 7.4 `adapter.toml` 模板

```toml
[target]
repo          = "/target/agents/GEAK"
worktree_base = "/target/agents/.geak-worktrees"
runner        = "interface/run_kernel.py"
budget        = 6
timeout_seconds = 7200

editable  = ["kernel_workflow/knowledge/self_monitoring.md"]
max_changed_files = 1
max_diff_lines    = 200

protected = [
  "kernel_workflow/kernel_workflow.js",
  "kernel_workflow/roles/verify_engineer.md",
  "kernel_workflow/roles/benchmark_engineer.md",
  "kernel_workflow/roles/director.md",
  "kernel_workflow/scripts/**",
  "perf_knowledge/index/capability_index.yaml",
]

snapshot = "git"
branch   = "penguin-evolution"
human_approval_before_commit = true

[scoring]
mode   = "program"
script = ".penguin/score.sh {result_json}"
gate   = "gate.sh"
decide = "decide.py"

[benchmark]
canary_runs  = 1
dev_runs     = 3
holdout_runs = 3
expose_holdout_details_to_optimizer = false

[resources]
gpus          = 8
gpus_per_case = 1
lease_script  = ".penguin/gpu_lease.sh"
```

### 7.5 Evaluator 返回的协议 YAML

**必须逐字段沿用原格式**（`agent-evaluation/SKILL.md:101-114`），否则 scoreboard 与 Web UI 无法解析：

```text
protocol_version: 1
status: ok
case_id: <case_id>
run: <run>
expected_version: <version>
provider: <actual_provider>
model_id: <actual_model_id>
thinking_level: <configured_thinking_level>
score: <0_to_100>
cost: <number_or_null>
duration_ms: <non_negative_integer>
session_id: <test_session_id>
```

失败时 `status: failed`，缺失身份字段填 `null`，且**不得包含** score / cost / duration / session_id
（`agent-evaluation/SKILL.md:137`）。

约束沿用：评分项总分恰好 100；`score` 两位小数；`duration_ms` 非负整数；成本缺失时 `cost: null` 且不因此判无效。

### 7.6 从原 Skill 必须继承的规则

新 Skill 是改写而非重写，以下条款**逐条保留**：

- **污染隔离**（`agent-optimization/SKILL.md:44`）：rubric / gold / 私有评分条件 / Evaluator 状态 / Trace
  一旦进入 Optimizer 上下文，立即恢复当前 candidate 并以「已污染」停止
- **rubric 隔离**（`agent-evaluation/SKILL.md:56`）：只把 `statement/` 复制进工作区
- **一轮一个 bounded candidate**
- **无效评测与修复重跑不计入轮数**；执行失败时保持同一 candidate，只补齐失败单元
- **格式不合规时**由同一 Evaluator 基于已有结果重发协议 YAML，**不重新运行**被测对象

---

## 8. 手把手步骤

### 阶段 0：准备（半天）

**0.1 确认工具链**

```bash
penguin --version    # 0.2.0
node --version       # v24.19.0
claude --version     # 2.1.220
rocm-smi --showid | grep -c '^GPU\['   # 应为 8
```

改了 Penguin 的 TS 源码需 `pnpm build` 才生效（`penguin` 软链指向 `packages/cli/dist/index.js`）。

**0.2 为 GEAK 建专用分支，不要在日常 checkout 上跑**

```bash
cd /target/agents/GEAK
git checkout -b penguin-evolution
mkdir -p .penguin
mkdir -p /target/agents/.geak-worktrees
```

**0.3 建 holder agent**，记下 `PENGUIN_HOME` / `projectId` / `agentId`。

**0.4 记录环境指纹**（§6.6 清单），存入项目目录。

---

### 阶段 1：建 GEAK runner（数天，第一阻塞项）

**这是整个项目最大的一块工程，不能跳过，也不要低估。**

1. 通读 `interface/run_e2e.py`，识别可复用部分
2. 抽出 `interface/workflow_driver.py`：Claude SDK 调用、`claude -p` 回退、
   `enableWorkflows` 设置、Workflow 同步与后台完成等待、timeout 与进程清理、
   结果 JSON 解析、磁盘产物恢复、错误分类
3. 实现 `interface/run_kernel.py`（§7.1 的调用形式与结果协议）
4. 写 `test_run_kernel.py`，覆盖：正常完成、超时、Workflow 启动失败、结果 JSON 损坏
5. **验收**：对 `examples/tasks/knn` 手工跑通一次，产出合法 `result.json`

**在这一步跑通之前，不要开始写任何 Skill。**

---

### 阶段 2：准备 benchmark case（数天，第二阻塞项）

按 §6.3 补足 case。最低要求：

- Canary：1–2 个（挑最快的）
- Dev：5 个起步（10 个更好）
- Holdout：5 个，与 Dev 不重合

目录结构：

```
<agentDir>/benchmarks/geak-selfmon/
├── adapter.toml
├── benchmark.yaml          # canary / dev / holdout 三组清单
├── <case-id>/{statement/,rubric/}
└── scoreboard.yaml
```

`statement/`（被测对象可见）：kernel 路径、shape、dtype、目标、轮数预算。
`rubric/`（被测对象**永不可见**）：本方案下极简 —— 声明满分 100 由 `score.sh` 按 verified 加速比给出，
正确性不过为 0，总分恰好 100。

Holdout 的 case 清单建议存在 Optimizer 无权读取的路径。

---

### 阶段 3：写确定性脚本（1–2 天）

**3.1 `.penguin/gpu_lease.sh`** —— 从 8 张卡取一张独占

```bash
#!/usr/bin/env bash
# 用法: gpu_lease.sh <command...>
set -euo pipefail
LEASE_DIR=/tmp/geak-gpu-leases
mkdir -p "$LEASE_DIR"
for gpu in 0 1 2 3 4 5 6 7; do
  exec {fd}>"$LEASE_DIR/gpu$gpu.lock"
  if flock -n "$fd"; then
    export HIP_VISIBLE_DEVICES="$gpu"
    exec "$@"
  fi
  exec {fd}>&-
done
echo "no free GPU" >&2; exit 75    # EX_TEMPFAIL
```

> kernel benchmark 必须独占 GPU，否则测得的加速比不可信 —— 这是整套评分可信度的物理前提。

**3.2 `.penguin/score.sh`** —— verified 加速比 → 0–100

```bash
#!/usr/bin/env bash
# 用法: score.sh <result_json>
set -euo pipefail
python3 - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
v = r.get("verification", {})
if not v.get("correctness_passed"):
    print("0.00"); sys.exit(0)          # 正确性不过 = 0 分，无条件
if v.get("validation_status") != "accepted":
    print("0.00"); sys.exit(0)          # Director 未通过 = 0 分
s = float(v["final_speedup"])            # 只用 verified 数字
pts = [(1.0,0),(1.5,60),(2.0,85),(2.5,100)]
if s <= 1.0: out = 0.0
elif s >= 2.5: out = 100.0
else:
    for (x0,y0),(x1,y1) in zip(pts, pts[1:]):
        if x0 <= s <= x1:
            out = y0 + (y1-y0)*(s-x0)/(x1-x0); break
print(f"{out:.2f}")
PY
```

**两条硬规则**：正确性或 Director 验证不过 → 直接 0 分，不给部分分（否则 Optimizer 会学会牺牲正确性换速度）；
只认 verified 数字，绝不用 engineer 自报值 —— `verify_engineer.md` 开头写明自报数字
"may be noisy, measured against the wrong baseline, or wrong"。

**3.3 `gate.sh`** —— 静态门禁，按 §7.2 职责实现。核心是 protected paths 的哈希在候选应用前后都校验。

**3.4 `decide.py`** —— 统计判定，按 §6.5 门槛实现。

---

### 阶段 4：验证评测系统本身（半天）—— 不可跳过

**在接入 Penguin 之前，先用人工候选给评测系统打靶。** 三类：

| 人工候选 | 期望结果 |
|---|---|
| 无行为变化（如只改注释） | 判「不确定」或拒绝，**不应**报告显著提升 |
| 故意劣化（如把停滞阈值改成 1） | 识别为回归并拒绝 |
| 越权（尝试改 `verify_engineer.md`） | 在 `gate.sh` 阶段拒绝，**零 GPU 消耗** |

三项全过，才说明评测系统可信。**任一项不过，先修评测系统，不要往下走。**

---

### 阶段 5：建立 Reference Baseline（一夜）

1. `self_monitoring.md` 保持未修改状态
2. 全 Dev 集各跑 3 runs（8 卡并行）
3. 统计：正确运行率、`final_speedup` 分布、suite geomean、**Reference 自身噪声范围**
4. 按实测噪声**重新校准** §6.5 的门槛

**硬闸门**：若 Reference 的 95% 波动已接近或超过计划的接受门槛，说明方差压过信号。
此时必须先收紧（固定种子、减少轮数、换更稳的 kernel、增加 runs），**不得进入阶段 6**。

---

### 阶段 6：写两个 Skill（1 天）

按 §7.3、§7.6 改写。关键：**Skill 不自行裁决接受与否**，只负责调 `gate.sh` / `decide.py` 并遵从其输出。

---

### 阶段 7：跑优化循环

在**新的顶层 Session**（与建基线的 Session 分开）启动：

```
使用 external-agent-optimization skill。
adapter: <agentDir>/benchmarks/geak-selfmon/adapter.toml
dev runs=3, holdout runs=3。最多 8 个有效 candidate 轮次。
```

每轮：提假设 → 改文件 → `gate.sh` → Canary → Dev → Holdout → `decide.py` → 接受则等人工审批后提交 evolution 分支。

---

## 9. 成本预期

单次 GEAK run 按 20–40 分钟估算，8 卡并行：

| 环节 | 规模 | 耗时 |
|---|---|---|
| 静态门禁 | — | 秒级，零 GPU |
| Canary | 2 case × 1 run = 2 | ~20 min |
| Dev | 5 case × 3 runs × 2（配对交错含 Reference）= 30 | ~2–3 h |
| Holdout | 5 case × 3 runs = 15 | ~1–1.5 h |
| **单个 candidate（全通过）** | ~47 | **~4 h** |
| **8 轮**（多数候选在 Canary 或 Dev 被拒） | — | **1–3 天** |

漏斗效应显著：被 `gate.sh` 拒的零成本，被 Canary 拒的只花 20 分钟。

**成本控制手段**：先过静态门禁 → 短 Canary → Dev 通过才跑 Holdout → 每轮一个 candidate →
限制 GEAK budget → 设 token/墙钟/GPU 小时硬上限 → 连续无改进提前停止。

注意有**两套模型开销**：Penguin 模型（生成候选）+ Claude 模型（跑 GEAK 内部 agent），需各自预算。

---

## 10. 验收标准

**评测系统可信**（阶段 1–5）：
- [ ] `run_kernel.py` 对 knn 稳定产出合法 `result.json`（含超时/失败路径）
- [ ] 三类人工候选（无变化/劣化/越权）全部得到期望结果
- [ ] Reference 完整，噪声范围已量化，接受门槛已按噪声校准
- [ ] 每个分数可通过 `session_id` 回溯到具体 Session

**优化有效**（阶段 7）：
- [ ] 至少 1 个 candidate 通过 Dev **且**通过 Holdout
- [ ] 无单 kernel 回归超过 `per_kernel_regression_limit`
- [ ] 提升可归因：Optimizer 的假设在预期 case 上得到支持
- [ ] `git log` 上每个被接受的 candidate 是独立 commit，可 review 可回滚
- [ ] 对照实验：还原改动，分数回落

> **MVP 的成功不是「自动改了提示词」，而是「系统能可靠拒绝越权、错误和回归候选」。**
> 先让评测和回滚可信，再让 Penguin 自动产生改动。

---

## 11. 风险清单

| 风险 | 表现 | 对策 |
|---|---|---|
| **Reward hacking** | 分数涨但 GEAK 没变强 | §4.1 protected + `gate.sh` 哈希强制；正确性/Director 不过判 0；只认 verified |
| **过拟合 Dev** | Dev 涨、真实场景退化 | Holdout 隐藏裁判（§6.1） |
| **方差淹没信号** | 同 case 多次 runs 分差巨大 | 固定种子 + 独占 GPU；runs ≥ 3；配对交错；按噪声校准门槛 |
| **时间漂移** | GPU 热/负载变化被误读为效果 | Reference↔Candidate 交错执行 |
| **均值掩盖单点崩** | 平均涨、某 kernel 崩 | `per_kernel_regression_limit` |
| **归因失效** | 说不清改动来自哪 | editable 一次开 1 个文件；一轮一个 bounded candidate |
| **污染** | 私有 rubric/holdout 进了 Optimizer 上下文 | 沿用污染检测，发生即停 |
| **误改工作区** | 改坏日常 checkout | 专用分支 + worktree；最小 glob；哈希校验 |
| **GPU 争抢** | 并发 benchmark 互相干扰 | `gpu_lease.sh` flock 独占 |
| **环境漂移** | 与旧 Reference 不可比 | 环境指纹，关键项变则重建 Reference |

---

## 12. 已知未决问题

1. **`run_kernel.py` 的具体实现**：需通读 `run_e2e.py` 后确定可复用边界，以及
   `kernel_workflow.js` 的参数传递方式（单 kernel / 轮数 / 关闭 deep_explore / `use_expert_skills` 保持 OFF）。
2. **`score.sh` 的字段名**：`verification.correctness_passed` / `final_speedup` /
   `validation_status` 需与 `verify_engineer.md`、`director.md` 的实际 Return JSON 对齐。
3. **Case 来源**：§6.3 的三个候选途径需实际评估可行性与工作量。
4. **Web UI 是否需改**：若要在评测页区分「程序评分」与「LLM 评分」，需改 scoreboard schema 与
   `BenchmarkService`；不改则显示为普通分数，功能不受影响。
5. **snapshot 导入导出**：Web 快照功能认 tar.gz，git ref 形式暂不被 UI 支持。

---

## 13. 后续阶段

**GEAK 深化**：按 §4.2 逐步解锁阶段二、三。特别值得做的是 `e2e_workflow/knowledge/learned/` ——
那是 GEAK 自己沉淀经验的目录（`INDEX.md` + `method-*.md` + 具体案例），说明它已有自我进化雏形。
在那里 Penguin 该优化的不是替它写条目，而是「怎么写条目」的元规则，
让沉淀的经验在后续 run 里更易被检索复用。这是两个自进化系统最自然的接合点。

**Hyperloom**：端到端优化官方 budget 24 小时/次，乘以评测矩阵后不可行 —— 这是物理约束，改代码也解决不了。
可行路径是**不优化「优化器」，优化它内部的「判断者」**：`src/hyperloom/agents/critic/`
（`AGENTS.md` + `SKILL.md` + `actions/*.md`）是纯判断任务，不需要 GPU，且有现成 ground truth ——
历史 session 里每个 patch 后续实测的 `cumulative_gain_pct_validated` 就是答案。
可做离线重放 benchmark，成本远低于 GEAK 真实评测。`critic` 优先，因为它是把关者，
误判代价最大（放过坏 patch 浪费一整轮 24 小时）。不建议碰
`orchestrator/prompts/orchestration.md`（影响整个长流程，无法离线评测）。

**与 `workload_spec` 的关系**：GEAK 有一个 `workload_spec` 概念，描述真实模型运行时某 kernel 的
shape/dtype/调用次数/性能权重（可由 `e2e_workflow/scripts/parse_profile.py` 从 torch trace 生成）。
它回答的是「GEAK 该针对什么负载优化」，本文回答的是「谁来优化 GEAK」。
两者可结合（用它给 benchmark case 加权），但不能互相替代。

---

## 附录 A：关键文件速查

**GEAK**
```
kernel_workflow/kernel_workflow.js          编排（protected）
kernel_workflow/roles/tech_lead.md      329 策略大脑，4 个 PHASE
kernel_workflow/roles/director.md       274 setup + validate（protected）
kernel_workflow/roles/benchmark_engineer.md 186 测量契约（protected）
kernel_workflow/roles/deep_engineer.md  135 开放式深度优化
kernel_workflow/roles/author_engineer.md 132 从零写实现
kernel_workflow/roles/engineer.md       109 专家工人
kernel_workflow/roles/verify_engineer.md 79 信任锚点（protected）
kernel_workflow/roles/profile_engineer.md 72 瓶颈分类
kernel_workflow/roles/integrator.md      65 合并 patch
kernel_workflow/knowledge/self_monitoring.md 92 ★阶段一目标
interface/run_e2e.py                        抽取 driver 的来源
interface/run_kernel.py                     ★待新建
examples/tasks/knn                          唯一现成 case
```

**Penguin**
```
packages/core/src/state/paths.ts                    目录布局
packages/core/src/environment/tools/exec-command.ts 任意 shell（方案基石）
packages/core/src/environment/tools/run-subagent.ts 并行派发
packages/server/src/services/benchmark-service.ts   只读，381 行
packages/skills/skills/agent-evaluation/SKILL.md    改写模板
packages/skills/skills/agent-optimization/SKILL.md  改写模板
packages/docs/content/self-improvement.zh.md        机制说明
```

## 附录 B：本文已核对的事实

| 结论 | 出处 |
|---|---|
| exec_command 支持 workdir 并以之 spawn | `exec-command.ts:68-85` |
| BenchmarkService 全只读 | `benchmark-service.ts`（381 行） |
| 被测对象须为 Penguin Agent 是提示词约束 | `agent-evaluation/SKILL.md:71` |
| 可编辑状态限制 | `agent-optimization/SKILL.md:65` |
| 污染即停止 | `agent-optimization/SKILL.md:44` |
| 只复制 statement，隔离 rubric | `agent-evaluation/SKILL.md:56` |
| 协议 YAML 字段 | `agent-evaluation/SKILL.md:101-114` |
| 失败时不得含 score/session_id | `agent-evaluation/SKILL.md:137` |
| role 按 PHASE 加载 | `kernel_workflow.js:393-394` |
| perf_knowledge 为 REFERENCE ONLY | `kernel_workflow.js:68` |
| self_monitoring 总是加载 | `engineer.md:31` |
| specialty→knowledge 映射 | `engineer.md:24-31` |
| expert_skills 默认 OFF 且行为字节一致 | `roles/_fragments/expert_skills.md` 开头 |
| capability_index.yaml 为自动生成 | 该文件第 1 行 |
| benchmarksDir 路径 | `paths.ts:142-144` |
| **GEAK 无 single-kernel runner** | `interface/` 仅含 run_e2e 系列 |
| **GEAK 自带 case 仅 1 个** | `examples/tasks/` 仅 `knn` |
| `penguin run --goal 500k` 存在 | `packages/cli/src/commands/run.ts:15` |
| 本机 8 张 MI355X | `rocm-smi` |
