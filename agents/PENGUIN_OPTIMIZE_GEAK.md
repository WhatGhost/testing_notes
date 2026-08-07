# 用 Penguin 优化 GEAK：落地手册

> 文档目的：让人或 agent 照着本文，从零搭起「Penguin 自动优化 GEAK 提示词资产」的闭环。
> 文档状态：设计 + 步骤已就绪；标注为「待实现」的文件尚不存在，需按本文创建。
> 事实标注：带 `文件:行号` 的结论均已在代码中核对（汇总见附录 B）；未标注者为设计决策。
> 环境基线（已核对）：**8× AMD Instinct MI355X**、ROCm 7.2、torch 2.9.1+rocm7.2、Python 3.12、Node 24.19、pnpm 11.18、Claude Code 2.1.220、profiler 三件套齐全。

**开工前必读**：§2.3 指出的 runner 缺口是整个项目的第一个阻塞项，§6.3 指出的 case 荒是第二个。
这两件事不解决，后面所有步骤都无法执行。不要跳过。

> 本文只描述**方案与步骤**。实际执行进度、验证结果、实测数据记录在
> [`PENGUIN_OPTIMIZE_GEAK_PROGRESS.md`](PENGUIN_OPTIMIZE_GEAK_PROGRESS.md)。

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

**好消息**：启动机制已通过 `run_e2e.py --dry-run` 完全查明（§2.4），`run_kernel.py` 不再有猜测成分。

### 2.4 GEAK Workflow 的确切启动机制（已实测查明）

GEAK README 给人类的启动方式是交互式的：

```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions
# 然后自然语言：use <GEAK>/kernel_workflow to optimize <GEAK>/examples/tasks/knn
```

自动化路径则由 `interface/run_e2e.py` 封装。**以下每条都是坑，`run_kernel.py` 必须逐条照做**：

| 事项 | 细节 | 出处 |
|---|---|---|
| **不能用 `--effort ultracode`** | 公开版 claude（≥2.1.x，含本机 2.1.220）**会拒绝**该参数。必须改用 `--settings` 注入 `{"enableWorkflows":true,"ultracode":true}`（最高优先级 flag settings 层） | `run_e2e.py:72-80` |
| **SDK 会用自带 CLI** | `claude_agent_sdk` 优先驱动 `claude_agent_sdk/_bundled/claude` 而非 `$PATH` 里的，换系统 claude 无效，须显式覆盖 | `run_e2e.py:82-84` |
| **Workflow 是后台任务** | Workflow 调用被路由成非阻塞后台任务，主 agent turn 几乎立刻结束，需保持 CLI 进程存活才能保住后台 workflow | `run_e2e.py:642-650` |
| **`claude -p` 有 600s 后台等待上限** | 超时后**直接 terminate 后台 workflow**，且退出码仍是 0（静默腰斩）。GEAK 的 kernel_workflow 必然超过 10 分钟，所以这是致命的。解法：设 `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` 无限等待，或走 SDK 持久 client | **实测触发**，提示语出自 CLI |
| **最终 JSON 的获取** | `run_e2e.py:89-95,651-656` 称后台路径不经 transcript 暴露返回值，须轮询磁盘 marker。**但实测**：只要 prompt 里明确要求 "print EXACTLY ONE final line of compact JSON"，CLI 路径**能在 transcript 末行拿到**完整返回值。两条路都要实现：末行解析为主，磁盘哨兵兜底 | `run_e2e.py` + 实测 |
| **`workflow_return.json` 不会自动生成** | 即使 prompt 里写了兜底要求，agent 成功打印后就不再落盘。**`run_kernel.py` 必须自己写这个哨兵**，不能指望 agent 自觉 | 实测 |
| **`claude -p` 会等 stdin 3 秒** | 输出里混入 `Warning: no stdin data received in 3s`。加 `< /dev/null` 显式跳过 | 实测 |
| **args 必须是真 JSON 对象** | 若以 JSON 字符串传入，workflow 读不到 `args.workflow_dir` 会**立即中止** | prompt 内 CRITICAL 段 |
| 允许的工具 | `["Workflow", "Bash", "Read", "Write"]` | `run_e2e.py:70` |
| 默认模型 | `GEAK_CLAUDE_MODEL`，默认 `claude-opus-4-8` | `run_e2e.py:69` |
| SDK 可选 | 本机**未安装** `claude_agent_sdk`，会走 `claude -p` CLI 回退路径 | 实测 |

**prompt 模板**（`run_kernel.py` 照抄结构，替换 scriptPath 与 args）：

```
Invoke the Workflow tool exactly once with:
  scriptPath: "<GEAK>/kernel_workflow/kernel_workflow.js"
  args: {<真正的 JSON 对象>}
CRITICAL: pass `args` as a real JSON OBJECT (a mapping), NOT as a JSON-encoded string.
  If args arrives as a string the workflow cannot read args.workflow_dir and aborts immediately.
<跑完整流水线的说明>
The workflow persists its full return value to "<eval_dir>/workflow_return.json" as its final act;
that file is the source of truth. When it finishes, print EXACTLY ONE final line of compact JSON
that is the Workflow tool's full return value. If for ANY reason that file does not exist when the
tool returns, write that exact return value there yourself with the Write tool before printing.
Print nothing after the JSON line.
```

**「永远产出合法 result.json」的四层递降**（`run_e2e.md` 已给出成熟实现，直接复用）：

1. 完整 transcript 捕获（不只取最后一条 assistant 文本）
2. 扫描整个 transcript 提取最后一个含 `eval_dir` 的 JSON（容忍单行/```json``` 围栏/多行/尾随散文）
3. 落盘哨兵 `<eval_dir>/workflow_return.json`
4. 兜底从磁盘产物重建；重建的结果标记 `recovered_from_disk: true`

**`kernel_workflow.js` 参数清单**（源码读出）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `kernel_path` | **必需** | kernel 目录绝对路径 |
| `workflow_dir` | **必需** | 含 js/roles/knowledge/scripts 的目录 |
| `budget` | 6 | 方向预算 |
| `gpu_ids` | `'0'` | GPU 列表 |
| `min_improve` | **0.02** | round winner 提交门槛 |
| `deep_cost` | 2 | deep_explore 预算消耗（且独占一轮） |
| `exp_root` | `<parent>/exp` | 运行产物根目录 |
| `apply_to_original` | `false` | 是否回写原目录 |

> **注意**：GEAK 自己的 `MIN_IMPROVE` 默认就是 **2%**，与本文 §6.5 独立定出的
> `dev_suite_relative_gain_min: 0.02` 是同一量级。实测噪声是否支持该门槛，见进度文档。

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
  │        gpu_lock.sh 取独占 GPU（+ workspace 级编译缓存隔离）
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
| `interface/run_kernel.md` | 契约文档（照 `run_e2e.md` 写） | — |
| `interface/test_run_kernel.py` | runner 的测试 | — |
| `.penguin/score.sh` | verified 加速比 → 0–100 | 1h |

（GPU 独占**不新建脚本**，复用 GEAK 自带的 `kernel_workflow/scripts/gpu_lock.sh`，见 §8 阶段 3.1。）

`run_kernel.py` 的调用契约 —— **照抄 `run_e2e.py` 的形式，用 JSON 文件而非命令行参数**
（可版本化、参数多时不爆炸、便于程序化生成与留档）：

```bash
python3 interface/run_kernel.py <handoff.json> <result.json> [--dry-run]
```

| 退出码 | 含义 |
|---|---|
| `0` | `result.json.status` 为 `ok` 或 `no_gain` |
| `1` | crash，`result.json.status == "error"` 且含 `error` 字段 |
| `2` | 用法错误 / handoff 不可读 |

`--dry-run` 打印映射后的 workflow args 与 prompt 后退出 0，**不做任何 GPU 工作** ——
用它在 CI 里校验参数映射，也是集成期排障的第一手段。

**无论成功、失败还是超时，都必须写出一个合法 `result.json`**，否则上层无法区分「跑挂了」和「没跑」。
实现方式直接复用 §2.4 的四层递降（完整 transcript 捕获 → 扫描提取 → 落盘哨兵 → 磁盘重建）。

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
gpu_lock      = "kernel_workflow/scripts/gpu_lock.sh"   # GEAK 自带，勿自行实现
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
claude --version     # 需 >= 2.1.177（dynamic Workflow 特性）
rocm-smi | tail -n +5 | grep -cE '^[0-9]'   # GPU 数量，应为 8
        # 别用 `rocm-smi --showid | grep -c '^GPU\['`，它每卡多行会返回 40

# GEAK README 要求的 profiler（三选一即可，缺失会让 profile_engineer 降级）
for p in rocprof-compute rocprofv3 rocprof; do command -v $p || echo "$p 缺失"; done

# 凭据（GEAK 内部要调 Claude，缺了 runner 建好也跑不了）
printenv ANTHROPIC_API_KEY  >/dev/null && echo "API key 已设置"
printenv ANTHROPIC_BASE_URL >/dev/null && echo "BASE_URL 已设置（走网关）"
```

> `claude_agent_sdk` 是**可选**依赖（不在 GEAK 的 `pyproject.toml` 里）。装了走 SDK 路径
> （transcript 捕获更完整），没装则回退 `claude -p` CLI，两条路都可用。

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

**3.0 ⚠️ 先解除容器的单卡限制，否则后面全是白费**

本容器在 PID 1 层面注入了 `ROCR_VISIBLE_DEVICES=0`，**默认所有 ROCr/HIP 应用只能看到 1 张卡**。

后果很隐蔽：`rocm-smi` 照常显示 8 张卡（它走 sysfs，不受 ROCr 约束），`gpu_lock.sh` 设置的
`HIP_VISIBLE_DEVICES=<gpu_id>` 也不会报错 —— 但 **ROCr 层在更底层先把可见设备集砍成 1 张，
HIP 层的分配被直接架空**。实测中 4 个 specialist engineer 因此全部挤在物理 GPU 0 上互抢，
GEAK 日志报 `co-tenant noise (MaxNs up to ~500ms, huge StdDev)`。

它只是个**可覆盖的默认值**，容器本身有全部 8 卡的访问权（实测 GPU 0/3/7 均可正常执行 matmul）。
所以每个评测命令的**最外层**必须先解除：

```bash
unset ROCR_VISIBLE_DEVICES        # 或 export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

验证方法：

```bash
python3 -c "import torch; print(torch.cuda.device_count())"   # 应为 8，不是 1
```

**顺序不能反**：先解除 ROCr 限制，再由 `gpu_lock.sh` 分配 `HIP_VISIBLE_DEVICES`。

**`.bashrc` 只能解决一半**。`/root/.bashrc` 末尾已追加 `unset ROCR_VISIBLE_DEVICES`，
但该文件第 6 行就是 `[ -z "$PS1" ] && return` —— **非交互式 shell 根本不读它**。实测：

| 场景 | 结果 |
|---|---|
| 交互式 shell（人工进容器） | ✅ 已解除，`device_count = 8` |
| 非交互式（脚本 / subprocess / `exec_command`） | ⚠️ 仍为 `0`，`device_count = 1` |

而 `Penguin exec_command → run_kernel.py → claude → Workflow → GEAK` 整条链都是非交互式的。
因此 **`run_kernel.py` 必须自己 unset，并且做启动前断言**：

```python
os.environ.pop("ROCR_VISIBLE_DEVICES", None)
n = torch.cuda.device_count()
assert n == EXPECTED_GPUS, f"只看到 {n} 张 GPU，期望 {EXPECTED_GPUS} —— 检查 ROCR_VISIBLE_DEVICES"
```

**断言比 unset 更重要**：这个故障是静默的（不报错、`rocm-smi` 照常显示 8 卡、只是噪声莫名变大），
没有断言就只能在事后对着异常数据反查。

**3.0b ⚠️ 必须跳过 rocprof-compute，否则每次 profile 白扔约 4 分钟**

`rocprof-compute` 在本环境（ROCm 7.2 / rocprofiler-compute 3.4.0）**不可用**，且失败方式很贵：
它会跑完 13 轮 counter replay（每轮完整执行一遍 benchmark）才失败，实测耗时 **3 分 44 秒**。
GEAK 一次 run 会 profile 多次（baseline + 每轮 reprofile），再乘以评测矩阵，浪费相当可观。

所有评测命令固定设置：

```bash
export PROFILER_PRIORITY="rocprofv3 rocprof"
```

`profile_kernel.sh:74-77` 的选择逻辑只按 `command -v` 取列表中第一个存在的 profiler，
因此这样即可跳过；第 63 行会打印 `Profiler priority:`，可用于核验生效。

**不要卸载 rocprof-compute** —— 它是 ROCm 包的一部分，删除破坏包完整性且不可逆；
用这个官方 env 开关是可逆的，将来上游修好后去掉即可。

失效根因与已定位的三处调用 bug 记录在进度文档，此处不展开。降级到 `rocprofv3` 后仍可获得
durations、dispatch counts、per-dispatch occupancy（VGPR/SGPR/Scratch/LDS/grid），
缺的只有 SoL / VALU / VMEM / HBM / cache-hit 计数器——GEAK 的 7.8× 就是在这种降级状态下取得的。

**3.0c ⚠️ `rocm-smi` 与 HIP 的 GPU 编号顺序不一致 —— 指定卡号前必须建映射**

两套编号用的是不同的枚举顺序，直接按 `rocm-smi` 的编号传 `gpu_ids` **可能跑到别人的卡上**。
唯一可靠的关联键是 **PCI Bus ID**。

本机实测映射（仅供参考，**换机器必须重测**）：

| HIP / torch 索引 | PCI（十进制→十六进制） | = rocm-smi |
|---|---|---|
| cuda:0 | 117 → `0x75` | GPU[3] |
| cuda:1 | 5 → `0x05` | GPU[0] |
| cuda:2 | 101 → `0x65` | GPU[2] |
| cuda:3 | 21 → `0x15` | GPU[1] |
| cuda:4 | 245 → `0xF5` | GPU[7] |
| cuda:5 | 133 → `0x85` | GPU[4] |
| cuda:6 | 229 → `0xE5` | GPU[6] |
| cuda:7 | 149 → `0x95` | GPU[5] |

建映射与校验的方法：

```bash
rocm-smi --showbus                      # rocm-smi 索引 → PCI（十六进制）
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    print(i, hex(torch.cuda.get_device_properties(i).pci_bus_id))"   # HIP 索引 → PCI
```

**指定卡后必须回验**落到的物理卡是否都在允许范围内：

```bash
env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=4,5,6,7 python3 -c "
import torch
allowed = {0x85, 0x95, 0xE5, 0xF5}     # 本次获准使用的物理卡（rocm-smi 4-7）
buses = [torch.cuda.get_device_properties(i).pci_bus_id for i in range(torch.cuda.device_count())]
assert set(buses) <= allowed, f'越界: {[hex(b) for b in buses]}'
print('✅ 全部在允许范围内')"
```

`gpu_lock.sh` 与 GEAK 的 `gpu_ids` 用的都是 **HIP 索引**，不是 rocm-smi 索引。

**3.1 GPU 独占 —— 直接复用 GEAK 自带的 `kernel_workflow/scripts/gpu_lock.sh`，不要自己写**

```bash
cd <workspace> && bash <GEAK>/kernel_workflow/scripts/gpu_lock.sh <gpu_id> <command...>
```

它做了四件事，自己写的版本几乎必然只覆盖第一件：

1. **flock per GPU id** —— 多个 engineer 安全共享 GPU（命令期间独占）
2. **`TORCH_EXTENSIONS_DIR` 按 workspace 隔离** —— 否则 `cpp_extension.load(name=...)` 会把不同
   engineer 的**不同源码**编进**同一个全局缓存**，既让并行编译串行化在一把全局锁上，
   **又会让一个 engineer benchmark 到另一个 engineer 的 `.so`**（静默的正确性灾难）
3. **`PYTORCH_ROCM_ARCH` 只设本机 arch** —— 环境默认常是 ~9 个架构的列表，编译慢约 9 倍且毫无收益。
   同时固定 `GPU_ARCHS`，避免 aiter JIT 反复 shell 到 `rocm_agent_enumerator`
4. **回收孤儿 `rocm_agent_enumerator` 进程** —— 注释记录了真实事故：GPU/KFD 竞争下这些进程会挂起堆积
   （观察到 561 个 enumerator / 37k tasks 导致整机卡死）。只杀 `ppid==1` 且存活 >60s 的，不误伤

> kernel benchmark 必须独占 GPU，否则测得的加速比不可信 —— 这是整套评分可信度的物理前提。
> 第 2、3 条则是并行评测能否既快又正确的前提，**不要绕过这个 wrapper 直接跑测量命令**。

**3.2 `.penguin/score.sh`** —— verified 加速比 → 0–100

GEAK 返回值是**扁平结构**（实测确认），没有 `verification` 嵌套，也**没有 `correctness_passed` 字段** ——
正确性隐含在 `validation_status` 里：

```bash
#!/usr/bin/env bash
# 用法: score.sh <result_json>
set -euo pipefail
python3 - "$1" "${SCORE_ANCHOR:?先按 baseline 校准，见下}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
anchor = float(sys.argv[2])              # baseline geomean，阶段 5 实测后填入

if r.get("validation_status") != "accepted":
    print("0.00"); sys.exit(0)           # Director 未接受 = 0 分，无条件

s = float(r["final_speedup"])            # verified geomean，绝不用 tech_lead_reported_geomean
# 相对 baseline 的分段线性：0.9x anchor→0, anchor→60, 1.15x anchor→85, 1.3x anchor→100
pts = [(0.90*anchor,0),(anchor,60),(1.15*anchor,85),(1.30*anchor,100)]
if s <= pts[0][0]: out = 0.0
elif s >= pts[-1][0]: out = 100.0
else:
    for (x0,y0),(x1,y1) in zip(pts, pts[1:]):
        if x0 <= s <= x1:
            out = y0 + (y1-y0)*(s-x0)/(x1-x0); break
print(f"{out:.2f}")
PY
```

**三条硬规则**：

1. `validation_status != "accepted"` → 直接 0 分，不给部分分（否则 Optimizer 会学会牺牲正确性换速度）
2. 只认 `final_speedup`（verified geomean），**绝不用 `tech_lead_reported_geomean`** ——
   实测两者差 2.2%（7.9729 自报 vs 7.7956 verified），`verify_engineer.md` 开头就写明自报数字
   "may be noisy, measured against the wrong baseline, or wrong"
3. **曲线必须按实测 baseline 校准，不能预设绝对值**

> ⚠️ **第 3 条是踩过的坑**：本文初稿把曲线写死为「1.0x→0, 1.5x→60, 2.0x→85, ≥2.5x→100」。
> 但实测 GEAK 在 knn 上单轮就做到 **7.8×** —— 用那条曲线的话**所有 candidate 都是满分 100**，
> 评分完全失去区分度，优化循环直接失效。
>
> 正确做法：阶段 5 建完 Reference Baseline 后，取各 case 的 baseline geomean 作为 `anchor`，
> 曲线围绕它展开（上面用的是相对锚点写法）。**不同 case 的 anchor 差异可能很大，应逐 case 设定。**

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

> ⚠️ **本节的估算比初稿悲观得多，因为初稿有一个错误假设。**
> 初稿以为「8 卡 = 8 路 Evaluator 并行」，但**每个 Evaluator 内部跑的 GEAK 自己就要占多张卡**
> （它的 specialist engineer 是并行的）。真实并行度 = `8 / 每个 GEAK run 分配的卡数`。

**实测锚点**：knn，`budget=2`，4 张卡，1 轮 → **29 分钟**，verified geomean 7.8×。

并行度是一个**待实测的权衡**，两端差异很大：

| 每 run 卡数 | 并行 Evaluator | 单 run 耗时 | 说明 |
|---|---|---|---|
| 1 | **8** | 慢 2–3×（engineer 全串行等锁） | 并行度高，总吞吐可能反而最优 |
| 2 | 4 | 中 | |
| 4 | **2** | 29 min（实测，budget=2） | 单 run 快但并行度低 |

**总吞吐 ≈ 并行度 ÷ 单 run 耗时**，两端相乘后可能接近，必须实测才能定。
建议在阶段 5 建 baseline 时顺便测出这条曲线。

按 `budget=6`（约为 budget=2 的 2–3 倍耗时）粗估单个 candidate 的完整评测：

| 环节 | 规模（runs） | 备注 |
|---|---|---|
| 静态门禁 | — | 秒级，零 GPU |
| Canary | 2 | 快速崩坏筛查 |
| Dev | 5 case × 3 runs × 2（配对交错含 Reference）= 30 | 大头 |
| Holdout | 5 case × 3 runs = 15 | 仅 Dev 通过者 |
| **合计** | **~47 runs** | |

47 个 run × 单 run 1–1.5 小时 ÷ 并行度，**乐观（8 路）约 6–9 小时，悲观（2 路）超过 24 小时**。
8 轮优化在悲观情形下会拖到 **一周以上**，这不可接受。

**因此必须先做的取舍**（在阶段 5 用实测数据决定）：
- 调低 GEAK `budget`（代价：GEAK 行为偏离真实使用场景）
- 减少 Dev case 数或 runs 数（代价：噪声容忍度下降、覆盖变窄）
- 提高并行度，每 run 只给 1 卡（代价：单 run 变慢，需实测总吞吐是否真的更好）
- 缩短 case（选编译+benchmark 更快的 kernel）

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
| **GPU 争抢** | 并发 benchmark 互相干扰 | GEAK 自带 `gpu_lock.sh` flock 独占 |
| **编译缓存串味** | engineer A 测到 engineer B 的 `.so` | `gpu_lock.sh` 的 `TORCH_EXTENSIONS_DIR` 按 workspace 隔离（勿绕过） |
| **单卡限制被静默架空** | 传了多卡却全挤在 GPU 0，噪声暴涨且不报错 | 最外层 `unset ROCR_VISIBLE_DEVICES`（§8 阶段 3.0），并用 `device_count==8` 验证 |
| **GPU 索引错位** | `rocm-smi` 与 HIP/torch 的编号顺序**不一致**，可能跑到别人的卡上 | 用 PCI Bus ID 建映射后再指定 `gpu_ids`（§8 阶段 3.0c） |
| **profiler 空转** | 每次 profile 白扔约 4 分钟在注定失败的 rocprof-compute 上 | `PROFILER_PRIORITY="rocprofv3 rocprof"`（§8 阶段 3.0b） |
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
kernel_workflow/scripts/gpu_lock.sh         ★GPU 锁 + 编译隔离 + arch 收窄（勿自行实现）
kernel_workflow/scripts/profile_kernel.sh   profiler 封装
interface/run_e2e.md                        ★外部集成契约文档，run_kernel 照抄
interface/run_e2e.py                        抽取 driver 的来源
interface/run_kernel.py                     ★待新建
examples/tasks/knn                          唯一现成 case
examples/tasks/knn/config.yaml              case 的测量接口约定
examples/tasks/knn/scripts/task_runner.py   compile/correctness/performance 三合一
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
| **公开版 claude 拒绝 `--effort ultracode`，须 `--settings` 注入** | `run_e2e.py:72-80` |
| SDK 优先驱动自带 bundled CLI 而非 `$PATH` | `run_e2e.py:82-84` |
| Workflow 调用被路由为非阻塞后台任务 | `run_e2e.py:642-650` |
| `ALLOWED_TOOLS = [Workflow, Bash, Read, Write]` | `run_e2e.py:70` |
| `GEAK_CLAUDE_MODEL` 默认 `claude-opus-4-8` | `run_e2e.py:69` |
| `args` 必须为真 JSON 对象，否则 workflow 立即中止 | `run_e2e.py --dry-run` 输出的 prompt |
| `kernel_workflow.js` 的 `MIN_IMPROVE` 默认 0.02 | `kernel_workflow.js` |
| `gpu_lock.sh` 含锁 / 编译缓存隔离 / arch 收窄 / 进程回收四项 | `kernel_workflow/scripts/gpu_lock.sh` |
| `task_runner.py` 有标准 warmup（10 warmup + 100 iter + event 计时） | `examples/tasks/knn/scripts/task_runner.py:66` |
| profiler 三件套本机齐全 | `command -v rocprof-compute/rocprofv3/rocprof` |
| `claude_agent_sdk` 本机未安装（可选依赖） | `python3 -c "import claude_agent_sdk"` |
| **容器在 PID 1 注入 `ROCR_VISIBLE_DEVICES=0`，默认只 1 卡可用** | `/proc/1/environ` |
| 该限制可在子进程解除，8 卡均可真实计算 | `env -u ROCR_VISIBLE_DEVICES` + 实跑 matmul（GPU 0/3/7） |
| `rocprof-compute` 原为缺 14 个 Python 依赖，已修复 | `/opt/rocm-7.2.0/libexec/rocprofiler-compute/requirements.txt` |
