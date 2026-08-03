# FlagTree AMD 后端 TLE 完整支持路线图

- 编制日期：2026-08-03
- 目标环境：4 × AMD Radeon Graphics `gfx1201`（RDNA4），ROCm 7.2.3，PyTorch 2.10
- 代码基线：`/target/FlagTree`，`main` @ `273af404b`
- 最终目标：**在 gfx1201 上支持 TLE 中一切硬件可达的能力**

## 文档体系

本文档是总纲，与另两份文档配合使用：

| 文档 | 定位 | 回答的问题 |
|---|---|---|
| `FlagTree-AMD-TLE-support-report.md` | 现状验证报告 | 现在是什么状态、哪一步断了 |
| **本文档** | **总体路线图** | **做全的边界在哪、分几步走** |
| `FlagTree-AMD-TLE-implementation-plan.md` | 阶段一详细实施方案 | 第一步的每一行怎么改 |

---

## 一、"做全"的定义与边界

TLE 的规模是：**31 个 op**（`TleOps.td` 共 36 个 `def`，减去第 86-90 行的 5 个 type 定义）、**20 个 populate 函数**（17 个 LLVM lowering + 3 个供 `raw_passes` 使用的 transform pattern）、**26 个 pass**（`tle.passes` 23 个 + `tle.raw_passes` 3 个，后者即现状报告中实测的"23 个 pass"）。

"做全"不等于全部接入——其中相当一部分绑定 NVIDIA 专有硬件单元，在 RDNA4 上不存在物理对应。

因此本路线图把目标定义为：**接入一切不依赖 NVIDIA 专有硬件单元的 TLE 能力，对硬件缺失项给出明确且友好的报错而非崩溃。**

三类边界：

| 类别 | 判定依据 | 处置 |
|---|---|---|
| **可达** | lowering 不引用 NVVM / nvgpu / NVWS / PTX，或引用仅限头文件耦合 | 接入（Phase 0-2） |
| **需重写** | 语义在 AMD 有对应硬件，但现有 lowering 硬绑 NVIDIA 实现 | 新写 AMD lowering（Phase 3，研究性） |
| **不可达** | RDNA4 无对应硬件单元 | 不做，但需保证报错清晰——P0 的 `addIllegalDialect<tle::TleDialect>()` 已顺带解决：未接入的 TLE op 会在转换阶段给出明确诊断，而非拖到 `llvm.to_module` 才报"failed to translate module to LLVM IR" |

### 1.1 参照系：NVIDIA 后端接入了什么

`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp` 中共调用 13 个 populate：主 TLE 块内 11 个（第 171-191 行）、FlagCX 1 个（第 195 行）、remote pointers 1 个（第 222 行，在另一轮转换中）。这是当前的能力上限参照。

海光 HCU 接了 4 个（`TritonGPUToLLVM.cpp:211-220`：`ExtractTile`、`InsertTile`、`LocalPointers`、`ExclusiveCumsum`）加 6 个 pass（`compiler_hcu.py:410-429`）。

本路线图的终点约为 NVIDIA 的 8-9 个。

---

## 二、全景盘点

### 2.1 二十个 populate 函数的定性

| # | populate 函数 | 实现文件 / 行数 | NVIDIA 依赖 | 判定 | 阶段 |
|---|---|---|---|---|---|
| 1 | `LocalPointersOpToLLVM` | `LocalPointersOpToLLVM.cpp` 109-349 | 无（4 处均在同文件的 remote pattern） | 可达 | **P0** |
| 2 | `ExtractTileOpToLLVM` | `ExtractTileToLLVM.cpp` 431 行 | 2 处 `isHCU` 分支发 `NVVM::Barrier0Op` | 可达（需 `isAMD()`） | **P0** |
| 3 | `InsertTileOpToLLVM` | `InsertTileToLLVM.cpp` 445 行 | 同上 | 可达（需 `isAMD()`） | **P0** |
| 4 | `MemDescWGMMAViewOpToLLVM` | `ViewOpToLLVM.cpp` 65 行 | **0 处**（名字带 WGMMA，实现是纯 offset permute） | 可达 | P1 |
| 5 | `PackOpToLLVM` | `PackOpToLLVM.cpp` 83 行 | 0 处 | 可达，但**仅服务 TLE-Raw**，见下 | P3 |
| 6 | `ExtractOpToLLVM` | `ExtractOpToLLVM.cpp` 212 行 | 1 处，仅第 29 行头文件 | 同上 | P3 |
| 7 | `ExclusiveCumsumOpToLLVM` | `ExclusiveCumsumOpToLLVM.cpp` 497 行 | 第 125 行 `isHCU` 分支；HCU 路径用 `targetInfo.shuffleUp`，非 HCU 发 PTX `shfl.sync.up.b32` | 可达（需 `isAMD()`） | P1 |
| 8 | `DeviceIntraBarrierOpToLLVM` | barrier 系列 | `ttng::NamedBarrier*`，硬编码 NVIDIA-only 报错 | 需重写 | P3 |
| 9 | `DSLRegionOpToLLVM` | `DSLRegionOpToLLVM.cpp` 95 行 | 1 处头文件；本体是 TLE-Raw 载体 | 需重写 | P3 |
| 10 | `WGMMASharedOperandFenceOpToLLVM` | `WGMMAFenceOpToLLVM.cpp` 56 行 | 5 处，硬绑 `NVVM::FenceProxyOp` | **不可达** | — |
| 11 | `TMAStoreCommitGroupOpToLLVM` | `TMAStoreCommitGroupOpToLLVM.cpp` 53 行 | 硬绑 `NVVM::CpAsyncBulkCommitGroupOp` | **不可达** | — |
| 12 | `DistributedBarrierOpToLLVM` | `DistributedBarrierOpToLLVM.cpp` 534 行 | **21 处**，全库最重 | **不可达** | — |
| 13 | `RemotePointersOpToLLVM` | `LocalPointersOpToLLVM.cpp` 487-564 | `NVVM::MapaOp`、cluster addrspace 7 | **不可达** | — |
| 14-17 | `FlagCxOp` / `GetDeviceIdToFlagCx` / `GetNumPes` / `GetLocalRank` | 分布式支撑 | FlagCX 限定 nvidia（`FlagTreeOptions.cmake:30`） | **不可达** | — |
| 18-20 | `ConvertArgToMemDesc` / `DSLRegionInline` / `RemoveRedundantCopy` | `raw_passes` 辅助 | 随 DSLRegion | 需重写 | P3 |

### 2.2 二十六个 pass 的定性

| 分组 | pass | 判定 | 阶段 |
|---|---|---|---|
| tile 降级 | `lower_extract_tile`、`lower_insert_tile` | 可达 | **P0** |
| local pointer 核心 | `select_encodings`、`insert_local_pointer_barriers`、`optimize_local_pointer_loads`、`optimize_local_pointer_stores` | 可达 | **P0** |
| local pointer 辅助 | `early_assign_memory_space` | 可达，但 **P0 默认不加**（HCU 未用，metax/NVIDIA 用了）；作为出问题时的回退项，见实施方案 3.3 节 | P0 备选 |
| 异步加载 | `lower_async_load`（170 行，**0 处 NVIDIA 引用**）、`downgrade_invalid_async_copy`（366 行，**0 处**）、`optimize_local_pointer_async_stores`（483 行，**0 处**） | 可达，但受阻于 RDNA4 硬件缺口，见 4.1 | P2 |
| tile 流水线 | `tile_style_pipeline_schedule`（325 行，**0 处**）、`materialize_tile_style_pipeline`（1125 行，仅 2 处 `isa<ttng::WarpGroupDotOp>` 检测） | 可达 | P2 |
| staging | `promote_local_store_staging` | 可达 | P2 |
| barrier | `lower_barriers`、`allocate_named_barriers` | 需重写 | P3 |
| pipe | `lower_pipe_to_nvws` | 需重写 | P3 |
| WGMMA / TMA | `lower_wgmma`、`lower_tma_copy`、`schedule_tma_store_sync` | **不可达** | — |
| 分布式 | `params_for_distribution` | **不可达** | — |
| raw | `tle_convert_arg_to_memdesc`、`tle_remove_redundant_copy`、`tle_dsl_region_inline` | 需重写 | P3 |
| **无人调用** | `assign_local_pointers_encoding`、`lower_exclusive_cumsum`、`optimize_exclusive_cumsum_layouts` | 见下 | — |

**关于三个"无人调用"的 pass**：全库 Python 后端（nvidia / hcu / metax / sunrise / iluvatar / mthreads）均未调用这三个 pass，它们只是注册在 `triton_tle.cc` 中。特别是 cumsum——**它不需要任何 pass**，NVIDIA 与 HCU 都是直接在 LLVM 转换阶段用 `populateExclusiveCumsumOpToLLVMPatterns` 一步到位（`nvidia/...:187`、`hcu/...:220`）。因此接入 cumsum 时不要去调 `add_lower_exclusive_cumsum`，那是条死路。

**关于 `pack` 与 `extract_*` 系列的归属**：这两组 op（`tle.pack` 与 `extract_allocated_ptr` / `extract_aligned_ptr` / `extract_offset` / `extract_sizes` / `extract_strides` / `extract_ptr`）操作的是 `LLVMStructType` 与 `LLVMPointerType`，本质是 memref descriptor 的解构与重组。

它们的唯一产生者是 `third_party/tle/utils/lib/Protocol.cpp`（第 61、88、152 行），而该文件的唯一调用者是 `utils/lib/TleRawMaterialize.cpp`——即 **TLE-Raw 的参数与返回值 materialization 协议**。Python 侧对这些 op 零引用，用户不会直接书写。

因此单独接入它们没有意义：不接 DSLRegion 的话，没有任何路径会产生这些 op。它们必须随 TLE-Raw 一起做，故归入 P3。

### 2.3 一个反直觉的发现

`TleLowerAsyncLoad.cpp` 降级的目标全部是**架构中立的 TritonGPU op**（第 119-165 行）：

```cpp
builder.create<triton::gpu::LocalAllocOp>(loc, memDescType);
builder.create<triton::gpu::AsyncCopyGlobalToLocalOp>(...);
builder.create<triton::gpu::AsyncCommitGroupOp>(...);
builder.create<triton::gpu::AsyncWaitOp>(loc, commit->getResult(0), 0);
builder.create<triton::gpu::LocalLoadOp>(...);
```

而 AMD 后端**已经实现**了 `AsyncCopyGlobalToLocalOpConversion`（`LoadStoreOpToLLVM.cpp:959`，注册于 `:2111`）。这意味着 `tle.load(is_async=True)` 这条链路在代码层面是通的——但被一个硬件门控卡住，详见 4.1。

---

## 三、五阶段路线图

```
P0 接线打通 ──> P1 低成本补齐 ──> P2 异步与流水线 ──> P3 重写攻坚（研究）
     │                                    │
     └────────────> P4 工程化与上游化 <────┘（可与 P1/P2 并行）
```

### Phase 0：接线打通（已有详细方案）

**目标**：让 `local_pointers` + `extract_tile` / `insert_tile` 在 gfx1201 上编译并执行正确。

**内容**：12 个接入点、约 60 行改动，全部照搬 HCU 模板。详见 `FlagTree-AMD-TLE-implementation-plan.md` 第三、四章。

**为什么必须先做**：这一步建立的 C++ 转换目标（`TleLLVMConversionTarget`）、CMake 链接、Python pass 编排三条基础设施，是后续所有阶段的公共地基。P1 之后的每一项都只是往这个框架里加 populate 调用。

**关键风险**：RDNA4 warp size 32 与 HCU 的 CDNA 64 差异，症状多为编译通过但数值错误。

**工作量**：乐观 2 天，悲观 6 天。

**退出条件**：TLE 单测中 local pointer 与 tile 相关用例通过，含数值校验。

---

### Phase 1：低成本 op 补齐

**目标**：把两个"几乎零成本"的 op 接进 P0 建好的框架。

| op | 增量工作 | 说明 |
|---|---|---|
| `tle.memdesc_wgmma_view` | 加 1 行 populate 调用 | 名字有误导性，`ViewOpToLLVM.cpp` 65 行实现是纯 offset permute，零 NVIDIA 引用 |
| `tle.exclusive_cumsum` | 加 1 行 populate + 1 处 `isHCU()` → `isHCU() \|\| isAMD()` | 复用 P0 建好的 `isAMD()` 钩子；AMD `TargetInfo.h:57` 已有 `shuffleUp`；**不需要任何 pass** |

**依赖**：P0 完成，且 P0 已引入 `isAMD()` 钩子。

**工作量**：编码不足 0.5 天，主要成本在构建与验证迭代，合计 1-2 天。

**退出条件**：`test_tle_cumsum.py` 的 12 个用例通过。`exclusive_cumsum` 需重点做数值校验——warp32 下的 warp scan 语义与风险 4.2 同源，这是本阶段唯一的实质风险。

**性价比**：全路线图最高，增量代码约 3 行。但需注意本阶段能接的 op 比初看时少——`pack` 与 `extract_*` 系列虽然同样零依赖，却只服务 TLE-Raw（见 2.2 节末），单独接入无意义。

---

### Phase 2：异步加载与 tile 流水线

**目标**：打通 `tle.load(is_async=True)` 与 tile-style 软件流水，这是 TLE 性能收益的第二来源。

**内容**：接入 6 个 pass —— `lower_async_load`、`downgrade_invalid_async_copy`、`optimize_local_pointer_async_stores`、`tile_style_pipeline_schedule`、`materialize_tile_style_pipeline`、`promote_local_store_staging`。

**好消息**：这 6 个 pass 共约 2500 行，NVIDIA 引用总计只有 2 处，且都只是 `isa<ttng::WarpGroupDotOp>` 的类型检测（`TleMaterializeTileStylePipeline.cpp:369,386`），在 AMD 上恒为 false 走另一分支，属软依赖。

**坏消息**：卡在一个 AMD 后端自身的硬件缺口上，见 4.1。这是整个路线图唯一需要改 AMD 后端本体（而非 TLE 接线）的阶段。

**工作量**：若 4.1 走规避路线，2-3 天；若补齐 RDNA4 direct-to-LDS，需先做硬件调研，工作量不可控（5 天到数周）。

**退出条件**：异步加载用例编译通过且不触发 fatal error；流水线用例数值正确。

---

### Phase 3：重写攻坚（研究性）

这一阶段的三项都需要**新写 AMD lowering**，而非接线。建议在 P0-P2 稳定后按实际需求取舍，不建议一次全做。

#### 3.1 barrier 系列

`TleLowerBarriers.cpp` 第 102、125 行硬编码报错 `"named barrier lowering is only supported on NVIDIA"`，第 98-164 行全部生成 `ttng::` op（mbarrier / NamedBarrier）。

AMD 侧的对应物是 `s_barrier`，语义弱于 NVIDIA named barrier（后者支持按 id 分组、指定参与线程数）。RDNA4 没有分组 barrier，只能全 workgroup 同步。因此这是**语义降级**而非等价重写，需要评估降级后的正确性边界。

#### 3.2 pipe 系列（6 个 op）

`TleLowerPipeToNvws.cpp` 把 pipe 转成 NVIDIA NVWS token / mbarrier。要在 AMD 上支持，需新写一条 `TleLowerPipeToAMD`，用 LDS 上的 flag + `s_barrier` 实现 ring buffer 的 ready/free 协议。

这是 TLE-Lite 层唯一的重写项，工作量最大但价值明确：pipe 是 `tle_cn.md` 第 4.1 节 SparseMLA 案例中 `TLE-Pipe-Pipelined` 提速 2 倍的核心机制。

#### 3.3 TLE-Raw（DSLRegion）

理论路径是把 `tle.raw.language(name="mlir")` 的载体接到 AMDGCN inline asm 或直接生成 ROCDL。`DSLRegionOpToLLVM.cpp` 仅 95 行且只有头文件耦合，框架侧改动不大，但要让用户能写 AMD 原生代码，需要设计整套 Python 侧的方言暴露。

本项还需连带接入 `pack` 与 `extract_*` 系列（共 7 个 op）及三个 `raw_passes`（`tle_convert_arg_to_memdesc`、`tle_remove_redundant_copy`、`tle_dsl_region_inline`）——它们构成 TLE-Raw 的参数 materialization 协议，单独接入无意义，但作为 Raw 的配套则是必需项。好消息是这部分本身零架构依赖，真正的工作量在 Python 侧的方言设计。

**工作量**：3.1 约 3-5 天，3.2 约 2-4 周，3.3 不可估。

---

### Phase 4：工程化与上游化（可与 P1/P2 并行）

| 项 | 内容 | 价值 |
|---|---|---|
| 解耦 NVIDIA 头 | 把 `ExtractOpToLLVM.cpp:29`、`DSLRegionOpToLLVM.cpp:28`、`ExclusiveCumsumOpToLLVM.cpp:30` 对 nvidia 私有头的引用换成 `TargetInfoBase.h` | 对全项目有益，建议独立 PR 提上游 |
| `isHCU()` 语义清理 | 重命名为 `usesGenericLowering()` 之类中立名，替代 `isHCU() \|\| isAMD()` 的累加模式 | 避免后续每加一个后端就改一遍全部分支点 |
| AMD CI | 现状是 40 个 CI workflow 中零 amd/rocm 项 | 防止接入成果被后续提交打回 |
| 贡献上游 | 连同 CI 一起提 PR | — |

---

## 四、三个关键攻坚项

### 4.1 RDNA4 的 direct-to-LDS 缺口（P2 的唯一阻塞）

**这是全路线图技术上最硬的一处，且问题不在 TLE 而在 AMD 后端本体。**

`third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp:617-629`：

```cpp
bool TargetInfo::supportsDirectToLDSScattering() const {
  switch (getISAFamily()) {
  case ISAFamily::GFX1250:  return true;
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:    return false;
  default:
    llvm::report_fatal_error(
        "Unsupported architecture for direct to lds loads");
  }
}
```

`ISAFamily` 枚举（`TargetUtils.h:9-20`）中 RDNA4 是独立成员，**不在上述任何 case 中**，因此 gfx1201 会命中 `default` 分支直接 fatal error——不是优雅降级，是进程中止。

而 TLE 的 `downgrade_invalid_async_copy` 帮不上忙：它的判定函数 `hasLegalCpAsyncWidth`（`TleDowngradeInvalidAsyncCopy.cpp:58`）按 cp.async 的字节宽度（NVIDIA 语义）筛选，不查询目标架构能力，不会因为 RDNA4 不支持而降级。

三条处置路线：

| 路线 | 做法 | 代价 | 评价 |
|---|---|---|---|
| **A（推荐先行）** | 在 TLE 的 `lower_async_load` 入口加 target 门控，RDNA4 上不生成 async copy，退化为同步 load | 小，改 TLE 侧 | 保证不崩溃，但拿不到异步收益；可先解锁 P2 的流水线部分 |
| B | 给 AMD 后端补 RDNA4 的 direct-to-LDS 实现 | 需硬件调研 | RDNA3/4 有 `global_load_lds` 类指令，但 Triton 尚未适配；属上游缺口，收益大但周期不可控 |
| C | 只把 `default` 分支从 fatal error 改成 `return false` | 一行 | 治标，可能只是把崩溃点后移；需确认下游对 false 的处理是否完备 |

建议 A + C 组合先行，B 作为独立课题。

### 4.2 warp size 32（P0 的最大不确定性）

已在实施方案第 6.1 节详述。此处只补一点全局影响：**这个风险不止影响 P0**。`exclusive_cumsum`（P1）的 warp scan、tile 流水线（P2）的 layout 计算，都建立在同一套 LinearLayout 基础设施上。因此 P0 的 warp32 验证结果，实质上是对后续所有阶段的一次前置探测——若 P0 顺利，后续阶段的同类风险大幅降低。

### 4.3 `isHCU()` 的可扩展性债务

当前全库 6 处 `isHCU()` 分支点。P0 引入 `isAMD()` 后变成 `isHCU() || isAMD()`，P1 的 cumsum 再加 1 处。每接入一个新后端就要遍历修改所有分支点，是明显的设计债务。

建议在 P1 完成后、P2 开始前，把 Phase 4 的"语义清理"提前做掉，避免债务继续累积。

---

## 五、里程碑与工作量汇总

| 阶段 | 内容 | 工作量（估计） | 累计能力 |
|---|---|---|---|
| P0 | 接线 + local_pointers + tile | 2-6 天 | 共享内存显式管理 + tile 重排 |
| P1 | `memdesc_wgmma_view` + `exclusive_cumsum` | 1-2 天 | 上述 + 2 类算子扩展 |
| P2 | 异步加载 + tile 流水线 | 2-3 天（路线 A）<br>不可控（路线 B） | 上述 + 软件流水 |
| P4 | 解耦 + CI + 上游 | 3-5 天 | 工程可持续性 |
| P3.1 | barrier 重写 | 3-5 天 | 语义降级版 barrier |
| P3.2 | pipe 重写 | 2-4 周 | TLE-Lite 完整 |
| P3.3 | TLE-Raw（含 `pack` / `extract_*` 配套） | 不可估 | — |

表中按**建议执行顺序**排列，故 P4 位于 P3 之前——P4 是工程债务清理，越早做越能降低后续成本，且可与 P1/P2 并行。

**到 P2 为止（不含 P3），累计 5-11 天，可覆盖 TLE 在 AMD 上约 8 成的实用价值。** P3 是长尾，建议按实际业务需求驱动，不必追求形式上的"全"。

### 最终可达状态

完成 P0-P2 + P4 后，gfx1201 上的 TLE 支持面将达到：

| 层 | 覆盖情况 |
|---|---|
| TLE-Lite | `load`（同步路径）、`extract_tile`、`insert_tile` 可用；`pipe` 需 P3.2；分布式不可达 |
| TLE-Struct GPU | `alloc`、`local_ptr`、`copy` 可用；`warp_specialize` 不可达 |
| TLE-Struct DSA | 不适用（面向 Ascend 等 DSA 族） |
| TLE-Raw | 需 P3.3 |

即：**接近 HCU 当前水平并略有超出**。HCU 接了 4 个 populate，本路线图到 P2 为止是 5 个（多出 `memdesc_wgmma_view`），外加 HCU 完全没有的异步加载与 tile 流水线能力。

---

## 六、决策点

三个需要在动手前明确的选择：

1. **P0 是否走最小路径先探**。只接 `local_pointers`、跳过 tile 与 `isAMD()` 钩子，约 1 天拿到第一个信号，再决定是否投入完整 P0。适合对 warp32 风险没把握时使用。

2. **P2 的 4.1 走哪条路线**。路线 A 保证不崩溃但无异步收益，路线 B 收益大但周期不可控。建议先 A 后 B。

3. **P3 是否做**。pipe 重写 2-4 周是全路线图最大单项投入。若业务上没有 FlashMLA 类的深度流水需求，P0-P2 已足够。
