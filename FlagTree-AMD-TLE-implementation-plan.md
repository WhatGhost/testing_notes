# 为 FlagTree AMD 后端接入 TLE 的实施方案

- 编制日期：2026-08-03
- 目标环境：4 × AMD Radeon Graphics `gfx1201`（RDNA4），ROCm 7.2.3，PyTorch 2.10
- 代码基线：`/target/FlagTree`，`main` @ `273af404b`
- 前置文档：`FlagTree-AMD-TLE-support-report.md`（现状验证报告）
- 上级文档：`FlagTree-AMD-TLE-roadmap.md`（完整支持路线图）。本方案是其中 Phase 0 的详细展开；若关心"做全"的整体分期与边界，先读路线图
- 文档中所有行号均已在基线提交上核实

---

## 一、可行性结论

**结论：`local_pointers` + `extract_tile` / `insert_tile` 这一组 op 接入 AMD 后端是可行的，且工作量远小于直觉预期——约 60 行改动，全部照搬现成模板。**

（这组 op 跨越了 TLE 的两个抽象层，不等同于"TLE-Lite"。准确的层次归属见第二章开头。）

支撑这个结论的是四条独立证据。

### 1.1 `local_pointers` 的 LLVM lowering 本身架构中立

公共实现 `third_party/tle/dialect/lib/Conversion/TleToLLVM/LocalPointersOpToLLVM.cpp` 中的 `LocalPointersOpConversion`（第 109-349 行）**没有任何一处 NVIDIA 专有引用**：没有 NVVM op、没有 nvgpu、没有 PTX 内联汇编、没有 mma layout、没有 TMA/descriptor、没有 cluster/DSMEM。

它做的事情是通用 MLIR/LLVM 操作：

- 地址空间从 TypeConverter 推导，不硬编码（第 137-141 行）
  ```cpp
  auto llvmPtrTy = cast<LLVM::LLVMPointerType>(typeConverter->convertType(ptrTy));
  auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), llvmElemTy, rewriter);
  ```
- 偏移计算走 Triton 通用的 LinearLayout 基础设施（第 276-320 行），AMD 后端的 swizzled shared encoding 已经在用这套东西
- 最终地址生成是标准的 i8 GEP + bitcast（第 331-334 行）

整个 pattern 里唯一使用 `targetInfo` 的地方是第 232-235 行的 `emitIndices`，这是 Triton 通用的 thread/lane id 发射工具，AMD 后端已实现。

### 1.2 NVIDIA 依赖被隔离在另一个 pattern 中

`LocalPointersOpToLLVM.cpp` 里有两个 pattern，但**由两个独立的 populate 函数分别注册**（第 568-579 行）：

```cpp
void tle::populateLocalPointersOpToLLVMPatterns(...) {
  patterns.add<LocalPointersOpConversion>(typeConverter, targetInfo, benefit);
}

void tle::populateRemotePointersOpToLLVMPatterns(...) {
  patterns.add<RemotePointersOpConversion>(typeConverter, benefit);
}
```

所有 NVIDIA 依赖（`NVVM::MapaOp`、cluster shared addrspace 7、FlagCX 外部符号）都落在 `RemotePointersOpConversion` 及其辅助函数里。**只调用 `populateLocalPointersOpToLLVMPatterns` 不会把这些东西带进 pattern set。**

### 1.3 三份独立实现的算法字节级一致

| 实现 | 路径 | 与公共实现的差异 |
|---|---|---|
| 公共 | `third_party/tle/.../LocalPointersOpToLLVM.cpp` | 基准 |
| 天数 iluvatar | `third_party/iluvatar/tle/lib/Conversion/TleToLLVM/LocalPointersOpToLLVM.cpp` | 核心算法（第 98-249 行 vs 公共第 180-335 行）**逐行相同**；仅删掉 remote 半边、多一条 addrspace 断言、错误上报方式不同 |
| 摩尔线程 mthreads | `third_party/mthreads/tle/dialect/lib/Conversion/MUSATLEToLLVM/LocalPointersOpToLLVM.cpp` | 与 iluvatar 版仅 4 处差异（命名空间、rank-0 兼容、padding API、populate 命名） |

这两家另起炉灶的原因**不是架构差异，而是构建隔离**。`cmake/FlagTreeOptions.cmake` 第 50-56 行和第 61-62 行把它们的 `FLAGTREE_TLE` 整个关掉、移除 `-D__TLE__`，导致 `third_party/tle` 子目录不参与编译，只能自带一份。

### 1.4 海光 HCU 已在类 AMD 架构上验证通过

这是最强的证据。HCU（海光 DCU）就是 AMD 后端的 fork——同样用 ROCDL、同样的 LDS、同样的 `membarFilter` 模式，两者 `TritonGPUToLLVM.cpp` 的 `runOnOperation()` 几乎逐行同构：

| 步骤 | AMD 行号 | HCU 行号 |
|---|---|---|
| `TargetInfo` 构造 + ISAFamily 检查 | 112-116 | 150-154 |
| `overrideIndexBitwidth(32)` | 119 | 157 |
| TypeConverter（含 `convertTensorDescType`，实现完全相同） | 66-92 | 104-130 |
| `annotateLocalLoadsSyncedViaAsyncWait` | 130-131 | 168-169 |
| `initSharedMemory` | 154 | 194 |
| `ModuleAxisInfoAnalysis` | 165 | 205 |
| **TLE 转换块** | **缺失** | **207-227** |
| "Emit logics to get threadId..." 注释 | 167 | 229 |

HCU 直接复用公共实现，**零改写**，只加了 21 行接线代码。

### 1.5 共享内存抽象完全一致

AMD 使用与 NVIDIA 完全相同的 TritonGPU 共享内存抽象：`MemDescType`、`SharedMemoryObject`、`ModuleAllocation`、`global_smem` 全局符号。地址空间数值也一致：

```cpp
// third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp:599
int TargetInfo::getSharedAddressSpace() const { return 3; }
```

NVIDIA（`TargetInfo.cpp:714`）与 HCU（`TargetInfo.cpp:599`）同为 3。TLE 代码中即便有硬编码也不会出错，而实际上并没有硬编码。

---

## 二、范围界定

### 2.0 TLE 三层抽象与本方案的对应关系

在界定范围之前需要先厘清一个容易误解的问题：TLE 的三层抽象是**面向用户的 API 分层，不是编译器实现的模块边界**，因此无法按层切一刀说"只实现某一层"。

#### 2.0.1 三层的定义

依据仓库根目录官方架构文档 `tle_cn.md` 第 3 章，三层沿"可移植性 ↔ 控制力"递进：

| 层 | 设计哲学 | 面向人群 | Lowering 路径 |
|---|---|---|---|
| TLE-Lite（3.2 节） | 一次编写，到处运行；用语义 hint 而非强制约束引导编译器，最小侵入 | 算法工程师、快速优化场景 | 经 FLIR 到 LLVM IR |
| TLE-Struct（3.3 节） | 架构感知，精细调优；按硬件族（GPGPU / DSA）暴露存储层级与并行结构 | 需要一定硬件理解的开发者 | 经 FLIR 到 LLVM IR |
| TLE-Raw（3.4 节） | 原生透传，极致掌控；内联厂商原生代码，绕过通用中间层 | 性能优化专家 | 走厂商私有编译管线 |

#### 2.0.2 API 的实际层次归属

| API | 所属层 | `tle_cn.md` 位置 | AMD 可行性 |
|---|---|---|---|
| `tle.load(is_async=True)` | Lite | 3.2.1.1 | 可行（Python 层 hint 包装，无需专门 lowering） |
| `tle.extract_tile` / `insert_tile` | Lite | 3.2.2 | **可行，本方案覆盖** |
| `tle.pipe` | Lite | 3.2.3 | **不可行**，见 2.0.3 |
| `device_mesh` / `sharding` / `reshard` / `remote` | Lite | 3.2.4 | 不可行，依赖 cluster + DSMEM + FlagCX |
| `tle.gpu.alloc` | **Struct GPU** | 3.3.1.1.2 | **可行，本方案覆盖** |
| `tle.gpu.local_ptr` | **Struct GPU** | 3.3.1.1.3 | **可行，本方案覆盖** |
| `tle.gpu.copy` | **Struct GPU** | 3.3.1.1.5 | 可行 |
| `tle.gpu.warp_specialize` | Struct GPU | 3.3.1.2.1 | 不可行，无对应硬件 |
| `tle.dsa.*`（Ascend UB/L1/L0A…） | Struct DSA | 3.3.2 | 不适用，DSA 族 |

关键一点：**共享内存的显式管理（`alloc` + `local_ptr` + `copy`）属于 TLE-Struct GPU，不属于 Lite。** 这也正是 TLE 在 GPU 上性能收益的主要来源——`tle_cn.md` 第 4.2 节（MoeAlignBlockSize，对 sglang_cuda 最高 4.06 倍）和第 4.3 节（radix-select TopK，对 Torch 最高 2.5 倍）两个案例都明确标注"利用 `tle-struct` 中 shared memory 相关扩展"。

#### 2.0.3 为什么不能"只实现 Lite"

两个原因叠加。

其一，Lite 层内部本身就有强 NVIDIA 依赖。`tle.pipe` 的 lowering 位于 `third_party/tle/dialect/lib/Transforms/TleLowerPipeToNvws.cpp`，第 32 行直接 include `nvidia/include/Dialect/NVWS/IR/Dialect.h`，整个 pass 的作用是把 pipe 转成 NVIDIA NVWS token / mbarrier 同步；`tle_cn.md` 第 425 行亦自陈"当前 lowering 将 CTA-scoped SMEM pipe 转成 GPU NVWS token/mbarrier 同步"。分布式全家桶的情况同 2.2 节。Lite 五个部分中，AMD 上真正能落地的只有 `tle.load` 与 `extract_tile`/`insert_tile`。

其二，剩下这点能力价值有限。`extract_tile`/`insert_tile` 是纯寄存器重排，作用是省去手写地址计算，属易用性改进而非性能突破。

#### 2.0.4 本方案的实际切分依据

既然分层不能作为实现边界，本方案按 **lowering 的后端依赖**切分：凡不依赖 NVVM / NVWS / cluster 的即纳入，依赖的即排除。这个切法得到的结果恰好等于海光 HCU 的接入组合。

两个非 NVIDIA 后端的实际接入情况可作对照：

| 后端 | README 层次标注 | 实际接入的 pass / pattern |
|---|---|---|
| 海光 HCU | Lite + Struct GPU | `extract_tile`、`insert_tile`（Lite）+ `local_pointers`（Struct GPU）+ `exclusive_cumsum` |
| 沐曦 metax | 仅 Lite | `early_assign_memory_space` + `select_encodings` + 3 个 local pointer 优化 pass（`compiler.py:276-280`），**均属 Struct GPU；未接 tile 两个 op** |

metax 标注为 Lite 却只实现了 Struct GPU 的内存部分，说明 README 的层次标注是市场层面的粗分类，与架构文档的 API 归属并不严格对应，不宜作为实现依据。

附带一个对 5.2 节风险的正面信号：metax 接入了 `add_early_assign_memory_space`，而 HCU 没有。该 pass 是否必需这一待定项，因此有了两个方向相反的参照。

### 2.1 阶段一目标（本方案覆盖）

让以下三个 op 在 gfx1201 上编译并执行正确：

| op | 定义位置 | 能力 |
|---|---|---|
| `tle.local_pointers` | `TleOps.td:92` | 共享内存指针张量，配合 `tl.load`/`tl.store` 使用 |
| `tle.extract_tile` | `TleOps.td:44` | 静态 offset 抽取 tile，纯寄存器重排 |
| `tle.insert_tile` | `TleOps.td:70` | 反向插入 |

按 2.0.2 的层次归属，这三个 op 横跨两层：`local_pointers` 属 TLE-Struct GPU（共享内存暂存，性能收益主要来源），`extract_tile` / `insert_tile` 属 TLE-Lite（tile 重排）。这也正是当前 31 个失败单测中占比最大的部分。

### 2.2 明确排除（硬件层面不可行）

| 能力 | 依赖硬件 | 排除原因 |
|---|---|---|
| WGMMA 系列 | Hopper warp-group MMA | `tle.wgmma_shared_operand_fence` 硬绑 `NVVM::FenceProxyOp`；RDNA4 用 WMMA，语义不同 |
| TMA 系列 | Hopper TMA 引擎 | `tle.tma_store.commit_group` 硬绑 `NVVM::CpAsyncBulkCommitGroupOp` |
| `tmem` | Blackwell Tensor Memory | 无对应硬件 |
| `warp_specialize` | Hopper warp group | 无对应硬件 |
| `tle.pipe` | NVIDIA NVWS / mbarrier | 虽属 Lite 层，但 lowering 位于 `TleLowerPipeToNvws.cpp`，第 32 行硬 include `nvidia/include/Dialect/NVWS/IR/Dialect.h`，无后端无关路径 |
| 分布式全家桶 | NVIDIA cluster + DSMEM + NVSHMEM/FlagCX | `tle.distributed_barrier` 重度依赖 NVVM cluster + nvgpu + PTX；且 `FlagTreeOptions.cmake:30` 已把 `FLAGCX_SUPPORT_BACKENDS` 限定为 nvidia |

### 2.3 推迟到阶段二

`tle.exclusive_cumsum`。HCU 实际上接了这个 op（`TritonGPUToLLVM.cpp:220`），本方案仍选择推迟，理由是**缩小首轮变更面**，而非技术不可行。

需要澄清两点，避免误读：

- 一旦按 2.4 加了 `isAMD()` 钩子，cumsum 就会走 HCU 分支使用 `targetInfo.shuffleUp`（`ExclusiveCumsumOpToLLVM.cpp:243,353`），**不会**发 PTX 内联汇编。PTX 路径（第 155-160 行）只在既非 HCU 也非 AMD 时触发。AMD 的 `shuffleUp` 已存在于 `TargetInfo.h:57`。
- 接入 cumsum **不需要任何 pass**，只需在转换块里加一行 `populateExclusiveCumsumOpToLLVMPatterns`，再把 `ExclusiveCumsumOpToLLVM.cpp:125` 的条件改为 `isHCU() || isAMD()`。`add_lower_exclusive_cumsum` 这个 pass 全库无人调用，不要去调。

真正的剩余成本在验证：warp32 下的 warp scan 语义需要独立的数值校验，与风险 1 同源。

### 2.4 阶段一必须附带的公共改动：`isAMD()` 钩子

**这一项在初版方案中被误判为可推迟，实测代码后确认是阶段一的硬阻塞项。**

`extract_tile` / `insert_tile` 的实现中有 5 处按 `targetInfo.isHCU()` 分支，非 HCU 分支直接发 NVVM op：

| 位置 | HCU 分支 | 非 HCU 分支 |
|---|---|---|
| `ExtractTileToLLVM.cpp:307` | `targetInfo.barrier(...)` | `NVVM::Barrier0Op` |
| `ExtractTileToLLVM.cpp:375` | `targetInfo.barrier(...)` | `NVVM::Barrier0Op` |
| `InsertTileToLLVM.cpp:312` | `targetInfo.barrier(...)` | `NVVM::Barrier0Op` |
| `InsertTileToLLVM.cpp:365` | `targetInfo.barrier(...)` | `NVVM::Barrier0Op` |
| `TleTileToLLVMUtils.cpp:118` | `getThreadId(rewriter, loc)` | `NVVM::ThreadIdXOp` |

AMD 若不加钩子会走进非 HCU 分支，生成 NVVM op。而 AMD 的 `TleLLVMConversionTarget` 不把 NVVM 声明为 legal，即便声明了也无法翻译成 AMDGCN，结果仍是编译失败。

好消息是 HCU 分支用的 `targetInfo.barrier()` 和 `getThreadId()` 都是 **target-agnostic 的通用接口**，AMD 直接复用即可，不需要写任何 AMD 专属实现。

**`tle.local_pointers` 不受此影响**——它没有任何 `isHCU()` 分支，文件中的 NVVM 引用（第 74、76、83、451 行）全部位于 remote pointer 半边（`mapSharedToClusterPointer` 与 `lowerDeviceSpace`），不在 `LocalPointersOpConversion`（第 109-349 行）内。因此若只想最快看到信号，可以先只接 `local_pointers`，跳过 tile 操作。

### 2.5 为什么以 HCU 而非 NVIDIA 为模板

NVIDIA 版本结构差异较大，不适合直接照抄：

| 维度 | AMD | NVIDIA | HCU |
|---|---|---|---|
| `applyPartialConversion` 轮数 | 3（func / call-ret / main） | 4（多一轮独立 CF） | 3 + TLE 轮 |
| 控制流下降 | 混在主 pattern 集内 | 单独一轮 `cfTarget` | 同 AMD |
| TLE target 合法 dialect | — | arith/LLVM/math/**NVVM**/gpu | arith/LLVM/**ROCDL**/**scf** |
| benefit 体系 | 双层 `commonBenefit`/`AMDBenefit` | 单一 `benefit` | 同 AMD |
| AxisInfo 分析 | `AMD::ModuleAxisInfoAnalysis` | TLE 下换成 `tle::ModuleAxisInfoAnalysis` | 保持 `HCU::` 不换 |
| TLE populate 数量 | — | 13（主块 11 + FlagCX 1 + remote 1） | **4** |

HCU 与 AMD 的差异仅限于符号改名（`AMD::` → `HCU::`）加少量 HCU 独有内容。因此本方案全部以 HCU 为模板。

---

## 三、完整接入点清单

| # | 文件 | 位置 | 改动 | 必需 |
|---|---|---|---|---|
| 1 | `third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp` | 第 28 行后 | 加 `#ifdef __TLE__` 头文件 include | 是 |
| 2 | 同上 | 第 60 行后 | 主 target 加 `addIllegalDialect<tle::TleDialect>()` | 是 |
| 3 | 同上 | 第 64 行后 | 新增 `TleLLVMConversionTarget` 类 | 是 |
| 4 | 同上 | 第 165 行与 167 行之间 | 插入 TLE 前置 `applyPartialConversion` 块 | 是 |
| 5 | `third_party/amd/lib/TritonAMDGPUToLLVM/CMakeLists.txt` | 第 1 行前 + 第 37 行后 | `if(FLAGTREE_TLE)` 定义并链接 `_TLE_LIBS` | 是 |
| 6 | `cmake/FlagTreeOptions.cmake` | 第 450-463 行列表 | 加入 `TritonAMDGPUToLLVM` | 是（易漏） |
| 7 | `third_party/amd/backend/compiler.py` | 第 2 行 | import 加 `tle` | 是 |
| 8 | 同上 | 第 210 行后 | 加 `add_lower_extract_tile` / `add_lower_insert_tile` | 是 |
| 9 | 同上 | 第 218 行后 | 加 local pointer pass 序列 | 是 |
| 10 | `include/triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h` | 第 138 行后 | 新增 `virtual bool isAMD()` 钩子 | 是（接 tile op 时） |
| 11 | `third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h` | 类内 | 覆写 `isAMD()` 返回 true | 同上 |
| 12 | `ExtractTileToLLVM.cpp` `:307,375`、`InsertTileToLLVM.cpp` `:312,365`、`TleTileToLLVMUtils.cpp` `:118` | 5 处 | 条件改为 `isHCU() \|\| isAMD()` | 同上 |

### 不需要改动的部分

| 项 | 原因 |
|---|---|
| include 路径 | 根 `CMakeLists.txt` 第 174-175 行已全局注册 `third_party` 与其 binary 目录 |
| `__TLE__` 宏 | `FlagTreeOptions.cmake` 第 37-44 行默认已开（`FLAGTREE_BACKEND` 为空时） |
| TLE dialect 注册 | `python/src/ir.cc:410-413` 在 `#ifdef __TLE__` 下已全局注册 |
| `TypeConverter` | TLE populate 函数接受 `mlir::LLVMTypeConverter&` 基类引用，AMD 的子类可直接传入 |
| 共享内存抽象 | AMD 已完全复用 TritonGPU 的 `MemDescType` / `SharedMemoryObject` |
| `third_party/amd/python/triton_amd.cc` | `tle.passes` 由 `third_party/tle/triton_tle.cc` 统一提供；只有需要 NVIDIA 专有的 `tle.raw_passes` 才需要动 |

---

## 四、分步实施

### 第 1 步：C++ 转换层（1 个文件，4 处改动）

文件：`third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp`

#### 1.1 头文件（第 28 行 `#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"` 之后）

```cpp
#ifdef __TLE__
#include "tle/dialect/include/Conversion/TleToLLVM/LocalPointersOpToLLVM.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#endif
```

对照 HCU 第 21-26 行。注意 `extract_tile` / `insert_tile` 的 LLVM pattern 声明在 `PatternTleToLLVM.h`（第 35-53 行）而非 `LocalPointersOpToLLVM.h`，两个头都要包。

#### 1.2 主 target 声明 TLE 非法（第 60 行 `addIllegalDialect<mlir::gpu::GPUDialect>();` 之后）

```cpp
#ifdef __TLE__
    addIllegalDialect<triton::tle::TleDialect>();
#endif
```

这一步直接修复当前的失败模式：现在 AMD 的 `TritonLLVMConversionTarget` 不认识 `tle.*` op，`applyPartialConversion` 把它们当作合法而放过，直到 `llvm.to_module`（`compiler.py:359`）才炸出 "failed to translate module to LLVM IR"。声明为 illegal 后，若 pattern 缺失会在转换阶段就给出明确诊断。

#### 1.3 新增 TLE 专用 ConversionTarget（第 64 行 `TritonLLVMConversionTarget` 类之后）

```cpp
#ifdef __TLE__
class TleLLVMConversionTarget : public ConversionTarget {
public:
  explicit TleLLVMConversionTarget(MLIRContext &ctx,
                                   LLVMTypeConverter &typeConverter)
      : ConversionTarget(ctx) {
    addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect,
                    ROCDL::ROCDLDialect, mlir::scf::SCFDialect>();
    addIllegalDialect<triton::tle::TleDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    markUnknownOpDynamicallyLegal([](Operation *) -> bool { return true; });
  }
};
#endif
```

完全照搬 HCU 第 89-102 行。三个设计要点：

- 合法 dialect 用 **ROCDL + SCF**，不引入 NVVM
- `addLegalOp<UnrealizedConversionCastOp>()` 满足 TLE 转换的合法性前提
- `markUnknownOpDynamicallyLegal` 让非 TLE op 在这轮 partial conversion 中不被触碰

不需要 NVIDIA 版本里的 `addDynamicallyLegalOp<tle::DSLRegionOp, tle::YieldOp>` 和 `addLegalOp<tle::RemotePointersOp>`，因为这些 op 本方案不支持。

#### 1.4 插入 TLE 前置转换块（第 165 行 `AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);` 之后、第 167 行 "Emit logics to get threadId..." 注释之前）

```cpp
#ifdef __TLE__
    {
      TleLLVMConversionTarget tleTarget(*context, typeConverter);
      RewritePatternSet tlePatterns(context);
      mlir::triton::tle::populateExtractTileOpToLLVMPatterns(
          typeConverter, tlePatterns, targetInfo,
          patternBenefitPrioritizeOverLLVMConversions);
      mlir::triton::tle::populateInsertTileOpToLLVMPatterns(
          typeConverter, tlePatterns, targetInfo,
          patternBenefitPrioritizeOverLLVMConversions);
      mlir::triton::tle::populateLocalPointersOpToLLVMPatterns(
          typeConverter, targetInfo, tlePatterns,
          patternBenefitPrioritizeOverLLVMConversions);
      if (failed(applyPartialConversion(mod, tleTarget, std::move(tlePatterns))))
        return signalPassFailure();
    }
#endif
```

对照 HCU 第 207-227 行（HCU 多一个 `populateExclusiveCumsumOpToLLVMPatterns`，本阶段不加）。

两个注意事项：

1. 内层 `RewritePatternSet tlePatterns(context)` 必须是独立变量，**不要**合并进第 174 行的主 `patterns`。TLE 需要在主转换之前单独跑完一轮。
2. 位置必须在 `axisInfoAnalysis` 之后（TLE pattern 不需要它，但保持与 HCU 一致可减少变量）、主 `RewritePatternSet` 之前。

### 第 2 步：CMake（2 个文件）

#### 2.1 `third_party/amd/lib/TritonAMDGPUToLLVM/CMakeLists.txt`

文件最开头（第 1 行 `add_triton_library(...)` 之前）插入：

```cmake
if(FLAGTREE_TLE)
    set(_TLE_LIBS
        TritonTLEAnalysis
        TritonNVIDIAGPUToLLVM
        TleToLLVM
        TritonTLETransforms
    )
else()
    set(_TLE_LIBS "")
endif()
```

然后在第 37-42 行的 `LINK_LIBS PUBLIC` 段中插入 `${_TLE_LIBS}`，位置放在 `TritonAMDGPUIR` 与 `LLVMCore` 之间（与 HCU 第 58 行的位置约定一致）：

```cmake
    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    TritonAMDGPUIR
    ${_TLE_LIBS}
    LLVMCore
    LLVMPasses
    LLVMSupport
)
```

**关于 `TritonNVIDIAGPUToLLVM`**：这个依赖看起来违和，但确实必需。公共 `TleToLLVM` 库的若干源文件硬 include 了 NVIDIA 后端的私有头：

- `ExtractOpToLLVM.cpp:29` → `nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h`
- `DSLRegionOpToLLVM.cpp:28` → 同上
- `ExclusiveCumsumOpToLLVM.cpp:30` → `third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h`
- `DistributedBarrierOpToLLVM.cpp:32-33` → `nvgpu/IR/Dialect.h` + `PTXAsmFormat.h`

这是**库级**（非 pattern 级）耦合，HCU 同样被迫链接它（`third_party/hcu/lib/TritonHCUGPUToLLVM/CMakeLists.txt:1-10`）。默认构建本来就会编译 nvidia 后端，所以短期无额外代价。长期清理方案见第七节。

**不需要加任何 include 路径**：根 `CMakeLists.txt` 第 174-175 行已全局注册。

#### 2.2 `cmake/FlagTreeOptions.cmake`

第 450-463 行 `flagtree_add_tle_generated_header_dependencies()` 的 target 列表中加入 `TritonAMDGPUToLLVM`：

```cmake
  foreach(_flagtree_tle_header_target IN ITEMS
      TritonAnalysis
      TritonToTritonGPU
      TritonGPUTransforms
      TritonNvidiaGPUTransforms
      TritonNVIDIAGPUToLLVM
      TritonAMDGPUToLLVM          # <-- 新增
      TritonGPUToLLVM
      ...
```

**这一步最容易漏。** 该函数的注释（第 446-449 行）说明了原因：TLE dialect 在核心库之后 `add_subdirectory`，不显式声明依赖的话，clean 状态下的并行构建可能在 `.inc` 文件生成前就去编译 AMD 库。症状是随机性的"找不到头文件"，串行构建或增量构建时不复现，排查成本高。

### 第 3 步：Python pass 编排（1 个文件，3 处改动）

文件：`third_party/amd/backend/compiler.py`

#### 3.1 import（第 2 行）

```python
from triton._C.libtriton import ir, passes, llvm, amd, tle
```

对照 HCU 第 2 行（HCU 还有 `distributed`，本方案不需要）。

#### 3.2 `make_ttgir` 第一个 pass manager（第 210 行之后、第 211 行 `pm.run(mod, 'make_ttgir_early')` 之前）

```python
        # flagtree tle: lower tle.extract_tile / tle.insert_tile
        tle.passes.add_lower_extract_tile(pm)
        tle.passes.add_lower_insert_tile(pm)
```

对照 HCU `compiler_hcu.py` 第 409-412 行。前一个 pass 是 `passes.ttir.add_convert_to_ttgpuir`，锚点明确。

#### 3.3 `make_ttgir` 第二个 pass manager（第 218 行 `add_optimize_thread_locality(pm)` 之后、第 219 行 `add_accelerate_matmul` 之前）

```python
        # begin flagtree tle
        tle.passes.add_select_encodings(pm)
        tle.passes.add_insert_local_pointer_barriers(pm)
        tle.passes.add_optimize_local_pointer_loads(pm)
        tle.passes.add_optimize_local_pointer_stores(pm)
        # end flagtree tle
```

对照 HCU 第 425-430 行。

**关于 `add_early_assign_memory_space`**：skill 文档 `skills/tle-developer/references/tle-sources.md` 第 127-135 行把 local pointer 流水线定义为 5 步，第 1 步是 `add_early_assign_memory_space`，并注明"Do not reorder without proof and tests"。NVIDIA（`compiler.py:325`）和 sunrise（`compiler.py:224`）都调用了它，但 **HCU 没有**。

本方案的取舍：**先按 HCU 不加**。理由是 HCU 是唯一在类 AMD 架构上验证通过的配置，先最小化变量。若 `add_select_encodings` 报编码冲突或 local pointer 的内存空间推断失败，再加回作为第一步。

**`make_llir` 不需要任何改动**——HCU 的 `make_llir` 里也没有 TLE pass。

### 第 3.5 步：`isAMD()` 钩子（接 tile op 时必需，只接 `local_pointers` 可跳过）

依据见 2.4 节。采用最小侵入做法：新增钩子而不重命名现有接口。

`include/triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h` 第 138 行后：

```cpp
  virtual bool isCuda() const { return false; }
  virtual bool isHCU() const { return false; }
  virtual bool isAMD() const { return false; }   // 新增
```

`third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h` 的 `TargetInfo` 类内覆写（参照 HCU 的 `TargetInfo.h:92`）：

```cpp
  bool isAMD() const override { return true; }
```

5 处调用点统一改为或条件，不改动分支体：

```cpp
  if (targetInfo.isHCU() || targetInfo.isAMD())
```

涉及 `ExtractTileToLLVM.cpp:307`、`:375`，`InsertTileToLLVM.cpp:312`、`:365`，`TleTileToLLVMUtils.cpp:118`。

这三个文件属于公共 TLE 代码，改动会影响所有后端。但由于只是在原有条件上追加 `|| isAMD()`，其他后端的 `isAMD()` 均返回 false，行为完全不变，风险可控。

### 第 4 步：构建

```bash
cd /target/FlagTree
unset FLAGTREE_BACKEND
export TRITON_BUILD_WITH_CCACHE=true
export MAX_JOBS=64
python3 -m pip install . --no-build-isolation -v 2>&1 | tee /target/flagtree-tle-amd-build.log
```

首次全量构建约 49 分钟；有 ccache 且只改了上述文件的话，增量构建应在几分钟量级。改动 CMake 文件会触发 CMake 重新配置，比纯 C++ 改动慢一些。

---

## 五、验证方案

现成的 31 个失败单测就是天然的验收标准，按依赖复杂度分四级推进。

### L0：编译不再报错（最小信号）

```bash
cd /target/flagtree-tests && python3 test_tle_on_amd.py
```

期望变化：`llir` 与 `amdgcn` 从 `MISSING` 变为有输出，TTGIR 中不再残留 `tle.local_pointers`。

### L1：local pointer 核心语义（12 个用例）

```bash
cd /target/FlagTree/python/test/tle
python3 -m pytest unit/test_tle_gpu_local_ptr.py -v
```

按预期难度排序：

| 用例 | 考察点 | 预期 |
|---|---|---|
| `test_local_pointer_store_populates_constant` | 最简单的 LDS 写入 | 最先绿 |
| `test_local_pointer_axpy_matches_torch` | 读-算-写完整链路 + 数值正确性 | 早期绿 |
| `test_local_pointer_none_generates_full_view_1d` / `_2d` | `indices=None` 全视图 | 早期绿 |
| `test_local_pointer_none_load_rewrites_to_local_load` | TTGIR 结构断言 | 中等 |
| `test_local_pointer_full_indices_load_rewrites_to_local_load` | 显式全索引的重写路径 | 中等 |
| `test_local_pointer_conditional_mask_store_compiles` | 掩码 + 控制流 | 中等 |
| `test_local_pointer_looped_elementwise_matches_torch` | 循环内复用 | 中等 |
| `test_local_pointer_scalar_dynamic_index_inserts_barrier` | barrier 插入正确性 | 关注点 |
| `test_local_pointer_axis_gather_matches_torch` | 张量索引模式 | 关注点 |
| `test_local_pointer_tiled_matmul_matches_torch` | LDS 与 `tl.dot`(WMMA) 交互 | **最可能暴露 RDNA4 问题** |
| `test_local_pointer_full_view_dot_avoids_pointer_convert_layout` | layout 转换消除 | **最可能暴露 RDNA4 问题** |

### L2：tile 操作（4 个用例）

```bash
python3 -m pytest unit/test_extract_tile_static_index.py unit/test_extract_tile_dynamic_index.py \
                  unit/test_insert_tile_static_index.py unit/test_insert_tile_dynamic_index.py -v
```

### L3：集成场景

```bash
python3 -m pytest integration/test_tle_local_store.py integration/test_tle_pipeline_e2e.py -v
```

`test_tle_pipeline_e2e.py` 的 4 个失败用例中若有涉及 `tle.pipeline` 的，可能仍不通过——pipeline op 走的是 `triton-tle-lower-pipe-to-nvws`，依赖 NVWS/mbarrier，不在本阶段范围。需要逐个确认失败原因是否已从 lowering 缺失变成其他问题。

### 回归验证（必须）

确保改动没有破坏原有 AMD 功能：

```bash
cd /target/flagtree-tests && python3 test_amd_basic.py
cd /tmp && python3 -c "import torch; ..."   # torch.compile 回归
```

由于 TLE 转换是独立的一轮 partial conversion，且 `markUnknownOpDynamicallyLegal` 放过所有非 TLE op，理论上对不含 TLE op 的 kernel 零影响。但 `addIllegalDialect<tle::TleDialect>()` 加进主 target 后需确认不影响正常路径。

---

## 六、风险与应对

### 风险 1：RDNA4 warp size 32 vs HCU 的 CDNA 64（最高优先级）

**这是本方案最大的不确定性。** HCU 验证过的全部路径建立在 wavefront 64 上，而本机 gfx1201 是 warp 32（已实测确认 `GPUTarget(..., warp_size=32)`）。

`local_pointers` 的偏移计算走 LinearLayout，`emitIndices` 依赖 lane id 布局。理论上 LinearLayout 框架是参数化的（AMD 后端的 swizzled shared encoding 已在 warp32 下工作），但 TLE 的 `add_select_encodings` pass 是否对 warp size 有隐式假设，只能实测。

**症状特征**：如果出问题，多半是**编译通过但结果错误**，而不是编译失败。因此 L1 中带 `matches_torch` 的数值校验用例比结构断言用例更重要。

**一条正面证据**：`TleTileToLLVMUtils.cpp` 第 126-129 行的 warp size 是从 layout 参数推导的，不是硬编码：

```cpp
  unsigned warpSizeVal = 1;
  for (auto t : threadsPerWarp)
    warpSizeVal *= t;
```

TLE 的 tile 工具至少在这一处是参数化的，说明作者没有普遍假设 warp 64。这降低了风险等级，但不能排除其他位置的隐式假设。

**应对**：优先跑数值校验用例；若失配，用 `TRITON_KERNEL_DUMP=1` 对比 TTGIR 中 `select_encodings` 前后的 layout 属性，检查 `sizePerThread` / `threadsPerWarp` 是否出现 64 相关的固定值。

### 风险 2：`add_early_assign_memory_space` 的取舍

已在第 3.3 步说明。若 `add_select_encodings` 阶段报错或 local pointer 的 memory space 推断失败，把它加为 pass 序列的第一步。

### 风险 3：`TleToLLVM` 对 NVIDIA 私有头的库级耦合

短期照抄 HCU 链接 `TritonNVIDIAGPUToLLVM` 即可，无额外代价（默认构建本就编译 nvidia 后端）。

长期清理方向：把 `ExtractOpToLLVM.cpp:29` 和 `DSLRegionOpToLLVM.cpp:28` 的 `nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h` 换成 `include/triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h`。这属于对公共代码的改动，会影响所有后端，建议作为独立 PR 提交上游，不要混在本次接入中。

### 风险 4：`isAMD()` 钩子缺失（已确认，已纳入阶段一）

初版方案曾把这一项判为阶段二，实测代码后确认是**阶段一硬阻塞项**，处理方案已写入第 3.5 步。

全库 `isHCU()` 调用点共 6 处：

| 位置 | 归属 op | 阶段一是否受影响 |
|---|---|---|
| `ExtractTileToLLVM.cpp:307` `:375` | `extract_tile` | 是 |
| `InsertTileToLLVM.cpp:312` `:365` | `insert_tile` | 是 |
| `TleTileToLLVMUtils.cpp:118` | tile 公共工具 | 是 |
| `ExclusiveCumsumOpToLLVM.cpp:125` | `exclusive_cumsum` | 否（阶段二） |

三个备选做法及取舍：

| 选项 | 做法 | 评价 |
|---|---|---|
| **A（采用）** | 新增 `isAMD()` 虚函数，5 处改为 `isHCU() \|\| isAMD()` | 语义清晰，其他后端行为不变 |
| B | 让 AMD 的 `TargetInfo` 覆写 `isHCU()` 返回 true | 一行改动，但语义谎报，不可接受 |
| C | 把 `isHCU()` 重命名为 `usesGenericLowering()` 之类语义中立名称 | 最干净，但改动面大，建议作为独立上游 PR |

**规避选项**：若只接 `local_pointers` 而不接 tile 操作，可完全跳过这一步——`LocalPointersOpConversion` 没有任何 `isHCU()` 分支。这是最快拿到第一个正向信号的路径。

### 风险 5：LDS 容量差异

RDNA4 每 workgroup 的 LDS 是 64KB，Hopper 的 shared memory 可配置到 228KB。TLE 的 tutorials 和部分测试用例的 tile 尺寸是按 Hopper 调的，可能超出 LDS 容量。

**症状**：编译期报 LDS 超限，或 `add_optimize_lds_usage`（`compiler.py:298`）介入后性能异常。

**应对**：这不是接入问题而是调参问题，遇到时缩小测试用例的 BLOCK 尺寸即可。

---

## 七、工作量估计

| 阶段 | 内容 | 估计 |
|---|---|---|
| 编码（步骤 1-3） | 3 个文件约 60 行，全部照搬 HCU 模板 | 0.5 天 |
| 编码（步骤 3.5） | `isAMD()` 钩子，3 个公共文件共 7 行 | 0.2 天 |
| 首次构建 + L0 验证 | 全量构建 49 分钟 + 冒烟 | 0.5 天 |
| L1 调试 | 若 warp32 无问题 | 0.5 天 |
| L1 调试 | 若 warp32 有隐藏假设 | 2-4 天 |
| L2/L3 + 回归 | tile 操作与集成场景 | 0.5-1 天 |

**乐观路径 2 天，悲观路径 6 天。** 主要变数是风险 1。

若采用 2.4 节末尾提到的"只接 `local_pointers`"最小路径（跳过步骤 3.5 与 tile 相关验证），拿到第一个正向信号的时间可压缩到**约 1 天**，适合先验证整体接线思路是否成立，再决定是否投入完整方案。

真正的时间消耗不在写代码，而在构建迭代（每次改 C++ 都要重新编译）和 warp size 相关的数值问题定位。建议第一轮只改 C++ 和 CMake，先把 L0 打通确认接线正确，再逐级推进。

---

## 八、阶段二展望

> 本章是阶段一完成后的近期展望。完整的分期规划、20 个 populate 与 26 个 pass 的逐项定性、以及"做全"的边界定义，见 `FlagTree-AMD-TLE-roadmap.md`。本章内容对应路线图的 Phase 1 与 Phase 4。

阶段一完成后，可选的后续工作按性价比排序：

1. **`exclusive_cumsum`**：直接复用阶段一已建的 `isAMD()` 钩子，只需再加 1 行 populate 调用与 1 处条件修改（详见 2.3 节）。AMD 的 `TargetInfo.h:57` 已有 `shuffleUp` 实现，语义上与 HCU 分支兼容。
2. **`tle.memdesc_wgmma_view`**：名字带 WGMMA 但 `ViewOpToLLVM.cpp`（65 行）的实现是纯 offset permute，**架构中立**，加 1 行 populate 即可。
3. **`tle.pack` / `tle.extract_*` 系列**：`PackOpToLLVM.cpp`（83 行）与 `ExtractOpToLLVM.cpp`（212 行）虽然同样无 NVIDIA 硬依赖，但**不要单独接入**——它们的唯一产生者是 `tle/utils/lib/Protocol.cpp`，而后者只被 `TleRawMaterialize.cpp` 调用，即它们是 TLE-Raw 的参数 materialization 协议。不接 DSLRegion 就没有任何路径会产生这些 op，接了是无用功。应随 TLE-Raw 一起考虑。
4. **上游清理 PR**：把 `TleToLLVM` 对 NVIDIA 私有头的耦合解开（风险 3 的长期方案），对整个 FlagTree 项目都有价值。
5. **贡献回上游**：AMD 后端目前无 CI 覆盖，若本方案跑通，可考虑连同一个 AMD CI workflow 一起提 PR。

**不建议尝试**的方向：WGMMA / TMA / tmem / warp_specialize / 分布式。这些不是工程投入问题，RDNA4 上没有对应硬件单元。若确实需要 TLE 的这些能力，唯一路径是 NVIDIA Hopper/Blackwell 环境。

---

## 九、参考位置速查

### TLE 侧

| 内容 | 路径 |
|---|---|
| 公共 lowering 目录（14 个 cpp / 2564 行） | `third_party/tle/dialect/lib/Conversion/TleToLLVM/` |
| `local_pointers` lowering | 同上 `LocalPointersOpToLLVM.cpp`，pattern 在 109-349 行 |
| `extract_tile`/`insert_tile` lowering | `third_party/tle/dialect/lib/Transforms/ExtractTileToLLVM.cpp` / `InsertTileToLLVM.cpp` |
| pattern 声明头 | `third_party/tle/dialect/include/Transforms/PatternTleToLLVM.h:35-53` |
| op 定义（31 个 op） | `third_party/tle/dialect/include/IR/TleOps.td` |

### 模板侧（HCU）

| 内容 | 路径 |
|---|---|
| TLE conversion target | `third_party/hcu/lib/TritonHCUGPUToLLVM/TritonGPUToLLVM.cpp:89-102` |
| TLE 转换块 | 同上 `:207-227` |
| CMake `_TLE_LIBS` | `third_party/hcu/lib/TritonHCUGPUToLLVM/CMakeLists.txt:1-10` |
| Python pass 编排 | `third_party/hcu/backend/compiler_hcu.py:409-412`（tile）与 `:425-430`（local ptr） |

### 目标侧（AMD）

| 内容 | 路径 |
|---|---|
| 转换主文件 | `third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp`（306 行） |
| 主 ConversionTarget | 同上 `:50-64` |
| TypeConverter | 同上 `:66-92` |
| 插入锚点 | 同上 `:165`（axisInfoAnalysis）与 `:174`（主 patterns）之间 |
| CMake | `third_party/amd/lib/TritonAMDGPUToLLVM/CMakeLists.txt`（44 行） |
| Python 流水线 | `third_party/amd/backend/compiler.py`，`make_ttgir` 在 `:205-266` |
| 报错位置 | 同上 `:359`（`llvm.to_module`） |
| 共享内存地址空间 | `third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp:599` |
