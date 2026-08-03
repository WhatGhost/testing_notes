# FlagTree 与 TLE 在 AMD GPU 环境的支持情况验证报告

- 验证日期：2026-08-03
- 仓库：https://github.com/flagos-ai/FlagTree
- 本地路径：`/target/FlagTree`
- 分支/提交：`main` @ `273af404b`（[TLE] Reland scoped helper inlining for warp-specialized local pointers, #859）
- 构建产物：`flagtree 0.6.0+git273af404`

---

## 一、结论速览

| 项目 | 结论 | 说明 |
|---|---|---|
| FlagTree 编译器本体 | **支持** | 在 gfx1201 上构建、编译、执行全部正常 |
| AMD 后端（`third_party/amd`） | **支持，但无官方 CI 覆盖** | 40 个 CI workflow 中没有任何 amd/rocm 项 |
| `tl.dot` / WMMA | **支持** | RDNA4 矩阵指令可用 |
| `torch.compile` + inductor | **不受影响** | 替换 triton 后仍正常工作 |
| TLE Python API / 语义层 | **可导入、可校验** | 模块与 C++ 绑定均加载成功（`tle.passes` 下 23 个 pass；另有 `tle.raw_passes` 下 3 个，全库合计 26 个） |
| TLE 实际 kernel 执行 | **不支持** | 编译在 TTGIR → LLVM IR 阶段失败 |
| TLE-Lite / TLE-Struct / TLE-Raw（三层均是） | **不支持** | AMD 后端未注册任何 TLE lowering pattern；各层 API 归属与移植可行性见 7.4 |
| TLE 分布式（distributed） | **不支持** | 依赖 NVSHMEM / FLAGCX / NVIDIA cluster |

一句话总结：**FlagTree 作为 Triton 编译器在 AMD 上完全可用；TLE 扩展在 AMD 上不可用，且不是配置问题，而是 AMD 后端根本没有接入 TLE 的编译流水线。**

需要特别注意的陷阱：由于构建时未设置 `FLAGTREE_BACKEND`，CMake 中 `FLAGTREE_TLE` 默认为 `ON`，TLE dialect 会被编译进产物。因此 `import triton.experimental.tle` **能成功**，纯语义层单测**能通过**，容易让人误判为"支持"。只有真正编译到 GPU 时才会暴露问题。

---

## 二、当前机器环境

### 硬件

| 项目 | 值 |
|---|---|
| GPU | 4 × AMD Radeon Graphics（Device ID `0x7551`） |
| GPU 架构 | `gfx1201`（RDNA4 / Navi 48） |
| warp size | 32 |
| 单卡功耗上限 | 300 W |
| CPU | AMD Ryzen Threadripper PRO 9995WX 96-Cores（192 线程） |

### 软件栈

| 组件 | 版本 |
|---|---|
| OS | Ubuntu 24.04 (noble)，内核 6.17.0-35-generic |
| ROCm | 7.2.3（`/opt/rocm-7.2.3`） |
| HIP | 7.2.53211-c2d9476115 |
| hipcc / clang | AMD clang 22.0.0git (roc-7.2.3) |
| Python | 3.12.3（venv：`/opt/venv`） |
| PyTorch | 2.10.0+rocm7.2.3.git1a270074 |
| torchvision / torchaudio | 0.25.0+rocm7.2.3 / 2.10.0+rocm7.2.3 |
| 原 Triton（安装前） | 3.6.0+rocm7.2.3.git4ed88892 |
| FlagTree（安装后） | 0.6.0+git273af404，`triton.__version__ == 3.6.0` |

### Triton 运行时识别结果

```
GPUTarget(backend='hip', arch='gfx1201', warp_size=32)
backends: ['amd', 'nvidia']
```

---

## 三、FlagTree 安装过程

### 3.1 依赖准备

系统依赖（`libxml2`、`zlib1g` 系列已预装，补装两项）：

```bash
apt-get update
apt-get install -y nlohmann-json3-dev ccache
```

Python 构建依赖：

```bash
cd /target/FlagTree
python3 -m pip install -r python/requirements.txt
# 装入 cmake 3.31.10、ninja 1.13.0、pybind11 3.0.4、lit 18.1.8
```

### 3.2 构建命令

AMD 属于默认后端，**不能**设置 `FLAGTREE_BACKEND`：

```bash
cd /target/FlagTree
unset FLAGTREE_BACKEND
export TRITON_BUILD_WITH_CCACHE=true
export MAX_JOBS=64
python3 -m pip install . --no-build-isolation -v
```

依据 `python/setup_tools/utils/tools.py`：

```python
default_backends: tuple = ("nvidia", "amd")
```

不设 `FLAGTREE_BACKEND` 时会同时构建 nvidia 与 amd 两个后端。

### 3.3 构建耗时与下载内容

- 总耗时：约 **49 分钟**（2930773 ms），退出码 0
- 生成 wheel：`flagtree-0.6.0+git273af404-cp312-cp312-linux_x86_64.whl`，**385 MB**
- 下载内容（即使只用 AMD 也会下载 NVIDIA 工具链）：
  - 预编译 LLVM：`llvm-f6ded0be-ubuntu-x64.tar.gz`，解压后约 **949 MB**（hash 见 `cmake/llvm-hash.txt`）
  - CUDA nvcc / ptxas 12.8.93、ptxas-blackwell 12.9.86
  - cuobjdump 12.8.55、nvdisasm 12.8.55
  - cudart 12.8.57、cupti 12.8.90 的 include 与 lib

构建日志：`/target/flagtree-build.log`

---

## 四、FlagTree AMD 后端验证结果：全部通过

测试脚本：`/target/flagtree-tests/test_amd_basic.py`

```
triton     : 3.6.0
triton path: ['/opt/venv/lib/python3.12/site-packages/triton']
torch      : 2.10.0+rocm7.2.3.git1a270074
hip        : 7.2.53211
device     : AMD Radeon Graphics
device cnt : 4
triton target: GPUTarget(backend='hip', arch='gfx1201', warp_size=32)
backends   : ['amd', 'nvidia']

PASS  vector_add
PASS  softmax
PASS  matmul (tl.dot -> WMMA on RDNA4)
PASS  asm stages: ['amdgcn', 'hsaco', 'llir', 'source', 'ttgir', 'ttir']

All AMD backend smoke tests passed
```

编译产物链完整：`ttir → ttgir → llir → amdgcn → hsaco`。

`torch.compile` 回归验证（确认替换 triton 未破坏 PyTorch 工作流）：

```
PASS: torch.compile + inductor works with FlagTree triton
```

### AMD 后端的已知局限

1. **无 CI 覆盖**。`.github/workflows/` 下 40 个配置覆盖了 nvidia、hcu、metax、mthreads、iluvatar、enflame、ascend、aipu、tsingmicro、xpu、sunrise、rpu、thrive、tileir，**没有任何 amd/rocm workflow**（`grep -rli "amd\|rocm" .github/workflows/` 返回 0）。AMD 后端基本是从上游 triton-lang 继承的，FlagTree 不做回归验证。
2. **README 无安装手册**。多后端支持表中 amd 一列的 Installation 是 `-`。
3. **专项优化只针对 CDNA**。`third_party/amd/backend/compiler.py` 中的架构特化仅覆盖 `gfx942`(CDNA3) 与 `gfx950`(CDNA4)，本机 gfx1201 走通用 RDNA 路径，拿不到这些优化。

---

## 五、TLE 支持情况

### 5.1 模块层：可加载（具有迷惑性）

```
libtriton.tle OK, passes: 23 add_* passes
tle.language OK -> ['B', 'BarrierKind', 'GroupKind', 'MemoryOrder', 'MeshConfig', 'P', 'S',
  'ShardedTensor', 'ShardingSpec', 'cleanup_communicator', 'communication', 'core',
  'create_dist_tensor', 'cumsum', 'device_mesh', 'distributed', 'distributed_barrier',
  'distributed_dot', 'extract_tile', 'get_mem_pool', 'gpu', 'insert_tile', 'load',
  'make_sharded_tensor', 'pipe', 'pipe_reader', 'pipe_slot', 'pipe_value',
  'pipe_wait_result', 'pipe_writer', 'range', 'raw', 'remote', 'reshard', 'shard_id', 'sharding']
tle.gpu -> ['PENDING', 'READY', 'alloc', 'alloc_barrier', 'alloc_barriers', 'barrier',
  'barrier_arrive', 'barrier_type', 'barrier_wait', 'buffered_tensor', 'buffered_tensor_type',
  'copy', 'core', 'iluvatar', 'layout', 'local_ptr', 'memory_space', 'mthreads',
  'nv_mma_shared_layout', 'pipeline', 'range', 'scope', 'shared_layout', 'smem',
  'storage_kind', 'swizzled_shared_layout', 'tensor_memory_layout', 'tmem', 'types',
  'warp_specialize', 'wgmma', 'wgmma_wait']
```

注意 `tle.gpu` 下存在 `iluvatar`、`mthreads` 两个厂商专用子模块，**没有 amd**。

### 5.2 具体不支持的环节：TTGIR → LLVM IR

复现脚本：`/target/flagtree-tests/test_tle_on_amd.py`（标准 TLE local-pointer axpy kernel）

```
target: GPUTarget(backend='hip', arch='gfx1201', warp_size=32)

--- launching TLE kernel on AMD ---
RESULT: FAILED -> RuntimeError: failed to translate module to LLVM IR

--- compilation stages that produced output ---
  ttgir          4005 bytes
  ttir           4390 bytes

  ttir  (Triton IR)      : OK
  ttgir (TritonGPU IR)   : OK
  llir  (LLVM IR)        : MISSING  <-- failure point
  amdgcn (GPU assembly)  : MISSING

--- unlowered TLE ops left in TTGIR ---
    "tle.local_pointers"
```

原始报错：

```
loc("offsets"(".../test_tle_on_amd.py":22:41)): error: LLVM Translation failed for
operation: builtin.unrealized_conversion_cast
/opt/venv/lib/python3.12/site-packages/triton/backends/amd/compiler.py:359:
RuntimeError: failed to translate module to LLVM IR
```

**失败位置精确定位**：

| 编译阶段 | AMD 上的状态 |
|---|---|
| Python 前端 / AST → Triton IR (ttir) | 成功 |
| Triton IR → TritonGPU IR (ttgir) | 成功，但 `tle.local_pointers` op 原样残留 |
| TritonGPU IR → LLVM IR (llir) | **失败** ← 断点在此 |
| LLVM IR → AMDGCN 汇编 | 未到达 |
| AMDGCN → HSACO | 未到达 |

**根因**：AMD 后端没有为 TLE op 注册 LLVM conversion pattern，`tle.local_pointers` 无人认领，被留成 `builtin.unrealized_conversion_cast`，翻译 LLVM 时报错。

### 5.3 代码层面证据

**证据 1：注册 lowering pattern 的后端里没有 amd**

`populateLocalPointersOpToLLVMPatterns` 的注册情况：

| 后端 | 注册位置 |
|---|---|
| nvidia | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:179` |
| hcu（海光） | `third_party/hcu/lib/TritonHCUGPUToLLVM/TritonGPUToLLVM.cpp:217` |
| mthreads | `third_party/mthreads/musa/lib/TritonMUSAGPUToLLVM/TritonGPUToLLVM.cpp` |
| metax | `third_party/metax/plugin/mctle/dialect/include/Conversion/TleToLLVM/` |
| iluvatar | `third_party/iluvatar/tle/lib/Conversion/TleToLLVM/LocalPointersOpToLLVM.cpp:269` |
| **amd** | **无** |

**证据 2：AMD 的 compiler.py 中零个 TLE pass**

`third_party/nvidia/backend/compiler.py` 中有 20+ 处 TLE pass 调用：

```python
tle.passes.add_lower_extract_tile(pm)
tle.passes.add_lower_insert_tile(pm)
tle.passes.add_early_assign_memory_space(pm)
tle.passes.add_select_encodings(pm)
tle.passes.add_insert_local_pointer_barriers(pm)
tle.passes.add_optimize_local_pointer_loads(pm)
tle.passes.add_optimize_local_pointer_stores(pm)
tle.passes.add_lower_wgmma(pm)
tle.passes.add_lower_barriers(pm)
...
```

在 `third_party/amd/backend/compiler.py` 中搜索 `tle` 结果为 **0**。

**证据 3：CMake 未接入**

nvidia、hcu、enflame 的 CMakeLists 均有 `if(FLAGTREE_TLE)` 分支链接 `TleIR`、`TleToLLVM`、`TritonTLETransforms`、`TritonTLEAnalysis`；`third_party/amd/` 下**一处都没有**。

**证据 4：TLE 子树内无 AMD 代码**

`third_party/tle/` 整个子树中没有任何 AMD / HIP / gfx 相关实现。tutorials 全部面向 Hopper：`tle_hopper_gemm_ws_persistent.py`、`tle_hopper_fa_ws_pipelined_pingpong_persistent.py`。

**证据 5：官方定位**

`third_party/tle/README.md`：

> This extension is specifically optimized for NVIDIA H100 devices.

README 的"扩展组件"表格中，TLE 覆盖 nvidia、enflame、mthreads、sunrise、hcu、iluvatar、metax、ascend、tsingmicro、aipu，**不含 amd**。

### 5.4 硬件层面的固有障碍

即使补齐软件接入，TLE 的部分能力在 RDNA4 上也无对应硬件：

| TLE 能力 | 依赖硬件 | gfx1201 (RDNA4) |
|---|---|---|
| `tle.gpu.alloc` + `local_ptr`（共享内存暂存） | 通用 shared memory | **有对应物**（LDS），可移植 |
| `tle.gpu.copy` TMA 路径 | Hopper TMA 引擎 | 无 |
| `tle.gpu.wgmma` | Hopper WGMMA | 无（RDNA4 用 WMMA，语义不同） |
| `tle.gpu.tmem` | Blackwell Tensor Memory | 无 |
| `warp_specialize` | Hopper warp group | 无 |
| distributed / cluster | NVIDIA cluster + DSMEM + NVSHMEM | 无 |

---

## 六、TLE 单元测试与集成测试结果明细

测试路径：`/target/FlagTree/python/test/tle/`

### 6.1 总体统计

| 测试集 | 通过 | 失败 | 跳过 | 收集错误 |
|---|---|---|---|---|
| `unit/` | **71** | **31** | 7 | 1 个文件 |
| `integration/` | **1** | **60** | 2 | 0 |
| 合计 | **72** | **91** | 9 | 1 |

复现命令：

```bash
cd /target/FlagTree/python/test/tle
python3 -m pytest unit/ -q --tb=line --ignore=unit/test_tle_get_local_pe.py
python3 -m pytest integration/ -q --tb=line
```

### 6.2 unit/ 通过的用例（71 个）

全部为**不需要真实 GPU 代码生成**的层次：Python API 契约、verifier 校验、IR 构造、缓存键计算。

| 文件 | 通过数 | 测试内容 |
|---|---|---|
| `test_tle.py` | 40 | TLE 基础 API、参数契约、错误信息 |
| `test_tle_distributed.py` | 25 | device_mesh / sharding 的 Python 层语义 |
| `test_tle_raw_cache_key.py` | 5 | TLE-Raw 缓存键计算 |
| `test_tle_cluster_tutorial_skip.py` | 1 | tutorial 跳过逻辑 |

### 6.3 unit/ 失败的用例（31 个）

全部为**需要编译到 GPU** 的用例，错误高度一致，均为 `failed to translate module to LLVM IR`。

| 文件 | 失败数 | 说明 |
|---|---|---|
| `test_tle_cumsum.py` | 12 | TLE cumsum，含 `test_tle_cumsum_amdgcn_fastpath_regression_guard` |
| `test_tle_gpu_local_ptr.py` | 12 | local_ptr 核心功能：axpy、tiled matmul、gather、barrier 插入、full view 等 |
| `test_extract_tile_dynamic_index.py` | 1 | `tle.extract_tile` 动态索引 |
| `test_extract_tile_static_index.py` | 1 | `tle.extract_tile` 静态索引 |
| `test_insert_tile_dynamic_index.py` | 1 | `tle.insert_tile` 动态索引 |
| `test_insert_tile_static_index.py` | 1 | `tle.insert_tile` 静态索引 |
| `test_tle_d2d_barrier.py` | 1 | 设备间 barrier |
| `test_tle_gpu_slot.py` | 1 | buffered tensor slot → memdesc index |
| `test_tle_raw_cache_key.py` | 1 | `ModuleNotFoundError: No module named 'mlir'`（缺 MLIR Python 包，与 AMD 无关） |

其中 `test_tle_gpu_local_ptr.py` 的 12 个失败最具代表性，它们正是 TLE 最基础的共享内存暂存能力。

### 6.4 unit/ 跳过与收集错误

| 文件 | 状态 | 原因 |
|---|---|---|
| `test_tle_wgmma_pipeline_routing.py` | 6 skipped | WGMMA 为 Hopper 专有，非 NVIDIA 环境自动跳过 |
| `test_tle_cumsum.py` | 1 skipped | 条件跳过 |
| `test_tle_get_local_pe.py` | **收集失败** | `NameError: name '_init_communicator_' is not defined`，需 NVSHMEM/FLAGCX 通信器初始化 |

### 6.5 integration/ 结果明细

| 文件 | 通过 | 失败 | 跳过 | 说明 |
|---|---|---|---|---|
| `test_tle_distributed.py` | 0 | **46** | 0 | 分布式 GEMM、DSMEM 原子操作、cluster barrier、peer smem 读取 |
| `test_tle_tma_copy.py` | 0 | 5 | 1 | TMA 拷贝，Hopper 专有 |
| `test_tle_pipeline_e2e.py` | **1** | 4 | 0 | 唯一通过的是不触发 lowering 的用例 |
| `test_tle_topk_smem_fallback.py` | 0 | 3 | 0 | 共享内存 topk 回退路径 |
| `test_tle_gemm.py` | 0 | 1 | 0 | TLE GEMM |
| `test_tle_local_store.py` | 0 | 1 | 0 | 局部存储 |
| `test_tle_ws_tma_gemm.py` | 0 | 0 | 1 | warp-specialized TMA GEMM，自动跳过 |

### 6.6 失败错误类型分布

| 错误 | 占比 | 含义 |
|---|---|---|
| `RuntimeError: failed to translate module to LLVM IR` | 绝大多数 | TLE op 未被 lowering，核心问题 |
| `RuntimeError: PassManager::run failed` | 次要 | TLE pass 在 AMD pipeline 中无法运行 |
| `AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'` | 2 例 | 分布式用例访问 NVIDIA 专有的 cluster 元数据 |
| `ModuleNotFoundError: No module named 'mlir'` | 1 例 | 环境缺 MLIR Python 包，与 AMD 无关 |

---

## 七、海光 HCU 路径：AMD 接入 TLE 的最佳参考

### 7.1 为什么 HCU 值得参考

海光 HCU（DCU）基于 AMD GPU IP 架构，在 FlagTree 中的运行时标识与 AMD **完全相同**。`python/test/tle/unit/test_tle_gpu_local_ptr.py` 中的后端判断：

```python
def _is_hcu_backend():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip"
```

HCU 与 AMD 共用 `backend == "hip"` 标识，而 HCU **有完整的 TLE 接入**。这说明 TLE 在类 AMD 架构上落地是已验证可行的，不是理论推测。

### 7.2 HCU 的 TLE 接入结构

**C++ lowering 层**——直接复用与 NVIDIA 相同的公共头文件，未另起炉灶：

`third_party/hcu/lib/TritonHCUGPUToLLVM/TritonGPUToLLVM.cpp`

```cpp
// line 23
#include "tle/dialect/include/Conversion/TleToLLVM/LocalPointersOpToLLVM.h"
// line 217
mlir::triton::tle::populateLocalPointersOpToLLVMPatterns(...);
```

**Python pass 编排层**：

`third_party/hcu/backend/compiler_hcu.py`

```python
from triton._C.libtriton import ir, passes, llvm, hcu, distributed, tle
...
# line 409-412
tle.passes.add_lower_extract_tile(pm)
tle.passes.add_lower_insert_tile(pm)
# line 425-430
tle.passes.add_select_encodings(pm)
tle.passes.add_insert_local_pointer_barriers(pm)
tle.passes.add_optimize_local_pointer_loads(pm)
tle.passes.add_optimize_local_pointer_stores(pm)
```

**CMake 层**：

- `third_party/hcu/lib/TritonHCUGPUToLLVM/CMakeLists.txt`：`if(FLAGTREE_TLE)` 链接 `TritonTLEAnalysis`、`TritonNVIDIAGPUToLLVM`、`TleToLLVM`
- `third_party/hcu/triton/Conversion/TritonToTritonGPU/CMakeLists.txt`：`if(FLAGTREE_TLE)` 设置 `_TLE_IR_LIBS TleIR`

### 7.3 若要为 AMD 后端接入 TLE，需改动的文件

按 HCU 的模式对照，最小改动集合：

| 层次 | 目标文件 | 参照 |
|---|---|---|
| C++ lowering | `third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp` | 引入 `LocalPointersOpToLLVM.h` 并调用 `populateLocalPointersOpToLLVMPatterns` |
| 转换合法性 | 同上 | 主 target 加 `addIllegalDialect<tle::TleDialect>()`，并新建一个以 ROCDL + SCF 为合法方言的 TLE 专用 `ConversionTarget` |
| CMake | `third_party/amd/lib/TritonAMDGPUToLLVM/CMakeLists.txt` | 加 `if(FLAGTREE_TLE)` 链接 TLE 库 |
| CMake | `cmake/FlagTreeOptions.cmake` | 在 tablegen 头文件依赖列表中加入 `TritonAMDGPUToLLVM` |
| Python pass 编排 | `third_party/amd/backend/compiler.py` | 在 `make_ttgir` 中插入 TLE pass 序列（`make_llir` 不需要改动） |

两点需要澄清（均已在实施方案中核实）：

- **`third_party/amd/CMakeLists.txt` 不需要改动**。HCU 除库级 CMake 外还改了 `third_party/hcu/triton/Conversion/TritonToTritonGPU/CMakeLists.txt`，那是因为 HCU fork 了一份自己的 `TritonToTritonGPU`；AMD 用的是公共库，而公共的 `lib/Conversion/TritonToTritonGPU/CMakeLists.txt` 第 1-4、22 行**已经**处理了 `_TLE_IR_LIBS`。
- 真正容易遗漏的是 `cmake/FlagTreeOptions.cmake` 中的 tablegen 依赖声明，遗漏会导致并行构建下随机的"找不到头文件"。

关于 pass 顺序：`skills/tle-developer/references/tle-sources.md` 把 local pointer 段定义为 5 步且注明不可随意重排：

1. `add_early_assign_memory_space`
2. `add_select_encodings`
3. `add_insert_local_pointer_barriers`
4. `add_optimize_local_pointer_loads`
5. `add_optimize_local_pointer_stores`

但实测 **HCU 只用了后 4 步**（`compiler_hcu.py:426-429`），未调用第 1 步；NVIDIA 与 metax 则 5 步齐全。第 1 步是否必需取决于后端，详见实施方案 3.3 节的取舍分析。

### 7.4 可移植范围的现实评估

层次归属依据仓库根目录官方架构文档 `tle_cn.md` 第 3 章。需注意共享内存管理（`alloc` / `local_ptr` / `copy`）属 TLE-Struct GPU（3.3.1.1 节）而非 Lite，这与直觉相反但对判断移植范围很关键。

| TLE 能力 | 所属层 | 可移植性 | 说明 |
|---|---|---|---|
| `tle.load(is_async)` | Lite（3.2.1） | **可行** | Python 层 hint 包装，无需专门 lowering |
| `extract_tile` / `insert_tile` | Lite（3.2.2） | **可行** | 纯寄存器重排，HCU 已验证 |
| `tle.pipe` | Lite（3.2.3） | **不可行** | lowering 位于 `TleLowerPipeToNvws.cpp`，硬依赖 NVIDIA NVWS dialect |
| 分布式全家桶 | Lite（3.2.4） | **不可行** | 依赖 NVIDIA cluster、DSMEM、NVSHMEM / FlagCX |
| `alloc` + `local_ptr` + `copy` 走 LDS | **Struct GPU**（3.3.1.1） | **可行** | 共享内存语义在 RDNA4 有直接对应，HCU 已验证 |
| `warp_specialize` | Struct GPU（3.3.1.2） | **不可行** | 无对应硬件 |
| TMA / WGMMA / tmem | Struct GPU | **不可行** | RDNA4 无对应硬件单元 |
| `tle.dsa.*` | Struct DSA（3.3.2） | 不适用 | 面向 DSA 族（如 Ascend UB/L1/L0A） |
| TLE-Raw | Raw（3.4） | 理论可行 | 需接厂商编译管线，本报告未评估 |

由此可见，TLE 的三层抽象是面向用户的 API 分层，**不构成编译器实现的模块边界**——Lite 层内部即有强 NVIDIA 依赖（`pipe`、分布式），而真正带来性能收益的共享内存控制反而在 Struct GPU 层。移植范围应按 lowering 的后端依赖切分，而非按层切分。

另需注意：HCU 基于 CDNA 架构，而本机 gfx1201 是 RDNA4，两者在 warp size（64 vs 32）、矩阵指令（MFMA vs WMMA）、LDS 容量上均有差异，照搬 HCU 实现仍需针对 RDNA4 调整。

---

## 八、环境状态与回滚

### 8.1 pip 元数据不一致（重要）

README 声称安装 flagtree 会卸载 triton，**实际并未卸载**：

```
flagtree          0.6.0+git273af404
triton            3.6.0+rocm7.2.3.git4ed88892   <- 元数据残留
```

`site-packages/triton/` 目录内容已被 FlagTree 覆盖（`triton.__version__` 现为 `3.6.0`，不带 rocm 后缀）。

> **警告**：此状态下**不要执行 `pip uninstall triton`**，那会连带删除 FlagTree 安装的文件。

### 8.2 备份位置

原 ROCm Triton 完整备份（该 wheel 为 ROCm 特供，PyPI 上不可获取）：

```
/root/triton-rocm-backup/
├── triton/                                          # 932 MB，541 个文件，已校验
├── triton-3.6.0+rocm7.2.3.git4ed88892.dist-info/
├── pip-freeze-before-flagtree.txt
└── triton-pip-show.txt
```

### 8.3 回滚步骤

```bash
pip uninstall -y flagtree
rm -rf /opt/venv/lib/python3.12/site-packages/triton
cp -a /root/triton-rocm-backup/triton /opt/venv/lib/python3.12/site-packages/
# 验证
python3 -c "import triton; print(triton.__version__)"   # 期望 3.6.0+rocm7.2.3.git4ed88892
```

---

## 九、相关文件清单

| 路径 | 说明 |
|---|---|
| `/target/FlagTree` | 仓库源码，main @ 273af404b |
| `/target/flagtree-build.log` | 完整构建日志 |
| `/target/flagtree-tests/test_amd_basic.py` | AMD 后端功能验证脚本 |
| `/target/flagtree-tests/test_tle_on_amd.py` | TLE 失败点定位脚本 |
| `/root/triton-rocm-backup/` | 原 ROCm Triton 备份 |
| `/root/.triton/llvm` | 预编译 LLVM 缓存（949 MB） |
| `/root/.triton/nvidia` | NVIDIA 工具链缓存 |

---

## 十、后续可选方向

1. **性能对比**：FlagTree 与原生 ROCm Triton 在 gfx1201 上的 GEMM / attention 性能对比。README 宣称的加速主要针对 NVIDIA、HCU、摩尔线程、燧原，AMD 未列入，实际差异需实测。
2. **完整 AMD 单测集**：跑 `python/test/unit/` 全量，核对 FlagTree 与上游 ROCm Triton 的行为差异。
3. **TLE 移植**：若确有需求，按第七节的文件清单参照 HCU 实现，先打通 `alloc` + `local_ptr` → LDS 这条最小链路（属 TLE-Struct GPU 层，非 Lite，见 7.4）。完整分期规划见 `FlagTree-AMD-TLE-roadmap.md`，第一步的逐行改动见 `FlagTree-AMD-TLE-implementation-plan.md`。
4. **换 NVIDIA 环境验证 TLE**：若目标是评估 TLE 本身的能力与收益，H100/H800 环境是唯一完整支持的路径。
