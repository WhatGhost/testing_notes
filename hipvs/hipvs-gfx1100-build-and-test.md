# hipVS 在 gfx1100（RDNA3）上的源码编译与测试指南

本文记录在**官方不支持**的 gfx1100 显卡上编译并验证 hipVS 的完整可用流程，包含测试套件的开启、运行与实测结果。

## 环境信息

| 项目 | 取值 |
|---|---|
| 仓库 | `https://github.com/AMD-Ecosystem/hipVS.git` |
| 分支 | `release/rocmds-26.03`（默认分支） |
| 提交 | `7ea3c731e15fdf2399b98097943555c4ac2c0f0a`（2026-07-27） |
| 操作系统 | Ubuntu 24.04 |
| ROCm | 7.2.1（官方推荐 7.2.3） |
| GPU | gfx1100 / RDNA3，wavefront 32 宽 |
| Python | 3.12.3 |

hipVS 官方支持的架构只有 **gfx90a、gfx942、gfx950**，全部属于 CDNA 系列（wavefront 64 宽）。gfx1100 不在其中，因此只能走源码编译。

---

## 一、克隆仓库

```bash
git clone https://github.com/AMD-Ecosystem/hipVS.git /root/hipVS
cd /root/hipVS
git log -1 --format="%H %cd" --date=short
```

> **若所处环境存在 TLS 拦截代理**，需先把代理的 CA 证书装入系统信任库，否则 `git clone` 会报证书校验失败，后续 CMake 通过 CPM 从 GitHub 拉取 raft、rmm、cuco、libhipcxx、hnswlib 等依赖也会一并失败：
>
> ```bash
> openssl s_client -showcerts -connect github.com:443 </dev/null 2>/dev/null \
>   | openssl x509 -outform PEM > /usr/local/share/ca-certificates/egress-proxy.crt
> update-ca-certificates
> ```

---

## 二、为什么 pip 安装不可用

官方提供的 PyPI 安装方式为：

```bash
pip install amd-hipvs==1.0.0 --extra-index-url=https://pypi.amd.com/rocm-7.2.3/simple/
```

这条命令**本身能装成功**，`import cuvs` 也能通过，但**一旦调用任何 GPU 接口就会段错误退出**。

根本原因是预编译 wheel 里的 GPU 代码架构不匹配。wheel 中的 `libcuvs.so`（约 135 MB）所含代码对象包（code object bundle）的目标架构只有 gfx90a、gfx942、gfx950，**没有任何 gfx1100 的代码对象**。可用 ROCm 自带工具确认：

```bash
roc-obj-ls <wheel 解包路径>/libcuvs/lib64/libcuvs.so | grep -o 'gfx[0-9a-z]*' | sort -u
```

在 gfx1100 上调用时，HIP 运行时找不到可用内核。开启 HIP 日志即可看到（默认不打印）：

```bash
AMD_LOG_LEVEL=1 python -c "import cuvs; ..."
```

```
No compatible code objects found for: gfx1100, value of HIP_FORCE_SPIRV_CODEOBJECT: 0
```

随后进程直接 `SIGSEGV`。以最简单的暴力搜索为例，崩溃点在 `cuvsBruteForceBuild()` → `raft::linalg::norm()` → `coalescedReductionThin` 内核启动处。

这类问题**无法通过环境变量或降级绕过**：GPU 内核是编译期烘进二进制的，架构不在其中就等于该内核不存在。唯一解决办法是针对 gfx1100 重新编译。

---

## 三、从源码编译（开启测试）

### 3.1 系统依赖

```bash
apt-get update
apt-get install -y --no-install-recommends \
  libopenblas0 libblas-dev liblapack-dev
```

`libcuvs.so` 在运行期会链接 `libblas.so.3`、`liblapack.so.3` 与 `libopenblas.so.0`；缺少 `libopenblas0` 会导致库加载失败。ROCm 侧所需的 hipBLAS、hipBLASLt、hipSOLVER、hipSPARSE、rocBLAS、rocPRIM、rocThrust、RCCL 由 ROCm 安装包提供，无需额外处理。

### 3.2 工具链

仓库要求 CMake ≥ 3.30.4，Ubuntu 24.04 自带的 3.28.3 不满足。用独立 venv 装新版 CMake 和 Ninja，避免污染系统 Python：

```bash
python3 -m venv /root/hipvs-venv
/root/hipvs-venv/bin/pip install --upgrade pip
/root/hipvs-venv/bin/pip install cmake==4.0.2 ninja
```

### 3.3 环境变量

把共用变量写成一个脚本，配置、编译、安装、测试各步骤统一 `source`：

```bash
cat > /root/hipvs-build-env.sh <<'EOF'
#!/bin/bash
export PATH="/root/hipvs-venv/bin:$PATH"
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake
export CMAKE_GENERATOR=Ninja

export HIPVS_ROOT=/root/hipVS
export HIPVS_BUILD_DIR=/root/hipVS/cpp/build
export HIPVS_INSTALL_PREFIX=/root/hipvs-install
export HIPVS_GPU_ARCH=gfx1100
EOF
```

### 3.4 配置

```bash
source /root/hipvs-build-env.sh
mkdir -p "$HIPVS_BUILD_DIR" && cd "$HIPVS_BUILD_DIR"

cmake -S .. -B . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_INSTALL_PREFIX="$HIPVS_INSTALL_PREFIX" \
  -DBUILD_CK_BACKEND=OFF \
  -DBUILD_TESTS=ON \
  -DBUILD_C_TESTS=ON \
  -DBUILD_MG_ALGOS=ON \
  -DCUVS_COMPILE_DYNAMIC_ONLY=ON \
  -DDETECT_CONDA_ENV=OFF
```

各选项含义：

| 选项 | 说明 |
|---|---|
| `CMAKE_HIP_ARCHITECTURES=gfx1100` | **核心开关**。指定生成 gfx1100 代码对象。也可写 `NATIVE` 让其自动探测本机架构 |
| `BUILD_CK_BACKEND=OFF` | 关闭 Composable Kernel 距离计算后端。它是 AMD 侧对标 CUTLASS 的 GEMM 路径（见 `cpp/CMakeLists.txt` 第 248 行），面向 CDNA 的矩阵指令，RDNA3 没有 MFMA 单元。关闭后距离计算走通用 HIP 内核。本文全部结果均基于此配置；开启后能否为 gfx1100 编译成功未做验证 |
| `BUILD_TESTS=ON` | 编译 C++ GoogleTest 测试目标 |
| `BUILD_C_TESTS=ON` | 编译 C API 测试。**必须显式开启**：`BUILD_TESTS=ON` 不会自动带上它 |
| `BUILD_MG_ALGOS=ON` | 编译多 GPU 算法（走 RCCL）。关闭时多 GPU 测试目标会因符号缺失而无法运行 |
| `CUVS_COMPILE_DYNAMIC_ONLY=ON` | 只编动态库，跳过静态库，显著缩短编译时间 |
| `DETECT_CONDA_ENV=OFF` | 不使用 conda 时关闭，避免 CMake 误判前缀路径 |

配置阶段会通过 CPM 从 GitHub 下载 raft、rmm、cuco、libhipcxx、hnswlib、gtest 等依赖并缓存到 `cpp/build/_deps`。

### 3.5 编译

```bash
source /root/hipvs-build-env.sh && cd "$HIPVS_BUILD_DIR"
ninja
```

编译量很大（主库加 37 个测试可执行文件），耗时以小时计，峰值内存较高。如内存吃紧可用 `ninja -j<N>` 限制并发。

### 3.6 安装

```bash
source /root/hipvs-build-env.sh && cd "$HIPVS_BUILD_DIR"
ninja install
```

> 若之后改动了任何 CMake 选项并重新 `ninja`，务必**再执行一次 `ninja install`**。否则安装目录里留着的仍是旧库，而运行测试时若 `LD_LIBRARY_PATH` 指向安装目录，就会加载到与新测试二进制不匹配的旧库。

### 3.7 验证产物架构

确认生成的库里确实只有 gfx1100 代码对象：

```bash
cd /root/hipVS/cpp/build
roc-obj-ls libcuvs.so | grep -o 'gfx[0-9a-z]*' | sort | uniq -c
```

预期输出只有一行 `220 gfx1100`——220 个代码对象包，全部为 gfx1100，无其他架构。

预期产物：

| 文件 | 大小 |
|---|---|
| `libcuvs.so` | 436 MB，含 220 个 gfx1100 代码对象包 |
| `libcuvs_c.so` | 1.1 MB |

---

## 四、运行测试

### 4.1 测试套件构成

仓库自带四套测试，本文覆盖的是 C++/C 部分：

| 层 | 位置 | 规模 |
|---|---|---|
| C++ / C（GoogleTest） | `cpp/tests/` | 110 个源文件，37 个可执行文件 |
| Python（pytest） | `python/cuvs/cuvs/tests/` | 22 个文件，90 个 test 函数 |
| Rust | `rust/` | 16 个 `#[test]` |
| C API（纯 C） | `cpp/tests/**/*.c` | 8 个文件 |

`ctest` 共注册 38 个条目：1 个 GPU 资源探测辅助项 + 37 个测试可执行文件。其中 35 个基于 GoogleTest，合计 **16,321 个测试用例**。

### 4.2 运行环境

运行测试必须让动态链接器优先找到**构建目录**里的库，并能找到 CPM 拉取的 rmm 与 rapids_logger：

```bash
source /root/hipvs-build-env.sh && cd "$HIPVS_BUILD_DIR"
export LD_LIBRARY_PATH="$HIPVS_BUILD_DIR:$HIPVS_BUILD_DIR/_deps/rmm-build:$HIPVS_BUILD_DIR/_deps/rapids_logger-build"
```

### 4.3 全量运行

`ctest` 默认单项超时 900 秒，对本套件远远不够——CAGRA 单个测试目标就有 3,000 多个用例。**直接把上限设到 7200 秒**，并把最慢的 `NEIGHBORS_TEST` 单独拎出来跑：

```bash
# 其余 37 项，每项上限 2 小时
ctest --test-dir ./tests --timeout 7200 --output-on-failure -E '^NEIGHBORS_TEST$'

# NEIGHBORS_TEST 直接运行，不经 ctest 超时限制（约 2 小时 10 分）
./gtests/NEIGHBORS_TEST
```

`NEIGHBORS_TEST` 之所以单独处理：它包含 prefiltered brute-force 用例，规模为 1,024 查询 × 8,192 数据 × 最高 2,052 维，其 **CPU 对照实现是单线程稠密-稀疏矩阵乘**，单个用例约 170 亿次乘加。运行期间 GPU 占用为 0%、单核 CPU 满载——瓶颈在校验基准而非 GPU 算子。

全量跑完约需 **7 小时**（各测试目标实际耗时之和）。测试之间通过 GPU 资源调度串行执行，大部分目标独占 GPU。

### 4.4 单项与筛选运行

按名称跑单个目标：

```bash
ctest --test-dir ./tests --timeout 7200 -R '^NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST$' --output-on-failure
```

只跑某个算法族：

```bash
ctest --test-dir ./tests --timeout 7200 -R 'CAGRA' --output-on-failure
```

直接调用二进制并筛选用例，适合定位或补跑：

```bash
./gtests/NEIGHBORS_TEST --gtest_list_tests                     # 列出全部用例
./gtests/NEIGHBORS_TEST --gtest_filter='RefineTest/*'          # 只跑某套件
```

### 4.5 读取结果

`ctest` 的完整输出写在 `cpp/build/tests/Testing/Temporary/LastTest.log`，**每次运行都会覆盖**，需要保留就先备份。注意 ctest 会截断**通过**项的输出，因此从该日志统计 `[       OK ]` 行数会偏少；判断是否通过应以 ctest 的 `Passed` / `Failed` 结论为准（GoogleTest 仅在全部用例通过时返回退出码 0）。

统计某次运行的用例通过与失败数：

```bash
L=cpp/build/tests/Testing/Temporary/LastTest.log
echo "通过: $(grep -c '^\[       OK \]' "$L")"
echo "失败: $(grep -c '^\[  FAILED  \]' "$L")"
```

---

## 五、测试结果

### 5.1 汇总

| 指标 | 数值 |
|---|---|
| 测试目标（基于 GoogleTest） | 35 |
| 测试用例总数 | 16,321 |
| 通过 | **16,321（100%）** |
| 失败 | **0** |
| 累计机时 | 约 7 小时 |

**全部用例通过，无一例数值错误。**

其中最值得关注的是 CAGRA：它的图搜索内核按 `team_size` 参数划分线程组，是整个库里最依赖 wavefront 宽度假设的部分（RDNA3 为 32 宽，官方支持的 CDNA 为 64 宽）。其 float、half、int8、uint8 四种数据类型共 9,417 个用例同样零失败。

### 5.2 近邻搜索算法（21 个目标）

| 测试目标 | 覆盖模块 | 用例数 | 耗时 |
|---|---|---:|---:|
| NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST | CAGRA | 3,139 | 49 分 23 秒 |
| NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST | CAGRA | 3,048 | 42 分 07 秒 |
| NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST | CAGRA | 1,615 | 28 分 24 秒 |
| NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST | CAGRA | 1,615 | 33 分 06 秒 |
| NEIGHBORS_ANN_CAGRA_TEST_LARGE_DATASET | CAGRA | 3 | 9 分 16 秒 |
| NEIGHBORS_ANN_CAGRA_TEST_BUGS | CAGRA | 1 | 21.50 秒 |
| NEIGHBORS_ANN_CAGRA_HELPERS_TEST | CAGRA | 1 | 0.81 秒 |
| NEIGHBORS_ANN_IVF_PQ_TEST | IVF-PQ | 1,085 | 14 分 04 秒 |
| NEIGHBORS_ANN_VAMANA_TEST | Vamana | 945 | 9 分 36 秒 |
| NEIGHBORS_ANN_NN_DESCENT_TEST | NN-Descent | 736 | 27 分 37 秒 |
| NEIGHBORS_ALL_NEIGHBORS_TEST | all-neighbors | 600 | 38 分 30 秒 |
| NEIGHBORS_ANN_IVF_FLAT_TEST | IVF-Flat | 388 | 6 分 44 秒 |
| NEIGHBORS_DYNAMIC_BATCHING_TEST | 动态批处理 | 280 | 22 分 21 秒 |
| NEIGHBORS_TEST | refine / 稀疏 KNN / 预过滤 | 279 | 2 小时 06 分 |
| NEIGHBORS_HNSW_TEST | HNSW | 256 | 4 分 49 秒 |
| NEIGHBORS_ANN_SCANN_TEST | ScaNN | 102 | 1 分 44 秒 |
| NEIGHBORS_ANN_BRUTE_FORCE_TEST | 暴力搜索 | 78 | 5.46 秒 |
| NEIGHBORS_TIERED_INDEX_TEST | 分层索引 | 72 | 19.73 秒 |
| NEIGHBORS_EPSILON_NEIGHBORHOOD_TEST | epsilon 邻域 | 56 | 3.68 秒 |
| NEIGHBORS_BALL_COVER_TEST | ball cover | 18 | 2.10 秒 |
| NEIGHBORS_MG_TEST | 多 GPU | 15 | 30.56 秒 |

### 5.3 基础模块（6 个目标）

| 测试目标 | 覆盖模块 | 用例数 | 耗时 |
|---|---|---:|---:|
| DISTANCE_TEST | 距离度量 | 1,651 | 1 分 47 秒 |
| PREPROCESSING_TEST | 预处理 / 量化 | 198 | 9.97 秒 |
| CLUSTER_TEST | kmeans 聚类 | 86 | 3 分 31 秒 |
| STATS_TEST | 统计（召回率等） | 8 | 0.78 秒 |
| SPARSE_TEST | 稀疏矩阵 | 5 | 1.04 秒 |
| INTEROP_TEST | DLPack 互操作 | 1 | 0.30 秒 |

### 5.4 C API（8 个目标）

| 测试目标 | 覆盖模块 | 用例数 | 耗时 |
|---|---|---:|---:|
| MG_C_TEST | 多 GPU | 18 | 10.62 秒 |
| ALL_NEIGHBORS_C_TEST | all-neighbors | 11 | 1.66 秒 |
| CAGRA_C_TEST | CAGRA | 4 | 2.86 秒 |
| BRUTEFORCE_C_TEST | 暴力搜索 | 3 | 1.06 秒 |
| DISTANCE_C_TEST | 距离度量 | 1 | 0.50 秒 |
| IVF_FLAT_C_TEST | IVF-Flat | 1 | 0.97 秒 |
| IVF_PQ_C_TEST | IVF-PQ | 1 | 1.66 秒 |
| HNSW_C_TEST | HNSW | 1 | 1.37 秒 |

另有 `cuvs_c_test`、`cuvs_c_neighbors_test` 两个纯 C 测试程序通过，未采用 GoogleTest 因此不计入用例统计。

### 5.5 C API 端到端抽查

除官方测试套件外，另用独立 C 程序对照 CPU 计算结果做了验证：

| 场景 | 结果 |
|---|---|
| 暴力搜索：`cuvsBruteForceBuild` + `cuvsBruteForceSearch` | 10 条查询的 top-1 邻居与 CPU 结果全部一致，距离误差 9e-06（float32 精度范围内） |
| CAGRA：10,000 × 64 数据集，100 条查询，`team_size` 取 0/8/16/32 | recall@10 = 0.976，各 team_size 结果一致 |

---

## 六、已知限制

测试全绿说明**数值正确性与 API 行为**在 gfx1100 上成立，但不等同于官方支持，以下几点需要注意：

1. **性能未做基准对比。** 本次只验证正确性，未运行 `BUILD_CUVS_BENCH` 基准测试，也未与官方支持的 CDNA 卡做吞吐对比。
2. **距离计算未走 CK 的 GEMM 优化路径。** 因关闭了 CK 后端，距离计算使用通用 HIP 内核。RDNA3 缺少 MFMA 单元，CK 在该架构上即便可用也是走 WMMA 而非 MFMA，性能特征与官方支持的 CDNA 卡不同。
3. **多 GPU 仅验证了代码路径。** 测试在单卡上完成，`NEIGHBORS_MG_TEST` 与 `MG_C_TEST` 证明多 GPU 代码路径可正常初始化与执行，但未做真正的跨卡通信测试。官方也将多 GPU 标注为实验性功能。
4. **ROCm 版本略低于推荐值。** 实测使用 7.2.1，官方推荐 7.2.3。
5. **Python 绑定未编译。** 本文只覆盖 C++/C 层。若需要 Python 接口，须从源码编译 `python/cuvs`，且要避免 pip 自动拉取为 gfx942 预编译的 `pylibraft`、`rmm` 等包。

---

## 附：关键路径速查

| 用途 | 路径 |
|---|---|
| 源码 | `/root/hipVS` |
| 构建目录 | `/root/hipVS/cpp/build` |
| 安装前缀 | `/root/hipvs-install` |
| 测试二进制 | `/root/hipVS/cpp/build/gtests/` |
| ctest 工作目录 | `/root/hipVS/cpp/build/tests` |
| ctest 详细日志 | `/root/hipVS/cpp/build/tests/Testing/Temporary/LastTest.log` |
| 环境变量脚本 | `/root/hipvs-build-env.sh` |
| 依赖缓存 | `/root/hipVS/cpp/build/_deps` |
