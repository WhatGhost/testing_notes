# W7900 上 vLLM(+gguf-plugin)与 llama.cpp 的对比压测 —— 结果与安装记录

- **机器**:AMD Radeon PRO W7900,gfx1100 / RDNA3,96 CU,48 GiB(单卡,`HIP_VISIBLE_DEVICES=0`)
- **模型**:Muse-Glimmer-30B,GGUF `Q4_K_XL`(Dynamic,18.30 GiB)+ dflash 草稿模型(1.52 GiB)
- **数据**:测试于 2026-08-31 ~ 09-03 完成,共 **46 次有效运行**(22 组配置 × 2 轮 + 2 次异常复跑)
- **对照文档**:MI300X 上的原始优化记录 `WhatGhost/testing_notes` → `vllm-gguf/opt-vllm-gguf-plugin.md`

> 本文的性能数字**全部来自 W7900 实测**。MI300X 报告里的倍数不适用于本机——
> 两者架构不同(96 CU RDNA3 vs 304 CU CDNA3),kernel 派发阈值也已按本机重新标定(见 §5.3)。

---

## 1. 结论速览

两个引擎没有全局赢家,**分界线在并发 4**:并发 1 归 llama.cpp,并发 4 及以上归 vLLM。

每个并发档下,各引擎**吞吐最优**的那一格及其延迟表现(A/B 两轮均值,延迟取 p50):

| 并发 | 引擎 | 最优配置 | 吞吐 tok/s | 吞吐加速比 | TPOT ms | TTFT ms |
|---|---|---|---|---|---|---|
| 1 | **llama.cpp** ✅ | DFlash on | **62.4** | 1.00×(基准) | **13.6** | 384 |
| 1 | vLLM | N=16 | 42.2 | **0.68×** | 20.5 | 382 |
| 4 | llama.cpp | DFlash on | 76.1 | 1.00×(基准) | 43.9 | 1767 |
| 4 | **vLLM** ✅ | N=5 | **103.1** | **1.35×** | **35.2** | **546** |
| 16 | llama.cpp | DFlash off | 143.9 | 1.00×(基准) | 92.7 | 1095 |
| 16 | **vLLM** ✅ | N=0 | **205.9** | **1.43×** | **70.9** | **332** |

**「吞吐加速比」以同并发下的 llama.cpp 为基准**,大于 1 表示 vLLM 更快。
粗体是该并发下该指标的胜者(并发 1 的 TTFT 382 vs 384 差 0.5%,**属并列**,未标)。

赢家在并发 4 和 16 上是**三项指标通吃**——吞吐更高的同时延迟也更低,没有权衡;
并发 1 上 llama.cpp 的 TPOT 只有 vLLM 的 2/3,优势同样干净。

**并发越高,vLLM 越占优**(0.68× → 1.35× → 1.43×),原因是连续批处理把并发吃得更满,
而不是单点算子更快:关掉投机后,两者在并发 4 上吞吐几乎完全相等(60.9 vs 60.6,差 0.5%),
差距只在并发 1 和并发 16 两端才拉开。

> **表中的 TPOT/TTFT 是"吞吐最优配置"下的值,不是各引擎能达到的最低延迟。**
> 若单独优化 TTFT,vLLM 在并发 1 关掉投机(N=0)可做到 **216 ms**(比表中 382 更低),
> 但吞吐会掉到 22.1。要按延迟/尾延迟选型请直接看 §9.3 的全分位总表。

### 三条值得记住的结论

1. **并发 16 时投机解码是负收益,两个引擎都是。** llama.cpp 从 143.9 掉到 88.3(只剩 61%),
   vLLM 从 205.9 掉到最好 178.4(损失 13.4%)。batch 已经被真实请求填满时,草稿模型的
   每步开销摊不掉,反而挤占了本可用于真实 token 的算力。
2. **关掉投机后,两个引擎在并发 4 上几乎完全持平**(60.9 vs 60.6,差 0.5%)。
   差距集中出现在并发 1(llama.cpp 领先)和并发 16(vLLM 领先),说明区别主要来自
   **投机解码实现 + 调度策略**,而不是底层算子快慢。
3. **llama.cpp 的 DFlash 在低并发极强**:并发 1 时 29.6 → 62.4,翻了 2.11 倍,
   这是全表最大的单项增益,也是它在并发 1 上反超 vLLM 的唯一原因。

---

## 2. 软硬件环境

| 项 | 值 |
|---|---|
| 基础容器 | `rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0` |
| ROCm | 7.2.3 |
| Python | 3.12.3(虚拟环境 `/opt/venv`) |
| torch | `2.12.0+git6bbd260`(gfx1100 在 `torch.cuda.get_arch_list()` 内) |
| triton | `3.7.1+gitf0b55c07` |
| vLLM | `0.28.1rc1.dev43+g6f7df92a8.rocm723`(nightly wheel) |
| vllm-gguf-plugin | `0.0.5`(editable,源码在 `/target/vllm-gguf-workspace/vllm-gguf-plugin`) |
| llama.cpp | tag `b10373`,HIP 后端,`gfx1100` 单目标 |

容器起法(宿主机上执行):

```bash
docker run -d --name vllm_gyy_1 \
  --network host --ipc host --privileged \
  --cap-add CAP_SYS_ADMIN --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --shm-size 10g --device /dev/kfd --device /dev/dri \
  -v /home/yuygao:/target \
  rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0 sleep infinity
```

> 进容器后确认 `which python pip` 都指向 `/opt/venv/bin/`。装错解释器是最常见的坑:
> `pip install` 明明成功,`vllm` 却 import 不到插件。

---

## 3. 安装:代码从哪来

两个仓库都是 **WhatGhost 的 fork**,分支都叫 **`gguf-kernel-perf`**,公开仓库,匿名 clone 即可。

| 组件 | 仓库 | 分支 | 本次使用的 commit |
|---|---|---|---|
| 插件 | `https://github.com/WhatGhost/vllm-gguf-plugin.git` | `gguf-kernel-perf` | `a48163b89814bc56bc55c54981e9140428045c28` |
| vLLM 源码 | `https://github.com/WhatGhost/vllm.git` | `gguf-kernel-perf` | `4b33d88c72f517b1ab7da3d8c6f43ee65bb5308e` |
| llama.cpp | `https://github.com/ggml-org/llama.cpp` | tag `b10373` | codeload tarball |

插件的上游是 `vllm-project/vllm-gguf-plugin`(vLLM 在 in-tree GGUF 支持下线后的外挂方案,
见 [vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583));
fork 的增量是 b10373 MMQ kernel 的 vendor 与接入。

```bash
mkdir -p /target && cd /target
git clone -b gguf-kernel-perf https://github.com/WhatGhost/vllm-gguf-plugin.git \
    /target/vllm-gguf-workspace/vllm-gguf-plugin
git clone -b gguf-kernel-perf --depth 50 https://github.com/WhatGhost/vllm.git \
    /target/vllm-gguf-workspace/vllm-src
ln -sfn /target/vllm-gguf-workspace/vllm-src /target/vllm-src
```

要严格复现本次结果,就钉死到上表的 commit:

```bash
cd /target/vllm-gguf-workspace/vllm-gguf-plugin
git fetch --depth 1 origin a48163b89814bc56bc55c54981e9140428045c28 && git checkout FETCH_HEAD

cd /target/vllm-gguf-workspace/vllm-src
git fetch --depth 1 origin 4b33d88c72f517b1ab7da3d8c6f43ee65bb5308e && git checkout FETCH_HEAD
```

`checkout FETCH_HEAD` 后是 detached HEAD,这里是**想要的**:要的是固定快照,不是会漂移的分支。

> **本机现状提示**:`/target/vllm-gguf-workspace/` 下的两个目录**当前没有 `.git`**
> (代码是拷贝进来的,不是原地 clone)。所以本机无法用 `git log` 反查版本,
> 上表的 commit 来自安装时的记录。新机器请按上面的命令重新 clone。

---

## 4. 安装 vLLM:nightly wheel + 源码覆盖(**不从源码编译**)

### 4.1 为什么不编译

在 ROCm 上从源码编 vLLM 等于重编全部 HIP kernel,耗时极长,而 fork 相对 wheel 的差异
**只有纯 Python 文件**(主要是 `vllm/config/speculative.py`),重编没有意义。所以用组合方案:

1. pip 装 ROCm nightly 的 vLLM wheel → 拿到编译好的 `.so` 和全套依赖;
2. fork 源码克到 `/target/vllm-src`;
3. 在源码树里**软链**回 wheel 的 `.so`(源码树无二进制,不链就 import 失败);
4. 起服务时 `PYTHONPATH=/target/vllm-src`,让源码树的 `.py` **盖过** site-packages。

> 前提是 fork 的改动必须是纯 Python。将来 fork 里若出现 C++/HIP 改动,这条路就不成立了。

### 4.2 装 wheel

```bash
python -m pip install vllm --pre \
    --extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm723 --upgrade
```

**关于版本漂移**:nightly 索引只保留近期构建。MI300X 原始文档记录的是
`0.27.2rc1.dev122+g8efa13b70.rocm723`,该版本在本次安装时**已被索引清理**,
本机实际装到的是 `0.28.1rc1.dev43+g6f7df92a8.rocm723`,**实测可用**。

先查索引上还剩什么:

```bash
curl -sSL https://wheels.vllm.ai/rocm/nightly/rocm723/vllm/ \
  | grep -oE 'vllm-[^"<]*\.whl' | sort -u | tail -5
```

若要与本机完全一致,钉住版本:

```bash
python -m pip install --pre --extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm723 \
    'vllm==0.28.1rc1.dev43+g6f7df92a8.rocm723'
```

若已被清理,装索引当前版本即可,但**必须重跑 4.3 的软链和 §7 的验证**——
新 wheel 的 `.so` 清单可能变化,源码树与新 wheel 之间也可能出现真实的 API 漂移。

> 别单独升 torch。`import vllm` 报 `undefined symbol` 基本都是 wheel 的 `.so`
> 与当前 torch ABI 对不上;上面那条 pip 命令会把整个 torch 栈换成配套版本。

### 4.3 软链回二进制

源码树里只有 `.py`。缺了软链,`PYTHONPATH` 一指过去就 import 失败。

```bash
SP=$(python -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "vllm"))')
cd /target/vllm-src/vllm

for item in "$SP"/*.so "$SP"/vllm-rs "$SP"/_version.py; do
    [ -e "$item" ] || continue
    base=$(basename "$item")
    [ -e "$base" ] && continue        # 源码树里已有同名文件就别覆盖
    ln -s "$item" "$base"
done

find /target/vllm-src/vllm -maxdepth 1 -type l | sort   # 本机实测 10 项
```

本机的 10 项:

```
_C.abi3.so                      _rust_tool_parser.abi3.so    spinloop.abi3.so
_C_stable_libtorch.abi3.so      cumem_allocator.abi3.so      vllm-rs
_moe_C_stable_libtorch.abi3.so  fs_io_C.abi3.so              _version.py
_rocm_C.abi3.so
```

### 4.4 源码覆盖为什么是必须的

fork 给 `speculative.py` 加了 `hf_config_path`。draft 是**单文件 GGUF**,它的 config
必须靠这个键单独指路。所以:**开投机解码 = 必须带 `PYTHONPATH=/target/vllm-src`**,
否则启动时该键会被 wheel 版 vLLM 判为非法字段而拒绝。

```bash
grep -c hf_config_path \
  /opt/venv/lib/python3.12/site-packages/vllm/config/speculative.py   # wheel → 0
grep -c hf_config_path /target/vllm-src/vllm/config/speculative.py    # 源码树 → 3
```

> **一个会骗人的地方**:`_version.py` 是软链到 wheel 的,所以即使走源码树,
> `vllm.__version__` 打印的仍是 wheel 的版本串。排查时别信版本串,查 `vllm.__file__`。

---

## 5. 安装插件(含 W7900 专属改动)

### 5.1 安装命令

```bash
cd /target/vllm-gguf-workspace/vllm-gguf-plugin
PYTORCH_ROCM_ARCH=gfx1100 python -m pip install -e . --no-build-isolation
```

三个要点:

- **`--no-build-isolation` 不能省**。插件要链接**已装的那个 torch**;开了构建隔离,
  pip 会在临时环境拉一个别的 torch 来编,编出的 `.so` ABI 对不上。
- **`PYTORCH_ROCM_ARCH=gfx1100` 建议显式设**。插件走 torch 的 `_get_rocm_arch_flags()`
  定目标架构,未设时**按当前可见显卡自动探测**,多卡混插或容器可见性异常时会探错。
- **不要加 `VLLM_GGUF_PLUGIN_NO_MMQ_B10373=1`**。见下条。

### 5.2 关键修正:W7900 上**不需要**跳过 b10373

MI300X 原始文档把 `VLLM_GGUF_PLUGIN_NO_MMQ_B10373=1` 列为 RDNA3 的安装前提,理由是
vendor kernel 未在 RDNA3 验证过。**本机实测推翻了这条**:不带该开关可以正常编译、加载、运行,
且 b10373 正是性能的来源。

编译无需任何源码适配:`mmq-config-rdna3.cuh`、`mma.cuh`、`vendors/hip.h` 均已 vendor 齐全,
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` 正常展开。

**快速判据**:编出的 `_C_gguf.abi3.so` 应为 **6,034,992 字节**(跳过 b10373 时是 5,158,320,
相差 876 KB,与单独编出的 b10373 静态库体积吻合)。

```bash
ls -l /target/vllm-gguf-workspace/vllm-gguf-plugin/vllm_gguf_plugin/_C_gguf.abi3.so
```

### 5.3 必须打的两处改动

装完之后,**还要改两处代码并重装**,否则 RDNA3 会静默回落到老 kernel(b2899),性能差一个数量级。

**① 放开架构门禁** —— `vllm_gguf_plugin/csrc/gguf/b10373/mmq_b10373_launcher.cu`
的 `supported()` 末尾:

```cpp
const int id = ggml_cuda_get_device();
const int cc = ggml_cuda_info().devices[id].cc;
// CDNA 走 MFMA,RDNA3 走功能对应的 WMMA。RDNA4 的 tile 配置虽已 vendor,
// 但没有硬件实测过,暂不放开。
return GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA3(cc);
```

**② 窄输出阈值按架构分流** —— `vllm_gguf_plugin/quantization/linear.py`:

```python
_NARROW_OUTPUT_DIM = 1024
_NARROW_MMVQ_SAFE_CDNA = 48      # 原值,MI300X(304 CU)标定
_NARROW_MMVQ_SAFE_RDNA3 = 80     # W7900(96 CU)重新标定
```

派发处改为 `return _NARROW_MMVQ_SAFE_RDNA3 if _is_rdna3() else _NARROW_MMVQ_SAFE_CDNA`。

阈值 80 的依据(attn_k/attn_v,256×6656 Q6_K,单位 ms):n ≤ 80 时 MMVQ 稳定快 1.5~2.4 倍,
88~112 为抖动区,n ≥ 112 后 MMQ 稳定胜出;取 80 是抖动区下沿的保守值。

> 这条改动的**实测增益只有 +3~4%,而测量噪声本身有 2.0%**,分离度约 1.5~2 个标准差,
> 不构成干净的显著性结论。保留它是因为有 kernel 级依据(n=64 处两条路径差 2.16 倍),
> 但**不应宣称为大幅优化**。真正的性能来源是 ① 的门禁。

改完重装(`pip install -e .` 会重新编译):

```bash
cd /target/vllm-gguf-workspace/vllm-gguf-plugin
PYTORCH_ROCM_ARCH=gfx1100 python -m pip install -e . --no-build-isolation
```

---

## 6. 安装 llama.cpp b10373

```bash
mkdir -p /target/vllm-gguf-workspace/llamacpp && cd /target/vllm-gguf-workspace/llamacpp
curl -L -o llama.cpp-b10373.tar.gz \
  https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/b10373
tar xf llama.cpp-b10373.tar.gz && mv llama.cpp-b10373 src && cd src

cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
install -Dm755 build/bin/llama /root/.local/bin/llama
```

本机产物记录(`llamacpp/meta/BINARY_VERSION.txt`):

```
tag: b10373
built: 2026-08-31T07:40:31+00:00
sha256: 0862379c4c3001815d2c75af37cd0c06811781a286c0f129ed91fda4709cc77c
flags: -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DLLAMA_CURL=OFF
codegen_check: Q4_K/Q5_K/Q6_K = 204/204/106 条 v_wmma_i32_16x16x16_iu8,v_dot4 均为 0
smoke_dflash: OK (block_size=16, acceptance=0.614, mean_len=2.84)
```

`codegen_check` 那行是关键验收:反汇编 fatbin 确认走的是 **WMMA 矩阵指令**而非 `v_dot4`
向量路径——这是 b10373 相对 b2899 的核心区别,也是唯一能证明"新 kernel 真的编进去了"的硬证据。

> 二进制装在 `/root/.local/bin/llama`,**该路径默认不在 PATH 里**,压测脚本用的是绝对路径。

---

## 7. 安装后验证

```bash
# 1) 解释器对不对
which python pip                      # 都应在 /opt/venv/bin/

# 2) GPU 可见、架构匹配
python -c "import torch; print(torch.cuda.get_device_name(0));
print('gfx1100:', 'gfx1100' in torch.cuda.get_arch_list())"
#   期望 AMD Radeon PRO W7900 / True

# 3) 源码覆盖生效
PYTHONPATH=/target/vllm-src python -c "import vllm; print(vllm.__file__)"
#   期望 /target/vllm-src/vllm/__init__.py

# 4) draft 侧 hf_config_path 存在(投机解码的前提)
grep -c hf_config_path /target/vllm-src/vllm/config/speculative.py    # 期望 3

# 5) 插件 C 扩展已注册
PYTHONPATH=/target/vllm-src python -c "
import torch, vllm_gguf_plugin
print('ggml_dequantize:', hasattr(torch.ops._C_gguf, 'ggml_dequantize'))
print('ggml_mul_mat_a8:', hasattr(torch.ops._C_gguf, 'ggml_mul_mat_a8'))"
#   期望两个 True

# 6) 插件已注册进 vLLM 量化后端
PYTHONPATH=/target/vllm-src python -c "
from vllm.plugins import load_general_plugins; load_general_plugins()
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
print('gguf registered:', 'gguf' in QUANTIZATION_METHODS)"
#   期望 True

# 7) 【W7900 关键】b10373 是否真的启用
PYTHONPATH=/target/vllm-src python -c "
from vllm_gguf_plugin.ops import mmq_b10373_available
import torch; torch.zeros(1, device='cuda')
print('mmq_b10373:', mmq_b10373_available())"
#   打完 §5.3 的改动后期望 True。若为 False,说明门禁没放开,性能会差一个数量级。
```

第 7 条的另一个查法:服务日志里应出现

```
INFO ... [ops.py:304] GGUF MMQ 后端: b10373 矩阵核心
```

注意该行由 **EngineCore 子进程**打印,要确认进程号与实际推理进程一致。

---

## 8. 压测方法(口径)

**两个引擎用的是同一套客户端测量口径**,这是结果可比的前提:

| 项 | 设定 |
|---|---|
| 数据集 | GSM8K 真实题目,**200 条**,两引擎同一份、同顺序 |
| 并发模型 | 闭环恒定并发(`asyncio.Semaphore`),一条完成立刻补一条 |
| 计时 | 纯客户端墙钟,从第一个请求发出到最后一个 token 收到 |
| 吞吐 | `out_tok_per_s = 输出 token 总数 / 墙钟` |
| TPOT | `(e2e_ms - ttft_ms) / (输出 token 数 - 1)` |
| 采样 | `temperature=1.0, top_p=0.95, top_k=64`(两引擎一致) |
| `max_tokens` | 2048 |
| 轮次 | **每格跑 2 轮(A/B)**,且 A/B 之间**颠倒 cell 顺序**,把位置效应与配置效应解耦 |

引擎侧关键参数:

```bash
# vLLM
HIP_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 PYTHONPATH=/target/vllm-src \
VLLM_GGUF_MMQ_B10373=1 \
  vllm serve "$TARGET" --tokenizer "$CONF" --hf-config-path "$CONF" \
  --speculative-config "{\"model\":\"$DRAFT\",\"hf_config_path\":\"$DRAFT_CONF\",\"num_speculative_tokens\":$N}" \
  --max-model-len 131072 --max-num-seqs 16 --gpu-memory-utilization 0.9 --port $PORT

# llama.cpp
HIP_VISIBLE_DEVICES=0 llama serve \
  -m "$M/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf" \
  --mmproj "$M/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf" \
  --spec-type draft-dflash -md "$M/dflash-Muse-Glimmer-30B-Q4_K_M.gguf" \
  -ngld 99 --spec-draft-n-max 16 \
  -a muse-glimmer-30B -ngl 99 -c $CTX -np $NP \
  --cache-ram 0 --slot-save-path "$SLOTDIR" \
  --host 127.0.0.1 --port $PORT \
  --jinja --temp 1.0 --top-p 0.95 --top-k 64
```

两个防污染措施,都很关键:

- **`--cache-ram 0`**:关掉 llama.cpp 的 RAM prompt cache。不关的话上一轮的题会被下一轮命中,
  跨轮污染。只影响跨 slot 的 prefill 复用,decode 在 GPU 上不受影响。
- **每格重起服务**(`FRESH_SERVE=1`)+ 开测前用 `/slots` erase 清 slot 级 KV。

`-c` 是**总量、按 `-np` 均分到各 slot**,本机取 `CTX_PER_SLOT=65536`
(MI300X 用 131072,W7900 只有 48 GiB 放不下,减半)。实际负载 prompt 约 114 token、
输出约 318 token,离上限很远,不构成约束。

### 已知的口径不对称

vLLM 的 `/metrics` 不暴露**逐请求的 prefill 计时**和**投机解码细节**,这些字段在
vLLM 侧记为 `null` 而**不是编一个近似值**。跨引擎比较时只用两边都有的量
(吞吐、TTFT、TPOT、E2E)。另外 SSE 分块粒度两边不同(vLLM 约 3.37 token/chunk,
llama.cpp 约 1.01),所以**不要用 chunk 级派生量做跨引擎比较**。

---

## 9. 测试结果

### 9.1 vLLM 全 N 扫描

`N` = `num_speculative_tokens`,`N=0` 表示关闭投机解码。吞吐单位 tok/s,取 A/B 两轮均值。

| 并发 | N=0 | N=1 | N=2 | N=3 | N=5 | N=8 | N=16 |
|---|---|---|---|---|---|---|---|
| 1 | 22.1 | 19.1 | — | 29.4 | 35.7 | 41.5 | **42.2** |
| 4 | 60.9 † | 66.5 | — | 102.8 | **103.1** | 92.5 | — |
| 16 | **205.9** | 158.8 | 170.7 | 178.4 | 173.5 | — | — |

† 该格 A 轮出现异常值(38.5 vs B 轮 60.6,轮差 44.6%),已于全部测试结束后在**独占卡上补跑两次**:
61.1 / 61.0(彼此差 0.16%),与 B 轮吻合。**A 轮 38.5 判为宿主侧争抢导致的坏点已剔除**,
表中 60.9 是 B 轮 + 两次补跑共 3 个样本的均值(60.6 / 61.1 / 61.0)。
判据是 N=0 的并发缩放曲线(22.1 → 60.9 → 205.9)线性度良好,38.5 偏离趋势 37%。

**草稿接受率**(A/B 两轮几乎完全一致,故只列一个值):

| 并发 | N=1 | N=2 | N=3 | N=5 | N=8 | N=16 |
|---|---|---|---|---|---|---|
| 1 | 0.871 | — | 0.678 | 0.533 | 0.412 | 0.270 |
| 4 | 0.869 | — | 0.678 | 0.532 | 0.410 | — |
| 16 | 0.869 | 0.776 | 0.680 | 0.534 | — | — |

接受率**只随 N 变化,几乎不随并发变化**——这符合预期,接受率是草稿模型与目标模型的
分布吻合度,与调度无关。N 越大,越靠后的草稿 token 越难命中,故单调下降。

**注意 N=1 是负收益**(并发 1 时 19.1 < 22.1,并发 16 时 158.8 < 205.9):
接受率虽高达 0.87,但每步只多赚不到 1 个 token,摊不平草稿模型一次前向的开销。

### 9.2 llama.cpp DFlash 开关对比

| 并发 | DFlash off | DFlash on | 变化 | 接受率(on) |
|---|---|---|---|---|
| 1 | 29.6 | **62.4** | **+111%** | 0.288 |
| 4 | 60.6 | **76.1** | +26% | 0.255 |
| 16 | **143.9** | 88.3 | **−39%** | 0.191 |

DFlash 是块扩散草稿,一次前向预测 16 token 的块。它在并发 1 上收益极大(翻倍还多),
但**并发 16 时是重度负收益**,吞吐只剩 61%,TPOT(p50)从 92.7 ms 恶化到 151.4 ms,
TTFT(p50)从 1095 ms 恶化到 4756 ms(**4.3 倍**)。

另外**llama.cpp 的接受率随并发明显下降**(0.288 → 0.255 → 0.191),
这一点与 vLLM 不同(vLLM 的接受率只随 N 变化)。接受率下滑与调度开销上升叠加,
是并发 16 上反转如此剧烈的直接原因。

### 9.3 全指标总表

全部 22 组配置,每格 A/B 两轮均值,延迟单位均为毫秒。**粗体 = 该引擎该并发下的吞吐最优格**。
由 `/target/bench-results/_gen_tables.py` 直接生成。

#### 表 1:吞吐与 TPOT(逐 token 间隔)

| 引擎 | 并发 | 配置 | 吞吐 tok/s | 总吞吐 tok/s | TPOT mean | TPOT p50 | TPOT p90 | TPOT p99 | 接受率 |
|---|---|---|---|---|---|---|---|---|---|
| vLLM | 1 | N=0(关) | 22.1 | 30.0 | 44.63 | 44.59 | 44.93 | 45.36 | — |
| vLLM | 1 | N=1 | 19.1 | 26.1 | 50.57 | 50.41 | 52.94 | 55.08 | 0.871 |
| vLLM | 1 | N=3 | 29.4 | 39.8 | 32.05 | 31.45 | 36.77 | 41.87 | 0.678 |
| vLLM | 1 | N=5 | 35.7 | 48.4 | 25.85 | 25.71 | 30.41 | 34.53 | 0.533 |
| vLLM | 1 | N=8 | 41.5 | 56.3 | 21.69 | 21.43 | 26.55 | 30.16 | 0.411 |
| vLLM | 1 | N=16 | **42.2** | 57.4 | 20.82 | **20.54** | 28.13 | 33.59 | 0.270 |
| vLLM | 4 | N=0(关) | 60.9 | 82.7 | 64.22 | 64.26 | 64.99 | 65.68 | — |
| vLLM | 4 | N=1 | 66.5 | 90.7 | 57.16 | 57.03 | 60.20 | 62.52 | 0.870 |
| vLLM | 4 | N=3 | 102.8 | 139.4 | 35.70 | 35.61 | 40.03 | 44.46 | 0.678 |
| vLLM | 4 | N=5 | **103.1** | 140.1 | 35.25 | **35.19** | 41.47 | 46.71 | 0.531 |
| vLLM | 4 | N=8 | 92.5 | 125.5 | 38.91 | 38.67 | 48.17 | 54.82 | 0.410 |
| vLLM | 16 | N=0(关) | **205.9** | 279.1 | 70.72 | **70.94** | 72.86 | 75.95 | — |
| vLLM | 16 | N=1 | 158.8 | 215.6 | 90.87 | 91.23 | 95.81 | 98.69 | 0.869 |
| vLLM | 16 | N=2 | 170.7 | 230.6 | 84.60 | 85.03 | 92.22 | 97.35 | 0.776 |
| vLLM | 16 | N=3 | 178.4 | 242.9 | 80.52 | 80.80 | 90.60 | 96.66 | 0.680 |
| vLLM | 16 | N=5 | 173.5 | 235.6 | 81.13 | 81.28 | 95.11 | 105.06 | 0.534 |
| llama.cpp | 1 | DFlash off | 29.6 | 40.2 | 32.58 | 32.71 | 32.99 | 33.33 | — |
| llama.cpp | 1 | DFlash on | **62.4** | 84.9 | 13.68 | **13.63** | 18.24 | 21.48 | 0.288 |
| llama.cpp | 4 | DFlash off | 60.6 | 82.2 | 63.39 | 63.53 | 65.20 | 67.04 | — |
| llama.cpp | 4 | DFlash on | **76.1** | 103.4 | 44.22 | **43.91** | 58.06 | 66.76 | 0.255 |
| llama.cpp | 16 | DFlash off | **143.9** | 195.9 | 94.50 | **92.75** | 102.90 | 144.45 | — |
| llama.cpp | 16 | DFlash on | 88.3 | 119.8 | 152.43 | 151.41 | 204.45 | 242.35 | 0.191 |

「总吞吐」= 输出 + 输入 token 之和除以墙钟,仅供参考;跨引擎比较请用「吞吐」(纯输出)。

#### 表 2:TTFT(首 token 延迟)与 E2E(整条请求端到端)

| 引擎 | 并发 | 配置 | TTFT mean | TTFT p50 | TTFT p90 | TTFT p99 | E2E mean | E2E p50 | E2E p90 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| vLLM | 1 | N=0(关) | 228 | 216 | 265 | 311 | 14407 | 12504 | 24477 | 38899 |
| vLLM | 1 | N=1 | 358 | 357 | 405 | 452 | 16470 | 14157 | 27634 | 50967 |
| vLLM | 1 | N=3 | 367 | 360 | 413 | 490 | 10974 | 9342 | 19083 | 33307 |
| vLLM | 1 | N=5 | 361 | 359 | 391 | 459 | 8980 | 7345 | 16778 | 29760 |
| vLLM | 1 | N=8 | 361 | 359 | 405 | 438 | 7739 | 6211 | 14227 | 28531 |
| vLLM | 1 | N=16 | 384 | **382** | 421 | 481 | 7511 | 5792 | 15078 | 32811 |
| vLLM | 4 | N=0(关) | 319 | 325 | 370 | 573 | 20754 | 18020 | 35054 | 60551 |
| vLLM | 4 | N=1 | 500 | 486 | 557 | 765 | 18621 | 15542 | 31900 | 55491 |
| vLLM | 4 | N=3 | 520 | 497 | 607 | 776 | 12297 | 10342 | 21910 | 38550 |
| vLLM | 4 | N=5 | 568 | **546** | 634 | 789 | 12216 | 10197 | 21446 | 40733 |
| vLLM | 4 | N=8 | 630 | 609 | 713 | 823 | 13700 | 11140 | 24167 | 54357 |
| vLLM | 16 | N=0(关) | 451 | **332** | 518 | 1701 | 23127 | 20160 | 41127 | 63627 |
| vLLM | 16 | N=1 | 842 | 687 | 956 | 2554 | 29980 | 25836 | 51360 | 91727 |
| vLLM | 16 | N=2 | 983 | 815 | 1096 | 2643 | 28780 | 24008 | 51259 | 95505 |
| vLLM | 16 | N=3 | 1030 | 885 | 1222 | 2661 | 26803 | 22996 | 45591 | 83692 |
| vLLM | 16 | N=5 | 1204 | 1075 | 1321 | 2649 | 27820 | 22609 | 49145 | 103563 |
| llama.cpp | 1 | DFlash off | 407 | 402 | 464 | 523 | 10660 | 9251 | 17886 | 31035 |
| llama.cpp | 1 | DFlash on | 394 | **384** | 455 | 513 | 5090 | 3947 | 10006 | 18780 |
| llama.cpp | 4 | DFlash off | 750 | 726 | 849 | 1300 | 20977 | 18235 | 35458 | 66351 |
| llama.cpp | 4 | DFlash on | 1652 | **1767** | 1923 | 2117 | 16537 | 13283 | 31438 | 53472 |
| llama.cpp | 16 | DFlash off | 1410 | **1095** | 1854 | 4473 | 31700 | 26719 | 55338 | 89364 |
| llama.cpp | 16 | DFlash on | 4669 | 4756 | 5600 | 6079 | 55413 | 44606 | 99941 | 176040 |

E2E 的绝对值受输出长度支配(GSM8K 平均输出约 318 token,p90 约 541),
**跨并发比较 E2E 没有意义**(并发越高排队越久),只在同一并发档内横向比。

#### 从全分位数据能看出、只看 p50 看不出的三件事

1. **投机解码把 TPOT 的尾部显著拉长。** vLLM 并发 1:N=0 的 p99/p50 只有 1.02
   (45.36/44.59,几乎无尾),而 N=16 是 1.64(33.59/20.54)。原因是接受长度逐步波动——
   接受多的步很快、接受少的步退化成普通解码。**投机解码换来的是更低的平均延迟和更抖的延迟。**
   对 SLA 按 p99 卡的场景,这个代价要单独评估。
2. **llama.cpp 并发 16 关投机时 TPOT 尾部异常重**:p50 92.75、p90 102.90,但 p99 冲到
   **144.45**(p99/p50 = 1.56),而同档 vLLM N=0 只有 1.07(75.95/70.94)。
   这与 §9.4 里该格 4.52% 的轮间抖动是同一现象,指向调度在满载时不够稳。
3. **llama.cpp 并发 16 开 DFlash 是全表最差的一格,且差在尾部**:E2E p99 达到
   **176 秒**(vLLM 同档最差 N=5 也只有 104 秒),TTFT 连 p50 都要 4.8 秒。
   这个配置在生产里不可用,不只是吞吐低的问题。

另外注意 **vLLM 并发 16 的 TTFT mean(451)远高于 p50(332),p99 是 1701**。
这是闭环压测的爬坡效应:开测瞬间 16 条请求同时到达,头几条排队等 prefill。
稳态下的 TTFT 更接近 p50,**p99 不代表在线服务的长期表现**。

### 9.3.1 加速比

| 并发 | 引擎间(快 / 慢) | vLLM 缩放(最优配置) | llama.cpp 缩放(最优配置) | vLLM 缩放(关投机) | llama.cpp 缩放(关投机) |
|---|---|---|---|---|---|
| 1 | **llama.cpp 1.48×** | 1.00× | 1.00× | 1.00× | 1.00× |
| 4 | **vLLM 1.35×** | 2.44× | 1.22× | 2.76× | 2.05× |
| 16 | **vLLM 1.43×** | 4.88× | 2.31× | 9.32× | 4.87× |

「缩放」= 该并发的吞吐 ÷ 同引擎同口径在并发 1 的吞吐,理想线性应为 1× / 4× / 16×。

**关投机那两列是最干净的对比**(同一配置从头跑到尾,不掺配置切换):
并发 1 → 16,**vLLM 拿到 9.32×(线性的 58%),llama.cpp 只有 4.87×(线性的 30%)**。
这就是并发 16 上 1.43× 差距的全部来源——不是单点算子更快,而是连续批处理把并发吃得更满。

「最优配置」那两列的缩放数看起来小很多(4.88× / 2.31×),是因为并发 1 的基数被
投机解码抬高了(vLLM 42.2、llama.cpp 62.4)。**这两列不适合用来判断调度能力**,
只说明"从最好调到最好"能提升多少。

### 9.4 轮间稳定性

22 组配置各跑 2 轮,**中位轮差 0.71%**,18/22 在 2% 以内,16/22 在 1% 以内。

轮差最大的 5 组:

| 配置 | 轮差 | 判读 |
|---|---|---|
| vLLM c4 N=0 | 44.60% | **坏点**,已剔除并补跑(见 §9.1 †) |
| llama.cpp c16 off | 4.52% | 真实抖动(147.2 / 140.7),并发 16 无投机时调度抖动较大 |
| vLLM c1 N=3 | 3.40% | 真实抖动(28.9 / 29.9) |
| vLLM c4 N=3 | 2.33% | 真实抖动 |
| vLLM c16 N=1 | 1.70% | 正常 |

除已剔除的坏点外,**没有任何一格超过 5%**,结论不依赖于单轮数据。

---

## 10. 结果解读

### 10.1 投机解码为什么在高并发反转

投机解码的收益来自"用一次目标模型前向验证多个 token"。低并发时 GPU 严重欠载,
草稿模型的额外前向几乎是白送的;并发升高后 batch 本身就填满了 GPU,草稿前向
**直接与真实请求争抢算力**,而接受率又随并发略降,于是收益转为损失。

llama.cpp 的反转比 vLLM 剧烈得多(−39% vs −13%),TTFT 恶化 4.5 倍尤其明显,
指向 DFlash 的 16-token 块与连续批处理的调度耦合较差。

### 10.2 关掉投机后,差距的来源变了

| 并发 | vLLM N=0 | llama.cpp off | 比值 |
|---|---|---|---|
| 1 | 22.1 | 29.6 | llama.cpp 1.34× |
| 4 | 60.9 | 60.6 | **持平(1.00×)** |
| 16 | 205.9 | 143.9 | vLLM 1.43× |

并发 4 上两者**完全持平**,说明底层 GGUF 矩阵乘的效率相当(本来也是同一代 b10373 kernel)。
并发 1 上 llama.cpp 领先是单请求路径开销更低(Python 栈 vs C++ 栈);
并发 16 上 vLLM 领先是连续批处理调度更好。这是**两个不同机制**,不是同一个原因的延伸。

### 10.3 与"同事测出 227 tok/s"的差异,全部来自口径

我们在同一台机器上复现了同事的合成 harness,结论是**两边的数都对,只是问的问题不同**。
拆解如下(并发 16):

| 环节 | 数值 | 倍数 |
|---|---|---|
| 我们的口径,19.7GB 模型,真实 GSM8K | 143.9 tok/s | 基准 |
| 换成合成 harness(256 token 定长、一次性发射、greedy) | 214.2 tok/s | **×1.49** |
| 再换成 17GB 的 Q4_K_M 模型 | 243.6 tok/s | **×1.14** |
| 同事报告值(同为 17GB / `-np 16` / `CTX_PER_SLOT=65536` / DFlash off) | 227.5 tok/s | — |

1.49 × 1.14 = 1.70,与实测 243.6/143.9 = 1.69 吻合——**两个乘数完整解释了我们自己
两个口径之间的全部差距**。

我们复现出的 243.6 比同事报告的 227.5 **高 7%**,这 7% 未被上面的拆解解释。
考虑到同事自己的两次记录也差 4%(227.49 vs Study2 的 218.56),该 harness 在
不同容器/批次下本身就有 ±5~7% 的散布,我们的值落在其上沿。**结论(差距来自口径而非引擎)
不受这 7% 影响,但不应把 243.6 与 227.5 说成"吻合"。**

两个来源各自的含义:

- **模型 1.14×**:19.7GB 的 Dynamic Q4_K_XL 有 83% 的字节是 Q5_K/Q6_K,比纯 Q4_K_M 重。
  这个惩罚在并发 1 是 1.11×、并发 16 是 1.14×,**基本不随并发放大**,
  说明压力仍以带宽为主,没有暴露出 Q5_K/Q6_K 在计算侧的额外劣势。
- **harness 1.49×**:合成 harness 测的是**定长定量下的稳态峰值**——所有请求同时开始、
  同时结束、长度一致,没有 ramp-up 和 drain,没有长尾请求拖尾。我们的口径测的是
  **真实变长负载的端到端**,包含爬坡、排空和长尾。前者是上界,后者是实际体验。

另外**同事表格里并发 32 那一行是排队假象**:服务端 `-np 16` 只有 16 个 slot,
发 32 路并发时有一半在排队。实测 `-np 16` 下 c=32 得 253.5 tok/s,
把 slot 真开到 32(`-np 32`)后是 **308.1 tok/s**,差 21%。

---

## 11. 复现步骤

```bash
# 主矩阵:两轮、两引擎、全档位(耗时约 17 小时)
cd /target/bench-vllm
./full200_both_engines.sh

# 单独扫 vLLM 的最优 N
/target/vllm-gguf-workspace/scan_bestn.sh

# 只跑 llama.cpp 侧(DFlash 开关 × 并发)
/target/run_matrix.sh

# 口径交叉验证(合成 harness / 换模型)
cd /target/bench-crosscheck && ./run_crosscheck.sh
```

产物:

| 文件 | 内容 |
|---|---|
| `/target/bench-results/full200-vllm.jsonl` | vLLM 32 格(16 配置 × 2 轮) |
| `/target/bench-results/full200-llamacpp.jsonl` | llama.cpp 12 格(6 配置 × 2 轮) |
| `/target/bench-results/redo-c4n0-vllm.jsonl` | 异常格补跑 2 次 |
| `/target/bench-results/crosscheck-c16.jsonl` | 口径交叉验证 15 组 |
| `/target/bench-results/_agg_for_doc.py` | 逐格汇总与轮间稳定性检查(纯文本输出) |
| `/target/bench-results/_gen_tables.py` | **直接生成 §1 和 §9.3 的 markdown 表格**,数据变了重跑即可 |

`BEST_N` 的取值(`full200_both_engines.sh` 里的 `BEST_N=([1]=8 [4]=3 [16]=3)`)是
**小规模扫描阶段的初值**,不是本文的结论。全 200 条跑完后的真实最优是
**并发 1 → N=16、并发 4 → N=5、并发 16 → N=0**。

> 尤其注意并发 16:40 条的小规模扫描曾给出 N=3 最优,原因是那批 N=0 测到 115.9 ms
> (比 200 条口径高 64%,本身就是坏点)。**200 条口径下 N=0 明确最优,N=3 反而差 13.4%。**
> 这是小样本扫描误导的实例,新机器上重扫时要留意。

---

## 12. 已知未做 / 局限

- **RDNA4 门禁未放开**,无硬件可验。
- **`dispatch_J` 未扩到 128**。J 决定 token 维分块数;本负载解码阶段每步最多 64 token,
  J=64 时分块数已是 1,不可能更少,**J > 64 一次都不会被选中**。仅长上下文预填充可能受益,
  且需同步改 `dispatch_J` 与 `MMQ_B10373_J_MAX` 两处,否则会越界。
- **并发只测了 1 / 4 / 16**。并发 4 是两个引擎的交叉点附近,2 和 8 没测,
  交叉点的精确位置未知。
- **vLLM 侧缺 prefill 与投机细节的服务端指标**,只能用客户端量做跨引擎比较(见 §8)。
- **冷启动代价未计入**。vLLM 首次起服务因 torch.compile 需约 **14 分钟**,缓存热后约 **7 分钟**;
  llama.cpp 没有 torch.compile 阶段,smoke 配置(`-c 8192 -np 1`)下模型加载仅 **8 秒**
  (压测用的大 `-c` 配置会更久,未单独计时)。压测只测稳态,长期在线服务这项可忽略,
  但**频繁重启或弹性扩缩的场景,两者差距是数量级的**。
