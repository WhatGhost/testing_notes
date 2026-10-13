# MinerU 4.0.4 在 AMD GPU 上的部署与验证报告

验证日期：2026-09-20。本文合并结论、测试证据和复现说明，作为唯一的结果报告。实现细节使用可展开的摘要呈现，不依赖本机目录或脚本链接；实际脚本和原始测试证据继续保留。

MinerU 用于将 PDF、扫描件等文档解析为 Markdown 和结构化数据，提取正文、表格、公式及图片。本次关注 **MinerU 4.0.4 在 AMD gfx1201 / ROCm 7.2.3 上能否真实完成解析**，不是只检查安装或显卡识别。

## 1. 核心结论

**可以运行，但不能笼统地说所有默认后端都已通过。**

| 路线 | 小模型 / VLM | 实际结果 |
| --- | --- | --- |
| A：CPU 基线 | ONNX CPU / llama.cpp CPU | Standard 文本页、纯扫描页均通过；这不是 GPU 验证 |
| B：ROCm vLLM | ONNX CPU / vLLM GPU | Standard 两份样本均通过 |
| A-Vulkan | ONNX CPU / llama.cpp Vulkan GPU | 两份样本均通过；VLM 25/25 层及视觉编码器使用 GPU |
| Torch Standard | Torch GPU / vLLM GPU | 默认设置和显式 OCR FP32 均通过；本组样本未调用小模型 OCR 识别器 |
| Torch Basic 默认设置 | Torch GPU / 不使用 VLM | **未通过**：文本页通过，扫描页正文和表格文字缺失 |
| Torch Basic，OCR FP32 | OCR 识别改用 GPU FP32 / 不使用 VLM | **两份样本均通过**；这是显式规避方案，不是默认配置成功 |

建议：

- **保守选择**：ONNX CPU 小模型 + ROCm vLLM GPU，避开本次发现的默认 Torch OCR 问题。
- **需要 llama.cpp GPU**：Vulkan 功能已通过，但仍有 RADV testing-only 警告，需另做生产稳定性验证。
- **需要 Torch 小模型**：Standard 已通过；Basic 使用已验证的 GPU FP32 OCR 规避方式，并补充真实文档回归。
- **不能宣称全流程 GPU 化**：ONNX 路线的小模型在 CPU 上；Torch Basic 的表格分类/结构组件仍包含 ONNX CPU 路径。

## 2. 样本与实测结果

两份人工构造的单页 PDF 内容相同，包含正文、库存表格和能量公式。文本 PDF 含 314 个可提取文字字符，扫描 PDF 没有原生文字层；未使用业务或私有文档。

| 配置 | 文本 PDF | 纯图片扫描 PDF |
| --- | ---: | ---: |
| A：ONNX + llama.cpp CPU | 5.731 秒，通过 | 28.710 秒，通过 |
| B：ONNX + vLLM GPU | 6.244 秒，通过 | 3.564 秒，通过 |
| A-Vulkan：ONNX + llama.cpp GPU | 4.051 秒，通过 | 2.527 秒，通过 |
| Torch Basic 默认设置 | 46.474 秒，通过 | 7.948 秒，**内容失败** |
| Torch Basic，OCR FP32 | 23.563 秒，通过 | 3.789 秒，通过 |
| Torch Standard + vLLM，默认设置 | 13.355 秒，通过 | 3.976 秒，通过 |
| Torch Standard + vLLM，OCR FP32 | 11.186 秒，通过 | 4.446 秒，通过 |

**这些是功能烟测计时，不是性能排名。** 时间包含解析及保存结果，不包含模型下载和独立 VLM 服务启动。首次初始化、内核缓存、VLM 缓存、精度和档位均存在差异，不能据此断言 FP32 比 FP16 更快或推算生产吞吐。

<details>
<summary>展开：正确性检查与输出摘要</summary>

- 检查单页结构、标题、参考答案 42、Widget/Cable/Total、表格值 45.50、能量公式以及非空结构化输出。
- 保存 Markdown、中间 JSON、结构化内容、模型输出及导出的图片素材；A/B 基线的扫描页 Markdown 完全一致。
- 公式比较对空白与等价的单个上标花括号做规范化，不把等价 LaTeX 排版当作识别错误。
- 默认 Torch Basic 扫描页的标题、Widget、Total、42 均失败，只留下缺字/空表格与公式；原结果没有被 FP32 成功结果覆盖。
- 七种配置、每种两份样本，共 **14 个解析结果：13 个通过、1 个已知失败**。不能把“审计通过”解释为所有配置通过。

</details>

## 3. 关键发现与使用边界

### 3.1 Vulkan 确实使用了 AMD GPU

真实设备为离散 AMD `RADV GFX1201`，不是软件 Vulkan。服务日志确认 **25/25 层 offload 到 GPU**，并记录 **`CLIP using Vulkan0 backend`**，说明视觉编码器也使用了 GPU。

驱动仅作为隔离的用户态组件下载、校验和解包，没有安装系统包或升级内核。但必须保留原始警告：

> radv is not a conformant Vulkan implementation, testing use only.

**本次证明的是功能可运行，不是 Vulkan 一致性认证或生产稳定性。**

<details>
<summary>展开：GPU 身份、运行配置与显存证据</summary>

- 本机测试卡：PCI `0000:03:00.0`，对应 **HIP 2 / Vulkan 0 / AMD-SMI 0**。不同工具编号不通用，换机或重新枚举后必须重查。
- llama.cpp CPU 基线：GPU layers=0，视觉投影器不上 GPU，device=none，8 个 CPU 线程。
- Vulkan：显式选择 Vulkan0、99 个 GPU layers、视觉投影器 offload、单设备；模型实际只有 25 层，日志确认全部 offload。
- vLLM：BF16、上下文 8192、最多 4 个序列、35% 显存预算、eager、`TRITON_ATTN` 解码注意力、`TORCH_SDPA` 视觉注意力，未启用 AITER。模型加载占用 2.16 GiB，并建立 GPU KV cache、完成真实多模态请求。
- Vulkan 主模型 buffer 500.79 MiB、KV 96 MiB、主计算 40.02 MiB、视觉计算 267.08 MiB；解析后整卡采样 1689 MB。
- vLLM 运行时整卡采样 10483 MB，包含 KV cache，不是权重大小。Torch Standard 测试的此项采样发生在前端退出后，不是并行峰值。
- Basic FP32 前端的 Torch 峰值 allocated 为 819114496 字节，约 0.763 GiB；不含 ORT、驱动或其他进程。
- **以上整卡采样都不是全程峰值**；测试卡空闲基线及本轮结束时均为 58 MB。

</details>

### 3.2 Torch 默认 FP16 OCR 存在质量问题

实际 GPU 前向完成并不代表输出正确。对同一扫描页标题裁剪，分别创建模型实例进行对照：

| 设备 / OCR 识别精度 | 识别文字 | 置信度 | 输出非有限值 |
| --- | --- | ---: | ---: |
| AMD GPU FP16 | `Ω` | 0.407227 | 0 |
| AMD GPU FP32 | `MinerU Parsing Smoke Test` | 0.999584 | 0 |
| CPU FP32 | `MinerU Parsing Smoke Test` | 0.999584 | 0 |

仅将 OCR 识别改为 GPU FP32 后，完整 Basic 的文本页和扫描页均通过；公式仍为 FP16，OCR 没有退回 CPU。

**尚未定位底层算子或库层根因。** 模型浮点状态和观测输出均无 NaN/Inf，不能误写成“已证实 FP16 溢出”，也不能推广为所有 AMD GPU 或所有 MinerU 4.x 都有此问题。

### 3.3 Standard 成功不能替代 Basic 验证

这组 Standard 样本使用了 VLM 识别相关内容，本地 OCR 识别器虽然已加载，但 **前向调用次数为 0**。因此它的成功不能否定默认 Basic 扫描页的失败。

<details>
<summary>展开：Torch 各模块实际完成的 GPU 前向</summary>

| 模块 | Basic 默认 | Basic OCR FP32 | 两次 Standard 各自 |
| --- | --- | --- | --- |
| 布局 | GPU FP32，2 次 | GPU FP32，2 次 | GPU FP32，2 次 |
| OCR 检测器 | GPU FP32，11 次 | GPU FP32，11 次 | GPU FP32，10 次 |
| OCR 识别器 | GPU FP16，4 次 | GPU FP32，3 次 | 已加载，但 0 次前向 |
| 公式识别 | GPU FP16，2 次 | GPU FP16，2 次 | 未调用本地公式小模型 |

观察钩子记录参数、实际输入/输出的设备与精度、开始/完成次数，不改变模型计算。设备名 `cuda:0` 在这里表示 AMD ROCm；构建有 HIP 标识，不是 NVIDIA CUDA。

Basic 表格分类的 ORT 会话使用 `CPUExecutionProvider`，表格结构组件也采用 ONNX，不能称为纯 Torch/GPU 流水线。

</details>

## 4. 环境与模型

测试环境为 Ubuntu 24.04.4、Python 3.12.3、ROCm 7.2.3；机器有 8 张 gfx1201 显卡，单卡约 31.86 GiB，本次只选一张空闲卡。A 与 B 使用独立环境，Vulkan 复用 A，Torch 小模型复用 B；未替换共享 ROCm/PyTorch。

<details>
<summary>展开：精确依赖版本及安装约束</summary>

| 组件 | A / A-Vulkan | B / Torch 补测 |
| --- | --- | --- |
| MinerU | 4.0.4 | 4.0.4，full extra |
| docvortex | 0.4.17 | 0.4.17 |
| mineru-vl-utils | 2.0.5 | 2.0.5 |
| ONNX Runtime | 1.30.0，CPUExecutionProvider | 1.30.0，CPUExecutionProvider |
| mineru-llama-cpp | 0.1.2 | 0.1.2，B 未使用此引擎 |
| vLLM | 未安装 | 0.28.0+rocm723 |
| PyTorch | 未安装 | 2.12.0+git6bbd260，HIP 7.2.53211 |
| Torchvision | 未安装 | 0.27.1+df56172 |
| Torchaudio | 未安装 | 2.11.0+34c52a6 |
| Triton | 未安装 | 3.7.1+gitf0b55c07 |
| Transformers | 未安装 | 5.17.0 |
| NumPy | 2.5.3 | 2.3.5 |

- 两个隔离环境的依赖一致性检查通过，完整依赖快照已保留。安装逻辑优先使用快照，不无约束升级。
- B 使用 [vLLM 0.28.0 / ROCm 7.2.3 专属 wheel 索引](https://wheels.vllm.ai/rocm/0.28.0/rocm723)及配套 Torch 栈，当前 wheel 需要 glibc ≥2.39。
- MinerU 4.0.4 约束 `vllm>=0.19.1,<0.29.0`、`transformers>=5.10.1,<6`。实测时浮动 ROCm 索引提供的 vLLM 0.29 超出约束，不能直接替换。
- 实际安装固定 PyPI 4.0.4；官方源码副本提交为 `25b16c40d18f9cf3d61a6b2e56148bfefe659dc7`，保持干净，未打源文件兼容补丁。
- 原共享环境仍是 Torch `2.10.0+rocm7.2.3.lw.git1a270074`、Torchvision `0.25.0+rocm7.2.3.git82df5f59`、Torchaudio `2.10.0+rocm7.2.3.git5047768f`、Triton `3.6.0+rocm7.2.3.git4ed88892`。

</details>

<details>
<summary>展开：模型来源、固定修订与下载校验</summary>

| 用途 | 上游模型仓库 | 本次修订 | 约体积 |
| --- | --- | --- | ---: |
| ONNX 小模型 | opendatalab/MinerU-4_models_onnx | `358310b4f64b95f9fefc372ad899356e4111f376` | 0.86 GB |
| llama.cpp 主模型 + 视觉投影器 | jinzhenj/MinerU2.5-Pro-2605-1.2B-GGUF | `9185688a0495e1577d521a757c7c0b62dd38ca48` | 1.24 GB |
| vLLM BF16 模型 | opendatalab/MinerU2.5-Pro-2605-1.2B | `bff20d4ae2bf202df9f45284b4d43681555a97ed` | 2.33 GB |
| Torch 小模型 | opendatalab/MinerU-4_models_torch | `2b3afb86f4d23fa623f6b2f4b2279ed4ef93a89d` | 0.89 GB |

- 模型由 MinerU 4.0.4 注册表指定，不替换为通用聊天模型。GGUF 为 Q8_0，vLLM 模型为 BF16。
- 官方模型完整性检查通过，各文件 SHA-256 与下载元数据修订已记录；Torch 下载资源合计 894881298 字节。
- 下载使用 `mineru-kit models download`，指定档位、小模型后端、VLM 引擎和 Hugging Face 来源；随后执行 `mineru-kit models verify`。
- 模型下载与推理隔离。推理显式设置 `MINERU_MODEL_SOURCE=local` 和 `HF_HUB_OFFLINE=1`；下载阶段不能带着离线限制，`download --source local` 也不是有效下载选项。
- 默认模型分支未来可能更新；严格复现应使用上述 commit 和记录的哈希，而不是重新下载浮动最新版。

</details>

## 5. ONNX CPU + 两种 GPU 引擎：启动、测试与结论

这里给出两条路线可直接展开阅读的命令和测试代码，不再只列脚本职责，也不要求跳转到本机脚本。**这两条路线的小模型都固定为 ONNX CPU，不是 Torch Standard。**

执行顺序：两个终端都先完成 **5.0 公共配置** → 终端 1 执行 **5.1 vLLM** 或 **5.2 llama.cpp Vulkan** 的启动块 → 服务就绪后，终端 2 执行对应测试配置，再运行 **5.3 完整测试代码** → 按 5.4 关闭服务。一次只运行一条 GPU 路线。

以下启动参数来自已实测脚本；路径改由变量传入，测试代码内嵌在本文中，并将新结果另存。第 2 节计时仍是原始实测记录，不是本次文档整理时重新测得的性能数据。

### 5.0 公共准备：两个终端各执行一次

前提是第 4 节对应的隔离环境和固定修订模型已准备完成。先在两个终端分别设置以下三个变量；**同一条路线的两个终端使用相同值，换路线或复跑时选新的运行目录**。

| 变量 | 应填写的内容 |
| --- | --- |
| `MINERU_VENV` | vLLM 路线填写 B 虚拟环境根目录；Vulkan 路线填写 A 虚拟环境根目录 |
| `MINERU_MODEL_BASE_DIR` | 已下载模型的公共根目录，包含官方模型管理器创建的各模型子目录 |
| `MINERU_RUN_DIR` | 本次新运行的独立工作目录，用于状态、缓存、日志、合成样本和输出 |

这三个变量由读者的实际部署决定，文档不硬编码任何机器的安装目录。环境变量不会自动从终端 1 传到终端 2。

<details>
<summary>展开：公共环境配置完整命令</summary>

```bash
set -euo pipefail
: "${MINERU_VENV:?请先设置本路线的虚拟环境根目录}"
: "${MINERU_MODEL_BASE_DIR:?请先设置已下载模型的公共根目录}"
: "${MINERU_RUN_DIR:?请先设置本次新运行目录}"
source "$MINERU_VENV/bin/activate"
export MINERU_MODEL_BASE_DIR MINERU_RUN_DIR
export PYTHONNOUSERSITE=1
export MINERU_HOME="$MINERU_RUN_DIR/mineru-state"
export MINERU_MODEL_SMALL_BACKEND=onnx
export MINERU_MODEL_SOURCE=local
export MINERU_MODEL_VLM_MAX_CONCURRENCY=1
export MINERU_MODEL_VLM_SERVER_URL=""
export MINERU_DEVICE_MODE=cpu
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export GRADIO_ANALYTICS_ENABLED=False
export HF_HOME="$MINERU_RUN_DIR/huggingface-cache"
export MODELSCOPE_CACHE="$MINERU_RUN_DIR/modelscope-cache"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MINERU_INTRA_OP_NUM_THREADS=8 MINERU_INTER_OP_NUM_THREADS=1
mkdir -p "$MINERU_HOME" "$HF_HOME" "$MODELSCOPE_CACHE"
python -c 'import importlib.metadata as m; assert m.version("mineru") == "4.0.4"; print("MinerU", m.version("mineru"), "ONNX Runtime", m.version("onnxruntime"))'
```

下载缺失模型时不要直接带着上述离线配置执行下载；模型准备与固定修订要求见第 4 节。不要向共享 Python 安装依赖，也不要跳过 GPU 占用预检。

</details>

### 5.1 ONNX CPU + vLLM GPU（B 路线）

**结论：Standard 文本页和扫描页均通过。** 小模型由 ONNX Runtime 在 CPU 上运行，VLM 由 ROCm vLLM 在单张 gfx1201 上运行；不是 CPU VLM，也不是 Torch 小模型路线。

| 项目 | 本次实测 |
| --- | --- |
| 环境 / 模型 | B 环境；vLLM 0.28.0+rocm723；MinerU VLM BF16 |
| 文本页 / 扫描页 | **6.244 秒 / 3.564 秒**，均通过正文、表格、公式及结构化输出检查 |
| GPU 证据 | HIP 设备身份、2.16 GiB 模型加载、GPU KV cache、真实多模态请求 |
| 整卡显存采样 | 58 → 10483 → 58 MB；10483 MB 不是全程峰值 |
| 使用边界 | 仅本次两份单页烟测；不代表整条流水线均在 GPU 或生产吞吐验收 |

<details>
<summary>展开：终端 1——vLLM GPU 预检与完整启动命令</summary>

先执行 5.0。下面保留本机已核验的默认映射 HIP 2 / PCI `0000:03:00.0`；其他机器必须先按实际 PCI 关系调整 `MINERUTEST_HIP_DEVICE` 与 `MINERUTEST_PCI_BDF`，不能把 AMD-SMI 编号直接填入 HIP 编号。

```bash
export HIP_VISIBLE_DEVICES="${MINERUTEST_HIP_DEVICE:-2}"
export MINERUTEST_PCI_BDF="${MINERUTEST_PCI_BDF:-0000:03:00.0}"
export MINERU_DEVICE_MODE=cuda
export MINERU_MODEL_VLM_ENGINE=vllm
export VLLM_NO_USAGE_STATS=1 VLLM_ROCM_USE_AITER=0
export VLLM_CACHE_ROOT="$MINERU_RUN_DIR/vllm-cache"
export TRITON_CACHE_DIR="$MINERU_RUN_DIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$MINERU_RUN_DIR/torchinductor-cache"
export TORCH_EXTENSIONS_DIR="$MINERU_RUN_DIR/torch-extensions"
export OMP_NUM_THREADS=1

python - <<'PY'
import importlib.metadata as metadata
import json
import os
import subprocess
import torch

assert metadata.version("vllm") == "0.28.0+rocm723"
assert torch.version.hip and not torch.version.cuda, "必须使用 ROCm PyTorch"
assert torch.cuda.device_count() == 1, "必须只暴露一张卡"
device = torch.cuda.get_device_properties(0)
bdf = f"{device.pci_domain_id:04x}:{device.pci_bus_id:02x}:{device.pci_device_id:02x}.0"
assert bdf == os.environ["MINERUTEST_PCI_BDF"], (bdf, "请重新核对设备映射")
gpu = json.loads(subprocess.check_output([
	"amd-smi", "metric", "--gpu", bdf, "--mem-usage", "--usage", "--json",
], text=True))["gpu_data"][0]
used = gpu["mem_usage"]["used_vram"]
print(json.dumps({"device": str(device), "pci_bdf": bdf, "used_vram": used}, indent=2))
assert used["unit"] == "MB" and used["value"] <= 1024, "显卡已被占用，不启动服务"
PY

mineru-kit models verify --tier standard --small-backend onnx --vlm-engine vllm
mineru-kit vlm-server --engine vllm \
	--host 127.0.0.1 --port 30081 \
	--served-model-name mineru-vlm --allowed-origins '[]' \
	--dtype bfloat16 --max-model-len 8192 --max-num-seqs 4 \
	--max-num-batched-tokens 8192 --gpu-memory-utilization 0.35 \
	--limit-mm-per-prompt '{"image":1,"video":0}' \
	--attention-backend TRITON_ATTN --mm-encoder-attn-backend TORCH_SDPA \
	--enforce-eager 2>&1 | tee -a "$MINERU_RUN_DIR/vllm-server.log"
```

`MINERU_DEVICE_MODE=cuda` 在这里供 ROCm/VLM 使用；`MINERU_MODEL_SMALL_BACKEND=onnx` 仍将小模型固定在 ONNX CPU，不能据此误认为小模型也上了 GPU。

**就绪判定**：等待 `Application startup complete`。同时核对日志中的 `TRITON_ATTN`、视觉 `TORCH_SDPA`、GPU 模型与 KV cache 初始化；仅进程存活或模型开始加载不算就绪。MinerU 官方包装入口会接入其自定义 logits processor，不直接换成未经验证的通用聊天服务。

</details>

<details>
<summary>展开：终端 2——vLLM 路线测试配置与验收</summary>

在 B 环境先执行 5.0，再执行：

```bash
export HIP_VISIBLE_DEVICES="${MINERUTEST_HIP_DEVICE:-2}"
export MINERU_DEVICE_MODE=cuda
export MINERU_MODEL_VLM_ENGINE=vllm
export VLM_ENGINE=vllm
export VLM_URL=http://127.0.0.1:30081
curl --fail --silent --show-error "$VLM_URL/v1/models"
```

然后在**同一终端直接执行 5.3 的完整测试代码**，无需寻找额外测试脚本。模型接口应返回 `mineru-vlm`；测试会新建文本/纯扫描样本，调用 Standard 解析并保存结果和断言摘要。

**验收**：两份样本均 `passed=true`，输出包含标题、答案 42、Widget/Cable/Total、45.50、能量公式和有效结构；终端 1 有真实多模态请求及 GPU 后端日志。HTTP 200 或生成了文件本身不代表内容通过。

</details>

### 5.2 ONNX CPU + llama.cpp GPU（A-Vulkan 路线）

**结论：Standard 文本页和扫描页均通过，GPU 后端是 Vulkan。** ONNX 小模型仍在 CPU，llama.cpp 的 VLM 和视觉编码器在 AMD GPU；不要与 A 的 llama.cpp CPU 基线混淆。

| 项目 | 本次实测 |
| --- | --- |
| 环境 / 模型 | A 环境，无 Torch；mineru-llama-cpp 0.1.2；GGUF Q8_0 + 视觉投影器 |
| 文本页 / 扫描页 | **4.051 秒 / 2.527 秒**，两份内容与结构检查均通过 |
| GPU 证据 | 真正的 AMD Vulkan0；`offloaded 25/25 layers to GPU`；`CLIP using Vulkan0 backend` |
| 整卡显存采样 | 解析后 1689 MB，清理后 58 MB；不是全程峰值 |
| 使用边界 | RADV 仍警告 `not conformant / testing use only`；不是标准认证或生产稳定性保证 |

<details>
<summary>展开：终端 1——Vulkan GPU 预检与完整启动命令</summary>

先执行 5.0，并设置 `MINERU_VULKAN_ROOT` 为**已解包的 Vulkan 用户态组件根目录**。这不是要求将驱动安装到系统；包版本和校验方式见下方“Vulkan 用户态驱动准备”折叠区。

下面固定选择已经实测的 Vulkan 0。若本机枚举或 PCI 地址不同，应先重新核对，而不是删除断言或跳过设备检查。

```bash
: "${MINERU_VULKAN_ROOT:?请先设置已解包的 Vulkan 用户态组件根目录}"
export MINERU_VULKAN_ROOT
export MINERUTEST_PCI_BDF="${MINERUTEST_PCI_BDF:-0000:03:00.0}"
export MINERU_DEVICE_MODE=cpu
export MINERU_MODEL_VLM_ENGINE=llama-cpp
export VK_DRIVER_FILES="$MINERU_VULKAN_ROOT/usr/share/vulkan/icd.d/radeon_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export LD_LIBRARY_PATH="$MINERU_VULKAN_ROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_VK_VISIBLE_DEVICES=0
export XDG_RUNTIME_DIR="$MINERU_RUN_DIR/xdg-runtime"
export MESA_SHADER_CACHE_DIR="$MINERU_RUN_DIR/mesa-shader-cache"
unset DISPLAY WAYLAND_DISPLAY
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

root = Path(os.environ["MINERU_VULKAN_ROOT"])
profile = Path(os.environ["MINERU_RUN_DIR"]) / "vulkan-device.json"
subprocess.run([
	str(root / "usr/bin/vulkaninfo"), "--json=0", "-o", str(profile),
], check=True, stdout=subprocess.DEVNULL)
props = json.loads(profile.read_text())["capabilities"]["device"]["properties"]
basic = props["VkPhysicalDeviceProperties"]
pci = props["VkPhysicalDevicePCIBusInfoPropertiesEXT"]
bdf = f"{pci['pciDomain']:04x}:{pci['pciBus']:02x}:{pci['pciDevice']:02x}.{pci['pciFunction']}"
assert basic["vendorID"] == 0x1002
assert basic["deviceType"] == "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
assert bdf == os.environ["MINERUTEST_PCI_BDF"], (bdf, "请重新核对 Vulkan 枚举")
spec = importlib.util.find_spec("mineru_llama_cpp")
assert spec and spec.origin, "当前环境缺少 mineru-llama-cpp"
binary = Path(spec.origin).parent / "bin/llama-server"
devices = subprocess.check_output([str(binary), "--list-devices"], text=True, stderr=subprocess.STDOUT)
print(devices)
assert re.findall(r"^\s+(Vulkan\d+):", devices, re.MULTILINE) == ["Vulkan0"]
gpu = json.loads(subprocess.check_output([
	"amd-smi", "metric", "--gpu", bdf, "--mem-usage", "--usage", "--json",
], text=True))["gpu_data"][0]
used = gpu["mem_usage"]["used_vram"]
print(json.dumps({"device": basic["deviceName"], "pci_bdf": bdf, "used_vram": used}, indent=2))
assert used["unit"] == "MB" and used["value"] <= 1024, "显卡已被占用，不启动服务"
PY

mineru-kit models verify --tier standard --small-backend onnx --vlm-engine llama-cpp
mineru-kit vlm-server --engine llama-cpp \
	--host 127.0.0.1 --port 30082 \
	--n-gpu-layers 99 --mmproj-offload --device Vulkan0 --split-mode none \
	--parallel 1 --ctx-size 8192 --threads 8 --threads-batch 8 --verbosity 4 \
	2>&1 | tee -a "$MINERU_RUN_DIR/llama-vulkan-server.log"
```

**就绪判定**：等待 30082 开始监听，并确认日志同时出现 **25/25 GPU offload** 和 **CLIP Vulkan0**。`MINERU_DEVICE_MODE=cpu` 在这条路线用于 CPU 前端；VLM 是否上 GPU 由 llama.cpp 的 Vulkan 设备和 offload 参数决定，不由该变量决定。

</details>

<details>
<summary>展开：终端 2——llama.cpp Vulkan 路线测试配置与验收</summary>

在 A 环境先执行 5.0，再执行；前端只连接本机 HTTP 服务，不需要在终端 2 加载 Vulkan 驱动，也不需要 Torch：

```bash
export MINERU_DEVICE_MODE=cpu
export MINERU_MODEL_VLM_ENGINE=llama-cpp
export VLM_ENGINE=llama-cpp
export VLM_URL=http://127.0.0.1:30082
curl --fail --silent --show-error "$VLM_URL/v1/models"
```

然后在**同一终端直接执行 5.3 的完整测试代码**。模型接口应返回 MinerU GGUF 模型；输入与 vLLM 路线相同，仍采用 Standard。

**验收**：两份样本均 `passed=true`，正文、表格、公式和结构化内容通过；终端 1 的真实设备、25/25 offload 与 CLIP Vulkan0 三类证据必须同时成立。仅 OCR 成功或 llama.cpp 进程启动不能排除 CPU 回退；RADV testing-only 警告必须保留。

</details>

### 5.3 两条路线共用的完整测试代码

此代码直接粘贴到完成上述配置的**终端 2**。它使用当前环境已有的 ReportLab 和 pypdfium2，按原实测样本的相同内容与布局生成两份 PDF，检查文字层，再解析、保存和验收。测试模式固定为 `standard`，`small_backend` 必须为 `onnx`。

每次运行创建唯一结果目录，不覆盖历史证据。与原流程相比，这里把事后表格值和公式审计也并入同一次测试；不能用本文中的新输出目录覆盖原基线摘要。

<details>
<summary>展开：完整测试代码——生成样本、Standard 解析、内容断言与结果保存</summary>

```bash
python -u - <<'PY'
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4

import onnxruntime as ort
import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

from mineru.config import VlmConfig, config
from mineru.model.vlm.runtime import shutdown_cached_models
from mineru.parser import parse
from mineru.parser.writer import FileBasedDataWriter


def make_samples(directory):
	directory.mkdir(parents=True, exist_ok=False)
	native = directory / "native.pdf"
	pdf = canvas.Canvas(str(native), pagesize=A4, invariant=1)
	pdf.setTitle("MinerU reproducible parsing smoke test")
	pdf.setFont("Helvetica-Bold", 20)
	pdf.drawString(48, 786, "MinerU Parsing Smoke Test")
	pdf.setFont("Helvetica", 12)
	for y, text in (
		(752, "This public synthetic document contains text, a table and a formula."),
		(732, "The reference answer is 42. No private or business data is included."),
		(700, "Inventory table"),
	):
		pdf.drawString(48, y, text)
	table = Table(
		[["Item", "Quantity", "Price"], ["Widget", "3", "12.50"],
		 ["Cable", "2", "4.00"], ["Total", "5", "45.50"]],
		colWidths=[220, 100, 120], rowHeights=32,
	)
	table.setStyle(TableStyle([
		("GRID", (0, 0), (-1, -1), 0.6, colors.black),
		("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
		("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
		("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
		("FONTSIZE", (0, 0), (-1, -1), 12),
		("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
	]))
	table.wrapOn(pdf, 440, 128)
	table.drawOn(pdf, 48, 548)
	pdf.setFont("Helvetica", 12)
	pdf.drawString(48, 506, "Energy formula")
	pdf.setFont("Times-Italic", 22)
	pdf.drawString(214, 466, "E = mc")
	pdf.setFont("Times-Roman", 13)
	pdf.drawString(282, 476, "2")
	pdf.setFont("Helvetica", 11)
	pdf.drawString(48, 416, "End of the one-page validation sample.")
	pdf.showPage()
	pdf.save()
	image_path = directory / "scan.png"
	with pdfium.PdfDocument(str(native)) as document:
		document[0].render(scale=2).to_pil().save(image_path)
	scanned = directory / "scanned.pdf"
	scan = canvas.Canvas(str(scanned), pagesize=A4, invariant=1)
	scan.drawImage(str(image_path), 0, 0, width=A4[0], height=A4[1])
	scan.showPage()
	scan.save()
	for path, expected in ((native, 314), (scanned, 0)):
		with pdfium.PdfDocument(str(path)) as document:
			characters = len(document[0].get_textpage().get_text_range())
		assert characters == expected, (path.name, characters)
		print(f"{path.name}: native_text_characters={characters}", flush=True)
	return {"native": native, "scanned": scanned}


engine = os.environ["VLM_ENGINE"]
url = os.environ["VLM_URL"].rstrip("/")
assert engine in {"vllm", "llama-cpp"}
assert urlsplit(url).hostname in {"127.0.0.1", "localhost", "::1"}, "仅允许本机服务"
assert metadata.version("mineru") == "4.0.4"
assert config.model.small_backend == "onnx", "必须显式选择 ONNX 小模型"
providers = ort.get_available_providers()
assert "CPUExecutionProvider" in providers
with urlopen(url + "/v1/models", timeout=15) as response:
	models = json.load(response)
print(json.dumps({"small_backend": config.model.small_backend, "engine": engine,
				  "onnx_available_providers": providers, "served_models": models}, ensure_ascii=False))

output = Path(os.environ["MINERU_RUN_DIR"]) / f"parse-{engine}-{uuid4().hex}"
output.mkdir(parents=True, exist_ok=False)
samples = make_samples(output / "inputs")
vlm = VlmConfig(engine=engine, server_url=url, max_concurrency=1, http_timeout=600)
summary = {
	"date_utc": datetime.now(timezone.utc).isoformat(),
	"mineru_version": metadata.version("mineru"),
	"small_backend": config.model.small_backend,
	"tier": "standard", "engine": engine, "server_url": url,
	"onnx_available_providers": providers, "served_models": models, "samples": [],
}
try:
	for name, source in samples.items():
		record = {"sample": name, "passed": False}
		summary["samples"].append(record)
		started = time.perf_counter()
		result = parse(source, tier="standard", ocr_mode="auto", vlm_config=vlm)
		destination = output / name
		result.save(FileBasedDataWriter(str(destination)))
		markdown = result.markdown()
		normalized = re.sub(r"\s+", "", markdown)
		structured = json.loads((destination / "structured_content.json").read_text())
		checks = {
			"one_page": len(result.pages) == 1,
			"title": "mineru" in markdown.lower(),
			"reference_answer": re.search(r"\b42\b", markdown) is not None,
			"table_values": all(value in markdown for value in ("Widget", "Cable", "Total", "45.50")),
			"energy_formula": re.search(r"E=mc\^(?:2|\{2\})", normalized) is not None,
			"nonempty_structure": bool(json.loads(result.to_json()).get("pages")),
			"structured_output": bool(structured),
		}
		record.update(seconds=round(time.perf_counter() - started, 3),
					  checks=checks, passed=all(checks.values()))
		print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
		if not record["passed"]:
			raise RuntimeError(f"Content validation failed: {name}: {checks}")
except Exception as exc:
	summary["error"] = f"{type(exc).__name__}: {exc}"
	raise
finally:
	(output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
	shutdown_cached_models()
	print(f"Results saved: {output}", flush=True)
PY
```

输出包括两份样本的 Markdown、结构化数据和本次摘要；即使内容断言失败，也保留已经产生的解析结果。`onnx_available_providers` 只是运行时可用 provider 列表，不能代替会话级设备证据；本次 ONNX CPU 结论基于固定的 CPU 版运行时和显式 ONNX 后端，VLM GPU 结论需结合服务日志。

</details>

### 5.4 停止服务与复查

在终端 1 按 Ctrl+C，只关闭本次服务。随后确认对应端口已关闭，并使用 AMD-SMI 复查测试卡显存是否回到启动前基线；本次记录为 58 MB，其他机器不要求恰好为这个值。PID 文件可能过期，不凭旧 PID 直接杀进程，也不使用全局 `pkill python`。

这两条路线分别占用 30081 和 30082。停止第一条后再启动另一条，尤其不能在预检报显卡繁忙时跳过检查。保留各次运行目录，不将新日志或输出覆盖原测试证据。

### 5.5 补充实现说明：Vulkan 准备与 Torch 对照

<details>
<summary>展开：Vulkan 用户态驱动准备与隔离方式</summary>

- 复用 `mineru-llama-cpp 0.1.2` wheel 中现成的 Vulkan 后端，无需源码编译。
- 从签名验证的 Ubuntu noble 官方源下载固定版 `mesa-vulkan-drivers=25.2.8-0ubuntu0.24.04.2` 和 `vulkan-tools=1.3.275.0+dfsg1-1`，只解包，不做系统安装。
- Mesa 包 SHA-256：`b3be471db0ec27eb28c1ea75a4d2e6bb3207c5904bdc3823dbba9ce530026939`。
- Vulkan tools 包 SHA-256：`994e429b5a4550b46c5bd9a938af1f60c4aa4890e08f661603b9518a8df824d1`。
- APT 使用隔离的索引、缓存与配置，禁止加载系统配置片段和后处理钩子；root 非沙箱下载提示不等于关闭签名校验。
- 仅为测试进程设置 `VK_DRIVER_FILES`、兼容变量 `VK_ICD_FILENAMES` 及库搜索路径；不向系统安装 ICD，不更改共享 shell 配置。无头计算时移除显示服务器变量，独立管理运行时目录和 shader cache。
- 依赖已有 Vulkan loader、LLVM 20、libdrm 等系统库，不是跨发行版自包含驱动包。依赖缺失或固定包下线时必须重新评估，不静默升级系统或换用最新版。

</details>

<details>
<summary>展开：公共 SDK 解析、设备追踪及 OCR FP32 规避</summary>

**解析逻辑**

- 使用公共 SDK `parse()`，显式设置 `tier` 和 `ocr_mode="auto"`。
- Standard 用 `VlmConfig` 指定本机引擎、服务地址、并发 1、HTTP 超时 600 秒；Basic 不创建 VLM 配置。
- 使用 `FileBasedDataWriter` 保存结果；内容断言失败时仍保存输出与失败摘要，再以非零状态退出。

**GPU 使用证据**

- 在模型工厂初始化后挂接只读观察钩子，分别追踪布局、OCR 检测、OCR 识别和公式模块。
- 记录参数设备/精度、实际输入/输出张量的设备/精度、前向开始/完成次数；不能仅凭模型已加载或显存上涨判定成功。
- 模型输出可能是嵌套字典或列表，观察逻辑递归提取张量；不修改数值、计算或设备放置。

**精度规避**

- 在第一次初始化 OCR 模型前，把进程内 `mineru.model._internal.pytorchocr.base_ocr_v20.OCR_INFERENCE_PRECISION` 设置为 `fp32`，测试结束后恢复原常量。
- 本测试 runner 的 `--ocr-precision fp32` 实现上述操作；**不是官方 `mineru-kit` 参数，也没有对应已验证的官方环境变量**。仅修改常量不会转换已经缓存的模型，应在独立测试进程中从初始化开始验证。
- 只改变 OCR 识别精度，仍使用 GPU；OCR 检测原本就是 FP32，公式保持 FP16，vLLM 保持 BF16。没有修改安装文件或上游源码。
- 精度诊断用同一标题裁剪，分别运行 GPU FP16、GPU FP32、CPU FP32，检查识别内容、置信度及状态/输出有限性；之后再对完整 Basic 两份 PDF 回归，而不是只凭单个裁剪判定问题已规避。

</details>

## 6. 审计与环境保护

本轮验证结束时：

- 原 A/B 输出、默认 FP16 失败和 FP32 规避结果分别保留；没有以成功结果覆盖失败证据。
- A/B 与共享环境记录的包版本未变，原模型文件哈希未变，官方源码副本保持干净。
- 没有系统包安装、内核/驱动升级或共享 PyTorch 替换；新增 Vulkan 包没有进入系统安装数据库。
- 30080～30083 均关闭，测试卡恢复 **58 MB**；四张已有负载的卡仍各约 **13157 MB**，未被停止。
- vLLM API 退出码为 0，但内部零超时 worker 终止和 1 个 semaphore 的 resource-tracker 警告已保留，不称为无警告关闭。没有为回收已退出 worker 重启容器或干预 PID 1。

<details>
<summary>展开：原始证据保留清单与审计含义</summary>

| 证据类别 | 保留内容 | 用途 |
| --- | --- | --- |
| 环境与模型 | 精确依赖快照、模型修订、逐文件 SHA-256、驱动包校验 | 避免未来版本漂移被当作相同测试 |
| 服务运行 | llama.cpp / vLLM 启动、注意力后端、GPU offload、多模态请求及关闭日志 | 证明实际推理引擎与设备 |
| 解析结果 | 每种配置的两份输出、内容断言、耗时、失败信息 | 区分真正通过与仅运行完成 |
| Torch 诊断 | 各模块前向设备/精度/次数、三组 OCR 精度对照 | 区分已加载、实际调用和识别正确 |
| 资源保护 | 显存采样、服务端口、共享版本、源码状态、系统包状态 | 核对隔离与清理 |

补测审计重新检查 14 个保存结果，明确将默认 Torch Basic 扫描页标记为已知失败。审计通过的含义是**证据与报告一致**，并非把该失败判为通过。原首轮审计也保留，没有被补测覆盖。

脚本和原始证据未因文档合并而删除；此处使用摘要而不是依赖本机路径，便于单独阅读和转发。

</details>

## 7. 尚未验证的范围

- 仅两份人工单页 PDF；未覆盖复杂中文、多栏、长文档、印章 OCR 或质量数据集。
- 未覆盖 Advanced、多卡、生产并发、异常恢复、长期稳定性或全程整卡峰值监控。
- Standard 验证的是官方独立 VLM 服务 + 公共 SDK 的本机 HTTP 组合，不是内嵌引擎或自动后端选择验收；Basic 不使用 VLM。
- 未定位 FP16 OCR 问题的底层根因，未验证其他 GPU/版本上的行为；不将本次 4.0.4 结论推广至全部 MinerU ≥4.0 版本。
- 面向生产前仍需真实但获授权的文档回归，并核对项目与模型许可证。旧版 AMD 专用适配文档限定 MinerU <4，不能代替本次 4.x 验证。

上游参考：[MinerU 官方仓库](https://github.com/opendatalab/MinerU)、[MinerU 4.x 快速入门](https://opendatalab.github.io/MinerU/zh/quick_start/)、[旧版 AMD 适配边界](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/AMD/)。