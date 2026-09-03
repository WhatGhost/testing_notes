# SenseNova-U1.5-8B-MoT 文生图 ROCm 部署报告（AMD Radeon PRO W7900）

在 AMD Radeon PRO W7900（gfx1100 / RDNA3）上运行 SenseNova-U1.5-8B-MoT 双分支文生图模型的完整记录，含启动配置、单卡与双卡性能实测、8 步蒸馏 LoRA 与 aiter 注意力后端加速、显存数据与复现步骤。

## 结论

文生图端到端跑通，成图内容与结构正确。1024×1024 出图：50 步基准约 **283 秒**，启用官方 8 步蒸馏 LoRA 后约 **33 秒**（提速 8.6 倍，质量相当且细节更锐利）。

再叠加可选的 aiter 注意力后端，50 步降到约 **164 秒**、8 步 LoRA 降到约 **16 秒**，出图质量无退化。它需要额外装 AMD 的 AITER 算子库并打两处补丁，属可选项；本报告其余章节的性能与显存数据均以默认的 `torch_sdpa` 为基线。详见「可选：aiter 注意力后端」。

**单卡即可运行，这是本机相对 32 GiB 级显卡的关键差异。** W7900 单卡 47.98 GiB，容得下 32.66 GiB 权重加全部运行时开销，实测峰值 42.01 GiB，余量约 6 GiB。R9700（31.86 GiB/卡）上必须分卡部署的限制在此不成立。

**单卡与双卡性能几乎相同**，稳态推理净耗时 282.2 秒对 283.6 秒，差异约 0.5%，在噪声范围内。原因是扩散阶段只有生成分支在算，理解分支完成提示词预填充后全程闲置（实测利用率 0–6%），把它挪到另一张卡并不能带来并行收益。**因此单卡是本机的推荐部署方式**，省一张卡且无性能代价。

代码改动共三处，与 gfx1201 上所需的完全一致，**其中只有一处属于 ROCm 适配**，另两处是上游在 NVIDIA 上同样会触发的缺陷：

| 改动 | 性质 | 仓库 | 文件 |
|---|---|---|---|
| 主机内存注册运行时由 libcudart 改为按平台解析 | ROCm 专属 | LightLLM | `lightllm/utils/kv_cache_utils.py` |
| torch_sdpa 补上分组查询注意力广播 | 上游缺陷，平台无关 | LightX2V | `lightx2v/common/ops/attn/torch_sdpa.py` |
| 纯文本图像请求传字符串 content | 上游缺陷，平台无关 | LightLLM | `lightllm/server/build_prompt.py` |

三处改动取自 `gyydev/rocm-gfx1201` 分支，在 gfx1100 上无需任何额外修改即可直接套用——说明这套适配是 RDNA 通用的，不绑定具体架构代次。

上表是默认 `torch_sdpa` 路径所需的全部改动。若改用可选的 aiter 后端，还需另外两处，但都落在 AMD ROCm 专属文件内，不触及平台无关代码，见「可选：aiter 注意力后端」一节。

**注意力后端的切换不算改动**：它通过一份仓库外的配置文件实现（`/target/sensenova-rocm/configs/neopp_dense_rocm.json`），两个上游仓库的配置文件保持原样。

## 环境

| 项 | 值 |
|---|---|
| GPU | 8 × AMD Radeon PRO W7900（gfx1100 / RDNA3），47.98 GiB/卡 |
| ROCm | 7.2.3 |
| PyTorch | 2.10.0+rocm7.2.3（HIP 7.2.53211） |
| Triton | 3.6.0+rocm7.2.3 |
| Python | 3.12.3 |
| transformers | 4.57.1 |
| 模型 | SenseNova/SenseNova-U1.5-8B-MoT，8 个分片共 32.66 GiB |

gfx1100 具备 WMMA 指令但没有 MFMA（那是 CDNA 独有），因此依赖 MFMA 的 kernel 不可用；本次用到的 Triton kernel 与 SDPA 路径均不涉及。

## 模型结构

混合专家式双分支（Mixture-of-Transformers），两个分支各 42 层且严格对称。本机扫描 8 个分片的张量元数据，构成如下：

| 部分 | 张量数 | 大小 |
|---|---|---|
| 理解分支 | 559 | 15.18 GiB |
| 生成分支（`_mot_gen` 后缀） | 551 | 15.13 GiB |
| 共享的词嵌入与输出投影 | 2 | 2.32 GiB |
| 视觉编码器 | 4 | 0.03 GiB |
| **合计** | **1116** | **32.66 GiB** |

理解分支由 LightLLM 承载（dense 结构，config.json 无 `num_experts` 字段，不涉及专家路由），加上共享部分需常驻 **17.50 GiB**；生成分支由 LightX2V 承载。

理解分支处理文本提示词后，把注意力键值缓存经共享内存传给生成分支，生成分支执行扩散去噪。这条跨进程缓存传递正是需要主机内存注册的地方。

磁盘上 8 个分片大小分别为 4.63 / 4.59 / 4.64 / 4.61 / 4.59 / 4.66 / 4.59 / 0.34 GiB。

## 单卡可行性

这是 W7900 相对 32 GiB 级显卡最实质的差异，因此单列一节。

单卡预算表：

| 项 | 大小 |
|---|---|
| 权重合计 | 32.66 GiB |
| KV 缓存池（8192 词元） | 1.31 GiB |
| 运行时开销（激活、试探推理、分配器预留、HIP 上下文） | 8.04 GiB |
| **实测峰值占用** | **42.01 GiB** |
| 单卡容量 | 47.98 GiB |
| **余量** | **5.97 GiB** |

运行时开销一项是实测峰值减去权重与 KV 缓存后的差值，未逐项拆分。余量接近 6 GiB，不是勉强挤进去的：`colocate` 模式下启动无 OOM，三次连续出图全部成功。

值得注意的是**单卡占用（42.01 GiB）反而低于双卡两侧之和（44.53 GiB）**。差出的约 2.5 GiB 来自双卡多出的一套 HIP 上下文（每进程约 0.3 GiB）与独立的分配器预留。分卡部署本身有固定成本，在容量够用时它是净开销。

## 代码改动详解

三处改动彼此独立，可分别应用。后两处是与平台无关的上游缺陷修复，可单独向上游提 PR。

### 一、主机内存注册运行时解析（ROCm 专属）

`LightLLM` — `lightllm/utils/kv_cache_utils.py`

跨进程传递 KV 缓存需要注册主机内存，原代码把 CUDA 运行时库的路径写死：

```python
ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
```

纯 ROCm 系统上没有这个文件。改为按平台解析：

```python
def _load_gpu_runtime():
    """Return (lib, symbol_prefix) for the host-memory-registration API.

    CUDA and HIP agree on both the call signatures and the numeric flag values used
    below, so only the library name and the symbol prefix differ.
    """
    if torch.version.hip is not None:
        candidates = ["libamdhip64.so", "/opt/rocm/lib/libamdhip64.so"]
        prefix = "hip"
    else:
        candidates = ["libcudart.so", "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so"]
        prefix = "cuda"
    for cand in candidates:
        try:
            return ctypes.CDLL(cand), prefix
        except OSError:
            continue
    raise OSError(f"cannot load GPU runtime, tried: {candidates}")
```

调用点相应改为按前缀取符号：

```python
lib, prefix = _load_gpu_runtime()
host_register = getattr(lib, f"{prefix}HostRegister")
host_get_device_pointer = getattr(lib, f"{prefix}HostGetDevicePointer")
```

`hipHostRegister` 与 `cudaHostRegister` 的调用签名和标志位数值一致，函数名前缀不同而已，因此只需替换库与前缀。

### 二、分组查询注意力广播（上游缺陷，平台无关）

`LightX2V` — `lightx2v/common/ops/attn/torch_sdpa.py`

本模型有 32 个查询头、8 个键值头。PyTorch 的 `scaled_dot_product_attention` 在两者不等时必须显式开启广播，否则报维度不匹配。上游漏了这个参数，补上：

```python
x = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attn_mask,
    dropout_p=drop_rate,
    is_causal=causal,
    enable_gqa=q.shape[1] != k.shape[1],
)
```

这与平台无关，在 NVIDIA 上走同一后端也会失败。

### 三、纯文本图像请求传字符串 content（上游缺陷，平台无关）

`LightLLM` — `lightllm/server/build_prompt.py`

`build_prompt_from_image_request` 原本无论请求是否携带图片，都把提示词包进多模态**列表**格式。而本模型自带的聊天模板（Qwen 系）对 `message.content` 调用 `.startswith()`，列表没有该方法，Jinja 抛出 `'list object' has no attribute 'startswith'`，接口返回 HTTP 417。改为按是否携带图片分支：

```python
images = request.images if isinstance(request, ImageEditRequest) else []
if images:
    user_content = [
        {"type": "image_url", "image_url": {"url": image.image_url[:100]}} for image in images
    ]
    user_content.append({"type": "text", "text": request.prompt})
else:
    # Text-only requests must pass a plain string: chat templates derived from Qwen call
    # .startswith() on message.content, which fails on the list form with
    # "'list object' has no attribute 'startswith'".
    user_content = request.prompt
kwargs = {"conversation": [{"role": "user", "content": user_content}]}
```

脱离服务即可复现，直接对模型自带 tokenizer 调 `apply_chat_template`：

| content 形式 | 结果 |
|---|---|
| `[{"type": "text", ...}]`（上游原始写法） | `UndefinedError: 'list object' has no attribute 'startswith'` |
| `"a red apple"`（修复后写法） | 成功 |
| `[{"type": "image_url", ...}, {"type": "text", ...}]`（图生图必需） | 同样 `UndefinedError` |

第三行说明这个模板根本没有多模态输入能力，这是图生图不可用的障碍之一。

## 启动

### 单卡（推荐）

```bash
CUDA_VISIBLE_DEVICES=1 \
LIGHTLLM_NEO_PREFILL_TRITON_BACKEND=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m lightllm.server.api_server \
  --model_dir /target/models/SenseNova-U1.5-8B-MoT \
  --enable_multimodal --enable_multimodal_x2i \
  --x2i_server_deploy_mode colocate --x2i_server_used_gpus 1 \
  --tp 1 --visual_gpu_ids 0 --disable_cudagraph \
  --max_total_token_num 8192 --max_req_total_len 2048 \
  --batch_max_tokens 1024 --chunked_prefill_size 1024 \
  --x2v_gen_model_config /target/sensenova-rocm/configs/neopp_dense_rocm.json \
  --host 127.0.0.1 --port 8081
```

`--visual_gpu_ids` 取的是 `CUDA_VISIBLE_DEVICES` 列表里的**逻辑下标**而非卡号，单卡场景下该列表只有一项，故填 `0`。

启用 8 步蒸馏 LoRA 只需把 `--x2v_gen_model_config` 换成 `/target/sensenova-rocm/configs/neopp_dense_8steps_rocm.json`，其余参数不变，出图从约 283 秒降到约 33 秒（详见"启用 8 步蒸馏 LoRA"）。

### 双卡

```bash
CUDA_VISIBLE_DEVICES=1,2 \
LIGHTLLM_NEO_PREFILL_TRITON_BACKEND=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m lightllm.server.api_server \
  --model_dir /target/models/SenseNova-U1.5-8B-MoT \
  --enable_multimodal --enable_multimodal_x2i \
  --x2i_server_deploy_mode separate --x2i_server_used_gpus 1 \
  --tp 1 --visual_gpu_ids 1 --disable_cudagraph \
  --max_total_token_num 8192 --max_req_total_len 2048 \
  --batch_max_tokens 1024 --chunked_prefill_size 1024 \
  --x2v_gen_model_config /target/sensenova-rocm/configs/neopp_dense_rocm.json \
  --host 127.0.0.1 --port 8081
```

两条命令只差三处：`CUDA_VISIBLE_DEVICES`、`--x2i_server_deploy_mode`、`--visual_gpu_ids`。

`separate` 模式下理解分支取列表中前 `tp × dp` 张卡，生成分支取紧随其后的 `--x2i_server_used_gpus` 张；`--visual_gpu_ids 1` 让视觉服务与生成分支共用第二张卡。

就绪标志是日志出现 `server start up ok`，不要依赖 `ss -ltn` 判断——容器内该命令可能看不到监听端口而服务其实正常。

### 加载耗时

从进程启动到 `server start up ok`：**单卡约 3.0 分钟，双卡约 4.2 分钟。**

理解分支的权重加载走 `LOADWORKER` 控制的多进程读取，默认值为 1（日志中可见 `Loading model weights with 1 workers`）。本机实测 8 个分片合计仅 13 至 54 秒，两次运行差异较大。

**这个数字不宜横向比较，它主要反映页缓存状态而非磁盘或 GPU 能力。** 本机内存 251.78 GB，33 GB 权重在首次下载后即完全驻留页缓存，后续加载几乎不触碰磁盘——最快的一次每分片仅 1.65 秒。内存不足以缓存整个模型的机器上，同样的代码路径会慢一个数量级（gfx1201 那台实测每分片 74.8 秒，理解分支权重耗时约 10 分钟）。在那种情况下调高 `LOADWORKER` 亦无收益，实测瓶颈不在并发度。

### 相对官方默认值的全部偏离

| 项 | 官方默认 | 本次取值 | 性质 |
|---|---|---|---|
| `LIGHTLLM_NEO_PREFILL_TRITON_BACKEND` | `0`（用 fa3-neo） | `1`（用 Triton） | ROCm 必需 |
| 配置中的 `attn_type` | `flash_attn3` | `torch_sdpa` | ROCm 必需 |
| `--disable_cudagraph` | 关（启用 graph） | 开（禁用） | ROCm 必需 |
| `--batch_max_tokens` | `16384 // dp` = 16384 | 1024 | 显存调优 |
| `--chunked_prefill_size` | `batch_max_tokens // 2` = 8192 | 1024 | 显存调优 |
| `--max_total_token_num` | 自动（按 `mem_fraction=0.9`） | 8192 | 显存调优 |
| `--max_req_total_len` | 16384 | 2048 | 显存调优 |
| `PYTORCH_ALLOC_CONF` | 未设置 | `expandable_segments:True` | 缓解碎片 |
| `--x2i_server_deploy_mode` | `colocate` | 单卡沿用 `colocate`；双卡改 `separate` | 见"单卡可行性" |
| `--tp`、`--x2i_server_used_gpus` | 1、1 | 1、1 | 未改动 |

注意与 32 GiB 级显卡的一处差异：`--x2i_server_deploy_mode` 在那类卡上必须改为 `separate`，而 W7900 上**可以保持官方默认的 `colocate`**。

### 关键参数说明

**`LIGHTLLM_NEO_PREFILL_TRITON_BACKEND=1` 必须设置。** 这是两处注意力后端切换中容易被忽略的一处。默认值 `0` 走 `fa3-neo`（FlashAttention 3 的定制版本），是 Hopper 特化实现，RDNA3 上不可用。它管的是**理解分支**的预填充注意力，与配置文件里 `attn_type` 管的**生成分支**注意力是两套独立代码，两处都要改，漏掉任何一处都起不来。

**`--disable_cudagraph` 必须开启。** LightLLM 的 CUDA graph 捕获依赖 `cupy` 的 `UnownedMemory`，而 `cupy` 在 PyPI 只有 CUDA 构建。其全部调用点均被 `torch.cuda.is_current_stream_capturing()` 或 `infer_state.is_cuda_graph` 条件包裹，该参数可完整绕开，对文生图无影响（graph 优化的是解码阶段，不是预填充）。

**`--batch_max_tokens 1024` 用于控制启动期试探推理的显存峰值。** 官方默认在单数据并行下取 16384，按此分配激活会显著抬高启动峰值。W7900 容量较宽裕，但保持 1024 可让单卡峰值稳定在 42 GiB 一线，留出安全余量。

## 注意力后端配置

模型自带配置里 `attn_type` 是 `flash_attn3`，Hopper 架构特化实现，RDNA3 上不可用。替代品是 `torch_sdpa`，在 ROCm 上经 AOTriton 落到 Triton flash kernel，无需额外依赖。

**这一项不通过修改上游文件实现。** `attn_type` 没有环境变量或命令行覆盖入口，只能来自配置文件；但 `--x2v_gen_model_config` 是直接 `open()` 任意路径，因此把配置放在仓库外即可：

```
/target/sensenova-rocm/configs/neopp_dense_rocm.json
```

该文件是 `configs/neopp/neopp_dense.json` 的副本，与上游原版逐行比对**只差 `attn_type` 一行**，键集完全一致。两个上游仓库的配置文件保持原样未动。完整内容：

```json
{
    "version": "dense",
    "load_kv_cache_in_pipeline_for_debug": true,
    "infer_steps": 50,
    "attn_type": "torch_sdpa",
    "cfg_scale": 4.0,
    "timestep_shift": 3.0,
    "cfg_interval": [-1, 2],
    "enable_cfg": true,
    "use_triton_qknorm_rope": true
}
```

## 可选：aiter 注意力后端（端到端提速约 1.7 倍）

`torch_sdpa` 之外还有一条更快的路。[AITER](https://github.com/ROCm/aiter) 是 AMD 官方的算子库，LightX2V 已经内置了对接实现（`lightx2v_platform/ops/attn/amd_rocm/flash_attn.py`，注册名 `aiter_attn`），本机实测有效：

| 场景 | `torch_sdpa` | `aiter_attn` | 提速 |
|---|---|---|---|
| 50 步 + CFG | 283.1 s | 167.7\* / 164.6 / 163.3 s | **1.72x** |
| 8 步蒸馏 LoRA | 33.1 s | 19.7\* / 17.2 / 16.3 / 16.2 s | **2.0x** |
| 注意力单算子，seq 4096 | 17.6 ms | 4.2 ms | 4.23x |

\* 首次请求，含 aiter 即时编译开销。

出图质量无退化，灰度分布与基准一致（`min=0`、`max=255`、`mean` 125–127）；单算子输出与 `torch_sdpa` 的最大绝对偏差约 `1e-3`，属 bf16 噪声量级。

单算子 4.23 倍的收益到端到端摊薄成 1.72 倍，是因为注意力只占总耗时的一部分，MoE 前馈与各类 GEMM 不受影响，属正常衰减。

### 为什么 gfx1100 上仍然有效

上游注释里写的「2.5x–6x，内部走 FA3（fmha_v3）」是 CDNA 的数字，**gfx1100 拿不到那条路径**。`aiter/ops/mha.py` 中所有快速路径都有架构门控，`can_impl_fmha_v3_fwd()` 要求 `gfx942`/`gfx950`，另有一条 `gfx1250` 的 ASM 路径，gfx1100 都不满足，最终落到 `else` 分支的 CK 通用实现 `mha_varlen_fwd`。

**但通用路径依然明显快于 AOTriton 的 SDPA**，所以结论是可用的——只是收益来源和上游宣传的不是同一条 kernel。AITER 官方文档也把 W7900 列为 *Experimental*，并注明「RDNA 上 Triton 与多数 FlyDSL kernel 可用，多数 CK 与 ASM kernel 仅限 CDNA」。

### 三个前提，缺一跑不起来

**一、安装 aiter 与 flydsl。** 官方 Release 提供预编译 wheel，矩阵覆盖 ROCm 7.0/7.1/7.2 × Python 3.10/3.12，本机的 ROCm 7.2.3 + Python 3.12 正好命中，无需源码编译：

```bash
# 约 1.59 GB；本机直连 GitHub 大文件不稳定，建议分块续传
curl -L -C - --retry 20 --retry-all-errors -o aiter.whl \
  "https://github.com/ROCm/aiter/releases/download/v0.1.21/amd_aiter-0.1.21%2Brocm7.2.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

# 必须 --no-deps：否则会拉入 PyPI 版 triton，覆盖掉 ROCm 定制栈
pip install --no-deps amd_aiter-0.1.21+rocm7.2.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
pip install --no-deps flydsl        # aiter 必需依赖，缺了会 ModuleNotFoundError
```

装完核对 `torch`、`triton` 版本未变。

**wheel 里没有 gfx1100 的代码对象**，预编译二进制只含 `gfx942` 与 `gfx950`（可用 `strings` 查 `.so` 里的 `amdgcn-amd-amdhsa--gfx*` 确认），gfx1100 上的 24 个文件全是 Triton 调优配置 JSON。因此首次调用会触发 hipcc 即时编译，**每种 kernel 配置约 100 秒**，编译结果落盘复用。上表首次请求的额外开销即来源于此。

**二、必须设 `PLATFORM=amd_rocm` 环境变量。** `lightx2v_platform/ops/__init__.py` 按该变量条件导入平台实现，只有取值为 `amd_rocm` 时才会执行 `from .attn.amd_rocm import *`。不设的话注册装饰器根本不运行，配置里写了 `aiter_attn` 只会得到 `KeyError`。注意主注册表通过 `ATTN_WEIGHT_REGISTER.merge(PLATFORM_ATTN_WEIGHT_REGISTER)` 做的是导入期一次性快照，事后手动 import 该模块也补不进去。

**三、需要两处额外代码改动**，与本报告「代码改动详解」中的三处相互独立，仅在走 aiter 时需要。两处都落在 AMD ROCm 专属文件内，**不触及任何平台无关代码**：

| 改动 | 文件 |
|---|---|
| sgl_kernel 注入循环跳过 `torch.ops` / `torch.classes` | `lightx2v_platform/base/amd_rocm.py` |
| aiter 补上 `cu_seqlens` 的设备搬运 | `lightx2v_platform/ops/attn/amd_rocm/flash_attn.py` |

**第一处：算子命名空间被误覆盖，import 阶段就崩。** `PLATFORM=amd_rocm` 时 `AmdRocmDevice.init_device_env()` 会把纯 Python 的 `AiterSglKernelCompat` 塞进 `sys.modules["sgl_kernel"]`——因为真正的 sgl_kernel 是 CUDA 专属的 C++ 扩展，ROCm 上装不了，只能用 aiter 的函数包装出同名方法来冒充。注入后它还会遍历 `sys.modules`，把已导入过 `sgl_kernel` 的模块属性一并替换：

```python
for mod_name, mod in list(sys.modules.items()):
    if mod is not None and hasattr(mod, "sgl_kernel"):
        setattr(mod, "sgl_kernel", sgl_kernel)
```

问题在于 **`torch.ops` 自身就以 `torch.ops` 为键存在于 `sys.modules` 中**（`_Ops` 是 `types.ModuleType` 的子类），而 `_Ops.__getattr__` 对**任意**名字都会当场创建 `_OpNamespace` 并缓存下来。也就是说循环里的 `hasattr(mod, "sgl_kernel")` 遍历到 `torch.ops` 时必然返回真——这一次 `hasattr` 调用本身就把命名空间造了出来——紧接着的 `setattr` 便把整个算子命名空间替换成了 shim 实例，`torch.classes` 同样被误伤。

之后 `lightx2v/common/ops/mm/sgl_kernel.py` 里的 `hasattr(torch.ops.sgl_kernel, "fp8_scaled_mm")` 因 shim 恰好有同名方法而为真，紧接着取 `.default` 拿到的却是绑定方法：

```
AttributeError: 'function' object has no attribute 'default'
```

**根源在注入循环，不在被它破坏的 `sgl_kernel.py`**，因此修法是在循环里跳过这两个伪模块，而不是去 `sgl_kernel.py` 里加防御性判空。

**这不是 gfx1100 或 RDNA 的问题，也不依赖 import 顺序。** 整条崩溃路径都在 Python 层，不碰任何硬件：把上面两段代码原样搬到一个 `CUDA_VISIBLE_DEVICES=""`、连 aiter 都没装的进程里，用几行的假 shim 就能复现同样的 `AttributeError`。判断注入与否的 `IS_AMD_ROCM` 取自 `torch.version.hip`，CDNA 与 RDNA 同样为真，循环也不看架构，因此**当前上游代码在 MI300/MI355 上启用 `PLATFORM=amd_rocm` 会同样崩溃**。

**它是一次回归，而非 ROCm 适配本身的缺陷。** 上游的时间线解释了为什么这条路径此前被验证通过却仍会坏：

| 时间 | 事件 |
|---|---|
| 2025-12-25 | PR #661 加入 ROCm 支持，含上面这段 shim 注入循环 |
| 2026-08-14 | PR #1381（MiniMax-H3 VAE 优化）新建 `sgl_kernel.py`，写下 `torch.ops.sgl_kernel.fp8_scaled_mm.default` |

`lightx2v/common/ops/mm/sgl_kernel.py` 全仓库只有 2026-08-14 那一次提交，`mm_weight.py` 中引入它的那行 import 也是同一天加的。ROCm 适配落地时这个文件尚不存在，当时的验证是有效的；八个月后的新代码撞上旧的注入逻辑才产生冲突，且未在 AMD 路径上复验。**因此这一处修复的意义不限于本机**，它是 main 分支上所有 ROCm 用户启用该平台开关时都会遇到的阻断性问题。

**第二处：aiter 漏了 `cu_seqlens` 的设备搬运。** 调用方允许把 `cu_seqlens` 建在 CPU 上、由后端负责搬运，这是 `lightx2v/common/ops/attn/flash_attn.py` 里 `flash_attn2` 与 `flash_attn3` 已有的约定（两者都写了 `is_cpu` 判断）。aiter 的 `apply` 漏了这步，neopp 调用时报：

```
RuntimeError: cu_seqlens_q must be on CUDA
Exception raised from mha_varlen_fwd at aiter_meta/csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu:423
```

**neopp 侧无需改动。** 它传的 q 是三维张量，`bs == 1`，`flash_attn3` 走的是 `flash_attn_func_v3` 稠密分支、根本不读 `cu_seqlens`，所以在 NVIDIA 上一直正常；aiter 的实现则不论 `bs` 是否为 1 都走 varlen 接口，才暴露出来。

两处改动已导出为一个可直接应用的 patch，基线对齐上游 `main`（`git apply` 或 `patch -p1` 均可，已验证应用后与本机实测代码逐字节一致）：

```
/target/sensenova-rocm/aiter-eval/lightx2v-rocm-aiter.patch
```

中文的根因分析与复现代码见同目录的 `aiter-fixes-notes.md`。

### 配置与启动

配置文件在对应的 ROCm 配置基础上只改 `attn_type` 一行：

```
/target/sensenova-rocm/configs/neopp_dense_rocm_aiter.json          # 50 步
/target/sensenova-rocm/configs/neopp_dense_8steps_rocm_aiter.json   # 8 步 LoRA
```

启动命令相对「单卡（推荐）」一节只多一个环境变量，其余参数不变：

```bash
CUDA_VISIBLE_DEVICES=1 PLATFORM=amd_rocm \
python -m lightllm.server.api_server \
  --model_dir /target/models/SenseNova-U1.5-8B-MoT \
  --enable_multimodal --enable_multimodal_x2i \
  --x2i_server_deploy_mode colocate --x2i_server_used_gpus 1 \
  --tp 1 --visual_gpu_ids 0 --disable_cudagraph \
  --max_total_token_num 8192 --max_req_total_len 2048 \
  --batch_max_tokens 1024 --chunked_prefill_size 1024 \
  --x2v_gen_model_config /target/sensenova-rocm/configs/neopp_dense_rocm_aiter.json \
  --host 127.0.0.1 --port 8081
```

启动日志出现下面两行即表示接线成功：

```
lightx2v_platform.ops.attn.amd_rocm.flash_attn - aiter flash_attn_varlen_func found (AMD ROCm optimized)
lightx2v_platform.base.amd_rocm - aiter sgl_kernel compatibility layer enabled (RMSNorm, GEMM)
```

### 是否采用

**追求出图速度就用**：8 步 LoRA 叠加 aiter 后单图约 16 秒，相对基准配置的 283 秒是 17 倍的总提速。代价是多一个 1.59 GB 依赖、首次调用的编译等待，以及两处额外补丁。

**求稳则留在 `torch_sdpa`**：零额外依赖，是上游默认路径，本报告的显存与性能数据也都以它为基线。AITER 对 W7900 的支持等级是 Experimental，长期跟随上游版本更新存在不确定性。

两处补丁的长期成本并不对等：`sgl_kernel` 那处是上游回归，一旦被修复即可撤掉；`cu_seqlens` 那处是 aiter 后端自身缺失的设备搬运，需要合入上游才能免去。

## 设备编号：本机两套编号一致

**这台机器上 HIP 的设备枚举顺序与 `rocm-smi` 的卡编号完全一致**，现场核验结果：

| `CUDA_VISIBLE_DEVICES` 里写的序号 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| PCI 地址（末段） | 03 | 23 | 43 | 63 | 83 | a3 | c3 | e3 |
| `rocm-smi` 中的卡 | card0 | card1 | card2 | card3 | card4 | card5 | card6 | card7 |

因此上面启动命令里的 `CUDA_VISIBLE_DEVICES=1,2` 拿到的就是 card1 与 card2，监控读数可直接对应。

**但这个结论不能推广。** 该映射取决于机器的 PCI 拓扑，同型号的另一台机器上就出现过错位（HIP 序号 0 对应 card3）。ROCm 没有 CUDA 那个 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 可用来强制对齐。每台机器都应现场核验：

```bash
# 一、rocm-smi 的卡 → PCI 地址
rocm-smi --showbus --csv

# 二、HIP 逻辑序号 → PCI 地址（注意清空 CUDA_VISIBLE_DEVICES）
env -u CUDA_VISIBLE_DEVICES python3 - <<'PY'
import ctypes
lib = ctypes.CDLL("libamdhip64.so")
n = ctypes.c_int(); lib.hipGetDeviceCount(ctypes.byref(n))
for i in range(n.value):
    buf = ctypes.create_string_buffer(64)
    lib.hipDeviceGetPCIBusId(buf, 64, i)
    print(i, buf.value.decode())
PY
```

忽略这一点的后果不只是报告写错卡号：排查显存问题时会盯着错误的卡看占用，从而误判是外部进程干扰还是自己的配置问题。

## 显存实测

单卡总量 47.98 GiB，`rocm-smi` 读数：

**双卡 `separate`**

| 卡 | 承载的服务 | 占用 | 余量 |
|---|---|---|---|
| card1 | 理解分支 | 26.12 GiB | 21.86 GiB |
| card2 | 生成分支 + 视觉服务 | 18.41 GiB | 29.57 GiB |
| | **合计** | **44.53 GiB** | |

**单卡 `colocate`**

| 卡 | 承载的服务 | 配置 | 峰值占用 | 余量 |
|---|---|---|---|---|
| card1 | 理解 + 生成 + 视觉 | 50 步基准 | 42.01 GiB | 5.97 GiB |
| card1 | 理解 + 生成 + 视觉 | 8 步 LoRA | 43.18 GiB | 4.80 GiB |

理解分支的构成是权重 17.50（含共享词嵌入与输出投影）+ KV 缓存 1.31 + 激活；读数高出的部分是 PyTorch 缓存分配器在启动期试探推理之后保留未归还的显存，属正常行为。

**读数必须减去外部占用后再归因。** 采集期间同机 card0 有 47.24 GiB 占用且利用率 100%、card3 有 44.54 GiB，均属其他负载，与本实例无关。只看单卡读数而不确认归属，会把别人的显存算到自己头上。

## 性能

1024×1024，50 步基准配置，两种部署各三次实测。本节数据均取默认的 `torch_sdpa` 后端；换用 aiter 后端的对比见「可选：aiter 注意力后端」。

先明确 NFE（前向传播次数）的口径，它决定了每步耗时的折算方式。配置中 `enable_cfg` 为真、`cfg_scale` 为 4.0，而 `cfg_interval` 是 `[-1, 2]`——调度器的时间步由 `linspace(0.0, 1.0, steps+1)` 生成并经 shift 变换，取值始终落在 [0, 1] 内，故区间判定 `t >= -1 and t <= 2` **恒为真，CFG 全程启用**。代码路径上此时每步跑两次前向（条件与无条件各一次），因此 **50 步 = 100 NFE**。实测各步耗时高度一致（50 步内标准差仅 0.03 秒）也印证了这一点：若 CFG 只在部分步生效，耗时会呈现两个明显的簇。

| 部署 | 端到端耗时 | 推理净耗时 | 单步耗时 |
|---|---|---|---|
| 单卡 `colocate` | 291.2\* / 283.6 / 283.1 s | 288.3\* / 282.2 / 282.2 s | 5.644 s |
| 双卡 `separate` | 285.1 / 293.0 / 286.4 s | 283.6 / 289.6 / 285.3 s | 5.673 s |

\* 首次请求，含 Triton 内核即时编译开销。

三项指标的口径：端到端是客户端从发起请求到拿到图片的墙钟时间；推理净是 LightX2V profiler 记录的 50 步累计（日志中的 `infer_main cost`）；单步取稳态最优一次的均值，折半即单次前向约 **2.82 秒**。端到端减推理净约 **2.0 秒**，是 VAE 解码与图像编码传输的固定开销。

**单卡与双卡性能相当，差异约 0.5%。** 稳态推理净耗时单卡 282.2 秒、双卡 283.6 秒，且单卡略优。这不是测量噪声导致的偶然结果，有两条独立证据：

其一，单步耗时几乎相同（5.644 对 5.673 秒），说明扩散阶段的算力消耗与分不分卡无关。

其二，推理期 GPU 利用率采样（每 3 秒一次）显示负载完全不均衡：

```
08:10:02 GPU1=0% GPU2=100%
08:10:29 GPU1=4% GPU2=100%
08:10:59 GPU1=6% GPU2=100%
```

生成分支所在的卡稳定 100%，理解分支所在的卡在 0–6% 之间。理解分支只在开头做一次提示词预填充，之后全程是扩散去噪，那张卡只是持有权重而已——端到端与推理净之间仅差 2.0 秒，而这 2.0 秒还要与 VAE 解码、图像编码传输分摊，可见预填充本身用时极短。**分卡部署并没有把工作分摊出去，只是把闲置的权重挪到了另一张卡上**，故不产生并行收益，反而多付一套 HIP 上下文与分配器预留。

**首次请求会额外付出约 6 秒的 Triton 内核即时编译开销**（单卡首次推理净 288.3 秒对稳态 282.2 秒）。该缓存在同机不同实例间共享，故后续新实例的首次请求只需补编译差异部分。

关于数据离散度：单卡三次中后两次完全一致（282.2 / 282.2 秒），50 步内单步标准差 0.030 秒；双卡三次里有两次出现异常慢步（最大 7.9 秒对正常 5.7 秒），单步标准差升至 0.17–0.40 秒。异常步与同机 card0 上的外部满载负载时间吻合，判断为共享资源争抢，非部署模式本身的问题。做性能对比时应确认同机负载状态，否则容易把外部干扰读成配置差异。

### 8 步蒸馏 LoRA

同为单卡 `colocate`、1024×1024，配置改用 `neopp_dense_8steps_rocm.json`（8 步，`enable_cfg` 为 false，故 8 NFE）：

| 配置 | 步数 | NFE | 端到端耗时 | 推理净耗时 | 单步耗时 |
|---|---|---|---|---|---|
| CFG 开，无 LoRA | 50 | 100 | 291.2\* / 283.6 / 283.1 s | 288.3\* / 282.2 / 282.2 s | 5.673 s |
| CFG 关，8 步 LoRA | 8 | 8 | 46.0\* / 33.1 / 35.7 / 33.1 s | 42.4\* / 32.0 / 34.6 / 32.0 s | 4.000 s |

\* 首次请求，含 Triton 内核即时编译开销（8 步配置下约 10 秒，高于 50 步配置的约 6 秒，因需补编译 batch=1 形状的 kernel）。

**8 步 LoRA 相对 50 步基准提速 8.6 倍**（端到端 283.1 对 33.1 秒；按推理净耗时算是 8.8 倍）。稳态三次为 33.1 / 35.7 / 33.1 秒，其中 35.7 秒那次的单步标准差达 0.93 秒，仍是外部负载干扰。

### NFE 不能单独决定耗时：批处理效应

这组数据揭示了一个容易导致误判的现象，值得单独说明。

两种配置的单次前向耗时并不相同：

| 配置 | 每步耗时 | 每步前向次数 | 折合单次前向 |
|---|---|---|---|
| 50 步，CFG 开 | 5.673 s | 2 | **2.837 s** |
| 8 步，CFG 关 | 4.000 s | 1 | **4.000 s** |

**同样是一次前向，CFG 关闭时反而慢 41%。** 原因是开启 CFG 时条件与无条件两路被批在同一次前向里计算（batch=2），一次权重读取服务两个样本；关闭后 batch=1，同样的权重访存只摊到一个样本上。该负载在 batch=1 时明显是访存受限而非算力受限，所以 batch 翻倍并不会让耗时翻倍——实测只增加 42%。

由此推论：**按 NFE 线性外推会低估 8 步配置的耗时**。若按 50 步实测的单次前向 2.837 秒外推，8 NFE 应为 22.7 秒加固定开销约 25 秒，而实测 33.1 秒，偏差 33%。用「解二元一次方程分离单次前向耗时与固定开销」的做法同样有问题，它隐含假设两种配置的单次前向等价，会把批处理效应错误地吸收进「固定开销」项，使该项被显著高估（本机实测固定开销仅约 1.1 至 2.0 秒）。

理论提速上限也应据此修正：NFE 从 100 降到 8 是 12.5 倍，但单次前向变慢 1.41 倍，两者相抵后上限约 8.9 倍，实测 8.6 倍已非常接近，剩余差距来自固定开销在 8 步场景下占比升高。

### 与 gfx1201 的横向参考

同一模型、同样 50 步配置，R9700（gfx1201 / RDNA4）实测约 161 秒，W7900 约 285 秒，后者是前者的约 1.76 倍；折合单次前向 2.82 秒对 1.61 秒。W7900 显存容量大出 50%（48 对 32 GiB），因而具备单卡部署能力，但单次前向的计算吞吐明显偏低。**该对比来自不同机器，仅供量级参考，未做控制变量的硬件基准测试，也未定位差距的具体来源。**

## 启用 8 步蒸馏 LoRA

官方已为 U1.5 发布 8 步蒸馏 LoRA，但它**不在基座模型仓库里**，而是单独一个仓库，容易漏找：

```bash
hf download sensenova/SenseNova-U1.5-8B-MoT-LoRAs \
  SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors \
  --local-dir /target/models/SenseNova-U1.5-8B-MoT-LoRAs
```

文件 777 MiB（814,867,236 字节）。

上游配置 `configs/neopp/neopp_dense_8steps.json` 里的 LoRA 路径是占位符 `/path/to/SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors`，**文件名指向上一代 U1**，必须替换为实际下载的 U1.5 文件路径，否则启动即失败。

本次使用的配置同样放在仓库外（`/target/sensenova-rocm/configs/neopp_dense_8steps_rocm.json`），相对上游只改两处，与上游的完整差异即：

```
"attn_type": "flash_attn3"  →  "torch_sdpa"                        （ROCm 必需）
"path": "/path/to/SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors"
     →  "/target/models/SenseNova-U1.5-8B-MoT-LoRAs/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"
```

完整内容（`infer_steps` 为 8、`enable_cfg` 为 false，两者共同决定了 8 NFE）：

```json
{
    "version": "dense",
    "load_kv_cache_in_pipeline_for_debug": true,
    "infer_steps": 8,
    "attn_type": "torch_sdpa",
    "timestep_shift": 3.0,
    "cfg_interval": [-1, 2],
    "enable_cfg": false,
    "use_triton_qknorm_rope": true,
    "lora_configs": [
        {
            "path": "/target/models/SenseNova-U1.5-8B-MoT-LoRAs/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors",
            "strength": 1.0
        }
    ]
}
```

**键名无需任何映射即可对上。** 该 LoRA 为 STANDARD 格式，882 个张量对应 294 组（每组 `alpha`、`lora_down.weight`、`lora_up.weight`），全部带生成分支的 `_mot_gen` 后缀。合并日志为 `Applied 294 LoRA weight adjustments out of 294 possible`，零条键不匹配警告。

**合并发生在启动期而非推理期**，因此 LoRA 无法热加载，启用或停用都要重启服务。含合并在内，服务从启动到就绪约 4.0 分钟，与 50 步配置相当。

显存占用 43.18 GiB，较 50 步配置的 42.01 GiB 高约 1.2 GiB，仍在单卡 47.98 GiB 内，余量 4.8 GiB。

出图质量与 50 步基准相当，未出现网格伪影；主体细节（果皮斑点、叶片脉络、木纹）反而更锐利，对比度略强。三次采样的灰度分布正常（mean 114 至 127）。

## 调用方式

**必须使用 OpenAI 兼容路由 `/v1/images/generations`。**

```python
payload = {
    "prompt": "A photorealistic red apple on a wooden table, soft natural lighting",
    "size": "1024x1024",          # 唯一生效的尺寸入口
    "output_format": "png",
    "response_format": "b64_json",
}
```

返回的 `data[i].b64_json` 是 base64 data URL，需去掉 `data:image/png;base64,` 前缀后解码。

**采样参数由服务端配置决定，不接受按请求指定。** 请求体里的 `img_config` 会被请求校验器整体丢弃并重建：

| 参数 | 实际行为 |
|---|---|
| `img_config.steps` | 无效。步数取配置文件的 `infer_steps` |
| `img_config.seed` | 无效。被强制改为 `42 + 随机数`，故输出不可复现 |
| `img_config.width/height` | 无效。分辨率取顶层 `size` 字段 |
| `img_config.guidance_scale`、`cfg_norm` 等 | 无效。取配置文件值 |

这是框架的分层设计而非缺陷：LightX2V 的推理接口本身不接受按次指定步数，步数是 pipeline 初始化时从配置读取的实例级属性。**尤其注意 `steps`——向 50 步配置的实例请求 8 步，拿到的是 50 步的耗时**，据此折算每步耗时会得出错误结论。

测试脚本通过环境变量 `T2I_PORT`、`T2I_TAG` 选择实例与输出标记，除标准库外只在像素校验时可选依赖 Pillow。

<details>
<summary><b>t2i_test.py 完整内容（点击展开）</b></summary>

```python
"""文生图冒烟测试。必须走 /v1/images/generations，原生 /generate_image 路由
对 PNG 字节做 JSON 序列化必然失败。"""
import base64, io, json, os, sys, time, urllib.request

PORT = os.environ.get("T2I_PORT", "8080")
URL = f"http://127.0.0.1:{PORT}/v1/images/generations"
OUT_DIR = "/target/sensenova-rocm/outputs"
TAG = os.environ.get("T2I_TAG", "")

prompt = sys.argv[1] if len(sys.argv) > 1 else "A photorealistic red apple on a wooden table, soft natural lighting, high detail"
width = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
height = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

# 分辨率必须走顶层 size 字段：img_config 会被服务端整体重建，里面传什么都会被丢弃
payload = {"prompt": prompt, "size": f"{width}x{height}",
           "output_format": "png", "response_format": "b64_json"}

print(f"  端点   : {URL}\n  提示词 : {prompt}\n  请求   : {width}x{height}（步数由服务端配置决定）\n")
req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"}, method="POST")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=3600) as resp:
        raw, status = resp.read(), resp.status
except urllib.error.HTTPError as e:
    print(f"  HTTP {e.code} after {time.time()-t0:.1f}s\n  body: {e.read().decode(errors='replace')[:1500]}")
    sys.exit(1)

elapsed = time.time() - t0
print(f"  HTTP {status} in {elapsed:.1f}s, 响应 {len(raw)} 字节")
data = json.loads(raw)
items = data.get("data") or []
if not items:
    print(f"  无图像, keys={list(data.keys())}"); sys.exit(1)
print(f"  实际尺寸: {data.get('size')}")

os.makedirs(OUT_DIR, exist_ok=True)
for i, item in enumerate(items):
    b64 = item.get("b64_json") or ""
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    blob = base64.b64decode(b64)
    path = f"{OUT_DIR}/t2i_{width}x{height}{TAG}_{i}.png"
    open(path, "wb").write(blob)
    # 注意力内核算错时通常表现为纯色或 NaN 帧，所以要检查像素而不是只看 HTTP 200
    note = ""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(blob)); img.load()
        px = list(img.convert("L").getdata())
        lo, hi = min(px), max(px)
        note = f" | {img.size[0]}x{img.size[1]} | 灰度 min={lo} max={hi} mean={sum(px)/len(px):.1f}"
        if lo == hi:
            note += "  <<< 纯色图，可疑"
    except Exception as e:
        note = f" | 像素检查失败: {e}"
    print(f"  已保存 {path} ({len(blob)} 字节){note}")
print(f"\n  总耗时 {elapsed:.1f}s")
```

</details>

## 依赖处理

两个项目的依赖合集里有若干 CUDA 专属包，直接安装会破坏已有的 ROCm PyTorch。策略是**约束文件 + 过滤清单**双保险。

`rocm-constraints.txt` 锁死 ROCm 栈，使 pip 在冲突时直接失败而不是静默替换：

```
torch==2.10.0+rocm7.2.3.lw.git1a270074
torchvision==0.25.0+rocm7.2.3.git82df5f59
torchaudio==2.10.0+rocm7.2.3.git5047768f
triton==3.6.0+rocm7.2.3.git4ed88892
numpy==2.4.4
```

版本号需与 `pip list` 的输出逐字一致（注意 torch 带 `.lw` 段，而 `torch.__version__` 打印的字符串不含该段，两者不可混用）。

需排除的 CUDA 专属包：`flashinfer-python`、`sgl-kernel`、`cuda_bindings`、`cupy-cuda12x`、`xformers`、`nixl`。

**依赖清单直接取自另一台已跑通机器（R9700）的 `pip freeze`，而非自行合并两个项目的 requirements。** 后者需要人工裁决大量版本冲突（例如 litellm 要求的 tiktoken 版本与 LightLLM 钉死的版本互斥、部分包锁定的版本早于 Python 3.12），实践中容易陷入 pip 的依赖解析失败。既然已有一套在同代 ROCm 栈上验证过的完整版本组合，直接复用它更可靠。清单文件为 `rocm-requirements-r9700.txt`，其中已排除 PyPI 上不存在的 ROCm 定制 wheel（`torch`、`torchvision`、`torchaudio`、`apex`），这些由本机现有环境提供。

安装：

```bash
pip install -r rocm-requirements-r9700.txt -c rocm-constraints.txt
```

清单含 146 项，装完连同传递依赖共 154 个包，ROCm 栈五个包版本未变。安装后应核对 `torch.version.hip` 非空（本机为 `7.2.53211`），确认未被 CUDA 版覆盖。

<details>
<summary><b>rocm-requirements-r9700.txt 完整内容（146 个包，点击展开）</b></summary>

```
# 源自 R9700 实测跑通环境的 pip freeze，版本全部精确指定
# 已排除本机已装且版本一致的 ROCm 定制 wheel（PyPI 上不存在）：
#   apex==1.10.0+rocm7.2.3.git751f5dd5
#   torch==2.10.0+rocm7.2.3.lw.git1a270074
#   torchaudio==2.10.0+rocm7.2.3.git5047768f
#   torchvision==0.25.0+rocm7.2.3.git82df5f59
#   triton==3.6.0+rocm7.2.3.git4ed88892
accelerate==1.14.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-types==0.8.0
antlr4-python3-runtime==4.9.3
anyio==4.14.2
atomics==1.0.3
attrs==26.1.0
audioread==3.1.0
av==18.1.0
build==1.5.0
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
click==8.5.0
coloredlogs==15.0.1
contourpy==1.3.3
cxxfilt==0.3.0
cycler==0.12.1
decorator==5.3.1
decord==0.6.0
diffusers==0.37.1
easydict==1.13
einops==0.6.0
environs==9.5.0
fastapi==0.115.6
filelock==3.29.0
fonttools==4.62.1
frozendict==2.4.6
frozenlist==1.8.0
fsspec==2026.4.0
ftfy==6.3.1
gguf==0.19.0
h11==0.16.0
h2==4.4.1
hf-xet==1.6.0
hf_transfer==0.1.9
hpack==4.2.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==0.36.2
humanfriendly==10.0
humanize==4.7.0
Hypercorn==0.18.0
hyperframe==6.1.0
hypothesis==6.152.4
idna==3.19
imageio-ffmpeg==0.6.0
ImageIO==2.37.4
importlib_metadata==9.0.1
iniconfig==2.3.0
interegular==0.3.3
Jinja2==3.1.6
joblib==1.5.3
jsonschema-specifications==2025.9.1
jsonschema==4.26.0
kiwisolver==1.5.0
langdetect==1.0.9
lazy-loader==0.5
librosa==0.11.0
llvmlite==0.49.0
loguru==0.7.3
lxml==6.1.2
MarkupSafe==3.0.3
marshmallow==3.19.0
matplotlib==3.10.9
modelscope-hub==0.3.0
modelscope==1.39.1
mpmath==1.3.0
msgpack==1.2.2
multidict==6.7.1
multiprocessing-logging==0.3.4
narwhals==2.25.0
networkx==3.6.1
ninja==1.13.0
numba==0.67.0
numpy==2.4.4
omegaconf==2.3.1
opencv-python==4.14.0.94
orjson==3.11.2
packaging==26.2
pandas==3.0.2
partial-json-parser==0.2.1.1.post6
peft==0.20.0
pillow==10.4.0
platformdirs==4.11.5
pluggy==1.6.0
plumbum==1.8.2
pooch==1.9.0
priority==2.0.0
prometheus_client==0.20.0
propcache==0.5.2
protobuf==4.22.3
psutil==7.2.2
py-cpuinfo==9.0.0
pybind11==3.1.0
pycparser==3.0
pydantic==2.10.3
pydantic_core==2.27.1
Pygments==2.20.0
PyJWT==2.13.0
pyparsing==3.3.2
pyproject_hooks==1.2.0
pytest==9.0.3
python-dateutil==2.9.0.post0
python-docx==1.2.0
python-dotenv==1.0.0
PyYAML==6.0.1
pyzmq==25.1.2
redis==8.1.0
referencing==0.37.0
regex==2026.8.31
requests==2.34.2
rpds-py==2026.6.3
rpyc==5.3.1
safetensors==0.4.5
scikit-learn==1.9.0
scipy==1.17.1
sentencepiece==0.2.0
setproctitle==1.3.6
setuptools==82.0.1
six==1.17.0
sortedcontainers==2.4.0
soundfile==0.14.0
soxr==1.1.0
starlette==0.41.3
sympy==1.14.0
threadpoolctl==3.6.0
tiktoken==0.7.0
tokenizers==0.22.1
tqdm==4.67.3
transformers==4.57.1
typing_extensions==4.15.0
ujson==5.10.0
urllib3==2.7.0
uvicorn==0.19.0
uvloop==0.22.1
wcwidth==0.8.3
websockets==15.0.1
wheel==0.47.0
wsproto==1.3.2
xxhash==3.6.0
yarl==1.24.5
zipp==4.1.0
zstandard==0.23.0
```

</details>

以下四个包容易被判为"本地不需要"而漏装，但启动时会因无条件 import 而失败，必须补装：`redis`（多模态视觉服务路径）、`av` 与 `decord`（视频编解码，导入链上无条件引用）、`uvloop`（HTTP 服务事件循环）。

## 已知限制

**图生图不可用，有两道独立障碍。** 一是 LightLLM 的适配层无条件调用 `self.pipe.runner.set_kvcache_i2i()`，而 LightX2V 未实现该方法。二是本模型的 chat template 只接受字符串形式的 content，而携带图片必须用列表形式，prompt 构造这一步就通不过。两者都与平台无关。

**原生路由 `/generate_image` 不可用。** 该路由把 PNG 原始字节直接交给 `json.dumps`，必然报 `reject_bytes is on` 错误。改用 OpenAI 兼容路由即可。

**采样参数不能按请求指定。** 步数、种子、引导强度只能在启动时通过配置文件设定，切换需重启；种子被强制随机化，故输出不可复现。

**LoRA 无法热加载。** 合并发生在启动期而非推理期，启用或停用 8 步 LoRA 都要重启服务，且需另备一份配置文件。

**"8 步且不加 LoRA"的质量未测。** 因步数属实例级配置，该组合需单独准备配置并重启才能测到，故无法量化蒸馏 LoRA 在质量上究竟补偿了多少。

**FP8 量化未验证。** 本机容量充裕，单卡已可运行，量化的必要性不高。`neopp_dense_fp8.json` 指定的 `fp8-sgl` 方案依赖 CUDA 专属的 `sgl_kernel`。不过在 `PLATFORM=amd_rocm` 下，LightX2V 会用 aiter 的 `gemm_a8w8_CK` 顶替 `sgl_kernel.fp8_scaled_mm`（日志中的「aiter sgl_kernel compatibility layer enabled」），这条路径未必真的走不通——本次只验证了该兼容层能正常加载，未实测 FP8 出图。

**autotune 配置未针对 gfx1100。** 现有 JSON 均为 H200 特化，尚有提速空间。

## 从零复现

```bash
# 1. 取代码
git clone https://github.com/ModelTC/LightLLM.git && cd LightLLM
git checkout neo_plus_clean && git checkout -b gyydev/rocm-gfx1201
git clone https://github.com/ModelTC/LightX2V.git && cd LightX2V
git checkout -b gyydev/rocm-gfx1201

# 2. 装依赖（约束文件保护 ROCm 栈）
pip install -r rocm-requirements-r9700.txt -c rocm-constraints.txt
pip install redis av decord uvloop
pip install -e . --no-deps        # 两个项目各执行一次

# 3. 应用三处补丁（完整代码见"代码改动详解"一节）

# 3b. 准备 ROCm 配置（复制上游 neopp_dense.json，仅把 attn_type 改为 torch_sdpa）
#     完整内容见"注意力后端配置"一节，可直接复制该 JSON
mkdir -p /target/sensenova-rocm/configs
sed 's/"flash_attn3"/"torch_sdpa"/' \
  /target/LightX2V/configs/neopp/neopp_dense.json > /target/sensenova-rocm/configs/neopp_dense_rocm.json

# 4. 下载模型
hf download SenseNova/SenseNova-U1.5-8B-MoT \
  --local-dir /target/models/SenseNova-U1.5-8B-MoT

# 4b.（推荐）下载 8 步蒸馏 LoRA 并准备对应配置，提速 8.6 倍
hf download sensenova/SenseNova-U1.5-8B-MoT-LoRAs \
  SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors \
  --local-dir /target/models/SenseNova-U1.5-8B-MoT-LoRAs
# 配置在上游 neopp_dense_8steps.json 基础上改 attn_type 为 torch_sdpa，
# 并把 lora_configs 的占位符路径替换为上面下载的实际路径
# 完整内容见"启用 8 步蒸馏 LoRA"一节，可直接复制该 JSON

# 4c.（可选）改用 aiter 注意力后端，再提速约 1.7 倍
#     需额外装 AITER 与 flydsl、多打两处补丁、启动时加 PLATFORM=amd_rocm
#     完整步骤见"可选：aiter 注意力后端"一节

# 5. 核验设备编号映射（见"设备编号"一节），挑空闲卡决定 CUDA_VISIBLE_DEVICES
rocm-smi --showbus --csv

# 6. 启动（见"启动"一节，推荐单卡 colocate）
#    用 8 步 LoRA 则把 --x2v_gen_model_config 指向 neopp_dense_8steps_rocm.json

# 7. 验证（脚本完整内容见"调用方式"一节；第二、三个参数是宽高，
#    步数由启动配置决定，请求无法指定）
python t2i_test.py "a red apple on a wooden table" 1024 1024
```

本报告自包含：三处补丁的代码、两份配置文件、依赖清单与测试脚本的完整内容均已内嵌，无需另取外部文件。

## 排障

**判断服务是否就绪** 看日志中的 `server start up ok`，不要依赖 `ss -ltn`——容器内该命令可能看不到监听端口，而服务其实正常。

**确认成图正确性不能只看 HTTP 200。** 注意力 kernel 若有数值错误，通常表现为纯色或噪声画面。`t2i_test.py` 会检查解码后像素的灰度分布，纯色图会被标记。本次三轮测试灰度分布均正常（如 min=0 / max=254 / mean=113.6）。

**排查显存前先确认卡号换算。** 报错信息里的 "GPU 0" 是 HIP 逻辑序号。本机两套编号恰好一致，但换机器前务必重新核验（见"设备编号"一节）。

**务必先确认没有残留进程再重启。** 主进程退出后子进程可能存活并继续占用显存，累积几轮能把一张卡占满。注意向 nohup 包装进程发信号不会终止实际的 `api_server` 进程，需定位到真实 PID。重启前检查：

```bash
for p in /proc/[0-9]*; do
  cl=$(tr '\0' '\n' < "$p/cmdline" 2>/dev/null | head -1)
  case "$cl" in lightllm::*) echo "${p#/proc/} $cl" ;; esac
done
```

**做性能测试时确认同机负载。** 本机 8 张卡常有其他任务运行，外部满载负载会通过共享资源争抢引入 1–2% 的波动甚至个别异常慢步。对比不同配置时应在相近负载条件下采样，并优先采用 profiler 记录的推理净耗时而非客户端墙钟时间。
