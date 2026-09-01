# SenseNova-U1.5-8B-MoT 文生图 ROCm 移植报告(AMD Radeon AI PRO R9700)

在 AMD Radeon AI PRO R9700(gfx1201 / RDNA4)上运行 SenseNova-U1.5-8B-MoT 双分支文生图模型的完整记录,含代码改动、依赖处理、启动配置、性能数据与复现步骤。

## 结论

文生图端到端跑通,成图内容、结构、光影均正确。1024×1024 出图:50 步基准约 **161 秒**,启用官方 8 步蒸馏 LoRA 后约 **17.6 秒**(提速 9.1 倍,质量相当)。分辨率对耗时几乎无影响,2048×2048 同样约 18 秒。

**需要两张卡。** 单卡装不下(权重 32.66 GiB 已超出单卡 31.86 GiB),但三张也不必:理解分支占一张,生成分支与视觉服务共用另一张,实测出图耗时与各占一张几乎无差(详见"卡位分配")。

代码改动共三处,**只有一处属于 ROCm 适配**,另两处是上游在 NVIDIA 上同样会触发的缺陷:

| 改动 | 性质 | 仓库 | 文件 |
|---|---|---|---|
| 主机内存注册运行时由 libcudart 改为按平台解析 | ROCm 专属 | LightLLM | `lightllm/utils/kv_cache_utils.py` |
| torch_sdpa 补上分组查询注意力广播 | 上游缺陷,平台无关 | LightX2V | `lightx2v/common/ops/attn/torch_sdpa.py` |
| 纯文本图像请求传字符串 content | 上游缺陷,平台无关 | LightLLM | `lightllm/server/build_prompt.py` |

三处改动刻意拆开,后两处 bug 修复可直接向上游提 PR,不夹带 ROCm 相关内容。已提交并推送至 fork 的 `gyydev/rocm-gfx1201` 分支(LightLLM 两个提交 `585ee06`、`4097515`;LightX2V 一个提交 `0def796`),`/target/patch-backup/` 下另存有可重放的补丁文件。

移植过程中另发现两处上游缺陷尚未修复,均与平台无关:图生图所需的 `set_kvcache_i2i` 未实现、原生路由 `/generate_image` 对 PNG 字节做 JSON 序列化必然失败。

此外有一处接口约定容易踩坑但**并非缺陷**:采样参数(步数、种子、引导强度)只能通过启动配置指定,请求体里的 `img_config` 会被整体重建;分辨率必须走顶层 `size` 字段。LightX2V 的推理接口本身不接受按次指定步数,两边是对齐的。详见"调用方式"。

**注意力后端的切换不算改动**:它通过一份仓库外的配置文件实现(`/target/configs/neopp_dense_rocm.json`),两个上游仓库的配置文件保持原样,详见"注意力后端配置"一节。

## 环境

| 项 | 值 |
|---|---|
| GPU | 8 × AMD Radeon AI PRO R9700 (gfx1201 / RDNA4),31.86 GiB/卡 |
| ROCm | 7.2.3 |
| PyTorch | 2.10.0+rocm7.2.3 (HIP 7.2.53211) |
| Triton | 3.6.0+rocm7.2.3 |
| Python | 3.12.3 |
| transformers | 4.57.1 |
| 模型 | SenseNova/SenseNova-U1.5-8B-MoT,8 个分片共 32.66 GiB |

gfx1201 具备 WMMA 指令但没有 MFMA(那是 CDNA 独有),因此依赖 MFMA 的 kernel 不可用;本次用到的 Triton kernel 与 SDPA 路径均不涉及。

## 模型结构

混合专家式双分支(Mixture-of-Transformers),两个分支各 42 层且严格对称:

- **理解分支**:由 LightLLM 承载,dense 结构(config.json 无 `num_experts` 字段,不涉及专家路由),权重约 17.45 GiB
- **生成分支**:由 LightX2V 承载,权重带 `_mot_gen` 后缀,约 15.13 GiB
- 两分支共享词嵌入与输出投影(各约 1.16 GiB)
- 共 1116 个张量,42 层每层 26 个

理解分支处理文本提示词后,把注意力键值缓存经共享内存传给生成分支,生成分支执行扩散去噪。这条跨进程缓存传递正是需要主机内存注册的地方。

## 单卡为何不可行

磁盘上 8 个分片合计 32.66 GiB(十进制 35.1 GB),按分支拆解:

| 部分 | 张量数 | 大小 |
|---|---|---|
| 理解分支 | 559 | 15.18 GiB |
| 生成分支(`_mot_gen` 后缀) | 551 | 15.13 GiB |
| 共享的词嵌入与输出投影 | 2 | 2.32 GiB |
| 视觉编码器 | 4 | 0.03 GiB |
| **合计** | **1116** | **32.66 GiB** |

单卡可用 31.86 GiB。**纯权重就超出 0.80 GiB**,且上表中共享部分已只计一份、无重复。这一点容易被"33G 模型 / 32G 显卡"的印象误导,实际方向是超的。

真实缺口远大于 0.80 GiB,因为还必须容纳:

| 开销 | 量级 | 说明 |
|---|---|---|
| KV 缓存池 | 1.31 GiB | 8192 词元;理解分支必需 |
| 启动期试探推理峰值 | 约 4 GiB | 已是 `batch_max_tokens` 调到 1024 后的值 |
| 扩散过程激活 | 数 GiB | 1024×1024 对应 4096 个潜空间词元 × 42 层 |
| HIP 上下文 | 每进程约 0.3 GiB | 无法避免 |

实测两侧占用分别约 20 GiB 与约 19.8 GiB,合计约 40 GiB,即单卡缺口在 6 到 8 GiB 量级。

**量化是唯一能压进单卡的路径。** FP8 可将权重降到约 16.3 GiB,加运行时约 6 GiB 共 22 GiB,余量充足。gfx1201 硬件支持 FP8——实测 `float8_e4m3fn` 与 ROCm 变体 `float8_e4m3fnuz` 的转换均成功。

障碍在软件侧:`neopp_dense_fp8.json` 指定的方案是 `fp8-sgl`,依赖 CUDA 专属的 `sgl_kernel`(导入即 `ModuleNotFoundError`)。LightX2V 的后端注册表里另有 `mxfp8`、`mxfp4`(AMD microscaling 格式)、`fp8-q8f` 等选项,方向对得上但都需要额外内核库,**本次未验证**。

## 代码改动详解

### 一、主机内存注册运行时解析(ROCm 专属)

`lightllm/utils/kv_cache_utils.py`

原代码把 CUDA 运行时库的路径写死:

```python
ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
```

纯 ROCm 系统上没有这个文件。改为按平台解析:

```python
def _load_gpu_runtime():
    """Return (lib, symbol_prefix) for the host-memory-registration API."""
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

`hipHostRegister` 与 `cudaHostRegister` 的 ABI 和标志位语义一致,函数名前缀不同而已,因此只需替换库与前缀。

实测:400 个共享内存页全部注册成功,耗时约 16 秒,零失败。

### 二、分组查询注意力广播(上游缺陷)

`lightx2v/common/ops/attn/torch_sdpa.py`

本模型有 32 个查询头、8 个键值头。PyTorch 的 `scaled_dot_product_attention` 在两者不等时必须显式开启广播,否则报维度不匹配。上游漏了这个参数:

```python
x = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attn_mask,
    dropout_p=drop_rate,
    is_causal=causal,
    enable_gqa=q.shape[1] != k.shape[1],
)
```

这与平台无关,在 NVIDIA 上走同一后端也会失败。

### 三、纯文本提示词构造(上游缺陷)

`lightllm/server/build_prompt.py`

`build_prompt_from_image_request` 无论请求是否携带图片,都把提示词包进多模态**列表**格式。而本模型自带的聊天模板(Qwen 系)对 `message.content` 调用 `.startswith()`:

```jinja
{%- if ns.multi_step_tool and message.role == "user" and
       not(message.content.startswith('<tool_response>') and ...) %}
```

模板既不遍历 content 列表、也不做类型判断,列表没有 `.startswith` 方法,Jinja 抛出 `'list object' has no attribute 'startswith'`,接口返回 HTTP 417。修复为:携带图片时保持列表格式,纯文本时直接传字符串。同样与平台无关。

脱离服务即可复现,直接对模型自带 tokenizer 调 `apply_chat_template`:

| content 形式 | 结果 |
|---|---|
| `[{"type": "text", ...}]`(上游原始写法) | `UndefinedError: 'list object' has no attribute 'startswith'` |
| `"a red apple"`(修复后写法) | 成功,得到 `<\|im_start\|>user\na red apple<\|im_end\|>...` |
| `[{"type": "image_url", ...}, {"type": "text", ...}]`(图生图必需) | 同样 `UndefinedError` |

第三行说明这个模板根本没有多模态输入能力,这是前述图生图不可用的第二道障碍。

## 注意力后端配置

模型自带配置里 `attn_type` 是 `flash_attn3`,它是 Hopper 架构特化实现,RDNA4 上不可用。替代品是 `torch_sdpa`,在 ROCm 上经 AOTriton 落到 Triton flash kernel,无需额外依赖。

**这一项不通过修改上游文件实现。** `attn_type` 没有环境变量或命令行覆盖入口,只能来自配置文件;但 `--x2v_gen_model_config` 是直接 `open()` 任意路径(`lightx2v/pipeline.py` 的 `set_infer_config_json`),因此把配置放在仓库外即可:

```
/target/configs/neopp_dense_rocm.json
```

该文件是 `configs/neopp/neopp_dense.json` 的副本,与上游原版逐行比对**只差 `attn_type` 一行**,键集完全一致。两个上游仓库的配置文件保持原样未动。

这样做的理由:改上游配置的默认值会扩大与上游的差异面,后续同步上游时产生无谓冲突,而且那不是缺陷修复、不该进上游。另外本次验证实际只用到 `neopp_dense.json` 一份配置,其余六份(8steps、fp8、三种并行、moe)都没有走到,改动它们等于提交未经验证的内容。

## 依赖处理

两个项目的依赖合集里有若干 CUDA 专属包,直接安装会破坏已有的 ROCm PyTorch。策略是**约束文件 + 过滤清单**双保险。

`rocm-constraints.txt` 锁死 ROCm 栈,使 pip 在冲突时直接失败而不是静默替换:

```
torch==2.10.0+rocm7.2.3.lw.git1a270074
torchvision==0.25.0+rocm7.2.3.git82df5f59
torchaudio==2.10.0+rocm7.2.3.git5047768f
triton==3.6.0+rocm7.2.3.git4ed88892
numpy==2.4.4
```

`rocm-requirements.txt` 是过滤后的依赖清单。被排除的 CUDA 专属包:`flashinfer-python`、`sgl-kernel`、`cuda_bindings`、`cupy-cuda12x`、`xformers`、`nixl`。另外 `cchardet` 在 Python 3.12 上编译不过,`numpy==1.25.1` 太旧不兼容 torch 2.10。

版本策略:LightLLM 把 99 个依赖全部精确锁定,其中行为敏感的库(transformers、rpyc、pydantic 等)保留上游版本;纯基础设施类(HTTP 栈、编码、日期处理)放开,因为部分上游锁定版本早于 Python 3.12 且与 numpy 2.x 冲突。两个项目版本冲突时以 LightLLM 为准(更严格)。

安装:

```bash
pip install -r rocm-requirements.txt -c rocm-constraints.txt
```

最终 152 个包(含传递依赖),ROCm 栈五个包版本未变。

### 清单之外还需补装的包

以下四个包在最初的过滤清单里被判为"本地不需要",但实际启动时会因无条件 import 而失败,必须补装:

| 包 | 触发点 |
|---|---|
| `redis` | `embed_cache/afs_utils.py` 在多模态视觉服务路径上无条件导入 |
| `av` | 视频编解码,导入链上无条件引用 |
| `decord` | 同上 |
| `uvloop` | HTTP 服务事件循环 |

## 启动

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LIGHTLLM_NEO_PREFILL_TRITON_BACKEND=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m lightllm.server.api_server \
  --model_dir /target/models/SenseNova-U1.5-8B-MoT \
  --enable_multimodal --enable_multimodal_x2i \
  --x2i_server_deploy_mode separate --x2i_server_used_gpus 1 \
  --tp 1 --visual_gpu_ids 1 --disable_cudagraph \
  --max_total_token_num 8192 --max_req_total_len 2048 \
  --batch_max_tokens 1024 --chunked_prefill_size 1024 \
  --x2v_gen_model_config /target/configs/neopp_dense_rocm.json \
  --host 127.0.0.1 --port 8080
```

加载约 16 分钟(生成分支约 5 分钟,理解分支约 10 分钟)。就绪标志是日志出现 `server start up ok`。

**两张卡即可,不需要三张。** `--visual_gpu_ids 1` 让视觉服务与生成分支共用第二张卡,详见"卡位分配"。

### 加载慢:提高并发度无效(已实测)

实测该分区(本地 NVMe + ext4)直读 4.4 GB/s、带缓存 14.1 GB/s,33 GB 权重理论上 7.5 秒读完;而实际每个 4.63 GiB 的分片耗时 74.8 秒,折合 63 MB/s,只用出磁盘能力的 1.4%。

理解分支的加载确实是串行的,并发度由环境变量控制且默认为 1:

```python
# lightllm/common/basemodel/layer_weights/hf_load_utils.py
worker = int(os.environ.get("LOADWORKER", 1))
with Pool(worker) as p:
    iterator = p.imap_unordered(partial_func, candidate_files, chunksize=1)
```

据此曾推断调高该值可大幅缩短启动。**实测结论是没有收益:**

| `LOADWORKER` | 8 个分片的加载方式 | 理解分支权重耗时 |
|---|---|---|
| 1(默认) | 串行,每片 74.8 秒 | 约 9 分 58 秒 |
| 8 | 并行,日志确认 `Loading model weights with 8 workers` | 10 分 09 秒 |

并发度提高 8 倍而总耗时不变,说明**瓶颈是八个进程共享的某个资源,不是并发度本身**。候选包括主机到设备拷贝共用同一条 PCIe 通道、safetensors 解析后的数据类型转换,以及页缓存未命中时的实际读取模式;本次未进一步定位。

因此 16 分钟启动暂时无解,`LOADWORKER=8` 可以设但不必期待效果。真要缩短,方向应是避免重复加载(如权重常驻或预转换为目标 dtype 的缓存),而非调并发度。

注意该值只影响理解分支。生成分支走 LightX2V 自己的加载器(`lightx2v/models/networks/base_model.py`),不读这个变量,实测约 5 到 6 分钟。

### 相对官方默认值的全部偏离

启动命令里每一项与官方默认不同的地方,以及原因:

| 项 | 官方默认 | 本次取值 | 性质 |
|---|---|---|---|
| `LIGHTLLM_NEO_PREFILL_TRITON_BACKEND` | `0`(用 fa3-neo) | `1`(用 Triton) | ROCm 必需 |
| 配置中的 `attn_type` | `flash_attn3` | `torch_sdpa` | ROCm 必需 |
| `--disable_cudagraph` | 关(启用 graph) | 开(禁用) | ROCm 必需 |
| `--x2i_server_deploy_mode` | `colocate` | `separate` | 单卡容量不足 |
| `--batch_max_tokens` | `16384 // dp` = 16384 | 1024 | 显存调优 |
| `--chunked_prefill_size` | `batch_max_tokens // 2` = 8192 | 1024 | 显存调优 |
| `--max_total_token_num` | 自动(按 `mem_fraction=0.9`) | 8192 | 显存调优 |
| `--max_req_total_len` | 16384 | 2048 | 显存调优 |
| `--visual_gpu_ids` | `None` | 1 | 与生成分支共卡 |
| `PYTORCH_ALLOC_CONF` | 未设置 | `expandable_segments:True` | 缓解碎片 |
| `--tp`、`--x2i_server_used_gpus` | 1、1 | 1、1 | 未改动 |

### 关键参数说明

**`LIGHTLLM_NEO_PREFILL_TRITON_BACKEND=1` 必须设置。** 这是两处注意力后端切换中容易被忽略的一处:

```python
# lightllm/models/neo_chat/layer_infer/transformer_layer_infer.py
_USE_TRITON_PREFILL = os.environ.get("LIGHTLLM_NEO_PREFILL_TRITON_BACKEND", "0")...
```

默认值 `0` 走 `fa3-neo`(FlashAttention 3 的定制版本),同样是 Hopper 特化、RDNA4 上不可用。它管的是**理解分支**的预填充注意力,与配置文件里 `attn_type` 管的**生成分支**注意力是两套独立代码,因此两处都要改,漏掉任何一处都起不来。`neo_chat_moe` 那份实现里有同名开关,行为一致。

**`--disable_cudagraph` 必须开启。** LightLLM 的 CUDA graph 捕获依赖 `cupy` 的 `UnownedMemory` 来构造无引用计数张量以避免捕获期显存碎片,而 `cupy` 在 PyPI 只有 CUDA 构建。核对过其全部调用点,均被 `torch.cuda.is_current_stream_capturing()` 或 `infer_state.is_cuda_graph` 条件包裹,该参数可完整绕开,对文生图无影响(graph 优化的是解码阶段,不是预填充)。

**`--x2i_server_deploy_mode separate` 是容量所迫,不是 ROCm 特有。** 官方默认 `colocate` 让两个分支共卡,而权重合计 32.66 GiB 超过单卡 31.86 GiB(详见"单卡为何不可行")。任何单卡容量低于此值的机器都需要改为分卡部署。

**`--batch_max_tokens 1024` 是避免启动期显存溢出的关键。** 启动时的试探推理按这个值分配激活。官方默认走 chunked 分支,取 `16384 // dp`,单数据并行下即 **16384**:

```python
# lightllm/server/api_start.py
if args.batch_max_tokens is None:
    args.batch_max_tokens = 16384 // args.dp
if args.chunked_prefill_size is None:
    args.chunked_prefill_size = args.batch_max_tokens // 2
```

按 2048 词元下实测的约 9 GiB(42 层中间量累积约 6.3 GiB、输出层 622 MB、softmax 同尺寸副本)线性外推,16384 词元需要约 18 GiB,叠加 17.58 GiB 权重后远超单卡的 31.86 GiB,**用默认值必然起不来**。降到 1024 后一次通过。

需要注意这个值只影响启动期的试探推理与预填充分块,不影响文生图的实际质量——提示词通常只有几十个词元。

### 卡位分配:两张卡即可

`separate` 模式下有三个服务要安置,但它们分属两根独立的轴,**最少两张卡就够**。

理解分支与生成分支共用一根轴,由 `x2i_server/manager.py` 的 `setup_devices` 按 `CUDA_VISIBLE_DEVICES` 顺序切分——前者取前 `tp × dp` 张,后者取紧随其后的 `--x2i_server_used_gpus` 张:

```python
llm_need_gpus = 0 if args.x2i_server_deploy_mode == "colocate" else args.tp * args.dp
x2i_need_gpus = args.x2i_server_used_gpus
if len(devices) < llm_need_gpus + x2i_need_gpus:
    raise ValueError(...)
```

`tp = dp = 1` 且 `--x2i_server_used_gpus 1` 时,这个校验只要求两张。

视觉服务走另一根轴 `--visual_gpu_ids`,它是 `CUDA_VISIBLE_DEVICES` 列表里的**逻辑下标**(不是卡号),与前两者**没有互斥校验**——`api_start.py` 仅检查下标个数不少于 `visual_dp × visual_tp`。把它指向第二张卡,就能与生成分支共用。

共用几乎不花代价。视觉编码器权重仅 4 个张量、0.03 GiB,即便单独占一张卡,实测本实例也只用掉约 0.78 GiB(其余是 HIP 上下文与分配器预留)。出图耗时实测:双卡 17.7 / 18.5 / 17.9 秒,三卡 17.5 / 17.6 / 17.6 秒,均值差约 2%,在噪声范围内。

**这个结论限于纯文生图。** 该路径上 ViT 基本不参与计算,只占显存。若要做图生图或图文理解,ViT 会真正投入运算,与生成分支争抢同一张卡可能引入排队延迟,那时分开更合适。

### 设备编号:两套编号并不一致

**这台机器上 HIP 的设备枚举顺序不是 PCI 顺序,而 `rocm-smi` 的卡编号是。** 也就是说,监控看到的卡号和选卡时写的序号是两套体系,必须换算。ROCm 没有 CUDA 那个 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 可用来强制对齐(HIP 运行时只识别 `HIP_VISIBLE_DEVICES` 与 `GPU_DEVICE_ORDINAL`)。

实测对照:

| `CUDA_VISIBLE_DEVICES` 里写的序号 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| PCI 地址(末段) | 63 | 43 | 03 | 23 | e3 | c3 | 83 | a3 |
| `rocm-smi` 中的卡 | card3 | card2 | card0 | card1 | card7 | card6 | card4 | card5 |

因此上面启动命令里的 `CUDA_VISIBLE_DEVICES=0,1` 实际拿到的是 **card3 与 card2**。若想真正使用 card0 与 card1,应写 `CUDA_VISIBLE_DEVICES=2,3`。

**这个映射不能照抄**,它取决于机器的 PCI 拓扑。每台机器都应现场核验:

```bash
# 一、rocm-smi 的卡 → PCI 地址
rocm-smi --showbus --csv

# 二、HIP 逻辑序号 → PCI 地址
python - <<'PY'
import ctypes
lib = ctypes.CDLL("libamdhip64.so")
n = ctypes.c_int(); lib.hipGetDeviceCount(ctypes.byref(n))
for i in range(n.value):
    buf = ctypes.create_string_buffer(64)
    lib.hipDeviceGetPCIBusId(buf, 64, i)
    print(i, buf.value.decode())
PY
```

两张表按 PCI 地址对接,即得换算关系。注意第二步会受当前 `CUDA_VISIBLE_DEVICES` 影响,核验时应清空该变量。

忽略这一点的后果不只是报告写错卡号:排查显存溢出时会盯着错误的卡看占用,从而误判是外部进程干扰还是自己的配置问题。本次移植中我就因此走了弯路。

### 显存实测

以启动命令中的 `CUDA_VISIBLE_DEVICES=0,1` 为例,按上表换算后实际落在 card3 与 card2。单卡总量 31.86 GiB,`rocm-smi` 读数如下:

| 卡 | 承载的服务 | 占用 | 余量 |
|---|---|---|---|
| card3 | 理解分支 | 27.04 GiB | 4.82 GiB |
| card2 | 生成分支 + 视觉服务 | 20.72 GiB | 11.14 GiB |

理解分支的构成是权重 17.58 + KV 缓存 1.31 + 激活,合计约 20 GiB;读数高出的部分是 PyTorch 缓存分配器在启动期试探推理(1024 词元约 4 GiB)之后保留未归还的显存,属正常行为。

作为对照,三卡布局(视觉服务独占一张)测得理解分支 26.79 GiB、生成分支 17.90 GiB;视觉服务所在卡读数 6.97 GiB,但其中仅约 0.78 GiB 属于本实例——停掉实例后该卡仍有 6.19 GiB,是同机其他负载。**读数必须减去外部占用后再归因**,否则会把别人的显存算到自己头上。

视觉服务并入生成卡后,生成卡由 17.90 升至 20.72 GiB。两个数来自不同实例,分配器预留有运行间差异,不宜精确相减;但量级上视觉服务的增量在 1 GiB 上下,远不值得为它单独留一张 31.86 GiB 的卡。

推理阶段只有生成分支所在的卡满载:理解分支处理提示词不到一秒,之后全程是扩散去噪。采样期间每 3 秒取样四次,card2 稳定 100%,card3 稳定 4%。

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

返回的 `data[i].b64_json` 是 base64 data URL,需去掉 `data:image/png;base64,` 前缀后解码。

### 采样参数由服务端配置决定,不接受按请求指定

请求体里的 `img_config` 声明为可选输入,实际会被请求校验器整体丢弃并重建:

```python
# lightllm/server/api_models.py  ImageGenerationRequest.parse_size
@model_validator(mode="after")
def parse_size(self):
    ...
    conf = get_default_image_config()          # 服务端默认(即启动时的配置文件)
    conf.update({
        "image_type": self.output_format,
        "height": height, "width": width,      # 来自 self.size,不是 img_config
        "seed": 42 + random.randint(0, 2**31-43),
    })
    self.img_config = ImageConfig(**conf)      # 用户传入的整个 img_config 被覆盖
    return self
```

因此各字段的实际行为:

| 参数 | 实际行为 |
|---|---|
| `img_config.steps` | 无效。步数取配置文件的 `infer_steps` |
| `img_config.seed` | 无效。被强制改为 `42 + 随机数`,故输出不可复现 |
| `img_config.width/height` | 无效。分辨率取顶层 `size` 字段 |
| `img_config.guidance_scale`、`cfg_norm` 等 | 无效。取配置文件值 |

实测验证:传 `size="1536x1536"` 得到 1536×1536;只传 `img_config` 宽高 1536 而省略 `size`,得到 1024×1024(即 `size` 的默认值)。

**这是设计使然,不是缺陷。** LightX2V 的 `LightX2VPipeline.generate()` 与 `NeoppRunner.set_inference_params()` 的签名里都没有步数参数——步数是 pipeline 初始化时从配置读取的实例级属性,框架本身不支持按次指定。而 LightLLM 侧 `get_default_image_config()` 的白名单恰好只有 `timestep_shift`、`cfg_interval`、`cfg_scale`、`cfg_norm` 四项,正是 `set_inference_params` 能接受的那几个,两边接口是对齐的。把步数绑在实例配置上也符合服务端的成本模型:8 步实例与 50 步实例的延迟和显存规划完全不同。

种子情形不同:下游 `pipe.generate(seed=...)` 是支持的,由 LightLLM 主动覆盖为随机值,目的是让同一提示词每次产出不同的图;OpenAI 自身的 images 接口同样不暴露 seed。

真正的不足只在接口清晰度:`ImageConfig` 声明了这些字段却在本路由上一律无效,且 neopp 无配套文档,极易误判。**尤其是 `steps`——向 50 步配置的实例请求 8 步,拿到的是 50 步的耗时**,本次移植中就曾因此误读了一组性能数据(见"性能"一节的说明)。

`/v1/chat/completions` 路径读的是 `request.image_config` 且未被重建,但步数同样无法生效,原因如上。

参数约束:`size` 形如 `1024x1024`,最终尺寸会经 `smart_resize` 按 32 对齐。`output_format` 限 `jpeg`/`png`/`webp`。

完整脚本见 `t2i_test.py`(通过环境变量 `T2I_PORT`、`T2I_TAG` 选择实例与输出标记)。

## 性能

1024×1024,两种配置各三次实测(热状态):

| 配置 | 步数 | NFE | 总耗时 |
|---|---|---|---|
| CFG 开,无 LoRA | 50 | 100 | 160.4 / 160.6 / 161.7 s |
| CFG 关,8 步蒸馏 LoRA | 8 | 8 | 17.6 / 17.6 / 18.6 s |

**8 步 LoRA 相对 50 步基准提速 9.1 倍。** NFE(前向传播次数)才是决定耗时的量:开启无分类器引导时每步跑两次前向,故 50 步是 100 NFE;蒸馏 LoRA 关闭引导,8 步即 8 NFE,两层因素相乘才有 12.5 倍的理论上限。

两组数据可解出耗时构成。设单次前向 t、与步数无关的固定开销 X:

```
100t + X = 161.0
  8t + X =  17.6
```

得 **t ≈ 1.56 秒,X ≈ 5.1 秒**(固定开销为 VAE 解码、图像编码与传输)。回代 50 步得 161.0 秒,与实测吻合。实测 9.1 倍略低于理论 12.5 倍,差距即来自这 5.1 秒固定开销在 8 步场景下占比升至 29%。

**首次请求会额外付出 Triton 内核即时编译开销。** 早期在全新缓存下实测为约 33 秒(首次 195.5 秒对热态 162.3 秒)。该缓存在同机不同实例间共享,故后续新实例的首次请求只需补编译差异部分,实测约 4.4 秒(22.0 秒对稳定 17.6 秒)。

> 早期报告曾把这项开销记为约 170 秒,那是因为当时误以为请求参数 `steps=8` 已生效、按 8 步折算每步耗时所致;实际该请求跑的是配置文件里的 50 步(见"调用方式"中的采样参数说明)。

### 分辨率几乎不影响耗时

8 步 LoRA 配置下实测三档分辨率:

| 分辨率 | 输出词元 | 耗时 | 锐度(拉普拉斯方差) | 同尺寸上采样对照 |
|---|---|---|---|---|
| 1024×1024 | 4096 | 17.6 s | 169.3 | — |
| 1536×1536 | 4096 | 17.8 s | 89.4 | 38.1 |
| 2048×2048 | 4096 | 18.0 s | 88.4 | 15.4 |

**像素量相差四倍而耗时几乎不变,原因是潜空间词元数恒定为 4096。** 决定扩散计算量的是词元数而非像素数;patch 边长随分辨率自适应放大(1024 对应 16 像素、2048 对应 32 像素),网格始终是 64×64。

高分辨率确实是原生生成,不是放大:2048 原生出图锐度 88.4,而把 1024 的图双三次插值到 2048 只有 15.4,相差近六倍。但细节密度不随分辨率提升——锐度由 1024 的 169.3 降到 88.4,因为每个 patch 要覆盖四倍像素。

实践含义:**高分辨率几乎是免费的**,想要大尺寸输出直接请求即可;但若目标是更丰富的细节,提高分辨率并不能达到。这也意味着高分辨率场景不需要多卡切分。

## 启用 8 步蒸馏 LoRA

官方已为 U1.5 发布 8 步蒸馏 LoRA,但它**不在基座模型仓库里**,而是单独一个仓库,容易漏找:

```bash
hf download sensenova/SenseNova-U1.5-8B-MoT-LoRAs \
  SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors \
  --local-dir /target/models/SenseNova-U1.5-8B-MoT-LoRAs
```

0.4B 参数、0.76 GB,与基座模型同在 2026-08-20 发布。

上游配置 `configs/neopp/neopp_dense_8steps.json` 里的 LoRA 路径是占位符 `/path/to/SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors`,**文件名指向上一代 U1**——该配置 2026-05-11 就为 U1 写好(同批提交的示例脚本里还留着开发者本机路径),U1.5 到 8 月才发布,配置未同步更新。留占位符让用户自行填写是 LightX2V 的惯例(QwenImage 的文档即如此说明),但 neopp 没有配套文档,所以这个约定无从得知。

本次使用的配置放在仓库外(`/target/configs/neopp_dense_8steps_rocm.json`),相对上游只改两处:`attn_type` 换成 `torch_sdpa`(ROCm 必需),LoRA 路径填实际路径。

键名无需任何映射即可对上。该 LoRA 为 STANDARD 格式(`.lora_up.weight`/`.lora_down.weight`),882 个张量对应 42 层 × 7 个投影 × 3 项,rank 128,每个键都带生成分支的 `_mot_gen` 后缀。`LoRALoader._remove_prefixes` 只剥离 `diffusion_model.`/`model.`/`unet.`,而这些键以 `language_model.` 开头不在其列,故推导出的模型键与基座权重键完全一致——预先比对为 294/294 命中,实际合并日志亦为 `Applied 294 LoRA weight adjustments out of 294 possible`,零条键不匹配警告。

合并发生在启动期而非推理期:有 `lora_configs` 且未开 `lora_dynamic_apply` 时,模型先加载权重但跳过 `_apply_weights`,待 LoRA 合并进 `original_weight_dict` 后再装入容器(该字典随后即被释放)。因此**LoRA 无法热加载,启用或停用都要重启服务**。

实测收益见"性能"一节:相对 50 步基准提速 9.1 倍(约 161 秒到约 17.6 秒),六次采样对比下出图质量相当,未出现 U1 那代文档提到的网格伪影。LoRA 版对比度与高光略强。

## 已知限制

**图生图不可用,有两道独立障碍。** 一是 LightLLM 的适配层无条件调用 `self.pipe.runner.set_kvcache_i2i()`,而 LightX2V 未实现该方法(图生图所需的三路无分类器引导尚未落地)。二是本模型的 chat template 只接受字符串形式的 content(详见下节),而携带图片必须用列表形式,故 prompt 构造这一步就已经通不过。两者都与平台无关。

**原生路由 `/generate_image` 不可用。** 该路由把 PNG 原始字节直接交给 `json.dumps`,必然报 `reject_bytes is on and '\x89PNG\r\n\x1a\n' is bytes`。这一缺陷同样与平台无关,改用 OpenAI 兼容路由即可。

**采样参数不能按请求指定。** 步数、种子、引导强度只能在启动时通过配置文件设定,切换需重启;种子被强制随机化,故输出不可复现。这是框架的分层设计而非缺陷,详见"调用方式"。

**"8 步且不加 LoRA"的质量未测。** 因步数属实例级配置,该组合需单独准备配置文件并重启服务才能测到,本次未做。因此无法量化蒸馏 LoRA 在质量上究竟补偿了多少。

**autotune 配置未针对 gfx1201。** 现有 JSON 均为 H200 特化,尚有提速空间。

## 从零复现

```bash
# 1. 取代码
git clone https://github.com/ModelTC/LightLLM.git && cd LightLLM
git checkout neo_plus_clean && git checkout -b gyydev/rocm-gfx1201
git clone https://github.com/ModelTC/LightX2V.git && cd LightX2V
git checkout -b gyydev/rocm-gfx1201

# 2. 装依赖(约束文件保护 ROCm 栈)
pip install -r rocm-requirements.txt -c rocm-constraints.txt
pip install redis av decord uvloop

# 3. 应用三处补丁(见"代码改动详解",或用 /target/patch-backup/ 下的补丁文件)
git am /target/patch-backup/lightllm/*.patch     # 在 LightLLM 内执行,2 个
git am /target/patch-backup/lightx2v/*.patch     # 在 LightX2V 内执行,1 个

# 3b. 准备 ROCm 配置(复制上游 neopp_dense.json,仅把 attn_type 改为 torch_sdpa)
mkdir -p /target/configs
sed 's/"flash_attn3"/"torch_sdpa"/' \
  /target/LightX2V/configs/neopp/neopp_dense.json > /target/configs/neopp_dense_rocm.json

# 4. 下载模型
hf download SenseNova/SenseNova-U1.5-8B-MoT \
  --local-dir /target/models/SenseNova-U1.5-8B-MoT

# 4b. (可选,推荐)下载 8 步蒸馏 LoRA 并准备对应配置,提速 9.1 倍
hf download sensenova/SenseNova-U1.5-8B-MoT-LoRAs \
  SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors \
  --local-dir /target/models/SenseNova-U1.5-8B-MoT-LoRAs
# 配置见 /target/configs/neopp_dense_8steps_rocm.json:在上游 neopp_dense_8steps.json 基础上
# 改 attn_type 为 torch_sdpa,并把 lora_configs 的占位符路径替换为上面下载的实际路径

# 5. 核验设备编号映射(见"设备编号"一节),挑两张空闲卡,据此决定 CUDA_VISIBLE_DEVICES 写什么
rocm-smi --showbus --csv

# 6. 启动(见"启动"一节);用 8 步 LoRA 则把 --x2v_gen_model_config 指向上面那份配置

# 7. 验证(第二、三个参数是宽高;步数由启动配置决定,请求无法指定)
python t2i_test.py "a red apple on a wooden table" 1024 1024
```

`verify_imports.py` 提供分层导入检查(13 项),可在启动前确认补丁生效与依赖完整。

## 排障

**启动期显存溢出。** 优先降 `--batch_max_tokens`(影响最大),其次 `--max_total_token_num`。若报错显示"PyTorch 已分配 XX GiB"而 XX 远小于卡容量,说明卡上有其他占用,不要盲目调参数。

**排查显存前先确认卡号换算。** 报错信息里的 "GPU 0" 是 HIP 逻辑序号,与 `rocm-smi` 的 card0 通常不是同一张卡(见"设备编号"一节)。盯着错误的卡看占用,会把自己的配置问题误判成外部进程干扰,或反之。

**务必先确认没有残留进程再重启。** 主进程崩溃后子进程可能存活并继续占用显存,累积几轮后能把一张卡占满。本次移植过程中有数轮显存溢出正是由此造成——在被自己污染的环境里反复调参,方向完全错误。重启前检查:

```bash
for p in /proc/[0-9]*; do
  cl=$(tr '\0' '\n' < "$p/cmdline" 2>/dev/null | head -1)
  case "$cl" in lightllm::*) echo "${p#/proc/} $cl" ;; esac
done
```

**判断服务是否就绪** 看日志中的 `server start up ok`,不要依赖 `ss -ltn`——容器内该命令可能看不到监听端口,而服务其实正常。

**确认成图正确性** 不能只看 HTTP 200。注意力 kernel 若有数值错误,通常表现为纯色或噪声画面。`t2i_test.py` 会检查解码后像素的灰度分布,纯色图会被标记。
