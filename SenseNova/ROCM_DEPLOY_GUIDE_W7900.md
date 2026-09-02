# SenseNova-U1.5-8B-MoT 文生图 ROCm 部署指南(AMD Radeon PRO W7900)

从零在一台装有 W7900 的 ROCm 机器上完成代码获取、依赖安装、模型下载、配置准备、启动与验证。按顺序执行即可,不需要参考其他文档。

---

## 0. 先读这一节:本文的可信度边界

本文的全部步骤**已在 AMD Radeon AI PRO R9700(gfx1201 / RDNA4)上实测跑通**,但 W7900 是 **gfx1100 / RDNA3**,是上一代架构。两者的差异集中在三处,文中凡涉及之处都会标注:

| 项 | R9700 (gfx1201) | W7900 (gfx1100) | 影响 |
|---|---|---|---|
| 单卡显存 | 31.86 GiB | 约 48 GiB | **单卡部署首次成为可能**,见第 7 节 |
| 原生 FP8 | 有 | **无**(FP8 自 RDNA4 才引入) | 量化压缩这条路在 W7900 上走不通 |
| Triton 内核成熟度 | 较新 | 支持更早、更成熟 | 预期只会更顺,但仍需实跑确认 |

两者的共同点是**都属于 RDNA、都没有 MFMA 指令**(那是 CDNA 独有),因此所有为"避开 Hopper / CDNA 特化实现"所做的配置调整在 W7900 上同样必要,一处都不能省。

标注约定:

- ✅ **已验证** —— 在 gfx1201 上实测通过,gfx1100 上极大概率一致
- ⚠️ **需现场确认** —— 与架构或本机环境相关,必须在 W7900 上重新核实

---

## 1. 环境核对

⚠️ **需现场确认。** 先确认基础环境,后面的依赖策略依赖这一步的输出。

```bash
# 显卡与架构:应看到 gfx1100
rocminfo | grep -m1 gfx
rocm-smi --showproductname

# 显存总量:记下这个数字,第 7 节要用
rocm-smi --showmeminfo vram

# ROCm 版本
cat /opt/rocm/.info/version

# PyTorch 是否为 ROCm 构建
python3 -c "import torch; print('torch', torch.__version__); print('hip', torch.version.hip); print('devices', torch.cuda.device_count())"
```

判断标准:

- `torch.version.hip` 必须非 `None`。若为 `None` 说明装的是 CPU 版或 CUDA 版,需先装 ROCm 版 PyTorch。
- `gfx1100` 是 ROCm 官方支持列表内的架构,通常无需设置 `HSA_OVERRIDE_GFX_VERSION`。**若某个内核报架构不支持再考虑**,不要预先设置,否则会掩盖真实问题。

参考环境(R9700 那台):ROCm 7.2.3、PyTorch 2.10.0+rocm7.2.3、Triton 3.6.0+rocm7.2.3、Python 3.12.3。W7900 那台不必完全一致,但 PyTorch 需 ≥ 2.5(`torch_sdpa` 的 `enable_gqa` 参数自 2.5 引入,见第 3 节补丁二)。

---

## 2. 克隆代码

✅ **已验证。** 两个仓库都需要,且都要切到打好补丁的分支。

```bash
mkdir -p /target && cd /target

git clone https://github.com/WhatGhost/LightLLM.git
cd LightLLM
git remote add upstream https://github.com/ModelTC/LightLLM.git
git checkout gyydev/rocm-gfx1201
cd ..

git clone https://github.com/WhatGhost/LightX2V.git
cd LightX2V
git remote add upstream https://github.com/ModelTC/LightX2V.git
git checkout gyydev/rocm-gfx1201
cd ..
```

**分支名里的 `gfx1201` 只是历史命名,内容与具体架构无关**,三处改动全是 ROCm 通用修复或平台无关的 bug 修复,在 gfx1100 上同样适用、同样需要。

分支基点(便于核对或自行重做):

| 仓库 | 上游基线分支 | 基点提交 | 补丁提交 |
|---|---|---|---|
| LightLLM | `neo_plus_clean` | `479f9b6` | `585ee06`、`4097515` |
| LightX2V | `main` | `6a3d442` | `0def796` |

### 如果你想自己打补丁而不用这个 fork

补丁文件可从 R9700 那台的 `/target/patch-backup/` 取得,共三个:

```bash
cd LightLLM && git checkout neo_plus_clean && git checkout -b my/rocm
git am /path/to/patch-backup/lightllm/*.patch      # 2 个

cd ../LightX2V && git checkout main && git checkout -b my/rocm
git am /path/to/patch-backup/lightx2v/*.patch      # 1 个
```

---

## 3. 三处代码改动是什么

✅ **已验证。** 已包含在上述分支里,此节仅供理解,无需手动操作。

**补丁一:主机内存注册运行时按平台解析**(LightLLM,`lightllm/utils/kv_cache_utils.py`)

上游硬编码 `/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so`,纯 ROCm 机器上必然加载失败。改为检测 `torch.version.hip` 后选择 `libamdhip64.so`。这是理解分支向生成分支传递 KV 缓存的必经路径,绕不开。

**补丁二:torch_sdpa 补上分组查询注意力广播**(LightX2V,`lightx2v/common/ops/attn/torch_sdpa.py`)

模型是 32 个查询头配 8 个键值头,`F.scaled_dot_product_attention` 需显式传 `enable_gqa=True` 才会广播。上游漏了,导致该后端在任何平台上都会报错。**这也是要求 PyTorch ≥ 2.5 的原因**。

**补丁三:纯文本图像请求传字符串 content**(LightLLM,`lightllm/server/build_prompt.py`)

上游无论请求是否带图都把提示词包成多模态列表,而本模型的聊天模板对 `message.content` 直接调 `.startswith()`,列表没有该方法,接口返回 HTTP 417。与平台无关。

---

## 4. 安装依赖

⚠️ **约束文件必须重新生成。** 这是换机器时最容易踩的坑。

两个项目的依赖合集里含 CUDA 专属包,直接 `pip install -r requirements.txt` 会用 CUDA 版 PyTorch 覆盖掉 ROCm 版。对策是约束文件锁死 ROCm 栈,让 pip 在冲突时**直接报错而不是静默替换**。

### 4.1 生成本机的约束文件

R9700 那台的 `rocm-constraints.txt` 锁的是那台机器的定制 wheel 版本串(形如 `2.10.0+rocm7.2.3.lw.git1a270074`),**在 W7900 那台必然不存在**,照抄会导致 pip 直接失败。请按本机实际版本重新生成:

```bash
cd /target
python3 - <<'PY' > rocm-constraints.txt
import importlib.metadata as md
print("# 锁死本机 ROCm 栈，禁止 pip 用 CUDA 构建覆盖")
for pkg in ("torch", "torchvision", "torchaudio", "triton", "numpy"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"# {pkg} 未安装")
PY
cat rocm-constraints.txt
```

若 `numpy` 版本低于 2.0,把该行改为 `numpy>=2.0`(PyTorch 2.10 需要 numpy 2.x)。

### 4.2 依赖清单

从 R9700 那台复制 `/target/rocm-requirements.txt`(89 行)。它是在两个项目的 requirements 合集基础上过滤而来,排除了以下 CUDA 专属包:

```
flashinfer-python    sgl-kernel    cuda_bindings
cupy-cuda12x         xformers      nixl
```

另外 `cchardet` 在 Python 3.12 上编译不过,`numpy==1.25.1` 与 torch 2.10 不兼容,两者也已剔除。

若无法复制,可自行生成:合并 `LightLLM/requirements.txt` 与 `LightX2V/requirements.txt`,删掉上述包,版本冲突时以 LightLLM 的为准(更严格)。行为敏感的库(transformers、rpyc、pydantic)保留上游锁定版本;纯基础设施类(HTTP 栈、编码、日期处理)可放开。

### 4.3 安装

```bash
cd /target
pip install -r rocm-requirements.txt -c rocm-constraints.txt

# 以下四个包在过滤清单里被判为不需要，但启动时会因无条件 import 而失败，必须补装
pip install redis av decord uvloop

# 最后以可编辑方式安装两个项目本身。--no-deps 不能省，原因见下
cd /target/LightLLM && pip install -e . --no-deps
cd /target/LightX2V && pip install -e . --no-deps
```

四个补装包的触发点:`redis` 来自 `embed_cache/afs_utils.py` 在多模态视觉服务路径上的无条件导入,`av` 与 `decord` 来自视频编解码导入链,`uvloop` 是 HTTP 服务的事件循环。

⚠️ **`--no-deps` 是整个安装流程里最容易翻车的地方。** 漏掉它,pip 会重新解析上游 `requirements.txt`——里面写着 `torch==2.9.1`——**又会把 CUDA 版 torch 拉回来覆盖 ROCm 版**,前面的约束努力当场作废。偏偏这一步发生在依赖都装完、看起来快完工的时候,最容易松懈。

### 4.4 装完立刻核对

```bash
python3 -c "import torch; assert torch.version.hip, 'PyTorch 被换成了非 ROCm 版本！'; print('ROCm PyTorch 完好:', torch.__version__)"
```

这一步必须做。若这里失败,说明约束没生效,后面全是白费功夫。

---

## 5. 下载模型

✅ **已验证。** 注意两个仓库的**大小写不同**,不是笔误。

```bash
# 基座权重，8 个分片共 32.66 GiB
hf download SenseNova/SenseNova-U1.5-8B-MoT \
  --local-dir /target/models/SenseNova-U1.5-8B-MoT

# 8 步蒸馏 LoRA（可选但强烈推荐，提速 9.1 倍），0.76 GiB
hf download sensenova/SenseNova-U1.5-8B-MoT-LoRAs \
  SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors \
  --local-dir /target/models/SenseNova-U1.5-8B-MoT-LoRAs
```

**LoRA 是纯加速手段,不是运行必需。** 基座仓库本身不含 LoRA,官方把它单独发在 `-LoRAs` 仓库。不加载 LoRA 走 50 步基准同样能出图,只是慢 9 倍。

关于权重构成:1116 个张量,42 层每层 26 个。带 `_mot_gen` 后缀的是生成分支,无后缀的是理解分支,两者共享词嵌入与输出投影。

---

## 6. 准备配置文件

✅ **已验证。** 关键原则:**配置文件放在两个仓库之外**,不修改上游文件。

`--x2v_gen_model_config` 是直接 `open()` 任意路径(见 `lightx2v/pipeline.py` 的 `set_infer_config_json`),所以外置完全可行,这样上游仓库保持干净,便于后续拉取更新。

```bash
mkdir -p /target/configs

# 50 步基准配置：在上游基础上只改 attn_type 一处
sed 's/"flash_attn3"/"torch_sdpa"/' \
  /target/LightX2V/configs/neopp/neopp_dense.json \
  > /target/configs/neopp_dense_rocm.json

cat /target/configs/neopp_dense_rocm.json
```

应得到:

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

8 步 LoRA 配置需改两处——`attn_type`,以及把上游那个指向**上一代 U1** 的占位符路径 `/path/to/SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors` 换成实际下载的 U1.5 权重:

```bash
cat > /target/configs/neopp_dense_8steps_rocm.json <<'JSON'
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
JSON
```

**为什么必须改 `attn_type`**:上游默认 `flash_attn3` 是 Hopper 架构特化实现,RDNA 上不可用。`torch_sdpa` 在 ROCm 上经 AOTriton 落到 Triton flash kernel,无需额外依赖。

---

## 7. 启动

### 7.1 双卡部署(推荐首选)

✅ **已验证。** 与 R9700 上跑通的布局一致,风险最低。先用这个确认全链路通,再去试单卡。

```bash
mkdir -p /target/logs && cd /target

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
  --x2v_gen_model_config /target/configs/neopp_dense_8steps_rocm.json \
  --host 127.0.0.1 --port 8080 \
  > /target/logs/lightllm.log 2>&1 &
```

卡位安排:理解分支占第一张,生成分支与视觉服务共用第二张。视觉编码器权重只有 0.03 GiB,共卡代价可忽略(实测耗时差约 2%)。

⚠️ **设备编号有坑。** HIP 的设备枚举顺序未必等于 `rocm-smi` 的显示顺序,`CUDA_VISIBLE_DEVICES=0,1` 拿到的**未必**是 `rocm-smi` 里的 card0、card1。ROCm 没有 CUDA 那个 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 可强制对齐。上机先核验:

```bash
# 一、rocm-smi 的卡 → PCI 地址
rocm-smi --showbus

# 二、HIP 逻辑序号 → PCI 地址（务必先清空 CUDA_VISIBLE_DEVICES）
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

两张表按 PCI 地址对接即得换算关系。忽略这一点的后果不只是记错卡号:排查显存问题时会盯着错误的卡看占用,把自己的配置问题误判成外部干扰。

### 7.2 单卡部署(48 GB 的新可能,值得一试)

⚠️ **未经验证,W7900 上首次尝试。** R9700 只有 31.86 GiB,单卡装不下(纯权重 32.66 GiB 就已超出),所以这条路当时无法验证。W7900 的 48 GB 有希望。

账面估算:

| 项 | 量级 |
|---|---|
| 权重 | 32.66 GiB |
| 启动期试探推理峰值 | 约 4 GiB(`batch_max_tokens=1024` 下) |
| 扩散过程激活 | 数 GiB |
| HIP 上下文 | 每进程约 0.3 GiB |
| **合计** | **约 40–45 GiB** |

W7900 若可用显存在 47 GiB 上下,余量偏紧但有机会。改动是把部署模式换成 `colocate`(两个分支共卡)并只给一张卡:

```bash
CUDA_VISIBLE_DEVICES=0 \
LIGHTLLM_NEO_PREFILL_TRITON_BACKEND=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m lightllm.server.api_server \
  --model_dir /target/models/SenseNova-U1.5-8B-MoT \
  --enable_multimodal --enable_multimodal_x2i \
  --x2i_server_deploy_mode colocate --x2i_server_used_gpus 1 \
  --tp 1 --visual_gpu_ids 0 --disable_cudagraph \
  --max_total_token_num 8192 --max_req_total_len 2048 \
  --batch_max_tokens 1024 --chunked_prefill_size 1024 \
  --x2v_gen_model_config /target/configs/neopp_dense_8steps_rocm.json \
  --host 127.0.0.1 --port 8080 \
  > /target/logs/lightllm_1card.log 2>&1 &
```

若显存溢出,依次尝试:把 `--batch_max_tokens` 与 `--chunked_prefill_size` 降到 512、把 `--max_total_token_num` 降到 4096。这些只影响启动期试探推理与预填充分块,**不影响文生图质量**——提示词通常只有几十个词元。

真的装不下就退回双卡,不必勉强。

### 7.3 加载时间

加载约 16 分钟(生成分支约 5 分钟,理解分支约 10 分钟)。就绪标志是日志出现 `server start up ok`。

```bash
tail -f /target/logs/lightllm.log | grep -E 'server start up ok|Applied [0-9]+ LoRA|out of memory|Traceback'
```

加载慢的原因是权重反序列化为单线程(实测只用出磁盘能力的 1.4%)。**注意:`LOADWORKER=8` 提高并发度实测无收益**(10 分 09 秒 vs 串行 9 分 58 秒),瓶颈不在并发度,别在这上面浪费时间。

LoRA 若加载成功,日志会出现 `Applied 294 LoRA weight adjustments out of 294 possible`。294/294 全中且零警告才算正常。

---

## 8. 验证

✅ **已验证。** 把下面的脚本存为 `/target/t2i_test.py`(或从 R9700 那台复制同名文件)。

```python
"""文生图冒烟测试。必须走 /v1/images/generations，原生 /generate_image 路由
对 PNG 字节做 JSON 序列化必然失败。"""
import base64, io, json, os, sys, time, urllib.request

PORT = os.environ.get("T2I_PORT", "8080")
URL = f"http://127.0.0.1:{PORT}/v1/images/generations"
OUT_DIR = "/target/outputs"
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

运行:

```bash
python /target/t2i_test.py "a red apple on a wooden table" 1024 1024
```

### 判断是否成功

| 检查项 | 期望 |
|---|---|
| HTTP 状态 | 200 |
| 灰度统计 | `min` 与 `max` 明显不等(纯色图说明注意力内核算错了) |
| 图像内容 | 肉眼确认与提示词相符 |
| 耗时(8 步 LoRA) | R9700 上约 17.6 秒,W7900 因架构不同会有出入 |
| 耗时(50 步无 LoRA) | R9700 上约 161 秒 |

⚠️ **首次请求会额外花约 170 秒做 Triton JIT 编译**,这是正常的,第二次起就没有了。测性能务必跑第二次。

### 调用接口时必须知道的两件事

**采样参数不能按请求指定。** 步数、种子、引导强度只能通过启动配置决定,请求体里的 `img_config` 会被 `ImageGenerationRequest.parse_size` 整体重建。这是框架的分层设计,不是缺陷——LightX2V 的推理接口本身就不接受按次指定步数。**分辨率必须走顶层 `size` 字段。**

**种子由服务端随机生成**,传入的 seed 会被覆盖,因此同一提示词多次采样结果不同。

---

## 9. gfx1100 上需要重新确认的项

⚠️ 以下几项在 gfx1201 上验证过,换架构后建议逐一确认。

**Triton 内核编译。** 涉及 `fused_qk_norm_3drope`(由 `use_triton_qknorm_rope: true` 启用)与理解分支预填充的 Triton 后端。gfx1100 的 Triton 支持比 gfx1201 更早更成熟,预期只会更顺。出图成功即等于这些内核全部验证通过,不必单独测。

**AOTriton 的 SDPA 后端。** `torch_sdpa` 在 ROCm 上依赖 AOTriton 提供 flash kernel。**这一项基本可以放心**:R9700 那台的 AOTriton 0.11.1 内核包里,`amd-gfx11xx` 与 `amd-gfx120x` 是并列存在的两套预编译镜像,gfx1100 有专门的内核,并非靠通用回退。上机可自行核对:

```bash
ls $(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib/aotriton.images/
# 期望看到 amd-gfx11xx
```

万一确实报"no kernel available",PyTorch 侧唯一相关的开关是 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`(用于放行实验性架构)。**没有**能强制回退到 math 后端的环境变量,那只能靠 `torch.nn.attention.sdpa_kernel()` 在代码里改。

**FP8 量化不可用。** RDNA3 没有原生 FP8,`float8_e4m3fn` 相关路径在 W7900 上无硬件加速。上游 `neopp_dense_fp8.json` 指定的 `fp8-sgl` 方案本就依赖 CUDA 专属的 `sgl_kernel`,在任何 ROCm 机器上都不可用。W7900 上这条路彻底关闭,**但 48 GB 本来也不需要靠量化压缩**。

**性能基准需重测。** R9700 上 8 步 LoRA 约 17.6 秒、50 步约 161 秒(单步约 3.21 秒)。W7900 的算力与带宽不同,数字会变,不要直接套用。

---

## 10. 已知限制

以下与平台无关,在 W7900 上同样存在。

**图生图不可用,有两道独立障碍。** 一是 LightLLM 适配层无条件调用 `set_kvcache_i2i()` 而 LightX2V 未实现;二是本模型的聊天模板只接受字符串形式的 content,携带图片必须用列表形式,prompt 构造这一步就通不过。

**原生路由 `/generate_image` 必然失败。** 它对 PNG 字节做 JSON 序列化,会报 `reject_bytes is on`。用 `/v1/images/generations`。

**autotune 配置未针对 RDNA。** 现有 JSON 均为 H200 特化,尚有提速空间,属可选优化项。

**分辨率对耗时几乎无影响。** 潜空间词元数恒为 4096,patch size 自适应缩放,所以 1024、1536、2048 耗时相近(R9700 上分别为 17.6 / 17.8 / 18.0 秒)。想出高分辨率图不需要额外代价。

---

## 11. 排错

**启动即显存溢出。** 先确认 `--batch_max_tokens 1024` 已设置。上游默认走 chunked 分支取 `16384 // dp` 即 16384,按实测线性外推需约 18 GiB 激活,叠加权重后必然溢出。降到 1024 可解。

**报错说 "GPU 0" 但你盯着 card0 看不出问题。** 报错里的编号是 HIP 逻辑序号,与 `rocm-smi` 的 card0 通常不是同一张卡,先做第 7.1 节的编号换算。

**重启前务必清理残留进程。** 主进程崩溃后子进程可能存活并继续占显存,累积几轮能把一张卡占满,然后你会在被自己污染的环境里反复调参、方向完全跑偏。

```bash
ps -eo pid,cmd | grep -E 'lightllm|api_server' | grep -v grep
# 优雅停止：对主进程发 SIGTERM，子进程会跟着退出
kill -TERM <主进程PID>
```

**出图是纯色。** 说明注意力内核算错了。检查配置里 `attn_type` 是否真的是 `torch_sdpa`,以及 `LIGHTLLM_NEO_PREFILL_TRITON_BACKEND=1` 是否设置。注意日志里打印的才是**实际生效值**:

```bash
grep 'lightllm get x2v config' /target/logs/lightllm.log
```

**HTTP 417。** 补丁三没生效,确认分支切换正确。

**pip 装完 PyTorch 变成 CUDA 版。** 约束文件没起作用。重做第 4.1 节,并确认 `pip install` 带了 `-c rocm-constraints.txt`。

---

## 附:与官方默认配置的全部偏离

| 项 | 官方默认 | 本方案 | 原因 |
|---|---|---|---|
| `LIGHTLLM_NEO_PREFILL_TRITON_BACKEND` | `0`(用 fa3-neo) | `1`(用 Triton) | ROCm 必需 |
| 配置中的 `attn_type` | `flash_attn3` | `torch_sdpa` | ROCm 必需 |
| `--disable_cudagraph` | 关 | 开 | ROCm 必需(cupy 只有 CUDA 构建) |
| `--x2i_server_deploy_mode` | `colocate` | 双卡时 `separate` | 单卡容量;W7900 上可尝试保持默认 |
| `--batch_max_tokens` | 16384 | 1024 | 显存调优 |
| `--chunked_prefill_size` | 8192 | 1024 | 显存调优 |
| `--max_total_token_num` | 自动 | 8192 | 显存调优 |
| `--max_req_total_len` | 16384 | 2048 | 显存调优 |
| `--visual_gpu_ids` | `None` | 双卡时 1 | 与生成分支共卡 |
| `PYTORCH_ALLOC_CONF` | 未设置 | `expandable_segments:True` | 缓解碎片 |

采样相关参数(步数、CFG、`timestep_shift`)全部保持官方默认,未做任何调整。

**注意力后端的切换不通过修改上游文件实现**,而是用仓库外的配置文件,两个上游仓库保持原样。

前三项标注"ROCm 必需"的是架构层面的硬约束,gfx1100 与 gfx1201 一样躲不开;显存调优那几项在 W7900 的 48 GB 上有放宽空间,可按第 7.2 节实际情况调整。
