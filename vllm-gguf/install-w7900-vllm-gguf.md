# 在另一台 AMD 机器上部署这套 vLLM + GGUF 插件

**目标**:在一台新的 AMD Radeon PRO W7900 机器上,把当前这套「vLLM + vllm-gguf-plugin」跑起来,
serve Muse-Glimmer-30B 的 GGUF 权重,并可选开启投机解码。

**前提**:新机器用的 Docker 镜像与本机一致(同一个 image)。所以镜像内的 OS / ROCm / hipcc / Python
自动就是对的,不需要你操心;但 **torch 栈和 vLLM 本身是镜像之外用 pip 装的**,必须照第 3 节重装一遍。

**代码从哪来**:两个仓库都在 GitHub 上、**都是公开的**,新机器上什么凭据都不用配,
匿名 `git clone` 即可。完整步骤见[第 2 节](#2-取代码两个仓库都用-gguf-kernel-perf-分支),
急用的话这张表就够:

| 仓库 | 分支 | 钉住的 commit |
|---|---|---|
| `https://github.com/WhatGhost/vllm-gguf-plugin.git` | `gguf-kernel-perf` | `a48163b89814bc56bc55c54981e9140428045c28` |
| `https://github.com/WhatGhost/vllm.git` | `gguf-kernel-perf` | `4b33d88c72f517b1ab7da3d8c6f43ee65bb5308e` |

**模型从哪来**:HuggingFace 的 `meta-models` 组织,三个仓库都公开非 gated,匿名可下。
**只下 GGUF,bf16 的 `.safetensors` 一个都不要**(约 60 GiB,用不到)。
需要下载的合计约 21 GiB,细节见[第 5 节](#5-模型文件从-huggingface-下载):

| 仓库 | 拿什么 |
|---|---|
| `meta-models/Muse-Glimmer-30B-GGUF` | 目标模型 + mmproj + draft 三个 GGUF(21.1 GiB) |
| `meta-models/Muse-Glimmer-30B` | 仅 6 个配置文件(27 MiB) |
| `meta-models/Muse-Glimmer-30B-assistant` | 仅 draft 的 `config.json`(883 B) |

阅读顺序:第 0 节必读(它决定了你对性能的预期),然后 2 → 3 → 4 → 5 → 6 顺序执行。

---

## 0. 必读:按默认装法,W7900 跑得通但拿不到本轮 MMQ 优化的收益

这是本文最重要的一节,别跳过。

W7900 是 **RDNA3 架构(gfx1100)**,而本轮做的 MMQ b10373 kernel 优化**当前只对 CDNA 开放**。

需要说清的是:**这是我们主动收紧的范围,不是硬件不支持**。CDNA 用的是 MFMA
指令(`v_mfma_i32_16x16x32_i8` 一类),RDNA3 上确实没有这条;但 RDNA3 有功能对应的
**WMMA** 指令(`v_wmma_*_iu8`),上游 b10373 也**已经支持**它。之所以只放开 CDNA,
是因为本轮只在 MI300X 上做了测量和阈值标定,不想让未验证的架构悄悄走新路径。
放开需要做什么,见 [0.2](#02-想在-rdna3-上真正启用需要做什么)。

门禁是**运行期**的,位于 `vllm_gguf_plugin/csrc/gguf/b10373/mmq_b10373_launcher.cu`:

```cpp
bool supported(QuantType type, int64_t n_tokens, int64_t k, int64_t n_rows) {
    ...
    return GGML_CUDA_CC_IS_CDNA(ggml_cuda_info().devices[id].cc);
}
```

非 CDNA 设备上 `supported()` 返回 false,插件会**自动回退到 b2899 的 dp4a 向量 ALU kernel**。
这个回退是设计好的、安全的:

| | 在 W7900(gfx1100)上 |
|---|---|
| 能否跑通 | 能,功能完整 |
| 输出正确性 | 不受影响,与 MI300X 一致 |
| 实际使用的 kernel | b2899 dp4a(旧路径) |
| 性能水平 | 相当于**优化前**,拿不到 `docs/opt-vllm-gguf-plugin.md` 里那些倍数 |
| `VLLM_GGUF_MMQ_B10373=1` 是否有意义 | 没有。它和架构门禁是**两道独立的关卡,必须都过**;置 1 只是「不主动禁用」,过不了架构那道 |

也就是说:**`docs/opt-vllm-gguf-plugin.md` 里记的加速比(并发 4 的 4.33×、并发 16 的 5.76×)
是 MI300X 的数,不要拿它作为 W7900 的验收标准。** W7900 上你应该期待的是「功能跑通」,
性能对标的是优化前的基线。

### 0.1 先按「跑通」装:构建时跳过 vendor kernel

下面 1–7 节走的是**先跑通**这条路。既然运行期一定会被门禁否掉,编它就是纯浪费
(而且能否在 gfx1100 上编译通过没有实测过——本机没有 RDNA3 卡)。插件预留了跳过开关:

```bash
export VLLM_GGUF_PLUGIN_NO_MMQ_B10373=1
```

带上它之后,插件只编 b2899 那套,构建更快、也绕开了编译风险。第 4 节的安装命令里已含这一条。

什么时候**不要**设它:

- 在 **MI300X / MI250(CDNA)** 机器上部署时。去掉它才能拿到本轮优化的收益,
  而且那边不需要任何代码改动,开箱即用。
- 想在 **W7900 上争取性能收益**时。这需要改代码,见 [0.2](#02-想在-rdna3-上真正启用需要做什么)。
  但建议先按 1–7 节把功能跑通、拿到 b2899 的基线数,再去动 kernel,
  否则出了问题分不清是哪一层的。

### 0.2 想在 RDNA3 上真正启用,需要做什么

**不需要再去 llama.cpp 里找 kernel 移植——RDNA3 的整条路径已经跟着本轮 vendor 的源码
一起进来了。** 逐环节核对如下(均为读码确认):

| 环节 | 现状 |
|---|---|
| RDNA3 的矩阵指令 | 有,WMMA(`v_wmma_*_iu8`) |
| `AMD_WMMA_AVAILABLE` 启用条件 | `common.cuh:269` — `GGML_USE_HIP && (RDNA4 \|\| RDNA3)`,我们的构建已设 `GGML_USE_HIP` |
| `RDNA3` 宏 | `vendors/hip.h:215` — `__GFX11__` 即定义,gfx1100 命中 |
| kernel 主体的 mma 路径 | 条件是 `AMD_MFMA_AVAILABLE \|\| AMD_WMMA_AVAILABLE`,**已含 RDNA3** |
| RDNA3 的 tile 配置 | `mmq-config-rdna3.cuh` 已在 vendor 目录,`mmq.cuh:240` 按 `cc` 运行期分派 |
| WMMA tile 原语 | `mma.cuh` 已 vendor |

所以要启用,改动集中在我们自己那几行:

1. **放开 `supported()` 的架构判断**(必需)。就是上面那一行,让它同时接受 RDNA3。
2. **重扫 MMVQ→MMQ 的切换阈值**(必需)。`quantization/linear.py` 里 `_mmvq_safe()`
   的阈值是 MI300X 上标定的。优化报告里有明确教训:**用错阈值比不改还慢**,
   所以这一步不能照搬,必须在 W7900 上重测。
3. **可选,为性能**:`dispatch_J` 目前只遍历 `{16, 32, 48, 64}`(按 CDNA 的合法 J 写的),
   而 RDNA3 的合法 J 到 128(`16/32/48/64/80/96/112/128`)。不改也**功能正确**——
   它会在这四档里挑最好的,只是少拿一点性能;`MMQ_B10373_J_MAX = 64` 的尾部余量
   也仍然成立(按实际选用的 J 算)。想吃满就把遍历和 `switch` 扩到 128,
   代价是静态库变大。

改完记得跑第 7 节第 8 条:在 gfx1100 上应从 `False` 变成 `True`。

---

## 1. 环境基线

同一镜像下,以下几项应当自动与本机一致。到新机器上先核一遍,不一致就说明镜像不是同一个:

```bash
cat /etc/os-release | grep -E '^(NAME|VERSION)='   # Ubuntu 24.04.4 LTS (Noble Numbat)
cat /opt/rocm/.info/version                       # 7.2.3
hipcc --version | grep -oE 'roc-[0-9.]+'          # roc-7.2.3
python3 -V                                        # Python 3.12.3
```

查 GPU 架构建议用 torch,它直接给出设备的 ISA,没有歧义:

```bash
python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, '|', p.gcnArchName)"
# 本机实测:AMD Instinct MI300X | gfx942:sramecc+:xnack-
# 新机器应为:AMD Radeon PRO W7900 | gfx1100
```

用 `rocminfo | grep -oE 'gfx[0-9a-z]+' | sort -u` 也行,但**它会多打出一个 `gfx9`**,
那不是第二块卡,而是 `amdgcn-amd-amdhsa--gfx9-4-generic` 这个通用 ISA 目标被正则
在连字符处截断的结果。认准完整那个(本机 `gfx942`)。

**装东西之前,先确认 Python 环境是对的。** 本镜像的包不装在系统 Python 里,而是在
`/opt/venv`,靠镜像的 `PATH` 生效(`VIRTUAL_ENV` 是空的,**不需要**手动 `activate`):

```bash
which python python3 pip vllm
# 期望四个都在 /opt/venv/bin/ 下
python3 -c "import sys; print(sys.prefix)"        # 期望 /opt/venv
```

如果它们指向 `/usr/bin`,说明 `PATH` 没带上 venv(常见于 `sudo`、`cron`、非登录 shell)。
**这时候千万别继续装**——包会落到系统 Python,而 `vllm serve` 用的是 venv 里那套,
表现是「明明装了却 import 不到」。先把 `/opt/venv/bin` 放到 `PATH` 最前面。

以下几项**不在镜像里**,是第 3 节要装的,列出来作为装完之后的核对目标:

| 包 | 本机版本(目标) |
|---|---|
| vllm | `0.27.2rc1.dev122+g8efa13b70.rocm723` |
| torch | `2.12.0+git6bbd260` |
| torchvision | `0.27.1+df56172` |
| torchaudio | `2.11.0+34c52a6` |
| triton | `3.7.1+git532137f0` |
| flash_attn | `2.8.3` |
| amd-aiter | `0.1.19` |
| amdsmi | `26.2.2+c2d9476115` |
| vllm-gguf-plugin | `0.0.5`(editable,指向源码目录) |

好消息:这一整套 torch 栈来自同一个 ROCm nightly 索引,彼此自洽,一条 pip 命令就能装齐。

torch 官方 wheel 的架构列表里**包含 gfx1100**,所以基础栈在 W7900 上是支持的:

```bash
python3 -c "import torch; print(torch.cuda.get_arch_list())"
# 本机实测:['gfx90a', 'gfx942', 'gfx950', 'gfx1100', 'gfx1101',
#            'gfx1200', 'gfx1201', 'gfx1150', 'gfx1151']
#                        ^^^^^^^^ W7900 在列
```

---

## 2. 取代码:两个仓库,都用 `gguf-kernel-perf` 分支

新机器上不需要预先准备任何代码,这一节从零开始拉。两个仓库都是 fork,**分支名相同
但远端地址不同**,别搞混:

| 用途 | 仓库地址 | 分支 | 钉住的 commit |
|---|---|---|---|
| GGUF 插件(本轮 kernel 优化在这) | `https://github.com/WhatGhost/vllm-gguf-plugin.git` | `gguf-kernel-perf` | `a48163b89814bc56bc55c54981e9140428045c28` |
| vLLM 源码(投机解码单文件支持) | `https://github.com/WhatGhost/vllm.git` | `gguf-kernel-perf` | `4b33d88c72f517b1ab7da3d8c6f43ee65bb5308e` |

两个仓库**都是公开的**,匿名就能拉,新机器上不用配 token 或 SSH key
(实测:清空凭据与全局配置后 `git ls-remote` 仍可访问)。

本文假设你把它们放在 `/target` 下,与本机布局一致(路径写死在若干脚本里,换路径要一起改)。
注意 vLLM 的目标目录名是 **`vllm-src`**,不是默认的 `vllm`,后面的脚本认这个名字。

### 2.1 推荐:按 commit 钉死

分支以后还可能往前走,而上表那两个 commit 才是本文全部结论对应的代码。
下面这种拉法**只下载指定 commit**,既钉得准,又比完整克隆省得多
(实测 vLLM 158 MB、插件 2.7 MB,两个加起来几秒钟):

```bash
mkdir -p /target && cd /target

# 插件
git init vllm-gguf-plugin && cd vllm-gguf-plugin
git remote add origin https://github.com/WhatGhost/vllm-gguf-plugin.git
git fetch --depth 1 origin a48163b89814bc56bc55c54981e9140428045c28
git checkout FETCH_HEAD
cd /target

# vLLM 源码(仅用于源码覆盖,不编译)
git init vllm-src && cd vllm-src
git remote add origin https://github.com/WhatGhost/vllm.git
git fetch --depth 1 origin 4b33d88c72f517b1ab7da3d8c6f43ee65bb5308e
git checkout FETCH_HEAD
cd /target
```

`git checkout FETCH_HEAD` 之后是 detached HEAD,这里**是想要的**:你要的就是这个固定
快照,不是一条会漂移的分支。想挂个名字方便看,可以 `git switch -c gguf-kernel-perf`。

浅克隆和 detached HEAD 都**不影响后面的构建**:插件的版本号取自 `pyproject.toml` 里的
静态字段(`base_version = "0.0.5"`),`setup.py` 全程不碰 git,所以装出来仍然是
第 4 节写的 `0.0.5`。

### 2.2 备选:直接按分支克隆

只想拿分支最新状态、不在意钉死的话:

```bash
mkdir -p /target && cd /target
git clone -b gguf-kernel-perf https://github.com/WhatGhost/vllm-gguf-plugin.git
git clone -b gguf-kernel-perf --depth 50 https://github.com/WhatGhost/vllm.git vllm-src
```

代价是:**如果分支之后有了新提交,你拿到的就不是本文验证过的那份代码**,
第 2.3 节的核对会对不上。真出现这种情况,按 2.1 回到钉住的 commit。

### 2.3 核对拿对了没有

只看 commit 号还不够(浅克隆、改名都可能骗过眼睛),连内容一起核:

```bash
# commit 号
git -C /target/vllm-gguf-plugin log --oneline -1   # a48163b [Kernel] Route Q4_K/Q5_K/Q6_K ...
git -C /target/vllm-src        log --oneline -1   # 4b33d88 [Spec Decode] Support single-file ...

# 内容:draft 侧的 hf_config_path 必须在(投机解码全靠它,见 3.4)
grep -c hf_config_path /target/vllm-src/vllm/config/speculative.py          # 期望 3

# 内容:b10373 的 vendor 源码必须完整
ls /target/vllm-gguf-plugin/vllm_gguf_plugin/csrc/gguf/b10373/ | wc -l      # 期望 32
```

上面三条的期望值都是在一份全新克隆上实测出来的。

---

## 3. 装 vLLM:nightly wheel + 源码覆盖(**不从源码编译**)

### 3.1 先理解方案,否则第 3.3 步会看不懂

这套环境**没有**用常规的 `pip install -e .` 装 vLLM。原因很实际:在 ROCm 上从源码编 vLLM
等于重编全部 HIP kernel,要花很久,而我们对 vLLM 的改动**只有纯 Python 文件**,重编毫无必要。

所以用的是「wheel 提供二进制 + 源码覆盖 Python」的组合:

1. 用 pip 装 ROCm nightly 的 vLLM wheel → 得到编译好的 `.so` 扩展和全套依赖;
2. 把 fork 源码克到 `/target/vllm-src`;
3. 在源码树里**软链**回 wheel 的那些 `.so`(源码树里没有二进制,不链就 import 不了);
4. 起服务时 `PYTHONPATH=/target/vllm-src`,让源码树的 Python 文件**盖过** site-packages 的。

> **这个方案成立的前提**:fork 相对 wheel 的差异必须是纯 Python。
> 我们那个提交(`4b33d88c7`)确实只改 `vllm/config/speculative.py` 一类 Python 文件,所以可行。
> 将来如果 fork 里出现了 C++/HIP 改动,**这条路就不能用了**,必须老老实实从源码编。

### 3.2 装 nightly wheel

```bash
python -m pip install vllm --pre \
    --extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm723 --upgrade
```

**这条命令会把整个 torch 栈一起换掉**(torch / torchvision / torchaudio / triton),
这是正常的,它们来自同一个索引、彼此自洽。建议先加 `--dry-run` 看清要动什么:

```bash
python -m pip install vllm --pre \
    --extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm723 \
    --upgrade --dry-run --report /tmp/pipreport.json
```

**nightly 会被清理,版本对不上怎么办?** 本机装的是 `dev122+g8efa13b70.rocm723`。
nightly 索引只保留近期构建,过一段时间这个版本就没了。先查索引上还有什么:

```bash
curl -sSL https://wheels.vllm.ai/rocm/nightly/rocm723/vllm/ \
  | grep -oE 'vllm-[^"<]*\.whl' | sort -u | tail -5
```

- 若 `dev122+g8efa13b70` 还在 → 显式钉住它,与本机完全一致:
  ```bash
  python -m pip install --pre --extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm723 \
      'vllm==0.27.2rc1.dev122+g8efa13b70.rocm723'
  ```
- 若已被清理 → 装索引上的当前版本即可。vLLM 主线在动,装新版通常没问题,
  但**装完务必重跑 3.3 的软链和第 7 节的验证**:新 wheel 的 `.so` 清单可能变化,
  且源码树(`4b33d88c7`)与更新的 wheel 之间可能出现真实的 API 漂移。
  真要严格复现,就把本机这一版 wheel 拷过去装:

  ```bash
  # ① 在【本机】下载 wheel(只下不装)。--no-deps 只拿 vllm 本体
  python -m pip download --no-deps --pre \
      --extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm723 \
      'vllm==0.27.2rc1.dev122+g8efa13b70.rocm723' -d /target/wheel-snapshot

  # ② 拷到新机器,例如
  #    rsync -avP /target/wheel-snapshot/ user@new-machine:/target/wheel-snapshot/

  # ③ 在【新机器】上装。依赖仍从 nightly 索引取,保证 torch 栈配套
  python -m pip install --pre \
      --extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm723 \
      /target/wheel-snapshot/vllm-*.whl
  ```

### 3.3 在源码树里软链回 wheel 的二进制

源码树里只有 `.py`,没有编译产物。缺了这些软链,`PYTHONPATH` 一指过去就会 import 失败。

需要链的一共 10 项:8 个 `.so`、Rust 前端目录 `vllm-rs`、以及构建期生成的 `_version.py`。
下面这段会自动枚举,不用手抄清单:

```bash
SP=$(python -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "vllm"))')
cd /target/vllm-src/vllm

for item in "$SP"/*.so "$SP"/vllm-rs "$SP"/_version.py; do
    [ -e "$item" ] || continue
    base=$(basename "$item")
    [ -e "$base" ] && continue        # 源码树里已有同名文件就别覆盖
    ln -s "$item" "$base"
done

# 核对:应列出 10 个软链
find /target/vllm-src/vllm -maxdepth 1 -type l | sort
```

本机实测的 10 项:

```
_C.abi3.so                      _rust_tool_parser.abi3.so    spinloop.abi3.so
_C_stable_libtorch.abi3.so      cumem_allocator.abi3.so      vllm-rs
_moe_C_stable_libtorch.abi3.so  fs_io_C.abi3.so              _version.py
_rocm_C.abi3.so
```

验证源码覆盖真的生效了:

```bash
PYTHONPATH=/target/vllm-src python -c "import vllm; print(vllm.__file__)"
# 期望:/target/vllm-src/vllm/__init__.py   ← 走源码树
# 若打印 site-packages 路径,说明 PYTHONPATH 没起作用
```

### 3.4 为什么非要做这个覆盖:draft 侧的 `hf_config_path`

如果你只想 serve 目标模型、**不开投机解码**,那其实用不着覆盖,装完 wheel 直接跑就行——
`--hf-config-path` 是 vLLM **原生**参数(wheel 版 `vllm/config/model.py` 里就有)。

但**一旦要开投机解码**,`--speculative-config` 里的 `hf_config_path` 键是我们那个提交加的,
wheel 里没有。实测对照:

```bash
SP=$(python -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "vllm"))')
grep -c hf_config_path "$SP/config/speculative.py"                  # wheel  → 0
grep -c hf_config_path /target/vllm-src/vllm/config/speculative.py  # 源码树 → 3
```

draft 是**单文件 GGUF**,它的 config 必须靠 `hf_config_path` 单独指路。
所以:**投机解码 = 必须走 `PYTHONPATH=/target/vllm-src`**,否则启动时该键会被拒。

而且插件这边**已经没有退路了**:插件分支上的 `ae13af7`
(*Drop the GGUF draft workarounds now that vLLM covers them*)把它原先在用户态兜底的
draft monkeypatch 整套删掉了,理由正是 fork 版 vLLM 已原生覆盖。所以「旧插件配新 wheel」
这种组合是跑不通 draft 的,**别试**。

对照之下,目标模型侧不受影响:vLLM 原生的 `create_model_config` 会把 `model` 改写到 config
目录、把 GGUF 文件留在 `model_weights`,单文件目标模型本来就是白拿的。

### 3.5 一个会骗人的地方:版本串不代表实际代码

`_version.py` 是软链到 wheel 的,所以即使你走的是源码树,`vllm.__version__` 打印出来
依然是 wheel 的版本串(`0.27.2rc1.dev122+g8efa13b70`),**看不出源码树其实是 `4b33d88c7`**。

排查时别信版本串,直接查文件内容(如 3.4 那两条 `grep -c`),或者查 `vllm.__file__`。

---

## 4. 装 GGUF 插件

插件是**真的要编译**的(它有自己的 HIP kernel),用 editable 模式装,以后改代码不用重装。

```bash
cd /target/vllm-gguf-plugin

# W7900(RDNA3)必须带这个变量,理由见第 0.1 节
VLLM_GGUF_PLUGIN_NO_MMQ_B10373=1 \
  python -m pip install -e . --no-build-isolation
```

三个要点:

- **`--no-build-isolation` 不能省**。插件要链接**已装的那个 torch**;开了构建隔离,
  pip 会在临时环境里拉一个别的 torch 来编,编出来的 `.so` ABI 对不上。
- **`VLLM_GGUF_PLUGIN_NO_MMQ_B10373=1`**:跳过 CDNA 专用的 vendor kernel 编译。
  在 MI300X 上部署时去掉它。(该变量认 `1` / `true` / `TRUE`。)
- **编译时 GPU 必须可见**。插件走 torch 的 `_get_rocm_arch_flags()` 定目标架构,而它在
  `PYTORCH_ROCM_ARCH` 未设时是**按当前可见显卡自动探测**的(本机实测探到 `gfx942`,
  并会打一行提示)。所以别在看不到卡的容器里编,否则架构探测没有依据。
  想显式钉住就设一下,W7900 上是:

  ```bash
  export PYTORCH_ROCM_ARCH=gfx1100
  ```

  顺带说明:因为是按可见卡探测、不是按 torch 支持的全部 9 个架构编,所以编译时间不会因此翻倍。

若嫌 pip 慢,插件 README 推荐 uv,等价命令:

```bash
VLLM_GGUF_PLUGIN_NO_MMQ_B10373=1 uv pip install -e . --no-build-isolation
```

装完核对(应显示 editable,Location 指向源码目录):

```bash
python -m pip list | grep vllm-gguf-plugin
# 期望:vllm-gguf-plugin  0.0.5  /target/vllm-gguf-plugin
```

---

## 5. 模型文件:从 HuggingFace 下载

模型全部来自 HuggingFace 的 `meta-models` 组织,**三个仓库都是公开的、非 gated**,
匿名就能下。要下的是 **21.1 GiB 的 GGUF 权重 + 27 MiB 的配置文件**;bf16 的
`.safetensors`(三个仓库合计约 60 GiB)**一个都不要**,权重全部来自 GGUF。

### 5.1 要下哪些文件

**① GGUF 权重** —— `meta-models/Muse-Glimmer-30B-GGUF`。本机这份就是下面 3 个,全都要:

| 文件 | 字节数 | 用途 |
|---|---|---|
| `Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf` | 19,653,960,832(18.30 GiB) | 目标模型 |
| `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` | 1,400,328,928(1.30 GiB) | 视觉塔 |
| `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` | 1,631,208,128(1.52 GiB) | draft 模型,开投机解码才用 |

**三个必须落在同一个目录,而且别改名。** mmproj 不是靠命令行参数传进去的:插件在
目标 GGUF 所在目录里按 `mmproj.gguf` → `mmproj-*.gguf` → `*mmproj*.gguf` 通配符找它
(`vllm_gguf_plugin/gguf_utils.py` 的 `detect_gguf_multimodal`),llama.cpp 基线侧则是
显式 `--mmproj`(`run_matrix.sh:113`)。所以启动命令里看不到它,但缺了它视觉路径就不成立
——目标 GGUF 的 731 个张量全是语言模型,视觉塔那 809 个 `v.blk.*` 只在 mmproj 里。

仓库里另有一个 `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`(另一档量化),本机没下过、
本文也没测过。

**② 目标模型的配置文件** —— `meta-models/Muse-Glimmer-30B`,只要这 6 个,合计 27 MiB:

| 文件 | 字节数 | 用途 |
|---|---|---|
| `config.json` | 5,109 | 模型结构,含 `text_config` 与 `vision_config` |
| `tokenizer.json` | 28,129,897 | tokenizer 本体 |
| `tokenizer_config.json` | 79,936 | tokenizer 配置 |
| `chat_template.jinja` | 9,992 | chat 接口的模板 |
| `generation_config.json` | 202 | 默认采样参数,vLLM 启动时会读并覆盖自己的默认值 |
| `processor_config.json` | 1,084 | 多模态 processor |

这 6 个就是 `--hf-config-path` 和 `--tokenizer` 要读的全部内容,实测能把 config、
tokenizer、多模态 processor 都正常加载起来。仓库里其余的 `LICENSE` / `README.md` /
`USAGE_POLICY.md` / `.gitattributes` 用不到;`model.safetensors.index.json` 也**不用**
(它只负责把张量名映射到 safetensors 分片,而我们不从 safetensors 取权重)。

**③ draft 的配置文件** —— `meta-models/Muse-Glimmer-30B-assistant`,只要 `config.json`
(883 字节),开投机解码时由 `--speculative-config` 里的 `hf_config_path` 指向它。

### 5.2 下载命令

```bash
mkdir -p /target/models

# ① GGUF 权重:3 个,同一目录
hf download meta-models/Muse-Glimmer-30B-GGUF \
    --include "Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf" \
    --include "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf" \
    --include "dflash-Muse-Glimmer-30B-Q4_K_M.gguf" \
    --local-dir /target/models/Muse-Glimmer-30B-GGUF

# ② 目标模型的配置文件:6 个
hf download meta-models/Muse-Glimmer-30B \
    --include "config.json" \
    --include "tokenizer.json" \
    --include "tokenizer_config.json" \
    --include "chat_template.jinja" \
    --include "generation_config.json" \
    --include "processor_config.json" \
    --local-dir /target/models/Muse-Glimmer-30B

# ③ draft 的配置文件:1 个
hf download meta-models/Muse-Glimmer-30B-assistant \
    --include "config.json" \
    --local-dir /target/models/Muse-Glimmer-30B-assistant
```

**每个文件都得自己带一个 `--include`**,别写成 `--include "a" "b"`。`hf download` 的
位置参数是「显式文件名」,第二个值会被当成位置参数,于是 `--include` 被整个忽略
(实测会打 `Ignoring --include since filenames have been explicitly set.`,结果只下到
一个文件,而且不是你想要的那个)。

匿名下载会警告限速(`Please set a HF_TOKEN ...`)。21.1 GiB 嫌慢就 `export HF_TOKEN=...`
再下。

### 5.3 校验

GGUF 的 sha256 已与 HF 上当前 `main` 对过,**与本文压测所用的文件逐字节一致**:

```bash
cd /target/models/Muse-Glimmer-30B-GGUF && sha256sum -c <<'EOF'
ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38  Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf
f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6  mmproj-Muse-Glimmer-30B-Q4_K_M.gguf
b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef  dflash-Muse-Glimmer-30B-Q4_K_M.gguf
EOF
```

配置文件齐不齐(7 个都得在,缺一个就报 `No such file`):

```bash
ls /target/models/Muse-Glimmer-30B/{config,tokenizer,tokenizer_config,generation_config,processor_config}.json \
   /target/models/Muse-Glimmer-30B/chat_template.jinja \
   /target/models/Muse-Glimmer-30B-assistant/config.json
```

### 5.4 备选:从本机直接同步

已经有一份的话,拷过去比重下 21.1 GiB 快:

```bash
# 在新机器上执行,把 SRC 换成本机地址
SRC=user@old-machine:/target/models

mkdir -p /target/models/Muse-Glimmer-30B{,-GGUF,-assistant}

# ① GGUF 权重:3 个
rsync -avP --include='*Q4_K_XL.gguf' --include='mmproj-*.gguf' \
    --include='dflash-*.gguf' --exclude='*' \
    "$SRC/Muse-Glimmer-30B-GGUF/" /target/models/Muse-Glimmer-30B-GGUF/
# ② 目标模型的配置文件:6 个(--exclude='*' 兜底,把 safetensors 挡在外面)
rsync -avP --include='config.json' --include='tokenizer.json' \
    --include='tokenizer_config.json' --include='chat_template.jinja' \
    --include='generation_config.json' --include='processor_config.json' \
    --exclude='*' "$SRC/Muse-Glimmer-30B/" /target/models/Muse-Glimmer-30B/
# ③ draft 的配置文件:1 个
rsync -avP "$SRC/Muse-Glimmer-30B-assistant/config.json" \
    /target/models/Muse-Glimmer-30B-assistant/
```

走这条路也请跑一遍 5.3 的校验:20 GiB 级别的传输出错不算罕见。

---

## 6. 起服务

### 6.1 先算显存:W7900 是 48 GB,本机 MI300X 是 192 GB

这是 W7900 上第二个需要调整的地方(第一个是第 0 节的 kernel 门禁)。

**别用「层数 × KV 头数」那套公式手推**:这个模型 `text_config` 里有 `sliding_window = 2048`,
大部分层是滑窗注意力,不为全上下文存 KV,按全注意力推会高估一倍多。直接用本机日志里的实测值:

| 档位 | 权重占用 | KV 池 | KV 池容量 | 折合每 token |
|---|---|---|---|---|
| 关投机解码 | 24.41 GiB | 142.07 GiB | 6,376,101 token | 23.4 KiB |
| 开投机解码(n=8) | 26.31 GiB | 140.03 GiB | 4,742,778 token | 31.0 KiB |

(以上为 MI300X、`--gpu-memory-utilization 0.9`、`--max-model-len 131072`、`--max-num-seqs 16`
的启动日志实测。注意**权重占用 24.41 GiB 明显大于目标 GGUF 文件的 18.30 GiB**——文件大小不等于
显存占用,别拿文件大小估。差额来自两处:一是同目录被自动发现的 mmproj 视觉塔也进了显存
(见 5.1),日志里 `MMEncoderAttention` 初始化、并按 4 个图像项预留了 16384 token 的
encoder cache;二是 F32/BF16 张量与运行时反量化的部分不按量化体积算。draft 再加约 1.9 GiB。)

反推出本机的固定开销:`172.8(=192×0.9) − 24.41 − 142.07 ≈ 6.3 GiB`,
这部分是 activation 与 profiling 占的。据此估 W7900:

```
预算       = 48 GiB × 0.9                    ≈ 43.2 GiB
关投机:KV = 43.2 − 24.41 − 6.3              ≈ 12.5 GiB → 约 56 万 token
开投机:KV = 43.2 − 26.31 − 6.5              ≈ 10.4 GiB → 约 35 万 token
```

(显卡标称的「48 GB」就是 48 GiB,不用换算:本机 MI300X 标称 192GB,`torch` 报
206141652992 字节,正好 192.0 GiB。`du -h` 报的 GiB 同理。)

结论:**`--max-model-len 131072` 在 W7900 上放得下**——关投机约合 4 条满长序列,
开投机约合 2.7 条,都远大于 1。可以先照原样起。但余量比 MI300X 小得多
(那边是 48 条和 36 条),如果启动时报 KV cache 容量不足,按下面的顺序调
(**按对症程度排的,不是按幅度**):

1. **降 `--max-model-len`**,131072 → 32768 或你实际需要的长度。
   这条最对症:该报错的字面意思就是「`max_model_len` 超过了 KV 池能装的 token 数」。
2. **调高 `--gpu-memory-utilization`**,0.9 → 0.93 左右。直接把 KV 池做大,
   但留给 activation 的余量会变小,调过头会在跑高并发时 OOM。
3. **降 `--max-num-seqs`**,16 → 8 或 4。注意它**不直接决定 KV 池大小**
   (池子是按空闲显存分配的),作用是压低 activation 峰值、间接给池子腾一点空间,
   所以效果有限,别指望靠它解决。
4. **不开投机解码**,权重占用从 26.31 GiB 降到 24.41 GiB(省 1.9 GiB),
   每 token 的 KV 也从 31.0 KiB 降到 23.4 KiB——两头都松一截。

### 6.2 启动命令

与本机压测所用的完全一致(见 `bench-vllm/serve_and_run.sh`):

```bash
TARGET=/target/models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf
DRAFT=/target/models/Muse-Glimmer-30B-GGUF/dflash-Muse-Glimmer-30B-Q4_K_M.gguf
CONF=/target/models/Muse-Glimmer-30B
DRAFT_CONF=/target/models/Muse-Glimmer-30B-assistant

HIP_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
PYTHONPATH=/target/vllm-src \
  vllm serve "$TARGET" \
  --tokenizer "$CONF" --hf-config-path "$CONF" \
  --speculative-config "{\"model\": \"$DRAFT\", \"hf_config_path\": \"$DRAFT_CONF\", \"num_speculative_tokens\": 3}" \
  --max-model-len 131072 --max-num-seqs 16 --gpu-memory-utilization 0.9 \
  --port 8077
```

几点说明:

- `HIP_VISIBLE_DEVICES=0`:本机压测固定用 4 号卡,新机器按实际卡号改。
- `PYTHONPATH=/target/vllm-src`:**开投机解码时必需**(理由见 3.4)。
- `--speculative-config`:**不想开投机解码就整行删掉**。此时 `PYTHONPATH` 也可以不设。
- `num_speculative_tokens`:本机在 MI300X 上的最优值随并发变化。W7900 的 kernel 路径不同,
  这个最优值**需要在新机器上重新扫**,别直接照搬。
- 这条命令**没有设 `VLLM_GGUF_MMQ_B10373`**,因为它在 W7900 上不起作用(第 0 节)。
  本机压测脚本里设它,是为了在 MI300X 上切换两代 kernel 做 A/B 对比,那是 CDNA 才有的需求。

服务就绪的标志是日志出现 `Application startup complete`。

**起得慢是正常的,别以为卡死了。** 本机实测从启动到就绪约 **3 分钟**,其中权重加载
就占 110 秒(日志里有 `Model loading took 24.41 GiB memory and 109.7 seconds`),
另有 `init engine (profile, create kv cache, warmup model) took 25.9 s`。
W7900 显存带宽低于 MI300X,只会更慢。想盯进度就 `tail -f` 那个日志。

### 6.3 发一个请求确认

请求里的 `model` 必须与服务注册的 ID 一致。这里的 ID 就是 **GGUF 的完整路径**
(不是 `--hf-config-path` 指的那个目录)。拿不准就先问服务:

```bash
curl -s http://127.0.0.1:8077/v1/models | python -m json.tool | grep '"id"'
```

```bash
curl -s http://127.0.0.1:8077/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "/target/models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf",
  "prompt": "用一句话说明什么是投机解码。",
  "max_tokens": 64, "temperature": 0
}' | python -m json.tool
```

### 6.4 停服务:必须按进程组关

`vllm serve` 会派生 **EngineCore 子进程**。只杀父进程的话,子进程会变成孤儿**继续占着显存**,
下一次起服务就会 OOM,而且从 `ps` 上不容易一眼看出来。正确的关法:

```bash
# ① 先看清楚要杀谁,别直接把 pgrep 的结果喂给 kill
pgrep -af "bin/vllm serve"

# ② 用上一步看到的父进程号,先关子进程再关父进程
PID=<上一步看到的 PID>
pkill -TERM -P "$PID"; kill -TERM "$PID"; sleep 10
pkill -KILL -P "$PID" 2>/dev/null; kill -KILL "$PID" 2>/dev/null

# ③ 确认显存真的还回来了(本机空载约 0.3 GB/卡)
rocm-smi --showmeminfo vram | grep -i used
```

第 ① 步不能省:在脚本或 heredoc 里跑 `pgrep -f`,它可能匹配到**包裹它自己的那层 shell**
——那层 shell 的命令行里就含着这个模式。写这份文档时就实地撞到过一次:明明没有服务在跑,
`pgrep -af` 仍返回一条,而且是一长串 `/usr/bin/bash -O extglob -c ...`。
分辨方法很简单:**真正的服务进程长得像 `/opt/venv/bin/python .../vllm serve ...`**,
不是一坨 shell 命令行。看清楚再动手。

这套顺序取自 `bench-vllm/serve_and_run.sh` 的 `cleanup()`,是压测里反复用的。

---

## 7. 验证清单

装完照着过一遍,任何一条不符就别继续往下:

```bash
# 1) Python 环境:必须是 venv 那套,不是系统 Python
which python pip                                  # 期望都在 /opt/venv/bin/

# 2) GPU 可见,且架构、torch 支持情况一次看全
python3 -c "
import torch
print('cuda available:', torch.cuda.is_available())
p = torch.cuda.get_device_properties(0)
print('device :', p.name, '|', p.gcnArchName)
print('gfx1100 in arch list:', 'gfx1100' in torch.cuda.get_arch_list())"
#   期望 True / AMD Radeon PRO W7900 | gfx1100 / True

# 3) 源码覆盖生效
PYTHONPATH=/target/vllm-src python -c "import vllm; print(vllm.__file__)"
#   期望 /target/vllm-src/vllm/__init__.py

# 4) draft 侧 hf_config_path 存在(投机解码的前提)
grep -c hf_config_path /target/vllm-src/vllm/config/speculative.py    # 期望 3

# 5) 插件的 C 扩展编出来了、算子已注册到 _C_gguf 命名空间
PYTHONPATH=/target/vllm-src python -c "
import torch, vllm_gguf_plugin
print('ggml_dequantize:', hasattr(torch.ops._C_gguf, 'ggml_dequantize'))
print('ggml_mul_mat_a8:', hasattr(torch.ops._C_gguf, 'ggml_mul_mat_a8'))"
#   期望两个都是 True

# 6) 插件的 entrypoint 能被发现
PYTHONPATH=/target/vllm-src python -c "
from importlib.metadata import entry_points
print([e.name for e in entry_points(group='vllm.general_plugins')])"
#   期望列表里含 'gguf'

# 7) 插件真的能注册进 vLLM 的量化后端
PYTHONPATH=/target/vllm-src python -c "
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
print('gguf registered:', 'gguf' in QUANTIZATION_METHODS)"
#   期望 True,并伴随两行 Registered ... 的 INFO 日志

# 8) MMQ b10373 的实际启用状态
PYTHONPATH=/target/vllm-src python -c "
from vllm_gguf_plugin.ops import mmq_b10373_available
print('mmq_b10373 available:', mmq_b10373_available())"
#   W7900 期望 False(架构门禁或构建时跳过);MI300X 期望 True
```

两个容易误判的地方:

- **第 7 条必须先 `load_general_plugins()`**。`gguf` 是通过 `vllm.general_plugins`
  entrypoint 注册的,vLLM 只在引擎初始化时才加载插件。**光 `import vllm` 去查
  `QUANTIZATION_METHODS` 会得到 `False`,那是正常的懒加载,不是装坏了。**
- **第 8 条在 W7900 上返回 `False` 是正确结果**,不是故障(见第 0 节)。

---

## 8. 排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 明明 `pip install` 成功了却 import 不到 | 装到系统 Python 去了,而 `vllm` 跑的是 `/opt/venv` 那套 | `which python pip` 应都在 `/opt/venv/bin/`;不是的话先修 `PATH`(见第 1 节) |
| 起服务报显存不足,但没跑任何东西 | 上次的 EngineCore 子进程成了孤儿,还占着显存 | 按 6.4 关干净,再用 `rocm-smi` 确认回到空载 |
| `import vllm` 报 `undefined symbol` | wheel 的 `.so` 与当前 torch ABI 不匹配 | 3.2 那条 pip 命令会把 torch 栈整套换成配套版本;别单独升 torch |
| `PYTHONPATH` 指过去后 import 失败,提示缺 `_C` | 3.3 的软链没做或做漏 | 重跑 3.3 的循环,核对 10 个软链都在 |
| `vllm.__file__` 仍指向 site-packages | `PYTHONPATH` 没传进去(比如被 `sudo`/`nohup` 丢掉) | 把 `PYTHONPATH=...` 写在命令同一行,如 6.2 |
| 启动报 `hf_config_path` 不是合法字段 | 走的是 wheel 版 vLLM,没走源码树 | 加 `PYTHONPATH=/target/vllm-src`(见 3.4) |
| 插件编译报 CDNA / MFMA 相关错误 | 在 RDNA3 上编了 CDNA 专用 vendor kernel | 加 `VLLM_GGUF_PLUGIN_NO_MMQ_B10373=1` 重装(第 0.1 节) |
| 插件 `.so` 报 ABI 不匹配 | 装的时候漏了 `--no-build-isolation` | 带上该参数重装(第 4 节) |
| 启动报 KV cache 容量不足 | 48 GB 显存放不下当前配置 | 按 6.1 的顺序调,**先降 `--max-model-len`**(最对症) |
| 性能远低于优化报告里的数字 | W7900 走的是 b2899 回退路径 | 这是预期行为,见第 0 节;报告里的倍数是 MI300X 的 |
| 起服务时 attention 后端报错 | `flash_attn` / AITER 在 CDNA 上才启用,RDNA3 需走 Triton 路径 | 未在 RDNA3 实测过。先看日志选了哪个后端,必要时用 `VLLM_USE_TRITON_FLASH_ATTN=1` 显式指定,并确认 `VLLM_ROCM_USE_AITER` 未开 |

### 8.1 本文哪些部分是实测的,哪些不是

写这份文档的机器是 **MI300X(gfx942)**,手上没有 W7900,所以要分清可信度:

**在本机实测过、输出即文中所写**:

- 第 1 节的环境核对命令;
- **第 2 节的取代码流程**:2.1 的钉 commit 拉法在空目录里从零跑通过,
  匿名访问、体积、2.3 的三条核对值全部实测;
- **第 5 节的模型文件**:三个仓库的可见性与文件清单用 HF API 查过,5.1 表里的字节数
  就是本机文件的实际大小;`hf download` 的 `--include` 重复写法实测过一次(那条
  「多个 `--include` 会被忽略」的坑就是实测撞出来的);三个 sha256 逐个核过(目标与
  draft 跑 `sha256sum -c` 都 OK、耗时 14 秒,mmproj 的 `f48b4523…` 与 HF `main` 的
  LFS oid 逐字符一致);**5.1 里那 6 个配置文件是实测出来的最小集**——把这 6 个单独拷到
  一个空目录,`get_config` / `AutoTokenizer` / `AutoProcessor` 三样都能加载
  (`model_type=muse_glimmer`、`vision_config` 在、chat template 在、processor 是
  `MuseGlimmerProcessor`),这也是「`model.safetensors.index.json` 不用」的依据;
  mmproj 是视觉塔这点是数 GGUF 张量表得到的(目标 731 个、视觉 0 个;mmproj 809 个,
  与 bf16 索引里的 809 个视觉张量对上),通配符发现逻辑是读插件源码确认的;
- 第 3.3 的软链循环(实测产出 10 项);第 3.4 的两条 `grep -c`(0 与 3);
- 第 7 节的验证 1–8 全部(其中第 8 条在本机返回 `True`,因为本机是 CDNA)。

**基于本机实测数据、但向 W7900 外推**:第 6.1 的显存估算。表里那四个数
(24.41 / 142.07 GiB、26.31 / 140.03 GiB)都是 MI300X 启动日志里的原值,
固定开销 6.3 GiB 也是从中反推的;但**把它套到 48 GiB 卡上是外推**,
W7900 的 activation 开销未必一样,没在 48 GiB 卡上验证过实际能不能起。
第 6.2 的启动耗时(约 3 分钟)同理,是本机的数。

**部分实测**:第 5.4 的 rsync——两个 `--include/--exclude` 过滤器在本机
`rsync -anv` 干跑过,分别正好命中 3 个 GGUF 和 6 个配置文件,但跨机传输本身没做过。

**从代码和仓库状态推出、未实机执行**:**第 0.2 节的 RDNA3
启用路径**——那张表的每一行都是读 vendor 源码确认的(文件、行号、宏条件都可自查),
但「放开门禁后在 gfx1100 上真能跑对、跑得快」没有任何实测支撑。

**没验过、文中也没声称验过的**:图像输入这条路。mmproj 会被加载(日志里视觉编码器
确实初始化了),但本文所有压测与质量核对都是纯文本,没发过一张图,所以「多模态输出
是对的」这句话本文给不出依据。只跑文本的话它仍然要下——它在加载路径里,不下就等于
让 loader 去找一个不存在的文件。

**W7900 上必须自己验的**:
- 带 `VLLM_GGUF_PLUGIN_NO_MMQ_B10373=1` 的插件编译能否通过(本机构建的是带 vendor 的版本);
- 第 7 节第 8 条在 gfx1100 上应返回 `False`——这是第 0 节结论的直接检验,**务必跑一下**;
- attention 后端在 gfx1100 上的自动选择结果(排查表最后一行),`flash_attn` 与 AITER
  都是 CDNA 导向的,RDNA3 大概率走 Triton 路径,但没有实测依据。

---

## 附:相关文档

| 文档 | 内容 |
|---|---|
| `docs/opt-vllm-gguf-plugin.md` | 本轮 MMQ kernel 优化的完整报告(**数据均为 MI300X**) |
| `docs/muse-glimmer-vllm-runbook.md` | 本机首次跑通该模型的完整过程,含 nightly 升级的风险分析 |
| `docs/bench-report-vllm-vs-llamacpp.md` | vLLM 与 llama.cpp 的横向压测报告 |
| `vllm-gguf-plugin/vllm_gguf_plugin/csrc/gguf/b10373/README.md` | vendor 源码的来源、改动清单与构建方式 |
| `bench-vllm/serve_and_run.sh` | 第 6.2 节启动命令的出处 |
