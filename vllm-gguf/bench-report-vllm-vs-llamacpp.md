# Muse-Glimmer-30B GGUF：vLLM 插件 vs llama.cpp 性能对比报告

配套基线报告：[`bench-report-dflash.md`](bench-report-dflash.md)（llama.cpp 侧的 DFlash 开关对比）。
本报告沿用它的数据集、采样参数与指标定义，把测试对象换成 vLLM + `vllm-gguf-plugin`。

阅读顺序：第 1～5 节是结论和可操作建议，第 6 节是完整数据，
第 7 节是可比性前提与配置，第 8 节是支撑上述结论的实验。

---

## 1. 一页速览

### 1.1 现在的性能水平

GSM8K，思考强度 high，单卡 MI300X。每个并发档取各自最优配置：

| 并发 | vLLM 最优配置 | vLLM TPOT ms | vLLM 单用户 tok/s | vLLM 聚合 tok/s | llama.cpp TPOT ms | llama.cpp 单用户 tok/s | llama.cpp 聚合 tok/s | vLLM 慢 |
|---|---|---|---|---|---|---|---|---|
| 1 | `num_speculative_tokens=1` | 18.33 | 54.6 | 52.3 | 6.99 | 143.1 | 131.0 | 2.50× |
| 4 | `num_speculative_tokens=16` | 46.62 | 21.4 | 75.9 | 15.30 | 65.4 | 235.7 | 3.11× |
| 16 | `num_speculative_tokens=16` | 67.61 | 14.8 | 195.2 | 33.83 | 29.6 | 407.4 | 2.09× |

三个口径沿用基线报告第 3.2 节的定义：**TPOT** 是稳态解码时每个输出 token 的平均间隔，
越小越好；**单用户 tok/s** 就是它的倒数（`1000 / TPOT`，即基线报告里的
"单用户吞吐（不含首字）"），表示单个用户实际感受到的出字速度；
**聚合 tok/s** 是整机所有请求的输出 token 总速率，随并发增长。
最后一列的"慢"按聚合口径算；因为单用户与 TPOT 是严格倒数，两者的倍数完全相同。

单看关闭投机的基础解码，差距在 1.44× ～ 5.13× 之间，最差的一格是并发 4。
完整数据见第 6 节。

### 1.2 已经做了哪些优化

**已启用并逐项验证生效：**

| 项 | 状态 | 依据 |
|---|---|---|
| torch.compile | 开，`mode=VLLM_COMPILE`，耗时 2.53 s（命中缓存） | 服务端日志 |
| CUDA graph | 开，`FULL_AND_PIECEWISE`，51 个 PIECEWISE + 16 个 FULL 尺寸 | 服务端日志 |
| 前缀缓存 | 开，`enable_prefix_caching=True` | 服务端日志 |
| chunked prefill | 开，`enable_chunked_prefill=True` | 服务端日志 |
| native HIP GGUF 扩展 | 开，非 Triton 回退，反量化数值已核对 | 第 8.6 节 |
| 显存利用率 | `--gpu-memory-utilization 0.9`，140 GiB KV cache | 服务端日志 |
| 并发上限对齐基线 | `--max-num-seqs 16` 对应 `-np 16` | 启动参数 |

**试过但无效：**

| 项 | 结果 | 原因 |
|---|---|---|
| `VLLM_ROCM_USE_AITER=1` | 四项指标全在 ±1.3% 噪声内 | AITER 覆盖注意力和非量化 GEMM，够不到 GGUF 量化 matmul（第 8.6 节） |

**这个模型开不了：**

| 项 | 结果 | 原因 |
|---|---|---|
| ROCm 自定义 paged attention | 强制回退 Triton | 该 kernel 要求 `sliding_window == 0`，本模型 52 层中 39 层是 window 2048 的 sliding attention。任何配置都绕不过（第 8.6 节） |

结论是：**vLLM 侧能开的都开了，开了也没用。** 差距不在配置。

### 1.3 差距主要在哪

把单次前向拆开测——GGUF 里 417 个量化 matmul 逐个按插件真实会选的分支计时再求和，
与实测的整个解码周期对照：

| 每次前向 token | 量化 matmul 合计 | 实测整周期 | matmul 占比 | 其余 | 若阈值不限制 | 阈值代价 |
|---|---|---|---|---|---|---|
| 1（并发1 关闭投机） | 14.19 ms | 21.72 ms | 65.3% | 7.53 ms | 14.19 ms | 0 |
| 4（并发4 关闭投机） | 106.20 ms | 113.61 ms | 93.5% | 7.41 ms | 39.15 ms | **+67.05 ms** |
| 16（并发16 关闭投机） | 164.90 ms | 162.63 ms | 101.4% | −2.27 ms | 108.56 ms | **+56.35 ms** |

据此差距可以拆成**两个彼此独立的问题**：

**问题一，量化 matmul kernel 本身效率低。** 单 token 解码是纯访存受限：
必须把 17.58 GiB 的权重读一遍，按 MI300X 峰值 5.3 TB/s 只需 3.56 ms，
实测 14.19 ms，**只达到峰值带宽的 25.1%**。这一项在任何并发下都存在，
与投机解码、与阈值都无关，是并发 1 关闭投机那 1.44 倍差距的主因——
vLLM 光 matmul 的 14.19 ms 就已经是 llama.cpp **整个周期** 15.13 ms 的 94%，
意味着即使把 vLLM 其余开销全部归零也只能勉强追平。详见第 3 节。

**问题二，kernel 切换阈值卡得过紧。** 插件用 `mmvq_safe` 决定走快 kernel（MMVQ）
还是慢 kernel（MMQ），现在取 2（大矩阵）/ 6（小矩阵），
而实测真实交叉点在 10～48 之间，**15 种权重形状无一例外全部偏低**。
这一项在单 token 时代价为零（那时本来就走 MMVQ），
但每次前向一超过 2 个 token 就立刻生效：4 个 token 时白花 67.05 ms，
占该周期的 59%。它解释了并发 ≥4 的大幅劣化、投机解码在低并发反而拖慢、
以及**并发 1 加到并发 4 总吞吐反降 25%** 这个反常现象。详见第 4 节。

两个问题的可修复性完全不同：问题二是一行判断逻辑，
放宽后端到端提速 2.55 倍且无质量代价（第 8.4、8.5 节）；
问题一要重写 kernel——插件的 GGUF kernel 整体移植自 llama.cpp **b2899**（2024 年），
而对比基线是 **b10373**，这期间上游把 MMQ 重写过。

### 1.4 读数前提

三处硬性差异，引用本报告任何倍数时都必须带上（详见第 7.1 节，
那里还有第四处，只影响一张缺失的表、不影响这里的任何倍数）：

- **llama.cpp 基线无法复验**：压测时二进制已从本机移除，ROCm 也从 7.2.1 升到 7.2.3。
  llama.cpp 的数字全部引自历史记录。
- **kernel 代际不同**：插件是 b2899 的移植，基线跑 b10373。这不是变量控制失败，
  而是被测对象的真实状态。
- **样本量不齐**：llama.cpp 全部 200 请求；vLLM 并发 16 为 200、并发 4 为 60、并发 1 为 50。
  均值可靠，故本报告不引用 vLLM 的 p99。

---

## 2. 投机解码参数怎么选

### 2.1 现状下的最优 N

`num_speculative_tokens`（下称 N）在三个并发档的最优值完全不同，
而且**与 llama.cpp 的结论方向相反**：

| 并发 | 最优 N | TPOT ms | 单用户 tok/s | TPOT 加速比 | 聚合加速比 | llama.cpp 同档加速比 |
|---|---|---|---|---|---|---|
| 1 | **1** | 18.33 | 54.6 | 1.16× | 1.14× | 2.14× |
| 4 | **16** | 46.62 | 21.4 | 2.46× | 2.20× | 1.42× |
| 16 | **16** | 67.61 | 14.8 | 2.40× | 2.10× | 0.67×（净亏） |

基线报告第 6 节建议"并发 ≥16 关闭 DFlash"，**在 vLLM 上必须反过来**：
并发 ≥4 时投机是净赚 2.4 倍，关掉才是最差选择。

并发 1 上的 N 尤其敏感，扫描结果（low 强度、20 请求）：

| N | 每次前向 token | TPOT ms | 单用户 tok/s | 相对关闭 |
|---|---|---|---|---|
| 0 | 1 | 20.51 | 48.8 | 1.00× |
| 1 | 2 | **17.43** | **57.4** | **1.18×** |
| 3 | 4 | 41.87 | 23.9 | 0.49× |
| 5 | 6 → 8 | 35.20 | 28.4 | 0.58× |
| 8 | 9 → 16 | 37.71 | 26.5 | 0.54× |
| 16 | 17 → 24 | 31.62 | 31.6 | 0.65× |

**N=1 与 N=3 之间是断崖，不是渐变。** N=1 时每次前向 2 个 token，
恰好贴着 `mmvq_safe = 2` 还能保住快 kernel；N=3 变成 4 个 token，
一脚踏空掉进慢 kernel，批间隔从 32.91 ms 跳到 128.53 ms。
所以并发 1 上"能且只能"用 N=1。

（"每次前向 token"一列标了两个值的，是 CUDA graph 捕获尺寸离散导致的向上补齐，
见第 7.2 节。）

### 2.2 改 kernel 之后最优 N 会变，而且变化不小

阈值一旦按实测交叉点放宽，上面这张表的形状就变了。已实测部分（同条件，第 8.4 节）：

| 并发 1 的 N | 现状 TPOT ms | 现状单用户 tok/s | 放宽后 TPOT ms | 放宽后单用户 tok/s | 变化 |
|---|---|---|---|---|---|
| 1 | 17.43 | 57.4 | 17.43（未受影响，本来就走 MMVQ） | 57.4 | — |
| 3 | 41.87 | 23.9 | **16.44** | **60.8** | 2.55× |
| 5 | 35.20 | 28.4 | 17.45 | 57.3 | 2.02× |
| 8 | 37.71 | 26.5 | 30.38 | 32.9 | 1.24× |

**最优点从 N=1 移到 N=3**（16.44 优于 17.43），新的断崖落在 N=5 与 N=8 之间。
N=2、4、6、7 未测，最优值有可能还在 N=3 附近但不是 N=3。

更值得注意的是**并发 4 的推荐配置可能整体翻转**。现状下并发 4 必须用 N=16
才能跳出悬崖（75.9 tok/s）；若阈值放宽，关闭投机时每次前向 4 个 token 会走回快 kernel，
按第 1.3 节的分解外推：

```
放宽后 并发4 关闭投机的周期 ≈ 39.15 (matmul) + 7.41 (其余) = 46.56 ms
每周期产出 4 个 token  =>  4 / 0.04656 = 85.9 tok/s
```

即 **85.9 tok/s，反超现状下最优的 N=16（75.9）**。

并发 16 同样外推（该格的"其余"实测为 −2.27 ms，是求和法的正偏差所致，
按 0 处理）：

```
放宽后 并发16 关闭投机的周期 ≈ 108.56 ms
每周期产出 16 个 token  =>  16 / 0.10856 = 147 tok/s
```

147 仍低于现状下 N=16 的 195.2，**所以高并发档投机依然该开**。

**这两个数是按实测分量算出来的外推，不是实测。** 阈值改动只在并发 1 上做过
端到端验证，并发 4 / 16 的验证是未覆盖项（第 9 节）。
另外"全走 MMVQ"在并发 16 时并非最优策略（第 8.3 节），
真实的最优会略好于 147。

---

## 3. 不开投机时也慢：基础解码的差距

这一节回答的是：即使完全不碰投机解码，vLLM 也比 llama.cpp 慢，为什么。

### 3.1 差距有多大

关闭投机，GSM8K high：

| 并发 | 每次前向 token | vLLM TPOT ms | llama.cpp TPOT ms | vLLM 单用户 tok/s | llama.cpp 单用户 tok/s | vLLM 聚合 | llama.cpp 聚合 | 慢 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 21.23 | 14.95 | 47.1 | 66.9 | 45.8 | 66.0 | 1.44× |
| 4 | 4 | 114.71 | 21.67 | 8.7 | 46.2 | 34.5 | 177.0 | 5.13× |
| 16 | 16 | 162.40 | 33.83 | 6.2 | 29.6 | 92.9 | 407.4 | 4.39× |

关闭投机时每个解码周期每请求恰好产出 1 个 token，所以批间隔与 TPOT 几乎相同
（vLLM 分别是 21.72 / 113.61 / 162.63 ms），本表不再单列；批间隔见第 4.3 节落点表。

**单用户口径下并发 4 那一格尤其难看**：8.7 tok/s 对 llama.cpp 的 46.2 tok/s，慢 5.3 倍。
而且相比 vLLM 自己在并发 1 的 47.1 tok/s，每个用户的出字速度掉到了 1/5.4，
**降幅超过并发增幅（4 倍）**——这就是聚合吞吐反而下降的另一种表述。
llama.cpp 同一区间是 66.9 → 46.2，只掉到 1/1.45，远小于 4 倍。

**并发 1 那一行要单独看。** 那里每次前向只有 1 个 token，
两个引擎都走各自的向量 kernel，`mmvq_safe` 阈值根本不起作用
（1 ≤ 2，插件本来就选 MMVQ）。所以这 1.44 倍的差距**与阈值无关**，
是一个独立的、更基础的问题。并发 4 和 16 那两行则是这个基础差距
叠加了阈值代价的结果，属于第 4 节的范畴。

### 3.2 单 token 前向的耗时构成

把 GGUF 里 417 个量化 matmul 逐个计时求和，与实测周期对照：

```
实测整周期      21.72 ms
├─ 量化 matmul  14.19 ms  (65.3%)
└─ 其余          7.53 ms  (34.7%)   注意力 + norm + rope + 采样 + 框架开销
```

matmul 部分按部件拆分（单 token，单位 ms）：

| 部件 | 耗时 | 占 matmul |
|---|---|---|
| ffn_up | 2.98 | 21% |
| ffn_gate | 2.86 | 20% |
| ffn_down | 2.58 | 18% |
| attn_q | 1.09 | 8% |
| attn_gate | 1.07 | 8% |
| attn_k | 1.06 | 7% |
| attn_v | 1.06 | 7% |
| attn_output | 1.04 | 7% |
| output（lm_head） | 0.44 | 3% |

方法有效性的自校验：同一套求和在 16 个 token 时给出 164.90 ms，
实测周期 162.63 ms，**只高出 1.4%**。说明"逐个 kernel 计时再求和"
足以近似真实前向，不是量级估算。单 token 那一格的 14.19 ms 可能含
一两毫秒的逐次调用开销（CUDA graph 在真实前向里会消掉这部分），
所以 65.3% 这个占比应理解为上界，真实值大约在 55%～65%。

### 3.3 主要原因：MMVQ kernel 只跑出四分之一的带宽

单 token 解码是**纯访存受限**的：算术量微不足道，
瓶颈就是把所有权重从 HBM 读一遍。于是有一条硬地板：

```
matmul 权重合计            17.58 GiB
MI300X HBM3 峰值带宽        5.3 TB/s
读一遍的理论下限            3.56 ms
实测                       14.19 ms   =>  达到峰值带宽的 25.1%
```

**这是本节的答案。** 移植过来的 MMVQ kernel 只跑出了 MI300X 四分之一的
访存带宽，而一个调好的"边读边反量化边累加"kernel 通常能到 60%～80%。
14.19 ms 里大约有 10 ms 是纯粹的效率损失。

对照 llama.cpp 就更直观：**vLLM 光 matmul 就花了 14.19 ms，
已经是 llama.cpp 整个周期 15.13 ms 的 94%。** 也就是说即使把
vLLM 的注意力、框架开销全部归零，也只能勉强追平，不可能反超。

出处是明确的：插件的 GGUF kernel 全部移植自 llama.cpp **b2899**，
源码头部就写着 `copied and adapted from .../llama.cpp/blob/b2899/ggml-cuda/`，
而对比基线跑的是 **b10373**。

其余 7.53 ms（注意力、norm、rope、采样、框架开销）里，
有一项已知劣势：ROCm 自定义 paged attention 因模型的 sliding window 无法启用，
只能走 Triton（第 8.6 节）。但这一项的绝对值没有单独测量，
**无法进一步把 7.53 ms 拆开归因**，这是本报告的一处限制。

---

## 4. 开投机后的反转：kernel 切换阈值

### 4.1 两个 kernel 的分工

插件从 llama.cpp 移植了两套量化 matmul：

- **MMVQ**（`ggml_mul_mat_vec_a8`，Mul Mat **Vec** Quantized）：为 token 极少的场景设计。
  权重流过来就地反量化、边读边累加，不做分块缓存。
- **MMQ**（`ggml_mul_mat_a8`，Mul Mat Quantized）：为 token 多的场景设计。
  先把一块权重解包搬进共享内存，再让批次里所有 token 复用。

分工依据是两种场景瓶颈不同：token 少时是纯访存受限，解包搬运纯属浪费；
token 多时权重可被复用，搬运成本被摊薄。
**关键在于 MMQ 的搬运成本与 token 数无关**——哪怕只有 3 个 token，这笔钱一分不少。

实测两条曲线（FFN gate，Q4_K，19968 × 6656，完整数据见第 8.1 节）：

| token 数 | MMVQ | MMQ |
|---|---|---|
| 1 | 0.055 ms | 0.323 ms |
| 3 | 0.122 ms | 0.324 ms |
| 8 | 0.233 ms | 0.325 ms |
| 16 | 0.440 ms | 0.326 ms |
| 32 | 0.850 ms | 0.327 ms |

**MMVQ 近似线性**（每多一个 token 稳定 +0.026 ms），
**MMQ 是水平线**（1 个 token 与 32 个 token 相差 1.2%）。两线交于 8 与 12 之间。

### 4.2 阈值现在取什么值

选择逻辑在 `vllm_gguf_plugin/quantization/linear.py`：

```python
if qweight_type in IMATRIX_QUANT_TYPES:
    mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
else:
    mmvq_safe = 2 if qweight.shape[0] > 5120 else 6
...
if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
    y = ops.ggml_mul_mat_vec_a8(...)      # 快
elif qweight_type in MMQ_QUANT_TYPES:
    y = ops.ggml_mul_mat_a8(...)          # 慢
```

`x.shape[0]` 是本次前向的 token 数，`qweight.shape[0]` 是权重输出维。
本模型是 K-quant，不属于 imatrix 类，**走的是 2 / 6 这一档**。

上一节的曲线说交叉点在 8～12，规则却在 **3** 个 token 处就切走，此时 MMVQ 还快 2.65 倍。
全形状扫描（第 8.2 节）确认这不是个别现象：**15 种权重形状的真实交叉点
全部高于当前阈值**，其中输出维 6656 的那批（`attn_output` 和 `ffn_down`，占模型很大比重）
被 5120 这个断点划进"大矩阵"给了阈值 2，而真实交叉点是 32～48，偏低 16～24 倍。

变量名中的 "safe" 暗示这是安全上限，但 git 历史显示它随一次重构一次性引入、
未留下依据，而实测 MMVQ 的数值误差在每一种形状上都不比 MMQ 差（第 8.2 节）。

### 4.3 用阈值解释每一格

"每次前向 token 数 = 并发数 × (投机深度 + 1)"。对照阈值 2：

| 场景 | 每次前向 token | 落点 | 实测批间隔 | 相对 1 token |
|---|---|---|---|---|
| 并发1 关闭 | 1 | MMVQ | 21.72 ms | 1.00× |
| 并发1 N=1 | 2 | MMVQ（贴线） | 34.40 ms | 1.58× |
| 并发1 N=3 | 4 | **MMQ** | 128.53 ms | 5.92× |
| 并发4 关闭 | 4 | **MMQ** | 113.61 ms | 5.23× |
| 并发4 N=16 | 68 → 72 | MMQ | 213.63 ms | 9.84× |
| 并发16 关闭 | 16 | MMQ | 162.63 ms | 7.49× |
| 并发16 N=16 | 272 | MMQ | 311.71 ms | 14.35× |

- **并发 1 投机为何几乎无收益**：N=1 是 2 个 token，贴着阈值保住 MMVQ，得 1.16×；
  N≥3 越界，前向贵 3.9 倍而 token 只多 1.6 倍，净亏。
- **并发 4 关闭投机为何最差**：4 个 token 刚跨过阈值，付了 MMQ 的全额搬运费，
  却只有 4 个 token 分摊。这是"最小的 MMQ 批次"，摊薄效率最低。
- **并发 4 / 16 投机为何大赚**：已经在 MMQ 里，而 MMQ 近乎水平线——
  并发 16 从 16 个 token 加到 272 个，前向只从 162.63 涨到 311.71（1.92 倍），
  token 却多 4.3 倍。**多塞 token 几乎免费**，于是拿到 2.40×。

按基线报告 5.2 节的分解方式（加速比 ≈ 每批产出比 ÷ 批间隔比）核对：

| 并发 | N | 产出比 | 周期比 | 预测加速比 | 实测加速比 |
|---|---|---|---|---|---|
| 1 | 1 | 1.81 | 1.58 | 1.14 | 1.16 |
| 1 | 16 | 4.28 | 8.09 | 0.53 | 0.56 |
| 4 | 1 | 1.80 | 1.55 | 1.17 | 1.20 |
| 4 | 16 | 4.24 | 1.88 | 2.25 | 2.46 |
| 16 | 1 | 1.80 | 1.13 | 1.60 | 1.63 |
| 16 | 16 | 4.30 | 1.92 | 2.24 | 2.40 |

预测与实测处处吻合到几个百分点，说明这套分解在 vLLM 侧同样成立，
两个引擎的差别只体现在"周期比"这一列上。

### 4.4 最刺眼的后果：并发扩展性出现负增长

聚合吞吐随并发的变化：

| 引擎 | 配置 | 并发1 | 并发4 | 并发16 | c4/c1 | c16/c1 | c4 效率 | c16 效率 |
|---|---|---|---|---|---|---|---|---|
| llama.cpp | 关闭投机 | 66.0 | 177.0 | 407.4 | 2.68× | 6.17× | 67.1% | 38.6% |
| llama.cpp | 开启投机 | 131.0 | 235.7 | 275.9 | 1.80× | 2.11× | 45.0% | 13.2% |
| vLLM | 关闭投机 | 45.8 | **34.5** | 92.9 | **0.75×** | 2.03× | **18.8%** | 12.7% |
| vLLM | 各自最优 N | 52.3 | 75.9 | 195.2 | 1.45× | 3.73× | 36.3% | 23.3% |

效率 = 相对并发 1 的加速比 ÷ 并发数，即线性扩展的达成率。

同样四行换成单用户口径（`1000 / TPOT`），看的是每个用户实际的出字速度：

| 引擎 | 配置 | 并发1 | 并发4 | 并发16 |
|---|---|---|---|---|
| llama.cpp | 关闭投机 | 66.9 | 46.2 | 29.6 |
| llama.cpp | 开启投机 | 143.1 | 65.4 | 19.8 |
| vLLM | 关闭投机 | 47.1 | **8.7** | 6.2 |
| vLLM | 各自最优 N | 54.6 | 21.4 | 14.8 |

单用户速度随并发下降是正常的（用户在抢同一份带宽），关键看降幅。
llama.cpp 关闭投机时并发 1→4 只掉到 1/1.45，vLLM 掉到 1/5.4，
**降幅超过了并发增幅 4 倍**——这与上表 c4/c1 = 0.75× 是同一件事的两种写法。

**vLLM 关闭投机时 c4/c1 = 0.75×，加并发反而掉吞吐。**
在正常的批处理实现里，把 batch 从 1 加到 4 时权重只需读一遍、由 4 个 token 分摊，
总吞吐应当接近 4 倍。llama.cpp 用 1.46 倍的周期换 4 倍 token，
vLLM 用 5.23 倍换 4 倍——多出来的成本全被 MMQ 的固定搬运费吃掉了。

按第 1.3 节的分解，并发 4 那一格 113.61 ms 里有 **67.05 ms（59%）纯属阈值代价**。

---

## 5. 部署建议

**这些建议只适用于 vLLM + 当前版本插件，与基线报告第 6 节的结论方向相反。**

**并发 1**：用 `num_speculative_tokens=1`。这是唯一还能保住 MMVQ 的投机配置，
给出 1.16×。不要用更大的 N——N≥3 会掉进 MMQ，反而慢一倍。

**并发 4 及以上**：**务必开启投机，且用大 N（16）**。
并发 4 给出 2.46×，并发 16 给出 2.40×。
关闭投机的并发 4 是整个配置空间里最差的点（34.5 tok/s，比单并发还低）。

**避开并发 2～3**。这一段每次前向 2～3 个 token，正好骑在阈值上，
行为在 MMVQ 和 MMQ 之间抖动。本次未直接测量，但机制上可以预期不稳定。

**如果吞吐是首要目标且可以改插件**：按第 8.2 节的实测交叉点放宽 `mmvq_safe`。
这是目前已知收益最大的单点改动（受影响工作点上 2.55×），且未观测到质量代价（第 8.5 节）。
注意改完之后最优 N 会变（第 2.2 节），需要重新扫一遍。
交叉点位置依赖具体硬件，本报告只在 MI300X 上测过。

**不要指望调 vLLM 的开关**。能开的都开了（第 1.2 节），AITER 无效，
注意力后端受模型架构限制无法改善。剩下的只能动 kernel。

---

## 6. 完整数据

数据集 GSM8K，思考强度 high，`max_tokens=2048`。

### 6.1 vLLM 总表

| 并发 | N | 请求数 | TPOT ms | 聚合 tok/s | 单用户 tok/s | 批间隔 ms | 每批 token | TTFT ms | 输出均长 | 接受率 | accLen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 50 | 21.23 | 45.8 | 47.1 | 21.72 | 1.01 | 200.0 | 706.3 | — | — |
| 1 | 1 | 50 | 18.33 | 52.3 | 54.6 | 34.40 | 1.83 | 232.8 | 698.4 | 0.8271 | 1.83 |
| 1 | 16 | 50 | 37.65 | 24.4 | 26.6 | 175.65 | 4.32 | 376.4 | 720.9 | 0.2086 | 4.34 |
| 4 | 0 | 60 | 114.71 | 34.5 | 8.7 | 113.61 | 1.01 | 409.4 | 763.9 | — | — |
| 4 | 1 | 60 | 95.82 | 40.3 | 10.4 | 175.62 | 1.82 | 561.2 | 768.3 | 0.8210 | 1.82 |
| 4 | 16 | 60 | 46.62 | 75.9 | 21.4 | 213.63 | 4.28 | 620.8 | 756.0 | 0.2059 | 4.29 |
| 16 | 0 | 200 | 162.40 | 92.9 | 6.2 | 162.63 | 1.01 | 546.2 | 755.0 | — | — |
| 16 | 1 | 200 | 99.81 | 145.9 | 10.0 | 183.14 | 1.82 | 696.0 | 762.1 | 0.8236 | 1.82 |
| 16 | 16 | 200 | 67.61 | 195.2 | 14.8 | 311.71 | 4.34 | 1041.6 | 755.2 | 0.2096 | 4.35 |

N = `num_speculative_tokens`，0 表示关闭投机。N=16 是与基线 `--spec-draft-n-max 16` 的对等点。

### 6.2 与 llama.cpp 对照

| 并发 | 配置 | vLLM TPOT | llama.cpp TPOT | 慢 | vLLM 单用户 tok/s | llama.cpp 单用户 tok/s | vLLM 聚合 | llama.cpp 聚合 | 慢 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 关闭投机 | 21.23 | 14.95 | 1.42× | 47.1 | 66.9 | 45.8 | 66.0 | 1.44× |
| 1 | 对等点 N=16 | 37.65 | 6.99 | 5.39× | 26.6 | 143.1 | 24.4 | 131.0 | 5.37× |
| 1 | 各自最优 | 18.33 (N=1) | 6.99 (开) | **2.62×** | 54.6 | 143.1 | 52.3 | 131.0 | **2.50×** |
| 4 | 关闭投机 | 114.71 | 21.67 | **5.29×** | 8.7 | 46.2 | 34.5 | 177.0 | **5.13×** |
| 4 | 对等点 N=16 | 46.62 | 15.30 | 3.05× | 21.4 | 65.4 | 75.9 | 235.7 | 3.11× |
| 4 | 各自最优 | 46.62 (N=16) | 15.30 (开) | 3.05× | 21.4 | 65.4 | 75.9 | 235.7 | 3.11× |
| 16 | 关闭投机 | 162.40 | 33.83 | 4.80× | 6.2 | 29.6 | 92.9 | 407.4 | 4.39× |
| 16 | 对等点 N=16 | 67.61 | 50.50 | 1.34× | 14.8 | 19.8 | 195.2 | 275.9 | 1.41× |
| 16 | 各自最优 | 67.61 (N=16) | 33.83 (关) | 2.00× | 14.8 | 29.6 | 195.2 | 407.4 | 2.09× |

单用户 tok/s 与 TPOT 是同一个量的倒数关系（`1000 / TPOT`），
两列的"慢"倍数相同，所以只在 TPOT 一侧列出。

两个观察：

- **并发 16 的对等点只差 1.34 倍**，是全表最接近的一格。原因不是 vLLM 变快了，
  而是 llama.cpp 在这一格因为投机反而变慢（50.50 vs 关闭时的 33.83）。
- **并发 4 关闭投机差 5.29 倍**，是全表最差的一格，且它是最常见的生产并发。

**投机草稿的质量 vLLM 更好，不是拖累。** 同为 16 的草稿深度，
vLLM 的平均接受长度 4.29～4.35，llama.cpp 是 3.07～3.62。
DFlash 头在 vLLM 里工作得更好，损失全部发生在 kernel 层。
**高并发的 TTFT vLLM 也更好**：并发 16 开投机时 1041.6 ms 对 llama.cpp 的 1551.1 ms。

### 6.3 并发扩展性

见第 4.4 节的表。

### 6.4 投机深度扫描

并发 1，思考强度 low，20 请求。用于选择调优点。

| N | TPOT ms | 单用户 tok/s | 相对关闭 | 聚合 tok/s | 批间隔 ms | 每批 token | 接受率 | accLen |
|---|---|---|---|---|---|---|---|---|
| 0 | 20.51 | 48.8 | 1.00× | 47.1 | 20.87 | 1.02 | — | — |
| 1 | **17.43** | **57.4** | **1.18×** | 54.4 | 32.91 | 1.87 | 0.8639 | 1.86 |
| 3 | 41.87 | 23.9 | 0.49× | 22.7 | 128.53 | 2.97 | 0.6633 | 2.99 |
| 5 | 35.20 | 28.4 | 0.58× | 26.0 | 131.69 | 3.49 | 0.5028 | 3.51 |
| 8 | 37.71 | 26.5 | 0.54× | 24.8 | 173.31 | 4.41 | 0.4304 | 4.44 |
| 16 | 31.62 | 31.6 | 0.65× | 27.3 | 173.95 | 4.86 | 0.2444 | 4.91 |

并发 1 时单用户 tok/s 与聚合 tok/s 只差一点，差值来自 TTFT 和请求间隙——
聚合口径把这些空档算进分母，单用户口径不算。

接受率随 N 单调下降（0.86 → 0.24）是投机解码的正常行为：草稿越长，
后面的 token 越难被接受。但 accLen 仍在涨（1.86 → 4.91），
所以 N 变大在"产出"一侧始终是赚的，亏的全在"周期"一侧。

---

## 7. 可比性声明与测试配置

### 7.1 可比性的硬性限制

这不是一次同机同时的 A/B，有四处限制：

**一、llama.cpp 基线无法复验。** 压测时二进制已从本机移除，
且 ROCm 从 **7.2.1** 升级到 **7.2.3**。本报告引用的全部 llama.cpp 数字来自
`bench-results/matrix.jsonl` 的历史记录，未重跑。
因此"慢 N 倍"这类结论包含一个无法量化的环境项。

**二、kernel 代际不同。** 插件的 GGUF kernel 整体移植自 llama.cpp **b2899**（2024 年），
基线跑的是 **b10373**。这期间 llama.cpp 的 MMQ 被重写过。
**这不是变量控制失败，而是被测对象的真实状态**——插件用户拿到的就是 b2899 的 kernel。

**三、样本量不一致。** llama.cpp 全部为 200 请求；vLLM 侧并发 16 为 200，
并发 4 为 60，并发 1 为 50（投机深度扫描为 20）。均值可靠，
p99 在小样本组不可靠，故本报告不引用 vLLM 的 p99。

**四、无法复现的表格。** 基线报告 4.5 节的 prefill / TTFT 分解依赖 llama.cpp 的
每请求 `timings.prompt_ms`，vLLM 的 OpenAI 兼容接口不提供该字段，此表在本报告中缺失。

### 7.2 服务端与依赖分支

硬件为单卡 AMD MI300X（`HIP_VISIBLE_DEVICES=4`，与基线同一张卡），
ROCm `7.2.3`，PyTorch `2.12.0+git6bbd260`。
被测对象是 **`vllm-gguf-plugin` @ `dflash-drop-upstream-workarounds` / `ae13af7e`**（editable 安装）。

**版本号有个坑，先说清楚。** 服务端日志打的是 `version 0.27.2rc1.dev122+g8efa13b70`，
但**实际执行的 vLLM Python 代码不是 `8efa13b70`**。压测脚本用
`PYTHONPATH=/target/vllm-src` 把源码树前置注入，真正跑的是分支
`single-file-draft-config` / `4b33d88c7`，比 `8efa13b70` 领先 196 个提交。
版本字符串来自 wheel 的 `_version.py`（被源码树符号链接过去），所以它反映的是
**编译扩展的构建点**，不是 Python 代码的提交点。
照着 `8efa13b70` 去复现会缺掉单文件 GGUF 草稿支持。

<details>
<summary><b>完整依赖清单（分支 / commit / 各自提供什么）</b></summary>

| 组件 | 分支 | commit | 提供什么 | 安装方式 |
|---|---|---|---|---|
| vLLM Python 代码 | `single-file-draft-config` | `4b33d88c7` | 调度、投机解码框架、单文件 checkpoint 支持 | `PYTHONPATH=/target/vllm-src` 前置注入 |
| vLLM 编译扩展 | 上游 | `8efa13b70` | `_C.abi3.so` 等，由源码树内符号链接指向 wheel | wheel `0.27.2rc1.dev122+g8efa13b70.rocm723` |
| **vllm-gguf-plugin** | `dflash-drop-upstream-workarounds` | `ae13af7e` | **被测对象**：GGUF 量化 matmul（MMVQ / MMQ）、dflash 草稿头 | editable，`_C_gguf.abi3.so` 就地编译 |

`/target/vllm-src/vllm/` 下的 `*.so` 与 `_version.py` 全是指向
`/opt/venv/lib/python3.12/site-packages/vllm/` 的符号链接——即"Python 代码用源码树、
二进制复用 wheel"的开发装法。这也是版本字符串与代码提交点脱节的原因。

**两侧各自相对上游的本地改动**（`git rev-list --count origin/main..HEAD`）：

```
vllm-src  (origin = WhatGhost/vllm，跟随上游)                      1 个
  4b33d88c7  [Spec Decode] Support single-file checkpoints for target and draft models

vllm-gguf-plugin  (origin = vllm-project/vllm-gguf-plugin)         3 个
  ae13af7    [Refactor] Drop the GGUF draft workarounds now that vLLM covers them
  7e9ecd4    [Models] Support the Muse Glimmer dflash draft model in GGUF
  92b85fd    [Models] Support Muse Glimmer multimodal GGUF (images only)
```

vLLM 那 196 个领先提交里只有 1 个是本地的，其余 195 个都是上游合并。

**关键归属：本报告诊断出的 `mmvq_safe` 阈值问题是上游插件的既有代码，不是这三个提交引入的。**
`quantization/linear.py` 里那两行阈值在 `origin/main` 上一字不差地存在，
最后一次改动是上游的 `87b598d Refactor and reorganize GGUF quantization params and methods (#6)`；
三个本地提交**没有一个碰过这个文件**。所以第 4 节的结论对所有插件用户都成立，
与这条 PR 分支无关。

| 其他 | 值 |
|---|---|
| PyTorch | `2.12.0+git6bbd260` |
| ROCm / HIP runtime | `7.2.3` / `7.2.53211` |
| GPU | AMD Instinct MI300X ×1 |
| 插件 GGUF kernel 来源 | 移植自 llama.cpp **b2899**（第 7.1 节限制二） |
| llama.cpp 基线 | **b10373**，历史数据，当时 ROCm **7.2.1**（第 7.1 节限制一） |

**被测 kernel 与 vLLM 的 `.so` 无关。** MMVQ / MMQ 都在插件自己的
`vllm_gguf_plugin/_C_gguf.abi3.so` 里就地编译，所以第 4、8 节的 kernel 测量
完全归属于插件工作区 `ae13af7e`，不受 wheel 构建点影响。

</details>

<details>
<summary><b>服务端启动命令（逐字，来自 <code>bench-vllm/serve_and_run.sh</code>）</b></summary>

```bash
HIP_VISIBLE_DEVICES=4 VLLM_SERVER_DEV_MODE=1 PYTHONPATH=/target/vllm-src \
  vllm serve /target/models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf \
  --tokenizer      /target/models/Muse-Glimmer-30B \
  --hf-config-path /target/models/Muse-Glimmer-30B \
  --max-model-len 131072 --max-num-seqs 16 --gpu-memory-utilization 0.9 \
  --port 8077 \
  --speculative-config '{"model": "/target/models/Muse-Glimmer-30B-GGUF/dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
                         "hf_config_path": "/target/models/Muse-Glimmer-30B-assistant",
                         "num_speculative_tokens": N}'
```

三点说明：

- **关闭投机就是整行删掉 `--speculative-config`**，其余参数逐字不变。
  开关投机与 `num_speculative_tokens` 都是启动参数，所以每个工作点必须独立起一次服务。
- **`method` 不用手传**，由草稿模型的 `config.json` 自动判定为 `dflash`；
  日志确认为 `SpeculativeConfig(method='dflash', ..., num_spec_tokens=N)`。
- `VLLM_SERVER_DEV_MODE=1` 只为暴露 `/reset_prefix_cache`（组间冲洗用，第 7.3 节）。

日志核对到的实际生效值：`max_seq_len=131072`、`load_format=gguf`、`quantization=gguf`、
`enforce_eager=False`、`seed=0`、`enable_prefix_caching=True`、`enable_chunked_prefill=True`；
KV cache 关闭投机时 142.07 GiB、开启时 140.03 GiB。

</details>

主模型 18.30 GiB（K-quant 动态混合，含 Q4_K / Q5_K / Q6_K），草稿模型 1.52 GiB。

CUDA graph 的捕获尺寸表是离散的：
`[1, 2, 4, 8, 16, 24, 32, 40, ..., 256, 272, 288, ..., 512]`。
本报告全部工作点都落在捕获范围内（≤512），**没有回退 eager**；
不在表上的尺寸会向上补齐到下一个捕获尺寸，涉及三处：
并发4 N=16 的 68 补到 72，扫描里 N=5 的 6 补到 8、N=8 的 9 补到 16。
相关表格已按补齐后的实际值标注。

### 7.3 客户端

`bench-vllm/bench_serve_vllm.py`，按基线客户端逐条复刻：

- **同一批 prompt**：GSM8K test split，同 seed 同洗牌顺序
- **同一组采样参数**：`temperature=1.0`、`top_p=0.95`、`top_k=64`、`max_tokens=2048`
- **同一种并发模型**：信号量维持恒定并发（闭环），而非 Poisson 到达（开环）。
  这一条很关键——vLLM 自带的 `benchmark_serving.py` 是开环的，它的 TTFT 含排队时间，
  与基线的 TTFT 定义不同，直接对比会得出错误结论。**本报告未使用 vLLM 自带工具。**
- **同样的预热与冲洗**：2 个独立 prompt 预热，组间调 `/reset_prefix_cache`

两处不得不改的地方：每请求 token 数改用 `stream_options.include_usage` 回报的
`completion_tokens`（llama.cpp 用 `timings.predicted_n`）；投机接受率改用 `/metrics`
计数器差分（llama.cpp 有每请求字段）。

### 7.4 SSE 分块口径

基线用"相邻 chunk 间隔 ≥1 ms 记为一个解码周期"来数批次。
vLLM 的分块语义不同：**一个 chunk 携带该周期产出的全部 token**，
而 llama.cpp 会把同周期的多个 token 拆成多个 chunk。
`bench-vllm/probe_chunks.py` 实测确认 vLLM 侧不存在 <1 ms 的簇，
1 ms 阈值仍能正确切分周期，两侧的"每批 token 数"与"批间隔"可比。

### 7.5 复现命令

全部命令按实验分组折叠在下面。前置条件：插件工作区在 `ae13af7e`、
`/target/vllm-src` 在 `single-file-draft-config`（第 7.2 节）。
`serve_and_run.sh` 负责"起服务 → 等就绪 → 压测 → 按进程组收服务"，
所以下面每条命令都是自包含的，跑完不留占显存的孤儿进程。

<details>
<summary><b>主表 9 组（第 6.1 / 6.2 节，表 1.1、3.1、4.4 的数据来源）</b></summary>

```bash
cd /target
# 并发 1：50 请求；并发 4：60 请求；并发 16：200 请求
for c in 1 4 16; do
  case $c in 1) np=50;; 4) np=60;; 16) np=200;; esac
  for n in 0 1 16; do
    BENCH_OUT=/target/bench-results/matrix-vllm.jsonl \
    bench-vllm/serve_and_run.sh "$n" "gsm8k-high-c${c}-n${n}" \
      --dataset gsm8k --strength high --concurrency "$c" \
      --num-prompts "$np" --max-tokens 2048 --seed 0
  done
done
```

`spec_n=0` 时脚本不传 `--speculative-config`，即关闭投机。
样本量不齐是刻意的取舍，代价见第 7.1 节限制三。

</details>

<details>
<summary><b>投机深度扫描 6 组（第 6.4 节，表 2.1 下半、2.2 的对照基准）</b></summary>

```bash
cd /target
for n in 0 1 3 5 8 16; do
  BENCH_OUT=/target/bench-results/scan-vllm.jsonl \
  bench-vllm/serve_and_run.sh "$n" "scan-n${n}" \
    --dataset gsm8k --strength low --concurrency 1 \
    --num-prompts 20 --max-tokens 2048 --seed 0
done
```

这组用 `low` 强度、20 请求，是为了快速扫参数；因此它的绝对值不与主表混用，
只在组内比较。

</details>

<details>
<summary><b>AITER 对照 2 组（第 8.6 节）</b></summary>

```bash
cd /target
# 与主表对应组逐字同参，唯一差别是多一个环境变量
VLLM_ROCM_USE_AITER=1 BENCH_OUT=/target/bench-results/opt-vllm.jsonl \
  bench-vllm/serve_and_run.sh 16 aiter-gsm8k-high-c16-n16 \
    --dataset gsm8k --strength high --concurrency 16 --num-prompts 200 --seed 0

VLLM_ROCM_USE_AITER=1 BENCH_OUT=/target/bench-results/opt-vllm.jsonl \
  bench-vllm/serve_and_run.sh 1 aiter-gsm8k-high-c1-n1 \
    --dataset gsm8k --strength high --concurrency 1 --num-prompts 50 --seed 0
```

</details>

<details>
<summary><b>阈值 A/B 3 组（第 8.4 节，需临时改插件源码）</b></summary>

改动只有一行。`vllm_gguf_plugin/quantization/linear.py:41` 的非 imatrix 分支：

```python
# 原（现状）
mmvq_safe = 2 if qweight.shape[0] > 5120 else 6
# 改后（放宽到与 imatrix 分支同值）
mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
```

插件是 editable 安装，改完直接生效、无需重装。然后重跑扫描里受影响最深的三组，
条件与上面"投机深度扫描"逐字相同：

```bash
cd /target
for n in 3 5 8; do
  BENCH_OUT=/target/bench-results/opt-vllm.jsonl \
  bench-vllm/serve_and_run.sh "$n" "mmvq8-n${n}" \
    --dataset gsm8k --strength low --concurrency 1 \
    --num-prompts 20 --max-tokens 2048 --seed 0
done
git -C vllm-gguf-plugin checkout vllm_gguf_plugin/quantization/linear.py   # 还原
```

</details>

<details>
<summary><b>Kernel 微基准（第 8.1 / 8.2 / 8.3 节，不需要起服务）</b></summary>

这几个脚本直接加载 GGUF 张量、直接调 kernel，不经过 vLLM，所以不用起服务，
也不受调度影响。

```bash
cd /target
# 8.1 两个 kernel 的成本曲线（绕开派发逻辑，两边都调一遍）
python bench-vllm/probe_mmvq_limit.py --warmup 5 --iters 50
# 8.2 全形状交叉点扫描（枚举每种 量化类型×输出维×输入维）
python bench-vllm/scan_mmvq_crossover.py --warmup 5 --iters 50
# 8.3 单次前向 matmul 求和 + 带宽利用率（417 个量化张量逐个计时）
python bench-vllm/sum_forward_matmul.py --warmup 5 --iters 50
# 单 matmul 微基准，按插件实际会选的分支计时
python bench-vllm/micro_gguf_matmul.py --warmup 5 --iters 50
```

</details>

<details>
<summary><b>质量验证：准确率三路 + 困惑度（第 8.5 节）</b></summary>

两个脚本都是 `run` 采集、`compare` 比对，温度硬编码为 0。
**它们不自己起服务**，需要先按目标配置起一个常驻服务——stock 用干净工作区，
relaxed 改 `linear.py` 但阈值按第 8.2 节实测交叉点分档
（≤4096 → 48，≤6656 → 32，≤19968 → 10，其余 6），而不是 8.4 节那种一刀切的 8/16：

```bash
HIP_VISIBLE_DEVICES=4 VLLM_SERVER_DEV_MODE=1 PYTHONPATH=/target/vllm-src \
  vllm serve /target/models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf \
  --tokenizer /target/models/Muse-Glimmer-30B \
  --hf-config-path /target/models/Muse-Glimmer-30B \
  --max-model-len 131072 --max-num-seqs 16 --gpu-memory-utilization 0.9 --port 8077
```

不传 `--speculative-config`：质量对比要的是 kernel 差异，投机会引入额外变量。
服务起好后：

```bash
cd /target
# 准确率:GSM8K 100 题、温度 0、并发 8、不开投机、low 强度
#   stock 跑两遍先量 vLLM 自身的噪声底,再跑 patched
python bench-vllm/check_gguf_quality.py run --tag stock-a
python bench-vllm/check_gguf_quality.py run --tag stock-b
python bench-vllm/check_gguf_quality.py run --tag mmvq-relaxed
python bench-vllm/check_gguf_quality.py compare stock-a stock-b mmvq-relaxed

# 困惑度:对同一段固定文本(GSM8K 官方参考解答 200 篇)算对数似然
python bench-vllm/check_gguf_ppl.py run --tag stock
python bench-vllm/check_gguf_ppl.py run --tag relaxed
python bench-vllm/check_gguf_ppl.py compare stock relaxed
```

`run` 的默认值已经是报告里用的配置（准确率：100 题 / 并发 8 / low；
困惑度：200 篇 / 并发 1），所以上面不用显式传参。
结果落在 `bench-results/quality/<tag>.json`。

</details>

<details>
<summary><b>口径校验与报告自检</b></summary>

```bash
cd /target
# 7.4 节:确认 vLLM 侧不存在 <1 ms 的 SSE chunk 簇(需先起服务)
python bench-vllm/probe_chunks.py --strength high --max-tokens 2048
# 把本报告所有表格的每个数字与 jsonl 重新比对一遍
python verify_report_vllm.py
```

`verify_report_vllm.py` 覆盖 11 张表：按"章节 + 列数 + 前几列取值"定位行，
从 jsonl 重算应有值，超出容差即报错。
不覆盖的是 4.1 / 8.1 / 8.2 / 8.3 / 8.5 那几张——它们来自一次性脚本的 stdout，
没有结构化数据源可比对。

</details>

---

## 8. 支撑实验与验证

前面几节的结论都来自这一节的实验。

### 8.1 微基准：两个 kernel 的成本曲线

`bench-vllm/probe_mmvq_limit.py` 绕开插件的判断逻辑，
对真实 GGUF 张量把两个 kernel 都调一遍并计时。

**FFN gate（Q4_K，19968 × 6656，插件阈值 2）**

| token 数 | MMVQ ms | MMQ ms | MMVQ 快 | 插件实际选择 |
|---|---|---|---|---|
| 1 | 0.055 | 0.323 | 5.92× | MMVQ |
| 2 | 0.095 | 0.323 | 3.40× | MMVQ |
| 3 | 0.122 | 0.324 | 2.65× | **MMQ**（越界） |
| 4 | 0.140 | 0.324 | 2.32× | MMQ |
| 6 | 0.189 | 0.324 | 1.72× | MMQ |
| 8 | 0.233 | 0.325 | 1.39× | MMQ |
| 12 | 0.334 | 0.325 | 0.97× | MMQ |
| 16 | 0.440 | 0.326 | 0.74× | MMQ |
| 32 | 0.850 | 0.327 | 0.38× | MMQ |

**attn_q（Q5_K，4096 × 6656，插件阈值 6）**

| token 数 | MMVQ ms | MMQ ms | MMVQ 快 | 插件实际选择 |
|---|---|---|---|---|
| 6 | 0.072 | 0.307 | 4.28× | MMVQ |
| 8 | 0.084 | 0.307 | **3.68×** | **MMQ**（越界） |
| 16 | 0.118 | 0.307 | 2.60× | MMQ |
| 32 | 0.193 | 0.309 | 1.60× | MMQ |

FFN 上两线交于 8 与 12 之间，插件却在 3 个 token 处切走。
attn_q 更极端：交叉点在 32 之外，插件在 8 个 token 处切走，此时 MMVQ 快 3.68 倍。

### 8.2 全形状交叉点扫描

单个形状不足以定性。`bench-vllm/scan_mmvq_crossover.py` 枚举模型里
**每一种** (量化类型, 输出维, 输入维) 组合，直接测两个 kernel：

| 类型 | 输出维 | 输入维 | 现阈值 | 实测交叉点 | 阈值+1 处 MMVQ 快 | MMVQ 误差 | MMQ 误差 |
|---|---|---|---|---|---|---|---|
| Q6_K | 256 | 6656 | 6 | >48 | — | 8.00e-03 | 8.00e-03 |
| Q5_K | 4096 | 6656 | 6 | >48 | — | 8.93e-03 | 1.46e-02 |
| Q6_K | 4096 | 6656 | 6 | >48 | — | 7.25e-03 | 7.25e-03 |
| Q4_K | 4096 | 6656 | 6 | >48 | — | 7.41e-03 | 1.81e-02 |
| Q4_K | 6656 | 4096 | 2 | 48 | **8.52×** | 1.37e-02 | 5.48e-02 |
| Q5_K | 6656 | 4096 | 2 | 40 | 6.59× | 8.24e-03 | 2.74e-02 |
| Q6_K | 6656 | 4096 | 2 | 32 | 3.41× | 1.01e-02 | 1.01e-02 |
| Q4_K | 6656 | 19968 | 2 | 40 | 8.40× | 1.23e-02 | 2.37e-02 |
| Q5_K | 6656 | 19968 | 2 | 40 | 7.53× | 9.85e-03 | 2.31e-02 |
| Q6_K | 6656 | 19968 | 2 | 32 | 7.37× | 1.48e-02 | 1.48e-02 |
| Q4_K | 19968 | 6656 | 2 | 12 | 2.59× | 7.81e-03 | 1.56e-02 |
| Q5_K | 19968 | 6656 | 2 | 10 | 2.32× | 8.85e-03 | 1.08e-02 |
| Q6_K | 19968 | 6656 | 2 | 10 | 2.24× | 8.66e-03 | 8.66e-03 |
| Q4_K | 202048 | 6656 | 2 | 6 | 1.81× | 6.76e-03 | 1.23e-02 |
| Q5_K | 202048 | 6656 | 2 | 6 | 1.81× | 6.99e-03 | 1.01e-02 |

误差列为交叉点之前区间内相对"反量化后稠密 matmul"的最大相对误差。

**一、15/15 全部偏低。** 没有一种形状的真实交叉点等于或低于当前阈值。

**二、规则方向对，断点错。** 现有规则只有 `输出维 > 5120` 一个断点。
交叉点确实随输出维增大而提前（4096 → >48，6656 → 32~48，19968 → 10~12，202048 → 6），
但 5120 这个断点把输出维 6656 的那批划进了"大矩阵"给阈值 2，
而它们的真实交叉点是 32～48，**偏低 16～24 倍**。

**三、没有精度理由。** MMVQ 的误差在每一种形状上都不比 MMQ 差：
Q6_K 上两者完全相同，Q4_K / Q5_K 上 MMVQ 反而小 2～4 倍（最极端一格 1.37e-02 vs 5.48e-02），
且不随 token 数恶化。**"safe" 这个命名没有数值依据支撑。**

### 8.3 单次前向的 matmul 耗时求和

`bench-vllm/sum_forward_matmul.py` 枚举 GGUF 里全部 417 个参与前向的 2D 权重张量
（排除 `token_embd`，它在前向里是查表不是 matmul），
逐个按插件真实会选的分支计时再求和，同时算出"若全程走 MMVQ"的反事实。

| 每次前向 token | matmul 合计 | 实测整周期 | 占比 | 其余 | 全走 MMVQ 则为 | 阈值代价 |
|---|---|---|---|---|---|---|
| 1 | 14.19 ms | 21.72 ms | 65.3% | 7.53 ms | 14.19 ms | +0.00 |
| 4 | 106.20 ms | 113.61 ms | 93.5% | 7.41 ms | 39.15 ms | **+67.05** |
| 16 | 164.90 ms | 162.63 ms | 101.4% | −2.27 ms | 108.56 ms | **+56.35** |

访存地板：matmul 权重合计 17.58 GiB，按 MI300X 峰值 5.3 TB/s 读一遍需 3.56 ms，
实测 14.19 ms，达到峰值带宽的 **25.1%**。

两点方法说明：

- n=16 那一行求和比实测周期高 1.4%，说明"逐个 kernel 计时再求和"能近似真实前向，
  这是方法有效性的自校验。偏高的部分来自逐次调用开销（真实前向里被 CUDA graph 消掉）。
- 同一偏差在 n=1 时约合一两毫秒，所以 65.3% 应理解为上界，真实占比约 55%～65%。
- "全走 MMVQ"是反事实对照，不是最优策略：n=16 时输出维 19968 的形状已过交叉点，
  最优应是逐形状取两者较小值，会略优于 108.56。

### 8.4 端到端 A/B：放宽阈值

把阈值从 `2 / 6` 临时抬到 `8 / 16`，其余一切不变，重跑扫描中受影响最深的三组
（并发 1，low 强度，20 请求，与第 6.4 节逐字同条件）：

| N | 每次前向 token（补齐后） | 改后 TPOT | 原 TPOT | 改后单用户 tok/s | 原单用户 tok/s | 改善 | 改后批间隔 | 原批间隔 |
|---|---|---|---|---|---|---|---|---|
| 3 | 4 | 16.44 | 41.87 | **60.8** | 23.9 | **2.55×** | 50.88 | 128.53 |
| 5 | 6 → 8 | 17.45 | 35.20 | **57.3** | 28.4 | **2.02×** | 65.87 | 131.69 |
| 8 | 9 → 16 | 30.38 | 37.71 | 32.9 | 26.5 | 1.24× | 134.62 | 173.31 |

N=8 那组本应是对照，却也改善了 1.24×，这一格恰好是机制的交叉验证。
补齐后是 16 个 token：对 FFN 大矩阵（输出维 19968，新阈值 8）来说 16 > 8，仍走 MMQ，没变；
对 `attn_q` 一类（输出维 4096，新阈值 16）来说 16 ≤ 16，切回了 MMVQ。
**一半矩阵变了、一半没变，于是得到介于两者之间的部分改善**，与预期一致。

改完之后并发 1 的最优点从 N=1 变成 N=3（16.44 vs 17.43），
相对关闭态的加速比从 1.18× 提升到 1.25×。**悬崖消失，但底子仍慢**——
llama.cpp 同格是 2.61×，因为第 3 节那个带宽利用率问题没有被触碰。

测完已用 `git checkout` 还原，插件工作区保持在 `ae13af7`。

### 8.5 生成质量验证

放宽阈值意味着更多矩阵走 MMVQ，必须确认生成质量不退化。

这里有个坑要先绕开：要让改动生效必须开并发，而连续批处理下每步的批次构成随调度而变，
数值可能随之微变——若真如此，即使温度为 0，同一配置跑两遍也会分叉，
那么"改后输出和改前不一致"就什么都说明不了。
因此保险起见设计为**三路对比**：stock 跑两遍先量出 vLLM 自身的噪声底，
patched 跑一遍，看它与 stock 的差异是否落在噪声底的同一量级。

配置：GSM8K 100 题，温度 0（贪心），并发 8，不开投机，low 强度。
此时每次前向 8 个 token：stock 下所有形状都超阈值走 MMQ，
patched 下除输出层外全部切回 MMVQ，是对比度最大的工作点。
patched 使用按第 8.2 节实测交叉点分档的规则（≤4096 → 48，≤6656 → 32，≤19968 → 10，其余 6）。

**结果一：噪声底是零，三路设计的前提不成立。**

| 运行 | 成功 | 准确率 | 平均输出长度 |
|---|---|---|---|
| stock-a | 100 | 42.0% | 297.5 |
| stock-b | 100 | 42.0% | 297.5 |
| mmvq-relaxed | 100 | 36.0% | 296.5 |

| 对比 | 文本完全一致 | 最终答案一致 |
|---|---|---|
| stock-a vs stock-b | **100/100** | 100/100 |
| stock-a vs mmvq-relaxed | 14/100 | 74/100 |
| stock-b vs mmvq-relaxed | 14/100 | 74/100 |

stock 两遍**逐字完全相同**。也就是说本工作点上 vLLM 是确定性的，
预想中的"连续批处理导致数值抖动"没有出现。这反而让测试更灵敏：
任何输出差异都可以完全归因于 kernel 改动。

**结果二：文本大面积分叉，但主要是混沌放大而非质量塌陷。**
86/100 的输出文本发生了变化，其中 60 例最终答案仍然相同。
平均输出长度 296.5 vs 297.5，没有出现胡言乱语或不收敛。
这是贪心解码的固有性质：一次 argmax 翻转就会让后续几百个 token 完全改道。

**结果三：准确率差异不显著，但样本量不足以下结论。**
42% → 36%。配对列联表为 stock 对/patched 错 9 例、stock 错/patched 对 3 例，
其余 88 例结论一致。McNemar 精确检验双侧 **p = 0.146**，
按常规阈值不显著；但 n=100 功效偏低，**也不足以排除真实存在的退化**，且方向偏负面。

**因此准确率这个指标在这里不好用**——它主要在测混沌抛硬币。
需要一个不受混沌影响的判据。

**结果四：固定文本的困惑度对比。**

让两个配置对**同一段**人类写的文本（GSM8K 官方参考解答，200 题，
已剔除 `<<...>>` 计算器标注）计算对数似然。文本固定则没有混沌，
文本不由任一配置生成则对谁都不偏袒；对数似然更高的一方，
就是对"反量化后的真实计算"更忠实的一方。

关键前提是让 prefill 也走小 batch：阈值只在 token 数少时起作用，
而 prefill 默认一次吃几千个 token，两个配置都会落在 MMQ 里、测不出差别。
因此服务端以 `--max-num-batched-tokens 8 --no-enable-prefix-caching` 启动，
把 prefill 切成 8 个 token 的小块，复现与并发 8 解码相同的 kernel 选择路径。

| 运行 | 篇数 | token 数 | 平均每 token 对数似然 | 困惑度 |
|---|---|---|---|---|
| stock | 200 | 28685 | −1.438964 | 4.2163 |
| patched | 200 | 28685 | −1.438654 | **4.2150** |

逐篇配对差（patched − stock）：均值 **+0.000287**，标准误 0.000382，**t = +0.75**，
patched 更优的篇数 103/200。

**两者在统计上无法区分。** 差异方向对 patched 略微有利（困惑度低 0.03%），
但 t = 0.75 远未达到显著，103/200 也就是抛硬币的结果。

**小结：** 三条证据合起来看——kernel 层面 MMVQ 的误差在 15 种形状上无一例外
不比 MMQ 差（第 8.2 节）；固定文本的对数似然检测不到差异；
GSM8K 准确率的 6 个百分点差异不显著且可由混沌发散解释——
**放宽阈值没有可观测的质量代价**。

但要如实说明：准确率那一项的 p = 0.146 并不是一个令人安心的数字，
n=100 的功效不足以排除中等幅度的退化。若要正式提交插件 PR，
建议把准确率样本扩到 400～500 题重测。

### 8.6 排除的其他解释

在把责任归给 kernel 之前，逐一排查了其他可能。

**AITER（无效）。** `VLLM_ROCM_USE_AITER` 默认关闭，它启用 AMD 为 MI300X 优化的
kernel 库，是唯一尚未验证的重量级开关。开启后重跑两个代表性工作点：

| 组 | 基线 TPOT | AITER TPOT | 基线单用户 tok/s | AITER 单用户 tok/s | 基线聚合 | AITER 聚合 |
|---|---|---|---|---|---|---|
| 并发16 N=16 | 67.61 | 68.27 | 14.8 | 14.6 | 195.2 | 197.7 |
| 并发1 N=1 | 18.33 | 18.39 | 54.6 | 54.4 | 52.3 | 52.0 |

四项全部落在 ±1.3% 的运行间噪声内。服务端日志确认 AITER 的
flash attention kernel 确实被 JIT 编译并启用。
**无效的原因是明确的**：AITER 覆盖注意力和非量化的 linear / MoE，
够不到 GGUF 的量化 matmul 这条路径。

**CUDA graph 与 torch.compile（已生效，非嫌疑）。** 服务端日志确认
`enforce_eager=False`、`mode=VLLM_COMPILE`、`cudagraph_mode=FULL_AND_PIECEWISE`，
torch.compile 耗时 2.53 秒（命中编译缓存），
图捕获 51 个 PIECEWISE 尺寸 + 16 个 FULL 尺寸，耗时 19 秒、占 6.20 GiB。

**注意力后端（受模型架构限制，非配置问题）。** 日志出现
`Cannot use ROCm custom paged attention kernel, falling back to Triton`。
排查后确认这不是配置失误：该 kernel 的启用条件要求 `sliding_window == 0`，
而本模型 `layer_types` 里 52 层中有 39 层是 `sliding_attention`（window 2048）、
13 层是 `full_attention`。**任何配置都无法启用，只能走 Triton 实现。**
这是本模型的固有代价，但它同时作用于所有并发档，
无法解释第 4.4 节那种随 token 数剧变的行为。

**native HIP 扩展（已生效）。** 确认插件的 C++/HIP 编译扩展在位并被调用，
反量化结果在 fp16 ULP 范围内正确，**不是 Triton 回退路径**
（回退会慢得多，且会掩盖 kernel 层的真实表现）。

---

## 9. 未覆盖的维度

- **ShareGPT 数据集**：基线报告第 7 节有四组对照，本次完全未测。
  该数据集 prompt 更长、输出更短，prefill 占比更高，MMQ 悬崖的相对影响可能不同。
- **思考强度 high / low 的完整交叉**：仅在并发 1 的扫描中用了 low，主表全部为 high。
- **并发 2、3、8**：机制预测 2～3 是抖动区、8 是 MMQ 深处，均未验证。
- **放宽阈值后的最优 N**：第 2.2 节的 85.9 / 147 tok/s 是按分量外推，不是实测。
  阈值改动只在并发 1 上做过端到端验证，并发 4 / 16 未验证；
  N=2、4、6、7 也未扫过。
- **放宽阈值的准确率验证功效不足**：GSM8K 只测了 100 题，McNemar p = 0.146。
  困惑度那一项（200 篇、28685 token）足以说明保真度无差异，
  但准确率这一项要做到显著需要 400～500 题。
- **7.53 ms「其余」的内部构成**：注意力、norm、rope、采样、框架开销未分开测量，
  因此无法量化 Triton 注意力回退单独造成多少损失。
- **多卡 / 其他 GPU 架构**：交叉点扫描只在 MI300X 上做过。
  提交插件 PR 前需要在其他架构上复测——交叉点位置依赖具体硬件的访存与算力比。
- **p99 尾延迟**：vLLM 侧并发 1 / 4 样本量偏小（50 / 60），未引用。
- **prefill 分解**：接口不提供，无法复现基线 4.5 节。

---

## 附：数据与脚本

复现命令见第 7.5 节，依赖分支与 commit 见第 7.2 节。

<details>
<summary><b>数据文件</b></summary>

| 文件 | 内容 |
|---|---|
| `bench-results/matrix-vllm.jsonl` | 主表 9 组（第 6.1 节） |
| `bench-results/scan-vllm.jsonl` | 投机深度扫描 6 组（第 6.4 节） |
| `bench-results/opt-vllm.jsonl` | AITER 2 组 + 阈值 A/B 3 组（第 8.4、8.6 节） |
| `bench-results/quality/stock-{a,b}.json`、`mmvq-relaxed.json` | 准确率三路对比原始输出（第 8.5 节） |
| `bench-results/quality/ppl-{stock,relaxed}.json` | 固定文本困惑度逐篇结果（第 8.5 节） |
| `bench-results/matrix.jsonl` | llama.cpp 基线，来自上一份报告 |

</details>

<details>
<summary><b>脚本</b></summary>

| 文件 | 用途 | 是否需要起服务 |
|---|---|---|
| `bench-vllm/bench_serve_vllm.py` | 压测客户端，与基线同口径 | 需要 |
| `bench-vllm/serve_and_run.sh` | 起服务 → 压测 → 收服务的编排 | 自带 |
| `bench-vllm/probe_chunks.py` | 校验 SSE 分块口径（第 7.4 节） | 需要 |
| `bench-vllm/probe_mmvq_limit.py` | 绕开判断逻辑，两个 kernel 并排计时（第 8.1 节） | 不需要 |
| `bench-vllm/scan_mmvq_crossover.py` | 全形状交叉点扫描（第 8.2 节） | 不需要 |
| `bench-vllm/sum_forward_matmul.py` | 单次前向 matmul 耗时求和与带宽利用率（第 8.3 节） | 不需要 |
| `bench-vllm/micro_gguf_matmul.py` | 单 matmul 微基准，按插件实际分支计时 | 不需要 |
| `bench-vllm/check_gguf_quality.py` | GSM8K 准确率三路对比（第 8.5 节） | 需要 |
| `bench-vllm/check_gguf_ppl.py` | 固定文本困惑度对比（第 8.5 节） | 需要 |
| `verify_report_vllm.py` | 机械校验本报告表格与 jsonl 一致 | 不需要 |

</details>
