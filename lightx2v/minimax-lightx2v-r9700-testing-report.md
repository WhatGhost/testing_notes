# MiniMax-H3 T2AV 性能与精度测试（AMD Radeon AI PRO R9700）

> 测试日期：2026-11-17 起
> 场景：文生音视频 `t2av`（`model_cls=minimax_h3`，`model_variant=fl2av`），口径对齐 RTX 5090（`minimax-lightx2v-5090-testing-note.md`）
> 用例与种子：`prompts.json` 原样使用，未改动

结论一页纸见第 1 节；与 5090 按阶段逐项对比见 `minimax-lightx2v-r9700-testing-note.md`，发现的问题见 `minimax-lightx2v-r9700-issues.md`。

---

## 1. 结论

**一句话质量结论**：FP8（`fp8-triton`）与 BF16 基线质量无可分辨差别——人工盲评 10 条用例逐条同分（B 与 A 均 3.95，F 与 E 均 3.87，无严重失败），自动指标只有 B 的 SQUIM PESQ 低 0.14（与 5090 同量级）；四步组的对白音质稍弱于 29 步（盲评声音 3.5 对 4.0）。

**推荐配置（单张 R9700，32GB）**

- 精度用 BF16。FP8 在这里收益很小（P2 上 B 比 A 快 1.08×，F 比 E 快 1.06×），B 的语音指标略低；自动调优（6.4）和 int32 溢出（6.5）还让 FP8 组的最大可跑规格更小，长视频会直接崩溃。只有主机内存紧张时 FP8 才有意义（主机内存少 28–32GB）。`fp8-rocm`、`fp8-torchao` 都不比 BF16 快。
- 快速出片：**E**（BF16 + 四步 LoRA v1.0，block 卸载），P2 端到端 231.5 s；以对白或画质为主：**A**（BF16 29 步），P2 1332.6 s。
- 平台算子 `aiter_attn` / `fp32_variance` / `torch_real_rope`；PyTorch 用默认显存分配器（不要设 `expandable_segments`，4.3 第 8 条）；进程用 `numactl --localalloc` 启动（block 卸载锁住约 100GB 主机内存，见 `minimax-lightx2v-r9700-issues.md` 第 12 条）。

**最大可跑规格**（第 9 节）：P3（768×1344×362 帧）单卡跑不了。BF16 四步（E）实测最大 768×1344×277 帧（11.5 s），或 544×960×362 帧（15 s）；FP8 组（F、B）按现有代码实测最大 768×1344×175 帧（7.3 s）；把 Triton GEMM 固定为唯一可用配置后 F 能跑 243 帧（10.1 s），260 帧及以上会因 FP8 GEMM 的 int32 地址溢出而崩溃（6.5）。

**P2 速度**（768×1344×124 帧，9 条请求的中位数，同卡同时段，6.2）

| 组 | R9700 | 5090 | 倍数 |
| --- | --- | --- | --- |
| A BF16 29 步 | 1332.6 s | 412.3 s | 3.23× |
| B `fp8-triton` 29 步 | 1234.7 s | 298.1 s（fp8-sgl） | 4.14× |
| E BF16 四步 | 231.5 s | 80.7 s | 2.87× |
| F `fp8-triton` 四步 | 217.6 s | 50.8 / 57.5 s（FP16 累加 + FP8 VAE） | 4.28× / 3.78× |

**与 5090 的主要差异**（细节见第 3 节和各节）

1. 慢 2.9–4.3 倍；FP8 收益小得多（A→B 1.08× 对 5090 的 1.38×），因为 `fp8-triton` 的 GEMM 只比 BF16 快 1.36×，注意力（`aiter_attn`，BF16）不受 FP8 影响。视频 VAE 解码 38 s，约为 5090 的 2 倍。
2. 5090 的 `fp8-sgl`、`fp8-f16-accum`、`sage_attn2`、sgl-kernel RMSNorm、Triton RoPE 都是 CUDA 专属，这里换成 `fp8-triton`、`aiter_attn`、`fp32_variance`、`torch_real_rope`。
3. 5090 单卡能跑 P3（A 峰值 30.6GB，E 32.1GB），R9700 不能。A、B 在 P3 上 OOM 时都在申请 2.19GiB，正好是 109222×5376 的 FP32 张量，推测是非融合的 FP32 RMSNorm / RoPE 中间结果（5090 用融合 kernel），没有逐算子确认。
4. `fp8-triton` 每遇到一个新的序列长度会自动调优，多占 6–10GiB 显存（6.4）；token 数 × 28672 超过 2³¹ 时它的 int32 地址溢出会让进程崩溃（6.5）。5090 的 B / H3 不走这条路径。
5. 敏感层 FP32 在这里没有作用（A1 与 A 逐位一致，第 7 节），5090 上 A' 与 A 有明显偏差。
6. 复现包默认的 `expandable_segments:True` 在 ROCm 7.2.4 上会让显存只增不减、连续请求后 OOM，这里改用默认分配器。

---

## 2. 测试环境

| 项 | 内容 |
| --- | --- |
| 系统 | Ubuntu 24.04.1，内核 7.0.0-42 |
| CPU / 内存 | 2× EPYC 9334（128 线程，8 NUMA 节点），503GB |
| GPU | 8× AMD Radeon AI PRO R9700（gfx1201 / RDNA4，64 CU，32624 MiB，PCIe 5.0 x16，卡间无直连，PPT 300W） |
| 驱动 / ROCm | amdgpu 7.1.9.31600000，宿主机 ROCm 7.2.3；容器内 ROCm 7.2.4 |
| 镜像 | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`（digest `sha256:4449f856…`），代替 LightX2V 官方 ROCm 镜像；依赖由 `tools/setup_container_rocm.sh` 安装 |
| 关键包 | torch 2.10.0+rocm7.2.4、triton 3.6.0、amd-aiter 0.1.21、sageattention 1.0.6、transformers 5.23.0、av 19.0.2（libx264） |
| LightX2V | `e8ddf40`（与 5090 相同，源码未改动） |
| 模型 | HF `MiniMaxAI/MiniMax-H3@42ed227e`，t2av 子集约 144GB |
| 权重校验 | 34 个文件 sha256 与 5090 `weights.sha256` 全部一致，包括本机 CPU 转出的 FP8 DiT（与 5090 B 组 fp8-sgl 权重逐字节相同） |
| 测试卡 | 单卡串行，每组所用的卡记录在各组 `meta.json` |
| 调用方式 | 同进程 `LightX2VPipeline.generate()`，顺序单请求；进程以 `numactl --localalloc` 启动 |
| 计时口径 | 端到端取中位数，扣除了 KFD 暂停 GPU 队列的时间（锁页主机内存被内核迁移时发生，与模型无关；P2 下影响不到 0.2%，见 `minimax-lightx2v-r9700-issues.md` 第 12 条） |

环境指纹：`logs/stage0/summary.json`；pip 清单：`logs/stage0/pip_freeze_container.txt`。

---

## 3. 与 5090 的差异

| 项 | 5090 | R9700 | 原因 |
| --- | --- | --- | --- |
| 注意力 | `sage_attn2`（INT8 QK / FP8 PV） | `aiter_attn`（CK FMHA，BF16 精确） | 官方 AMD Wan 示例；在 H3 形状上比 `sage_attn2_amd_rocm` 快 12–14%，误差 0.003 对 0.013 |
| RMSNorm | `sgl-kernel` | `fp32_variance` | 官方 AMD R9700 示例；sgl-kernel 仅 CUDA |
| RoPE | `minimax_h3_triton_rope` | `torch_real_rope` | 同上 |
| 29 步 FP8（B） | `fp8-sgl`（FP32 累加） | `fp8-triton`（FP32 累加），权重与 5090 B 相同 | `fp8-sgl` 依赖 CUDA 专属 sgl_kernel；见 4.1 |
| FP16 累加（H/H3/D/F/T3） | 主候选 | 不测 | `fp8-f16-accum` 仅 CUTLASS SM90/SM120 |
| 四步 FP8（F） | fp8-f16-accum DiT + FP8 VAE | `fp8-triton` DiT + BF16 VAE | 同上；F 的 VAE 耗时与 5090 F 不可比 |
| 卡数 | 2 卡（含双卡阶段 7） | 1 卡串行 | 本次只测单卡，阶段 7 不做 |
| 显存分配器 | `expandable_segments:True`（上游 `base.sh` 默认） | 默认分配器 | ROCm 7.2.4 / torch 2.10 上 `expandable_segments` 开启时 `empty_cache()` 不把显存还给驱动，连续请求后 OOM；见 4.3 第 8 条 |

其余公共参数（`enable_cfg=false`、`NoCaching`、`use_fused_qkv=false`、AdaLN 缓存、text encoder / VAE CPU offload、VAE 敏感层 fp32、29 步 shift 12/3、4 步 shift 6/3 + `training_euler` + LoRA alpha 128 动态加载、`crf 18 / medium`、`DTYPE=BF16`、`SENSITIVE_LAYER_DTYPE=BF16`、`use_compile=false`）与 5090 一致。

---

## 4. 阶段 0：准备与平台适配中的发现

### 4.1 FP8 方案选择（GEMM 微基准，`logs/stage0/fp8_gemm*.json`）

H3 DiT 的线性层形状（M=P2 约 3.1 万 token），单个 block 四类线性层合计（含激活量化）：

| 方案 | 实现 | 单 block 耗时 | 相对 BF16 | 相对 FP32 误差 |
| --- | --- | --- | --- | --- |
| BF16 | hipBLASLt | 187 ms | 1.00× | 0.0017 |
| `fp8-triton` | Triton GEMM | 138 ms | 1.36× | 0.037 |
| `fp8-torchao` | `torch._scaled_mm`（hipBLASLt），torch 激活量化 | 182 ms | 1.03× | 0.037 |
| `fp8-rocm`（官方 R9700 类） | 同上，权重加载时在线量化 | 199 ms | 0.94× | 0.037 |

- gfx1201 支持 FP8（OCP e4m3），无需像 W7900 那样退到 INT8。
- `fp8-rocm` 不在 H3 的量化白名单里，也不支持动态 LoRA 和 block 卸载缓冲，且比 BF16 慢，只做 GEMM 级测试。
- `fp8-torchao` 在 `e8ddf40` 里并不调用 torchao 包，就是 `torch._scaled_mm`；慢在多次 torch 小算子组成的激活量化。
- 初始化日志确认 B/F 实际为 312 个 `MMWeightWfp8channelAfp8channeldynamicTriton`，无回退。

### 4.2 注意力后端（`logs/stage0/attn.json`）

| 后端 | 1.6 万 token / 层 | 3.2 万 token / 层 | 相对 FP32 误差 |
| --- | --- | --- | --- |
| `aiter_attn` | 98.7 ms（78 TFLOPS） | 365.6 ms（80 TFLOPS） | 0.0030 |
| `sage_attn2_amd_rocm` | 109.9 ms | 415.7 ms | 0.0134 |
| `torch_sdpa` | 144.3 ms | 554.9 ms | 0.0029 |

`aiter_attn` 首次调用 JIT 编译约 46 秒（同一容器内之后复用）。

### 4.3 其他坑（复现包没有写）

1. HF 整库（该 revision）498GB，t2av 只需约 144GB；`hf` 2.x 每个 `--include` 只接一个模式，多余的模式会被当成文件名并静默让 `--include` 失效（`tools/fetch_models.sh`）。
2. 转换器导入的 qtorch 在能看到 GPU 时会 JIT 编译 CUDA 扩展，hipify 后在 ROCm 上编译失败；`tools/convert_rocm.py` 以仅 CPU 方式导入后原样运行 `converter.py`（转换 470 秒，与 5090 的 465 秒相当）。
3. `default_runner` 会导入 `lightx2v.server.metrics`，离线测速也需要 `prometheus-client`。
4. 装了 aiter 后 LightX2V 会把它注入为 `sgl_kernel`；H3 的 DiT（`fp32_variance` / `fp8-triton`）、VAE、文本编码器都不走 `sgl_kernel`，不受影响。
5. HIP 设备序号不是 PCI 顺序，`CUDA_DEVICE_ORDER` 在 ROCm 上无效；监控与选卡都按 PCI 地址对接（`tools/pick_gpu.py`）。
6. ROCm 上 VAE 解码阶段本进程有约 4GiB 不经 torch 分配器的显存，整卡显存会比 torch 保留量高约 4GiB；按进程统计显存要读 KFD（`/sys/class/kfd/kfd/proc/<pid>/vram_*`）。
7. 容器以 root 写挂载目录，产物需要 `chown` 回宿主机用户（`run_stage.sh` 每组结束时处理）。
8. **`expandable_segments:True` 在这套 ROCm 上会让显存只增不减。** 复现包的 `bench_h3.py` 沿用上游 CUDA `base.sh` 的默认，在导入 torch 之前设置 `PYTORCH_ALLOC_CONF` 和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。ROCm 7.2.4 / torch 2.10 上，这个模式下 `torch.cuda.empty_cache()` 只把 torch 的保留量降下来，物理显存不还给驱动（`hipMemGetInfo` 和 sysfs 的整卡用量都不降，进程退出才释放）；三个变量（`PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_HIP_ALLOC_CONF`）任一个单独设置都会触发（`logs/validate/vram_probe_envvars.txt`）。H3 每条请求里 block 卸载缓冲、VAE 解码都会大块分配和释放，多条请求后整卡显存逐条抬高，F 组 P2 预热和 01 号 3 条之后，从 08 号起 6 条全部 OOM（报错时 torch 自己只占 10.8GiB，整卡只剩约 1GiB）。
   - 排查：KFD 按进程统计显示这些显存都属于本进程；对照探针（`logs/validate/vram_probe.py`，结果 `logs/stage0/vram_probe.txt`）：开启时 `empty_cache` 后 torch 保留量 12.01→1.52GiB，而 sysfs 显存 12558MiB 不变；关闭时 3 秒内 12546→1794MiB。
   - 处理：`bench_h3.py` 在 `PLATFORM=amd_rocm` 时不再设置该变量（官方 AMD 脚本本来也没设置）。LightX2V 的 `lightx2v/pipeline.py` 第 7 行也会 `setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")`，但 `lightx2v/__init__.py` 在它之前已导入并初始化了 torch，设得太晚，不生效（探针 `pipeline` 模式：变量被设上，显存仍正常归还；修复后 F 组请求之间整卡显存回落到约 6GiB 也印证了这一点）。所以 `meta.json` 的 `pytorch_alloc_conf`（记录于导入 LightX2V 之前）为空，与实际生效的默认分配器一致。用 LightX2V 的 `scripts/base/base.sh` 启动（在 Python 启动前 export 这两个变量）会在 ROCm 上遇到同样的问题。
   - 影响范围：阶段 1（P1）和一次早期的 E（P2，HIP 5）是在开启该选项时测的，当时没有 OOM（P1 显存余量大）；阶段 2 的正式数据全部在默认分配器下重测。

---

## 5. 配置组

| 组 | 方案 | 步数 | DiT 放置 | 对应 5090 |
| --- | --- | --- | --- | --- |
| A `A_bf16_block_29` | BF16 | 29 | block 卸载 | A |
| B `B_fp8triton_block_29` | FP8 `fp8-triton` | 29 | block 卸载 | B（fp8-sgl，同一份权重） |
| E `E_bf16_block_4step` | BF16 + 四步 LoRA v1.0 | 4 | block 卸载 | E |
| F `F_fp8triton_block_4step` | `fp8-triton` + 四步 LoRA + BF16 VAE | 4 | block 卸载 | F（5090 为 FP16 累加 + FP8 VAE） |
| A1 / E1 | A / E 敏感层 FP32 | 29 / 4 | block | A' / E'（数值对照） |
| `B_fp8torchao_block_29` | `torch._scaled_mm` FP8 | 29 | block | 仅 P1 速度对照 |

---

## 6. 性能

### 6.1 阶段 1：P1 筛选（544×960，124 帧；2026-11-17）

单卡（HIP 5）。每组 01 号用例 3 次加 08、19 各 1 次（`B_fp8torchao` 只测 01 的 3 次），种子 42，预热 1 次。原始数据 `results/s1/<组>/P1/`，汇总 `logs/s1/summary_P1.json`。这一阶段按复现包默认开着 `expandable_segments:True`（见 4.3 第 8 条），没有出现 OOM；该选项对速度的影响没有单独测，阶段 2 的 E 组用默认分配器重测，可与早期开启该选项的 E 对照。

| 组 | 端到端中位数（范围） | 每步去噪 | 视频 VAE 解码 | 整卡显存峰值 | 主机内存 | 冷启动 | 5090 端到端 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A BF16 29 步 | **499.2**（499.0–499.8） | 16.40 s | 21.2 s | 19.7GiB | 137GiB | 69 s | 185.0 s |
| B `fp8-triton` 29 步 | **455.0**（453.0–455.5） | 14.84 s | 21.3 s | 16.6GiB | 105GiB | 50 s | 125.0 s（fp8-sgl） |
| B `fp8-torchao` 29 步 | **491.8**（490.9–498.1） | 16.14 s | 21.4 s | 16.8GiB | 105GiB | 63 s | — |
| E BF16 四步 LoRA | **95.3**（95.1–95.6） | — | — | 18.0GiB | 139GiB | 62 s | 无 P1 数据 |
| F `fp8-triton` 四步 LoRA | **89.0**（88.5–90.2） | 16.58 s | 21.8 s | 21.1GiB | 109GiB | 45 s | 无 P1 数据 |

说明：

- E 的 P1 运行早于逐阶段计时的加入，没有每步去噪和 VAE 解码的数字。
- 整卡显存峰值是 sysfs 读到的整卡值（MiB 换算成 GiB），含 torch 之外的分配（VAE 解码阶段约 4GiB）。torch 保留量：A 12.8GiB、B 12.0GiB、E 10.8GiB、F 15.3GiB。主机内存是进程 RSS 峰值。

结论：

1. **FP8 在 R9700 上的收益远小于 5090**：A 到 B（同样 29 步）只快 1.10×（5090 上 fp8-sgl 在 P1 快 1.48×）。四步里 F 比 E 只快 1.07×（5090 的 P2 上 F 比 E 快 1.59×，规格不同，仅作量级参考）。原因：`fp8-triton` 的线性层 GEMM 只比 BF16 快 1.36×（4.1），而注意力（`aiter_attn`，BF16）不受 FP8 影响。
2. **`fp8-torchao`（`torch._scaled_mm`）与 BF16 无差别**（491.8 对 499.2 s），不值得用。
3. **R9700 对 5090 的耗时倍数**：A 2.70×，B 3.64×（5090 的 B 为 fp8-sgl）。
4. **视频 VAE 解码在 R9700 上约 21–22 s**（BF16 VAE），占四步组端到端的约四分之一；5090 上同一阶段 12–13 s（BF16 VAE）。
5. **每步去噪的组成（推断，没有做逐算子 profile）**：4.1、4.2 的微基准按 50 层折算（线性层按 token 数线性缩放），P1 一步里注意力约 4.9 s、线性层约 4.8 s，合计约 9.7 s，而实测 16.4 s（A）。差的约 6.7 s 应落在 RMSNorm、RoPE、调制等小算子和音频分支上。

### 6.2 阶段 2：P2 正式测速（768×1344，124 帧；2026-11-18）

四组在同一张卡（HIP 4）上同一时段连续测完，默认显存分配器。每组 01、08、19 各 3 次，种子 42，预热 1 次，共 9 条正式请求，全部成功。原始数据 `results/s2/<组>/P2/`，汇总 `logs/s2/summary_P2.json`，表格由 `tools/report_tables.py perf s2 P2` 生成。

| 组 | 端到端中位数（范围） | 每步去噪 | 视频 VAE 解码 | 整卡显存峰值 | 主机内存 | 5090 端到端 | R9700 / 5090 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A BF16 29 步 | **1332.6**（1330.0–1334.7） | 44.52 s | 38.2 s | 22.0GiB | 136GiB | 412.3 s | 3.23× |
| B `fp8-triton` 29 步 | **1234.7**（1234.0–1236.5） | 41.15 s | 38.0 s | 30.6GiB（稳态约 23.6） | 108GiB | 298.1 s（fp8-sgl） | 4.14× |
| E BF16 四步 LoRA | **231.5**（231.0–232.1） | 47.36 s | 38.2 s | 22.3GiB | 142GiB | 80.7 s | 2.87× |
| F `fp8-triton` 四步 LoRA | **217.6**（216.8–219.0） | 43.91 s | 38.1 s | 31.4GiB（稳态约 22.1） | 110GiB | 50.8 s（调优）/ 57.5 s | 4.28× / 3.78× |

说明：

- 5090 的 F 是 FP16 累加 DiT 加 FP8 VAE（VAE 约 10.5 s），R9700 的 F 是 `fp8-triton` 加 BF16 VAE，F 的倍数不是同口径对比。
- 整卡显存峰值取 9 条请求的最大值。`fp8-triton` 的两组在每个用例的第一条请求上，去噪阶段整卡显存比同用例的重复请求高 6–10GiB（F 的 08 号首条 32197MiB，离 32624MiB 上限只差 427MiB；B 的 08 号首条 31323MiB），从去噪第一秒起一直保持到去噪结束；同用例的第 2、3 条回到稳态。三个用例的 packed 序列长度各不相同（01：37869，08：37848，19：37879），BF16 的组（A、E）没有这一现象（各请求 torch 分配峰值一致：A 12.8GiB、E 13.3GiB）。**原因是 Triton 自动调优**（6.4）。
- 分配器对速度没有可见影响（不是同卡对照）：早期在另一张卡上开着 `expandable_segments` 测的 E 是 229.5 s，本次 231.5 s；开着 `expandable_segments` 测到的 F 01 号三条 217.8–218.5 s、B 01 号首条 1230.4 s，与本次基本一致（`results/s2_expandable_oom/`）。
- 请求之间整卡显存基线每条请求上涨约 51MiB（F：VAE 解码后 5954MiB 升到 10 条后的 6409MiB），P2 下无影响。
- 冷启动（A 222 s、E 70 s、B 44 s、F 39 s）受页缓存中是否还有权重影响，不作比较。

结论：

1. **FP8 在 P2 上收益更小**：B 比 A 只快 1.08×（每步 41.2 对 44.5 s），F 比 E 快 1.06×；5090 上 A 到 B 是 1.38×。P2 序列更长，注意力（BF16，不受 FP8 影响）占比更大，P1 上 A 到 B 还有 1.10×。
2. **四步 LoRA 每步多约 2.8 s**：E 每步 47.4 s，A 每步 44.5 s（同为 BF16）；F 对 B 同样多 2.8 s。推测是动态应用 LoRA 的开销（没有逐算子 profile）。
3. **R9700 比 5090 慢 2.9–4.3 倍**。BF16 组差距最小（E 2.87×、A 3.23×），FP8 组差距更大（B 4.14×），因为 5090 上 FP8 的加速更明显。
4. **视频 VAE 解码 38 s，是 5090（BF16 VAE 18.7–19.0 s）的约 2 倍**，占四步组端到端的 16–18%。

### 6.3 FP8 备选 `fp8-rocm` 的端到端对照（P1）

用本地补丁（不提交）让 H3 接受官方 `fp8-rocm`（权重加载时在线量化、`torch._scaled_mm`），补丁 `logs/stage0/fp8_rocm_h3.patch`，代码在 `LightX2V-fp8rocm` 工作树，通过 `PYTHONPATH` 生效。P1、01 号 3 次、预热 1 次，HIP 4，2026-11-18。

| 组 | 端到端中位数（范围） | 每步去噪 | 整卡显存峰值 | 主机内存 |
| --- | --- | --- | --- | --- |
| B `fp8-rocm` | **509.2**（508.0–509.4） | 16.70 s | 15.9GiB | 106GiB |
| B `fp8-triton`（6.1，另一张卡，11-17） | 455.0（453.0–455.5） | 14.84 s | 16.6GiB | 105GiB |
| A BF16（6.1，另一张卡，11-17） | 499.2（499.0–499.8） | 16.40 s | 19.7GiB | 137GiB |

`fp8-rocm` 比 BF16 还慢约 2%（不是同卡同时段，跨卡差异在 1% 量级，见 6.2 说明），与 GEMM 微基准一致（4.1：0.94×）。不推荐。

### 6.4 `fp8-triton` 在新序列长度上的额外显存：Triton 自动调优（2026-11-19）

`fp8_gemm_kernel` / `fp8_gemm_bias_kernel`（`lightx2v/common/ops/mm/triton_kernels.py`）用 `@triton.autotune(key=["M","N","K"])` 在 4 个配置里选。序列长度一变 M 就变，每个新长度第一次出现时所有 FP8 线性层形状都要重新调优。

- 4 个候选里只有 `BLOCK 256×128×64、3 stages`（需要 48KB 共享内存）能在 gfx1201 上运行。第一个配置实测报 `OutOfResources: Required 98304, Hardware limit 65536`（`logs/diag/F_single_config_cfg0_oor_P2.log`）；另外两个按同一算法（块大小 ×（stages−1））需要 96KB、72KB，同样超过 64KB 上限，调优时会被当作无穷慢淘汰。也就是说调优结果是固定的，它只带来开销。
- 单算子复现（`logs/validate/fp8_spike_probe.py`，结果 `fp8_spike_probe.txt`，H3 线性层形状）：新长度第一次调用每个形状 196–456ms，之后 14–37ms；每遇到一个新长度，torch 保留量多约 2GiB（5.98→8.01GiB），单次调用的分配峰值没有变大。单算子复现不出整模型里的 +11GiB。
- 整模型对照（`logs/validate/bench_single_config.py`：启动前把两个 autotuner 的候选限定为上面唯一能用的配置，Triton 就不再调优；不改 LightX2V 源码）：F、P2、HIP 0（63），预热 01，再测 01、08。

| | 预热 01 整卡峰值 | 01 整卡峰值 | 08（新长度首条）整卡峰值 | 08 torch 分配峰值 | 01 / 08 端到端（原始） |
| --- | --- | --- | --- | --- | --- |
| 默认（自动调优，6.2） | 25868MiB | 21949MiB | 32197MiB | 23.77GiB | 218.8 / 219.6 s |
| 固定配置（不调优，另一张卡） | 21981MiB | 21947MiB | 21994MiB | 12.79GiB | 222.0 / 222.8 s |

结论：额外显存完全来自自动调优，固定配置后新长度首条请求与稳态一样（约 22GiB）。速度差 1.4% 不是同卡对照（跨卡差异在 1% 量级），配置本身相同，不应影响稳态速度。调优具体在哪里分配了约 10GiB 没有继续追。影响：B、F 每遇到一个新的序列长度，峰值都要多 6–10GiB，这直接限制了 FP8 组的最大可跑规格：默认时 F 最长 175 帧；用同样的方法固定配置（`BENCH_SCRIPT=logs/validate/bench_single_config.py`，`results/s6q/`）后 F 跑通 768×1344×243 帧（621.8 s），比默认多 68 帧。把这两个 kernel 的候选限定为能编译的那一个（或预先调优）属于对 LightX2V 的修改，没有做。

另外 `fp8_gemm_kernel` 读取 `A_SCALES` 时没有掩码，M 不是 `BLOCK_M`（256）的整数倍时最后一块会越界读最多 255 个元素（`minimax-lightx2v-r9700-issues.md` 第 3 条）；下面 6.5 的崩溃不是它造成的。

### 6.5 `fp8-triton` 在 M×N > 2³¹ 时崩溃：int32 地址溢出（2026-11-20）

固定配置的 F 跑 768×1344×277 帧（M=83740 token）时，去噪第 1 步 4 秒内 GPU 内存访问错误（`Memory access fault ... Page not present`，rc=134），当时整卡显存只有 29.3GiB（`logs/s6q/F_fp8triton_block_4step_768x1344x277.log`）；E（BF16）跑同规格正常。

H3 的 FFN 输入投影 `ff.net.0.proj` 是融合的 [28672, 5376]，FP8 GEMM 的输出是 M×28672；kernel 里 `C_ptr = C + rm * stride_cm + ...` 用 int32 计算，M×28672 超过 2³¹−1（M > 74898）就溢出。最小复现（`logs/validate/fp8_int32_probe.py`，只调这一个 GEMM，`fp8_int32_probe.txt`）：

| M | M×28672 | 结果 |
| --- | --- | --- |
| 73546（768×1344×243 帧） | 2.109×10⁹ | 正常，末 64 行相对误差 0.0017 |
| 74880 | 2.147×10⁹（≤ 2³¹−1） | 正常，0.0017 |
| 74920 | 2.148×10⁹（> 2³¹−1） | GPU 内存访问错误，rc=134 |
| 83740（768×1344×277 帧） | 2.401×10⁹ | GPU 内存访问错误，rc=134 |

F 在 P3（M=109222）上的崩溃也是这个原因。所以即使去掉自动调优，`fp8-triton` 在 768×1344 下也只能到 243 帧；260 帧（M≈78600）及以上要先修这个溢出（int64 偏移或按 M 分块），`minimax-lightx2v-r9700-issues.md` 第 4 条。BF16（hipBLASLt）不受影响。

## 7. 数值检查（阶段 3，P2，2026-11-19）

`--dump-tensors` 保存 DiT 输出的视频/音频 latent 和解码后的视频帧、音频波形，`tools/compare_tensors.py`（CPU）比较。阶段 4 的视频作参照，阶段 3 另起进程重跑。结果 `logs/s3/compare_tensors_P2.json`。

| 类别 | 对比 | 用例 | 视频 latent 相对 L2 | latent 余弦 | 音频 latent 相对 L2 | 视频 PSNR（最低帧） | 音频 SI-SNR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 下限 | A 同种子换进程 | 01 | 0 | 1 | 0 | 逐位一致 | 逐位一致 |
| 下限 | E 同种子换进程 | 01 | 0 | 1 | 0 | 逐位一致 | 逐位一致 |
| 上限 | A 换种子（42 对 123） | 01 | 0.819 | 0.667 | 1.038 | 9.7（9.4） | −28.3 dB |
| 上限 | E 换种子 | 01 | 1.038 | 0.456 | 0.928 | 7.9（7.2） | −24.3 dB |
| 良性扰动 | A1（敏感层 FP32）对 A | 01 | 0 | 1 | 0 | 逐位一致 | 逐位一致 |
| 良性扰动 | E1 对 E | 01 | 0 | 1 | 0 | 逐位一致 | 逐位一致 |
| 候选 | B 对 A | 01 | 0.274 | 0.963 | 0.084 | 22.3（21.2） | 13.4 dB |
| 候选 | B 对 A | 19 | 0.542 | 0.852 | 0.347 | 17.3（17.2） | −4.9 dB |
| 候选 | F 对 E | 01 | 0.570 | 0.838 | 0.165 | 14.0（12.7） | 9.4 dB |
| 候选 | F 对 E | 19 | 0.496 | 0.876 | 0.432 | 16.8（16.2） | −5.8 dB |

1. **完全确定**：同配置同种子换进程，latent 和解码结果逐位一致；mp4 之间的差异只来自 H.264 编码。没有 NaN / Inf。
2. **敏感层 FP32 在 R9700 上没有作用，A1 / E1 与 A / E 逐位一致。** H3 的敏感层是 `proj_in`、`audio_proj_in`、`time_embedder`、`proj_out`、`audio_proj_out`：四个投影层无论如何都按 FP32 计算（`MMWeightForceFp32`，两组都是 4 个），`time_embedder` 的输出已在 AdaLN 缓存里预先算好；R9700 用的 `fp32_variance` RMSNorm 也始终按 FP32 算方差，不看 `SENSITIVE_LAYER_DTYPE`。5090 上的 A' 有偏差，是因为那边的 sgl-kernel RMSNorm 会随这个设置换计算路径。因此本机没有"良性扰动"的参照，只能拿换种子作上限。
3. **FP8 带来的偏差远小于换种子，量级与 5090 相当**：B 对 A 在 01 号 PSNR 22.3、19 号 17.3；5090 上 FP8 / A' 对 A 在 01 号 21–36、08 / 19 号 14–17。19 号（两人对白）偏离更大，与 5090 一致；29 步采样对小的数值差异很敏感，画面会走向不同的细节，逐像素指标不能判断好坏，质量结论见第 8 节。

## 8. 质量（阶段 4、5，P2，种子 42）

范围：A、B、E、F 四组 × 10 条用例（阶段 4：01 05 06 07 18 19；阶段 5：15 17 20；08 取阶段 2 的视频）。全部视频 `check_av` 通过（阶段 2–5 共 78 个，无黑帧、冻帧、音频削波）。看片页 `results/s4/review/index.html`（标注组名）。

### 8.1 人工盲评（1 名评测者，2026-11-19，40 个视频，0–4 分）

页面 `results/s5/review/rate.html`（`tools/make_blind_rating.py --cases 01 05 06 07 08 15 17 18 19 20`，每条用例 4 个视频随机排序、文件名匿名），原始评分 `logs/s5/r9700_s5_rating_anonymous.json`，解盲汇总 `logs/s5/rating_summary.json`（`tools/aggregate_rating.py --key results/s5/review/blind_key.json`）。

| 维度 | A | B | E | F |
| --- | --- | --- | --- | --- |
| 视觉指令遵循 | 3.70 | 3.70 | 3.70 | 3.70 |
| 画面结构和细节 | 4.00 | 4.00 | 4.00 | 4.00 |
| 时间连续性 | 4.00 | 4.00 | 4.00 | 4.00 |
| 运动自然度 | 4.00 | 4.00 | 4.00 | 4.00 |
| 声音质量和遵循 | 4.00 | 4.00 | 3.50 | 3.50 |
| 音画同步 | 4.00 | 4.00 | 4.00 | 4.00 |
| 总平均 | 3.95 | 3.95 | 3.87 | 3.87 |

- 判定标准（与 5090 相同）：均分下降 ≤ 0.25 且无新增严重失败。B 对 A、F 对 E 下降都是 0，没有任何视频被标严重失败，**都通过**。逐条看，FP8 组和对应的 BF16 组每条用例每个维度的分数完全一样。
- 失分只有两类：四组都在 07（拍两次手，3 分）、08（敲三次，2 分）的指令遵循上失分，与自动计数一致；四步组（E、F）在对白 17、18、20（3 分）和 19（两人对白，2 分）的声音上失分，与 SQUIM 的结论一致。
- 5090 的盲评（8 条用例，部分不同）四组均分 3.88–3.92，结论一致：量化组不比基线差，四步组的声音稍弱。

### 8.2 自动指标

| 指标（10 条平均；对白 4 条） | A | B | E | F | B−A | F−E | 5090：H3−A / F−E（20 条） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLAP 音频-文本 | 0.293 | 0.294 | 0.277 | 0.279 | +0.001 | +0.002 | −0.001 / +0.010 |
| 对白识别错误率（17 18 19 20） | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| SQUIM PESQ | 3.21 | 3.07 | 2.17 | 2.14 | −0.14 | −0.03 | −0.06 / +0.41 |
| SQUIM STOI | 0.977 | 0.977 | 0.906 | 0.907 | 0 | +0.001 | −0.005 / +0.013 |
| SQUIM SI-SDR（dB） | 27.1 | 25.2 | 12.1 | 12.8 | −1.9 | +0.7 | −2.0 / +4.9 |

- 拍手 / 敲击次数（期望 → A / B / E / F）：07 号 2 → 3 / 3 / 4 / 4；08 号 3 → 7 / 7 / 9 / 9。与 5090 一样普遍数不准，属于模型能力问题，各组之间没有差别。
- B 对 A：CLAP 不变，对白全对，PESQ 低 0.14、SI-SDR 低 1.9 dB，四条对白里有三条 B 略低（17：3.30 对 3.50，19：3.91 对 4.22，20：2.40 对 2.53），18 号 B 略高（2.68 对 2.60）。降幅与 5090 上 FP8（H3）对 A 的降幅同一量级。
- F 对 E：各项差异都在 ±0.03 PESQ、±0.7 dB 以内，可以认为没有差别。5090 的 F 用 FP8 VAE，语音指标反而比 E 高，R9700 的 F 是 BF16 VAE，不可比。
- 四步组（E、F）的语音质量明显低于 29 步组（PESQ 2.1–2.2 对 3.1–3.2，SI-SDR 12–13 对 25–27 dB），与 5090 的结论相同；对白内容仍然全对。
- CLAP 只有 10 条用例，单条波动大（如 20 号四组都接近 0），只看组间差。

## 9. 长视频与最大规格（阶段 6，2026-11-19）

### 9.1 P3（768×1344×362 帧，约 15 s）单卡跑不了

packed 序列 109222 token，是 P2 的 2.9 倍。A、B、E 在去噪第 1 步几秒内 OOM（`results/s6/`、`results/s6b/`，A、B 各 3 个种子都一样）；F 在第 1 步出现 GPU 内存访问错误（`Memory access fault ... Page not present`），进程以 rc=134 退出（`logs/s6/F_fp8triton_block_4step_P3.log`），原因是 FP8 GEMM 的 int32 地址溢出（6.5）。

### 9.2 最大可跑规格探测（L1、种子 42、单条、无预热）

H3 帧数必须是 17n+5。768×1344 下对 E、F 在 P2（124 帧）和 P3（362 帧）之间二分，另测 544×960×362 帧（P1 分辨率的完整 15 s）。每次是新进程、第一次见到该长度。脚本 `tools/run_s6probe_r9700.sh`，记录 `logs/s6p/probe.log`，数据 `results/s6p/`。

| 规格 | 时长 | E BF16 四步 | F `fp8-triton` 四步 |
| --- | --- | --- | --- |
| 768×1344×175 | 7.3 s | — | 成功，390.5 s，整卡 31.1GiB |
| 768×1344×192 / 209 | 8.0 / 8.7 s | — | 第 1 步 OOM |
| 768×1344×243 | 10.1 s | 成功，651.0 s，31.5GiB | 第 1 步 OOM |
| 768×1344×260 | 10.8 s | 成功，715.5 s，31.6GiB | — |
| 768×1344×277 | 11.5 s | **成功，789.8 s，31.5GiB** | — |
| 768×1344×294 | 12.3 s | 第 2 步 OOM（运行 199 s 后） | — |
| 544×960×362 | 15.1 s | **成功，425.4 s，29.2GiB** | 第 1 步 OOM |

- 耗时为单条请求的端到端。
- 成功的几条整卡显存都贴近 32GiB 上限，余量不到 1GiB。
- F 只到 175 帧、而 E 到 277 帧，有两个原因：6.4 的自动调优额外显存（每个新长度首条请求 +6–10GiB；F 的 OOM 都在第 1 步几秒内），以及 6.5 的 int32 地址溢出。补测（2026-11-20，`results/s6q/`）：固定 GEMM 配置后 F 跑通 243 帧（621.8 s），277 帧因溢出崩溃。
- A、B 没有单独探测，A 与 E 同为 BF16（A 少 LoRA），B 与 F 同为 `fp8-triton`，下面在 175 帧上实测了它们。

### 9.3 A、B 多种子（768×1344×175 帧，L1，种子 42 / 123 / 2026）

规则（事先定好）：若 E、F 都能跑 544×960×362 就用它，否则用 E、F 都能跑的最长 768×1344 规格，即 175 帧。单卡，2026-11-19。

| 组 | 端到端中位数（范围） | 每步去噪 | 视频 VAE 解码 | 整卡显存峰值 | 主机内存 |
| --- | --- | --- | --- | --- | --- |
| A BF16 29 步 | **2288.4 s**（2287.2–2293.4，3 条） | 76.9 s | 53.0 s | 26.1GiB | 142GiB |
| B `fp8-triton` 29 步 | **2146.3 s**（2146.1–2146.4，2 条） | 72.0 s | 53.0 s | 27.9GiB | 110GiB |

- B 的种子 42 计时无效，速度只取种子 123、2026 两条；生成结果是确定的，三个种子的质量数据都有效。
- B 比 A 快 1.07×，与 P2 的 1.08× 一致。每步耗时是 P2 的 1.73 倍，而 token 数是 1.41 倍（约 5.3 万对 3.8 万），注意力的平方项占比更大。

L1 普通话旁白（`logs/s6m/l1_seeds.json`、`logs/s6m/audio_metrics.json`）：

| | A 42 / 123 / 2026 | B 42 / 123 / 2026 | F 42 |
| --- | --- | --- | --- |
| CER（整段旁白 40 字） | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| SQUIM PESQ | 2.73 / 2.97 / 2.83 | 2.64 / 2.93 / 2.79 | 1.98 |
| SQUIM SI-SDR（种子 42） | 14.3 dB | 13.6 dB | 7.7 dB |
| CLAP（种子 42） | 0.269 | 0.252 | 0.200 |

- L1 的旁白只有 40 个字，7.3 s 内完整说完，没有被截断；6 条的识别文本完全一致（同种子 B 对 A 的文本差异为 0）。
- B 的 PESQ 每个种子都比 A 低 0.04–0.09，方向与 P2（低 0.14）一致，幅度更小。每秒响度曲线两组相近。四步的 F 语音质量明显更低，与 P2 结论一致。

## 10. 产物索引

| 类型 | 位置 |
| --- | --- |
| 环境指纹 | `logs/stage0/summary.json` |
| 微基准 | `logs/stage0/fp8_gemm*.json`、`logs/stage0/attn.json` |
| 测速汇总 | `logs/s1/summary_P1.json`（含 `fp8-rocm`）、`logs/s2/summary_P2.json`、`logs/s6m/summary_768x1344x175.json` |
| 数值检查 | `logs/s3/compare_tensors_P2.json`、`results/s3/`、`results/s4/*/P2/*.tensors.pt` |
| 看片页 | `results/s4/review/index.html`（标注组名） |
| 人工盲评 | 页面 `results/s5/review/rate.html`（匿名，对应关系 `results/s5/review/blind_key.json`），评分 `logs/s5/r9700_s5_rating_anonymous.json`，汇总 `logs/s5/rating_summary.json` |
| 音频指标 | `logs/s5/audio_metrics.json`（P2 10 条）、`logs/s6m/audio_metrics.json`、`logs/s6m/l1_seeds.json`（L1） |
| P3 失败记录 | `results/s6/`、`results/s6b/`、`logs/s6/`、`logs/s6b/` |
| 最大规格探测 | `logs/s6p/probe.log`、`results/s6p/`；A、B 多种子 `results/s6m/` |
| 自动调优显存诊断 | `logs/validate/fp8_spike_probe.{py,txt}`、`logs/validate/bench_single_config.py`、`results/diag/`、`logs/diag/`；固定配置的 F 最大规格补测 `results/s6q/`、`logs/s6q/` |
| int32 溢出复现 | `logs/validate/fp8_int32_probe.{py,txt}` |
| 分配器环境变量探针 | `logs/validate/vram_probe_envvars.txt` |
| 问题清单 | `minimax-lightx2v-r9700-issues.md` |
| 每条请求 | `results/<stage>/<group>/<spec>/runs.jsonl`、`kfd_annotations.json`、`gpu_samples.csv`、`kfd.csv` |
| 开启 `expandable_segments` 时的阶段 2 旧数据（F 后几条 OOM，B 中途停止；仅作旁证） | `results/s2_expandable_oom/`、`logs/s2_expandable_oom/` |
| 早期开启 `expandable_segments` 时测的 E（P2） | `results/s2_hip5_c3/`、`logs/s2_hip5_c3/` |
| 分配器显存回收探针 | `logs/validate/vram_probe.py`、`logs/stage0/vram_probe.txt` |
