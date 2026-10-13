# FlagGems-sglang 算子测试说明（Radeon PRO W7900D，20261013_0218_numa1）

测试结果见 [W7900D_report.md](W7900D_report.md)，与 RTX 4090 的对比见 [W7900D_vs_RTX4090.md](W7900D_vs_RTX4090.md)。原始数据都在 `results_20261013_0218_numa1/` 下，下文的相对路径以这个目录为准（`diag/` 和脚本除外）。

## 测试方法

与 RTX 4090 的测试（[RTX4090_testing.md](RTX4090_testing.md)）用同一个提交、同一套测试文件和同样的测量参数，只把 CUDA 专用的部分换成 ROCm 的对应做法。

- 被测代码没有任何修改：`FlagGems-sglang-b7a575a/` 与 b7a575a 的源码包逐文件一致（`diff -r`，683 个文件）。56 个算子在本机全部解析到通用实现 `flaggems_sglang.ops`（`op_resolution.txt`），仓库的 `runtime/backend/_amd/ops` 没有注册任何算子。
- 每个算子单独起一个 pytest 进程（`run_suite_amd.sh`，由 `RTX4090_run_suite.sh` 移植），一个 kernel 出错不会影响其他算子。每个进程的时限与 4090 相同，为 1800 秒。
- 正确性：`tests/test_<算子>.py`，与仓库自带的 PyTorch 参考实现（`src/flaggems_sglang/reference/`）对比。
- 性能：`benchmark/test_<算子>.py --level comprehensive`，共 484 个形状。
- kernel 模式：`triton.testing.do_bench`（warmup 1000 ms、rep 100 ms，每次测量前清空 L2，取中位数），只计 GPU 时间。跑两遍，结果表取第 1 遍，第 2 遍用于检查稳定性。两遍不一致的点所在的算子再跑第 3 遍，用第 3 遍替换偏离的一遍，见「稳定性说明」。
- operator 模式：连续调用的墙钟时间，包含 Python 封装和 kernel 启动开销，不清空 L2。只跑一遍，warmup / iter 为调用次数。快慢分组直接沿用 4090 的分组，两台机器上每个算子的调用次数相同（见下方参数）。
- 加速比 = 参考实现延迟 / 算子延迟，大于 1 表示算子更快；每个算子取所有形状的几何平均。† 表示参考实现带 Python 循环或 `.item()` 等主机同步，加速比会被放大，与 4090 是同样的 21 个算子。
- 计时：HIP event 的刻度约 0.04 µs，比 CUDA event 的约 1.024 µs 细得多。但 kernel 模式有约 10 µs 的计时下限：只读写一个元素的 Triton kernel 测得 10.08 µs，PyTorch 单元素加法测得 9.56 µs（`calib_timer.json`），RTX 4090 上约 3 µs。所以小 kernel 的算子延迟和参考实现延迟都被抬到 10 µs 左右，加速比被压向 1。operator 模式下单次调用的下限是 Triton kernel 7.33 µs、PyTorch 单元素加法 3.45 µs。
- 环境：`HIP_VISIBLE_DEVICES` 选卡，并清除 `CUDA_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`（HIP 会把它们叠加起来，改变设备编号）；`DNN_VENDOR=amd`；所有阶段共用一个 Triton 缓存（`TRITON_CACHE_DIR`），与 4090 相同。
- **需要绑核**：性能测试进程必须绑定到 GPU 所在的 NUMA 节点（CPU 和内存都绑定），否则 CPU 开销大的参考实现和 operator 模式的耗时会明显波动。本次为 `numactl --cpunodebind=1 --membind=1`，方法见下方「绑核方法」。

### 绑核方法

由 `run_suite_amd.sh` 自动完成：先找到 GPU 的 PCI 地址和它所在的 NUMA 节点，再用 `numactl` 启动每个 pytest 进程，把进程的 CPU 和内存都限定在这个节点上。和 4090 的做法不同，PCI 地址直接向 HIP 运行时查询：本机 rocm-smi 的编号与 HIP / torch 的编号不保证一致，从运行时查到的设备一定是 `HIP_VISIBLE_DEVICES` 选中的那张卡。核心逻辑：

```bash
bus=$(python -c "import torch; p = torch.cuda.get_device_properties(0); print(f'{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0')")  # 0000:e3:00.0
node=$(cat "/sys/bus/pci/devices/${bus}/numa_node")  # 1
numactl --cpunodebind="$node" --membind="$node" python -m pytest benchmark/test_<算子>.py ...
```

- 本次用 HIP / torch 编号为 7 的 GPU（`HIP_VISIBLE_DEVICES=7`），PCI 地址 `0000:e3:00.0`，所在 NUMA 节点为 1。KFD 拓扑中这张卡的 `gpu_id` 为 37516，监控和进程统计都按它来认卡。
- `--cpunodebind=1`：进程只在节点 1 的 CPU 上运行，即 CPU 32–63、96–127，共 64 个逻辑 CPU。绑的是整个节点而不是某一个核，线程仍由系统在节点内的 CPU 之间调度。
- `--membind=1`：进程的内存只从节点 1 分配。
- 每个阶段目录下的 `pinning.txt` 记录了所用的命令、GPU 的 PCI 地址和 `numactl --show` 的输出，可以用来核对。本次 `perf_kernel_run1/`、`perf_kernel_run2/`、`perf_kernel_run3/`、`perf_operator/` 都绑在节点 1（`perf_operator/pinning.txt` 有快、慢两组各一条记录）。
- GPU 监控（`gpu_monitor_amd.sh`，每秒读一次 sysfs）绑在节点 0，不和测试进程争用 CPU。
- 设置环境变量 `NUMA_NODE=n` 可以手动指定节点，`NUMA_NODE=none` 关闭绑核。
- 正确性测试没有绑核，CPU 位置不影响数值结果。

### operator 模式参数

```
operator mode: warmup/iter are call counts; groups taken verbatim from the RTX 4090 run (RTX4090_testing.md)
fast: --warmup 100 --iter 1000: bmm_chunk causal_conv1d_update chunk_cumsum chunk_local_cumsum_scalar chunk_local_cumsum_vector chunk_scaled_dot_kkt chunked_embedding_lora_a chunked_sgmv_expand chunked_sgmv_shrink draft_topk1 embedding_lora_a ernie45_rope_fused fla_layernorm_gated fused_dual_residual_rmsnorm fused_gdn_gating fused_moe_router_cudacore fused_moe_router_tensorcore fused_rmsnorm gate_up_lora_b gelu_and_mul interleaved_rope l2norm log_scaling_tau mamba_layernorm_gated merge_state moe_fused_gate moe_fused_mul_sum moe_sum_reduce mrope_fused per_group_transpose per_token_group_quant_int8 per_token_quant_int8 qkv_lora_b rotary_embedding sgemm_lora_a sgemm_lora_b sigmoid_gate_topk_renorm silu_and_mul silu_and_mul_masked softcap_inplace_logits softcap_out state_passing
slow: --warmup 5 --iter 20: apply_token_bitmask causal_conv1d_fn chain_speculative_sampling chunk_state chunk_state_varlen context_attention decode_attention decode_grouped_attention extend_attention fused_moe_gemm fused_norm_rope_stacked hc_head selective_state_update w8a8_block_int8_matmul
```

如果按本机第 1 遍的参考实现延迟重新分组（以 5 ms 为界），只有 `rotary_embedding` 会换组（本机最大 7.7 ms），为保持调用次数一致，没有调整。

## 运行情况

| 项目 | 版本 / 配置 |
|---|---|
| 被测仓库 | flagos-ai/FlagGems-sglang @ b7a575a |
| GPU | AMD Radeon PRO W7900D（GPU 7，PCI 0000:e3:00.0，48 GB，gfx1100，功耗上限 241 W） |
| 驱动 | amdgpu 6.16.13 |
| ROCm | 7.2.3（HIP 7.2.53211） |
| PyTorch | 2.10.0+rocm7.2.3.git1a270074 |
| Triton | 3.6.0+rocm7.2.3.git4ed88892 |
| Python | 3.12.3 |
| pytest | 9.0.3 |
| 容器 | Docker 容器，PyTorch / Triton 为 ROCm 版 wheel（`/opt/venv`） |
| fp32 矩阵乘 | 完整 FP32：gfx1100 没有 TF32，进程中 `torch.backends.cuda.matmul.allow_tf32` 默认为 False，性能测试框架导入时也会设为 False |
| CPU | AMD EPYC 9334 32-Core Processor（2 路 × 32 核 × 2 线程，2 个 NUMA 节点） |
| 系统 | Ubuntu 24.04.4 LTS，内核 6.8.0-138-generic |

- GPU 型号：PCI 设备号 `1002:744b`、VBIOS 113-D7070910-100。本机的 pci.ids 版本较旧，`lspci` 只显示 "Device 744b"，型号按 pci-ids 数据库记为 Radeon PRO W7900D。
- 全部在 GPU 7 上运行，本机其他 GPU 没有使用。本机没有安装 sgl_kernel、aiter、flashinfer、vllm。
- 时间（2026-10-13，UTC）：正确性测试 02:26–03:31；kernel 模式第 1 遍 03:31–04:04、第 2 遍 04:04–04:17，operator 模式 04:17–04:31；第 3 遍 04:36–04:40；`silu_and_mul_masked` 的正确性补跑 04:40–04:41（见下文「超时与补跑」）。
- 性能测试的各个阶段，GPU 7 上只有测试进程本身（KFD 按进程统计的显存占用，进程数最多为 1），空闲时显存占用 27 MiB。正确性测试期间，02:41 前后约 6 秒有第二个进程，是我在同一张卡上运行的 `chain_speculative_sampling` 诊断脚本，不影响正确性结果。
- 频率、功耗和温度见下表，监控数据见 `gpu7_monitor.csv` 和 `perf_kernel_run3/gpu7_monitor.csv`。「GPU 忙」指 `gpu_busy_percent` ≥ 50% 的采样；功耗为 hwmon `power1_average` 的 1 秒采样，个别采样超过 241 W 的上限。

| 阶段 | GPU 忙时 sclk 中位数 | sclk 最高 | GPU 忙时功耗中位数 | 功耗最高 | 结温最高 | 显存温度最高 | GPU 上的进程数（最多） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 正确性 | 1984 MHz | 2255 MHz | 180 W | 284 W | 62°C | 54°C | 2 |
| kernel 第 1 遍 | 2172 MHz | 2267 MHz | 157 W | 306 W | 66°C | 60°C | 1 |
| kernel 第 2 遍 | 2183 MHz | 2266 MHz | 158 W | 289 W | 67°C | 60°C | 1 |
| operator | 2200 MHz | 2257 MHz | 138 W | 289 W | 68°C | 62°C | 1 |
| kernel 第 3 遍 | 2207 MHz | 2254 MHz | 141 W | 288 W | 64°C | 58°C | 1 |

- 主测试启动后（02:31），`run_all_amd.sh` 改为从 PCI 地址推导监控所用的节点和 rocm-smi 编号，此前监控节点写死为 0。本机 GPU 在节点 1，两种写法的结果相同。`env.txt` 由修改前的版本生成，所以没有 `monitor pin` 一行。
- 原始数据：`perf_detail.csv`（三遍 kernel 模式和 operator 模式每个形状的原始延迟，单位 ms）、`replacements.csv`（不稳定点的三遍数据和处理结果）、`perf_summary.csv`（替换后的各算子汇总）、`summary.json`、`accuracy.csv`，以及 `acc/`、`perf_kernel_run1/`、`perf_kernel_run2/`、`perf_kernel_run3/`、`perf_operator/`、`supp/acc/` 下每个算子的日志和 JSON。

## 正确性：`chain_speculative_sampling` 用例 7、8 失败的原因

- 用例 7：B=512，候选序列长度 4，词表 32000；用例 8：B=4096，其余相同。输入为 fp32。
- 只有输出的 token（predicts）不一致：用例 7 有 1 / 2048 个位置不同，用例 8 有 3 / 16384 个。接受位置（accept_index）和接受数（accept_token_num）完全一致。
- 每个不一致的位置，算子和参考实现选中的都是相邻的两个 token。采样阈值离这两个 token 之间的累积分布（CDF）边界只有总概率质量的 5.7e-9 到 5.5e-8，小于 fp32 的相对精度（约 1.2e-7），结果取决于累加顺序。
- 用 fp64 重新计算：3 个位置中有 1 个与算子一致、2 个与参考实现一致，两边都没有系统性错误。4090 上这 9 个用例全部通过。
- 诊断输出见 `diag_chain_speculative_sampling.txt`（脚本 `diag/diag_chain_speculative_sampling.py`）。

| 用例 | 行 | 算子 token | 参考实现 token | fp64 token | 与 CDF 边界的距离（占总概率质量） |
|---|---:|---:|---:|---|---:|
| 7（B=512） | 506 | 2564 | 2563 | 2564（同算子） | 5.67e-9 |
| 8（B=4096） | 284 | 27883 | 27882 | 27882（同参考实现） | 5.54e-8 |
| 8（B=4096） | 506 | 2564 | 2563 | 2564（同算子） | 5.67e-9 |
| 8（B=4096） | 1429 | 25477 | 25478 | 25478（同参考实现） | 2.33e-8 |

## 正确性：`fused_moe_router_tensorcore` 用例 7 失败的原因

- 用例参数：bs=4096、hidden=4096、256 个专家、topk=2，输入为 bf16。
- 4096 行的专家 ID，算子和参考实现都与 fp64 结果完全一致。
- 只有权重的第 (955, 1) 个元素超出容差：算子 0.4400331，参考实现 0.4401835，fp64 0.4401427。两边的误差符号相反（−1.10e-4 和 +4.1e-5），相差 1.50e-4，超过测试的 fp32 权重容差（atol 1e-4 + rtol 1e-4 × 0.44 ≈ 1.44e-4）。
- 以 fp64 为真值，两边精度相当：权重最大误差算子 1.20e-4、参考实现 1.23e-4；参考实现 logits 的最大误差为 6.15e-4（这一行 fp64 的前两大 logits 为 172.124487 和 171.883904）。
- 原因：两边都按 fp32 计算，只是累加顺序不同。参考实现走 PyTorch 的 fp32 矩阵乘；算子在 ROCm 上固定用 `input_precision="ieee"`（`ops/fused_moe_router_tensorcore.py` 第 580–581 行）；NVIDIA 上按 PyTorch 的 TF32 开关选择，开关关闭时用 tf32x3（拆成三次 TF32 乘法，精度接近 fp32）。hidden=4096、logits 约 172 时，fp32 舍入误差经 softmax 传到权重上已接近 1e-4，这个容差对 fp32 计算本身就很紧。4090 的正式结果关闭了 TF32，算子用 tf32x3、参考实现为完整 FP32，该用例通过。
- 诊断输出见 `diag_fused_moe_router_tensorcore.txt`（脚本 `diag/diag_moe_router.py`）。

| 指标（误差均相对 fp64 计算） | 算子 | 参考实现 |
|---|---|---|
| 权重最大误差 | 1.20e-4 | 1.23e-4 |
| 第 (955, 1) 个权重 | 0.4400331（误差 −1.10e-4） | 0.4401835（误差 +4.1e-5） |
| 专家 ID 与 fp64 不一致的行 | 0 / 4096 | 0 / 4096 |
| 该算子的 8 个用例 | 7 个通过，1 个失败 | - |

## 超时与补跑：`silu_and_mul_masked`

- 主测试的正确性阶段，这个算子的 pytest 进程在 1800 秒时限处被终止（03:00–03:30，退出码 124），没有产生结果。原因是 Triton autotune 在 gfx1100 上的编译时间过长，不是死循环或 kernel 出错。
- 测试走的是按行处理的 kernel `_silu_and_mul_masked_row_kernel`（E×T ≤ 1024 时），它的 autotune 有 57 个配置（BLOCK_T 1–32 × num_warps 4/8/16 × num_stages 2/3/4，另加 3 个 num_warps=2），按 (T, H) 分别调优。H 是编译期常量，每个程序要把 BLOCK_T × H/2 个 gate 和 up 元素都放在寄存器里。H=4096 时，BLOCK_T=32 的 tile 为 32×2048：
  - num_warps=4：每个版本 32,617 条指令，每条 lane 寄存器溢出 5.9 KB，编译约 173 秒。这个 kernel 没有循环，num_stages 2/3/4 生成的代码逐字节相同，但要分别编译 3 次。
  - num_warps=2：54,821 条指令，溢出 10.4 KB，单独编译了 972 秒。
- 正确性阶段被终止前，已编译出 170 个按行版本（相邻缓存写入的间隔合计 898 秒）。最后一个在 03:15:08 写入，此后约 15 分钟都在编译 32×2048、num_warps=2 这一个版本，直到进程被终止；它在 kernel 第 1 遍才编译完成。缓存里只有 num_stages 不同的版本，按行 kernel 有 54 组、按列 kernel 有 82 组，每组的代码都逐字节相同。4090 上第一次运行这个算子的正确性测试（容器默认 TF32 的对照）总共 144 秒；关闭 TF32 的正式那次为 54 秒，应是用上了之前运行留下的 Triton 缓存。
- 性能测试不受影响：kernel 第 1 遍里这个算子用了 1203 秒（接着完成编译，在时限内，退出码 0），第 2 遍和 operator 模式各 53 秒（缓存命中），484 个形状的数据都完整。
- 补跑：所有阶段结束后，只对这个算子的正确性测试用 `run_suite_amd_supp.sh` 重跑一次。这个脚本是 `run_suite_amd.sh` 的副本，唯一的区别是时限可以通过 `TIMEOUT` 设置（本次 10800 秒），测试代码、参数、GPU 和不绑核的设置都与主测试相同。5 个用例全部通过，用时 50 秒：Triton 缓存里已经有 kernel 第 1 遍编译好的版本，放宽的时限实际没有用到。`W7900D_report.md` 中这个算子的正确性标 ‡。两个脚本的差异：

```diff
-  timeout 1800 ${PIN[@]+"${PIN[@]}"} "$PY" -m pytest -p no:cacheprovider "${flags[@]}" "$target" \
+  timeout "${TIMEOUT:-1800}" ${PIN[@]+"${PIN[@]}"} "$PY" -m pytest -p no:cacheprovider "${flags[@]}" "$target" \
```

（另有文件头注释，以及在 `pinning.txt` 里记录 `timeout: 10800s` 的一行。）

- 诊断输出见 `diag_compile_cost.txt`（脚本 `diag/diag_compile_cost.py`，只读取 Triton 缓存）。编译时间按相邻两次缓存写入的间隔估算，autotune 逐个编译配置，所以这个间隔近似于单个版本的编译时间。

## 正确性：`fused_moe_router_tensorcore` 用了 1709 秒

- 没有超时，但接近 1800 秒的时限（4090 上第一次运行为 328 秒，关闭 TF32 的正式那次为 103 秒）。正确性阶段共编译了 268 个 GEMM 版本（`_router_gemm_kernel`，autotune 按 (M, N, K) 分别调优），相邻缓存写入的间隔合计 1646 秒。
- 其中 24 个版本需要 128 KiB 或 256 KiB 的 LDS，超过 gfx1100 每个 workgroup 64 KiB 的上限（KFD 拓扑中这张卡的 `lds_size_in_kb` 为 64），在这张卡上无法运行；但 Triton 先编译、启动时才检查资源，这 24 个版本占了 958 秒编译时间。最慢的是 256×256 tile、num_warps=8、256 KiB LDS 的版本，两次分别编译了 301 秒和 228 秒，每个约 16–17 万条指令。
- 算子在 ROCm 上用 `input_precision="ieee"`，这 268 个版本的 `tl.dot` 全部编译成 FMA，没有一条 WMMA 矩阵指令（`diag_dot_lowering.txt`），大 tile 因此生成很长的代码。
- 性能测试中这个算子每遍用时 96–99 秒（缓存命中）。

## 性能数据的解读

- 基线是仓库自带的 PyTorch 参考实现（ROCm 版 PyTorch），不是 SGLang 线上使用的 kernel（本机没有安装 sgl_kernel / aiter）。
- 标 † 的 21 个算子，参考实现带 Python 循环或 `.item()` 等主机同步，加速比被大幅放大（例如 `chunk_state_varlen` 约 254 倍、`w8a8_block_int8_matmul` 约 104 倍），不代表相对生产实现的收益。
- 受 10 µs 计时下限的影响，算子延迟小于 10 µs 的点有 109 个（4090 上 282 个），参考实现没有小于 10 µs 的点（4090 上 16 个）。这些点的加速比主要反映两边的启动和计时开销。
- 参考实现中有 20 个算子先转 fp32 再做矩阵乘 / einsum / 注意力。两台机器上它们都按完整 FP32 计算：本机没有 TF32，4090 的性能测试框架在导入时关闭了 TF32（`benchmark/performance_utils.py`），所以两边的加速比里没有 TF32 带来的差别。
- Triton 的 `tl.dot` 在 fp32 输入、又没有指定精度时，用的是 Triton 后端自己的默认值，不受 PyTorch 的 TF32 开关影响：NVIDIA 默认用 TF32，gfx1100 只能用 ieee fp32（本机 Triton 允许的精度为 ieee、bf16x3、bf16x6），编译出来是 FMA。bf16 / int8 输入的 `tl.dot` 在本机用上了 WMMA 矩阵指令（例如 `_w8a8_int8_dot`、`_bmm_chunk_kernel`、`_extend_attention_kernel`，见 `diag_dot_lowering.txt`）。
- kernel 模式下有 4 个点比参考实现慢：
  - `context_attention` 的两个小形状（0.71、0.85）。这个 benchmark 的 10 个形状都是 fp32，它的 `tl.dot` 在本机全部编译成 FMA，每条 lane 有 1.0–2.6 KB 寄存器溢出。4090 上同样两个形状的算子只要 50.2 µs 和 34.8 µs，本机为 429.2 µs 和 280.3 µs。较大的形状仍比参考实现快（最高 3.14）。
  - `fused_moe_router_tensorcore` 的 bs=4096 形状（0.76，算子 3780.5 µs 对参考实现 2871.5 µs）。如上所述，算子在 ROCm 上按 ieee fp32 做 FMA 矩阵乘。
  - `fused_moe_router_cudacore` 的 `[[83,512],[32,512],3,0.0,[32]]`（0.32，算子 261.6 µs 对参考实现 82.8 µs，4090 上算子 13.7 µs）。专家数 ≤ 32 时，算子走单 kernel 路径 `_router_kernel`，固定 `num_warps=1`；这个形状的 tile 为 32×512，本机编译结果每条 lane 寄存器溢出 2,056 字节，同一路径上的另外两个形状（tile 8×64、16×256）没有溢出。4090 上这个算子慢的是 256 个专家的两个大形状（0.82、0.67），在本机分别为 3.81 和 2.17。
- operator 模式下，484 个点中有 19 个比参考实现慢：`draft_topk1` 4 个、`chunk_local_cumsum_vector` 3 个、`moe_sum_reduce` 3 个、`chunk_local_cumsum_scalar` 2 个、`context_attention` 2 个、`fused_moe_router_tensorcore` 2 个、`log_scaling_tau` 2 个、`fused_moe_router_cudacore` 1 个。除了上面 kernel 模式就慢的 4 个点，其余 15 个在 kernel 模式下都比参考实现快，端到端变慢来自 Python 封装和 Triton kernel 启动的 CPU 开销：本机一次 Triton kernel 调用的墙钟下限为 7.33 µs，PyTorch 单元素加法为 3.45 µs。例如 `draft_topk1` 的 [[1,151936]] 形状，算子 59.8 µs，参考实现 33.0 µs。
- kernel 模式每次测量前清空 L2，operator 模式连续调用、L2 是热的，所以同一形状 kernel 模式的 GPU 时间可能比 operator 模式的墙钟时间还长（本机 484 个点中有 180 个）。
- operator 模式只跑了一遍，单个形状的波动没有量化，几十 µs 量级、以 CPU 开销为主的点可能有较大的单次偏差。
- 仓库的通用实现兼顾多个厂商平台的竞赛评分，本机上的数字不等同于竞赛成绩。

## 稳定性说明

kernel 模式跑了两遍。只看第 1 遍延迟 ≥10 µs 的点：第 1、2 遍的算子延迟或参考实现延迟相差超过 5%、且超过 2.048 µs 的，记为不稳定。2.048 µs 是 4090 上的 2 个计时刻度，这里沿用同一阈值以保持可比。10 µs 以下的点不做判定。

第 1、2 遍共有 20 个不稳定的点，分布在 6 个算子上：参考实现延迟不稳定的 19 个，算子延迟不稳定的 1 个。

这 6 个算子用相同的参数和绑核又跑了一遍 kernel 模式（第 3 遍，`results_20261013_0218_numa1/perf_kernel_run3/`）。对每个不稳定的点，看第 3 遍的不稳定项更接近第 1 遍还是第 2 遍，保留接近的一遍，另一遍整条换成第 3 遍的结果（参考实现延迟、算子延迟和加速比一起换）。如果第 3 遍离两遍的距离都比两遍之间的距离还大，就算作与哪一遍都不接近，这个点不替换。`W7900D_report.md` 只用替换后的第 1 遍；三遍的原始数据见 `results_20261013_0218_numa1/perf_detail.csv`，替换记录见 `results_20261013_0218_numa1/replacements.csv`。

结果：17 个点做了替换（替换第 1 遍 11 个、第 2 遍 6 个），其中 15 个点的第 3 遍与保留的一遍相差在 5% 以内，2 个点替换后两遍仍相差超过 5%；3 个点的第 3 遍与两遍都不接近，没有替换。仍不稳定的 5 个点在下表「处理」一列注明。

| 算子 | dtype | 形状参数 | 不稳定项 | 第 1 遍 (µs) | 第 2 遍 (µs) | 第 3 遍 (µs) | 处理 |
|---|---|---|---|---:|---:|---:|---|
| `chain_speculative_sampling` | fp32 | `[[4096,4],[4096,4],[4096,3],[4096],[4096,4,32000],[4096,3,32000],16384]` | 参考实现延迟 | 1917008 | 2060638 | 2005232 | 替换第 1 遍 |
| `chunk_state_varlen` | bf16 | `[[8192,8,128],[8192,64,64],[64,32,256],[64,32,256],[33],[32,64,64,128]]` | 参考实现延迟 | 219115 | 179437 | 193862 | 替换第 1 遍（替换后仍相差超过 5%） |
| `chunk_state_varlen` | bf16 | `[[32,2,16],[32,8,32],[8,2,16],[8,2,16],[5],[2,8,32,16]]` | 参考实现延迟 | 3498.2 | 3138.4 | 3233.9 | 替换第 1 遍 |
| `chunk_state_varlen` | bf16 | `[[96,4,32],[96,16,64],[16,3,32],[16,3,32],[6],[3,16,64,32]]` | 参考实现延迟 | 8990.5 | 7569.6 | 8111.6 | 替换第 1 遍（替换后仍相差超过 5%） |
| `extend_attention` | bf16 | `[[246,12,80],[246,4,80],[246,4,80],[554,4,80],[554,4,80],[5],[5],[308],96]` | 参考实现延迟 | 1507.4 | 1595.1 | 1553.1 | 替换第 1 遍 |
| `extend_attention` | bf16 | `[[79,8,64],[79,8,64],[79,8,64],[154,8,64],[154,8,64],[3],[3],[75],53]` | 参考实现延迟 | 724.8 | 765.1 | 743.7 | 替换第 2 遍 |
| `fused_moe_router_tensorcore` | bf16 | `[[4096,4096],[256,4096],2,0.0,null]` | 算子延迟 | 3563.9 | 3783.0 | 3780.5 | 替换第 1 遍 |
| `per_token_quant_int8` | bf16 | `[[1,2048]]` | 参考实现延迟 | 47.52 | 51.96 | 51.32 | 替换第 1 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[1,4096],[1024,4096],[1,32],[8,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 11564 | 12183 | 11376 | 替换第 2 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[1,4096],[7168,4096],[1,32],[56,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 81863 | 93042 | 81194 | 替换第 2 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[8,4096],[1024,4096],[8,32],[8,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 11603 | 13106 | 11355 | 替换第 2 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[8,4096],[4096,4096],[8,32],[32,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 56732 | 52378 | 45401 | 未替换（第 3 遍与两遍都不接近） |
| `w8a8_block_int8_matmul` | bf16 | `[[8,4096],[7168,4096],[8,32],[56,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 82354 | 86713 | 96236 | 未替换（第 3 遍与两遍都不接近） |
| `w8a8_block_int8_matmul` | bf16 | `[[64,4096],[1024,4096],[64,32],[8,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 11567 | 12540 | 13622 | 未替换（第 3 遍与两遍都不接近） |
| `w8a8_block_int8_matmul` | bf16 | `[[64,4096],[4096,4096],[64,32],[32,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 46573 | 51013 | 50397 | 替换第 1 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[64,4096],[7168,4096],[64,32],[56,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 82330 | 88142 | 80087 | 替换第 2 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[512,4096],[4096,4096],[512,32],[32,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 46410 | 50662 | 45116 | 替换第 2 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[512,4096],[7168,4096],[512,32],[56,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 81609 | 88179 | 86613 | 替换第 1 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[64,512],[1024,512],[64,4],[8,4],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 1357.9 | 1234.8 | 1196.4 | 替换第 1 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[256,4096],[1024,4096],[256,32],[8,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 13818 | 12473 | 12263 | 替换第 1 遍 |

作为对照，这 6 个算子中前两遍一致的点（≥10 µs），第 3 遍相对前两遍均值的偏差：算子延迟 54 个点，中位偏差 0.0%，没有与两遍都相差超过 5% 的点；参考实现延迟 46 个点，中位偏差 0.1%，其中 2 个点与两遍都相差超过 5%（-14% 到 +8%）。这些点仍使用前两遍的数据。

替换后两遍的相对差异（|第 1 遍 − 第 2 遍| / 两遍均值），按第 1 遍的延迟分档，每格依次为点数 / 中位数 / P90 / 最大：

| 延迟档 | 算子延迟 | 参考实现延迟 |
|---|---:|---:|
| <10 µs | 109 / 2.6% / 6.6% / 10.3% | 0 / - / - / - |
| 10–50 µs | 229 / 1.4% / 4.9% / 9.6% | 119 / 0.4% / 1.7% / 3.9% |
| 50–500 µs | 104 / 0.3% / 1.3% / 5.0% | 213 / 0.2% / 1.4% / 3.9% |
| ≥0.5 ms | 42 / 0.3% / 1.9% / 3.9% | 152 / 0.6% / 3.0% / 8.1% |

替换前，两遍几何平均加速比相差超过 10% 的算子：没有。

替换后，两遍几何平均加速比相差超过 10% 的算子：没有。

补充说明：

- 不稳定的点几乎都在参考实现上（20 个点中 19 个）。这 19 个点除 `per_token_quant_int8` 的 1 个外，都是带 Python 循环的参考实现，其中 `w8a8_block_int8_matmul` 占 12 个。替换后仍不稳定的 5 个点也都属于这种情况。
- 算子延迟不稳定的只有 `fused_moe_router_tensorcore` 的 bs=4096 形状：三遍分别为 3563.9、3783.0、3780.5 µs，第 1 遍被替换，报告用的是 3780.5 µs（加速比 0.76）。这个算子每个进程都会重新 autotune（结果不缓存），不同进程选中的配置可能不同，这可能是差异的来源，本次没有核实。
- 4090 上小于 50 µs 的两档，算子延迟两遍差异的中位数为 0.0%：CUDA event 的刻度是 1.024 µs，小 kernel 重复测量常落在同一个刻度上。HIP event 的刻度约 0.04 µs，能看到这部分抖动（中位数 2.6% 和 1.4%）。10–50 µs 档里相差超过 5% 的点，绝对差都不到 2.048 µs，不算作不稳定；10 µs 以下的点不做判定。
