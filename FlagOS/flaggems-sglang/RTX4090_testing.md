# FlagGems-sglang 算子测试说明（RTX 4090，20261012_0758_numa3）

测试结果见 [RTX4090_report.md](RTX4090_report.md)。

## 测试方法

- 每个算子单独起一个 pytest 进程（`run_suite.sh`），一个 kernel 出错不会影响其他算子。
- 正确性：`tests/test_<算子>.py`，与仓库自带的 PyTorch 参考实现（`src/flaggems_sglang/reference/`）对比。命令为 `NUMA_NODE=none TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 ./run_suite.sh acc <输出目录>`。
- 性能：`benchmark/test_<算子>.py --level comprehensive`，共 484 个形状。
- kernel 模式：`triton.testing.do_bench`（warmup 1000 ms、rep 100 ms，每次测量前清空 L2，取中位数），只计 GPU 时间。跑两遍，结果表取第 1 遍，第 2 遍用于检查稳定性。两遍不一致的点所在的算子再跑第 3 遍，用第 3 遍替换偏离的一遍，见「稳定性说明」。
- operator 模式：连续调用的墙钟时间，包含 Python 封装和 kernel 启动开销，不清空 L2。只跑一遍，warmup / iter 为调用次数，分组见下方参数。
- 加速比 = 参考实现延迟 / 算子延迟，大于 1 表示算子更快；每个算子取所有形状的几何平均。† 表示参考实现带 Python 循环或 `.item()` 等主机同步，加速比会被放大。
- CUDA event 计时的刻度约为 1.024 µs，10 µs 以下的 kernel 差一个刻度就是 10%–29%。
- **fp32 矩阵乘不用 TF32**：本容器默认设置 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，PyTorch 的 fp32 矩阵乘会走 TF32（输入只保留 10 位尾数）。正确性测试给测试进程设置 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`，`acc/tf32_check.txt` 记录了同样设置下进程内的实际状态；性能测试框架 `benchmark/performance_utils.py` 在导入时设置 `torch.backends.cuda.matmul.allow_tf32 = False`，代码里的设置优先于容器默认值。两类测试的参考实现因此都是完整 FP32。
- **需要绑核**：性能测试进程必须绑定到 GPU 所在的 NUMA 节点（CPU 和内存都绑定），否则 CPU 开销大的参考实现和 operator 模式的耗时会明显波动。本次为 `numactl --cpunodebind=3 --membind=3`，方法见下方「绑核方法」。

### 绑核方法

由 `run_suite.sh` 自动完成：先根据 GPU 的 PCI 地址找到它所在的 NUMA 节点，再用 `numactl` 启动每个 pytest 进程，把进程的 CPU 和内存都限定在这个节点上。核心逻辑：

```bash
bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i 1 | tr 'A-F' 'a-f')  # 00000000:d8:00.0
node=$(cat "/sys/bus/pci/devices/${bus: -12}/numa_node")  # 3
numactl --cpunodebind="$node" --membind="$node" python3 -m pytest benchmark/test_<算子>.py ...
```

- 本次用 GPU 1，PCI 地址 `00000000:D8:00.0`，所在 NUMA 节点为 3。sysfs 里的 PCI 地址是小写、域号只有 4 位，所以取 `nvidia-smi` 输出的后 12 个字符。也可以用 `nvidia-smi topo -m` 的 `NUMA Affinity` 列核对。
- `--cpunodebind=3`：进程只在节点 3 的 CPU 上运行，即 CPU 54–71、126–143，共 36 个逻辑 CPU。绑的是整个节点而不是某一个核，线程仍由系统在节点内的 CPU 之间调度。
- `--membind=3`：进程的内存只从节点 3 分配。
- 每个性能测试阶段目录下的 `pinning.txt` 记录了所用的命令和 `numactl --show` 的输出，可以用来核对。本次 `perf_kernel_run1/`、`perf_kernel_run2/`、`perf_kernel_run3/`、`perf_operator/` 都绑在节点 3。
- 设置环境变量 `NUMA_NODE=n` 可以手动指定节点，`NUMA_NODE=none` 关闭绑核。
- 正确性测试没有绑核（`NUMA_NODE=none`），CPU 位置不影响数值结果。

### operator 模式参数

```
operator mode: warmup/iter are call counts; grouped by max latency_base in perf_kernel_run1
fast (max latency_base <= 5 ms): --warmup 100 --iter 1000: bmm_chunk causal_conv1d_update chunk_cumsum chunk_local_cumsum_scalar chunk_local_cumsum_vector chunk_scaled_dot_kkt chunked_embedding_lora_a chunked_sgmv_expand chunked_sgmv_shrink draft_topk1 embedding_lora_a ernie45_rope_fused fla_layernorm_gated fused_dual_residual_rmsnorm fused_gdn_gating fused_moe_router_cudacore fused_moe_router_tensorcore fused_rmsnorm gate_up_lora_b gelu_and_mul interleaved_rope l2norm log_scaling_tau mamba_layernorm_gated merge_state moe_fused_gate moe_fused_mul_sum moe_sum_reduce mrope_fused per_group_transpose per_token_group_quant_int8 per_token_quant_int8 qkv_lora_b rotary_embedding sgemm_lora_a sgemm_lora_b sigmoid_gate_topk_renorm silu_and_mul silu_and_mul_masked softcap_inplace_logits softcap_out state_passing 
slow (max latency_base > 5 ms): --warmup 5 --iter 20: apply_token_bitmask causal_conv1d_fn chain_speculative_sampling chunk_state chunk_state_varlen context_attention decode_attention decode_grouped_attention extend_attention fused_moe_gemm fused_norm_rope_stacked hc_head selective_state_update w8a8_block_int8_matmul
```

## 运行情况

| 项目 | 版本 / 配置 |
|---|---|
| 被测仓库 | flagos-ai/FlagGems-sglang @ b7a575a |
| GPU | NVIDIA GeForce RTX 4090（GPU 1，48 GB，计算能力 8.9，功耗上限 450 W） |
| 驱动 | 590.48.01（支持的最高 CUDA 版本 13.1） |
| CUDA | 13.0（PyTorch 编译所用版本） |
| PyTorch | 2.10.0a0+b558c986e8.nv25.11 |
| Triton | 3.5.0 |
| Python | 3.12.3 |
| pytest | 8.1.1 |
| 容器 | NGC PyTorch 25.11（默认 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，PyTorch 的 fp32 矩阵乘默认走 TF32） |
| fp32 矩阵乘 | 完整 FP32，未用 TF32：正确性测试设 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`；性能测试框架导入时设 `torch.backends.cuda.matmul.allow_tf32 = False` |
| CPU | INTEL(R) XEON(R) GOLD 6554S（2 路 × 36 核 × 2 线程，4 个 NUMA 节点） |
| 系统 | Ubuntu 24.04.3 LTS，内核 6.8.0-138-generic |

- 全部在 GPU 1（RTX 4090 48GB，计算能力 8.9）上运行；GPU 0 被其他任务占用，没有使用。
- 正确性测试跑了两次：正式的一次（关闭 TF32）在 2026-10-13 05:40–05:47（UTC）运行，容器默认 TF32 的对照一次在 2026-10-12 05:48–06:02 运行。性能测试的前两遍 kernel 模式和 operator 模式在 2026-10-12 07:58–08:30 运行，第 3 遍在 09:10–09:14 运行。
- 前两遍和 operator 模式运行时，GPU 1 上没有其他进程（空闲时显存 23 MiB）。SM 频率约 2.5–2.6 GHz，最高 68°C，最大功耗 408 W，没有降频，监控数据见 `gpu1_monitor.csv`。
- 第 3 遍运行时，GPU 1 上多了一个其他容器的进程（PID 2566262），占用约 7 GB 显存且用量在变化，开跑前 10 秒内出现过一次 67% 的利用率。前两遍一致的点在第 3 遍中的算子延迟中位偏差为 0.0%（见「稳定性说明」），没有看到它对 GPU 端测量的明显影响。第 3 遍最高 70°C、410 W，SM 频率只有一次采样降到 2385 MHz，监控数据见 `perf_kernel_run3/gpu1_monitor.csv`。
- 关闭 TF32 的正确性测试运行时，GPU 1 上也有其他容器的进程（开始时占用约 0.8 GB 显存，利用率 19%）。正确性测试只比较数值，不受影响。开始和结束时的 `nvidia-smi` 输出见 `acc/nvidia_smi_start.txt`、`acc/nvidia_smi_end.txt`。
- 原始数据：`perf_detail.csv`（三遍 kernel 模式和 operator 模式每个形状的原始延迟，单位 ms）、`replacements.csv`（不稳定点的三遍数据和处理结果）、`perf_summary.csv`（替换后的各算子汇总）、`summary.json`、`accuracy.csv`（正式，关闭 TF32）、`accuracy_tf32.csv`（对照，容器默认 TF32），以及 `acc/`、`acc_tf32/`、`perf_kernel_run1/`、`perf_kernel_run2/`、`perf_kernel_run3/`、`perf_operator/` 下每个算子的日志和 JSON。

## 正确性：TF32 对照

同一套正确性测试跑了两次，只有 TF32 设置不同（两次都没有绑核）：

| 用例数 | 容器默认，走 TF32 | 关闭 TF32（正式结果） |
|---|---:|---:|
| 总数 | 458 | 458 |
| 通过 | 457 | 458 |
| 失败 | 1 | 0 |
| 跳过 | 0 | 0 |
| 错误 | 0 | 0 |

原始数据分别在 `acc_tf32/` 和 `acc/`，汇总分别在 `accuracy_tf32.csv` 和 `accuracy.csv`。

结果不同的用例：

| 用例 | 容器默认，走 TF32 | 关闭 TF32 |
|---|---|---|
| `test_fused_moe_router_cudacore[7]` | 失败 | 通过 |

- 为什么关闭 TF32：本容器（NGC PyTorch 25.11）默认设置 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，PyTorch 的 fp32 矩阵乘会走 TF32；原版 PyTorch 从 1.12 起默认是完整 FP32。仓库的性能测试框架会自己关闭 TF32，`fused_moe_router_tensorcore` 的代码注释也提到验收环境以 `allow_tf32=False` 运行，所以正式结果统一用完整 FP32。
- 这个设置只影响 fp32 矩阵乘，bf16 / fp16 的矩阵乘不受影响。FlagGems 的 Triton 算子不走 cuBLAS，但 `fused_moe_router_tensorcore` 会读取 `torch.backends.cuda.matmul.allow_tf32` 来选择自己的精度：开关打开时 M ≥ 1024 用 TF32，关闭时用 tf32x3（拆成三次 TF32 乘法，精度接近 FP32）。正式的正确性结果和性能测试中，它在 M ≥ 1024 时用的都是 tf32x3。

`fused_moe_router_cudacore` 用例 7 在 TF32 下失败的原因：

- 用例参数：bs=4096、hidden=4096、256 个专家、topk=8，输入为 bf16。
- 4096 行里只有第 103 行不一致：选出的专家集合相同，只是第 5、6 位（专家 136 与 183）顺序对调。这两个专家的 fp64 logits 分别为 125.168130 和 125.168110，只差 1.98e-5。
- 以 fp64 结果为真值：算子的顺序正确，参考实现的顺序错误。算子权重的最大误差为 7.0e-6，参考实现为 3.7e-4，超出了测试容差 1e-4。
- TF32 下参考实现 logits 的最大误差为 3.75e-3；关闭 TF32 后降到 3.2e-4，4096 行的专家顺序都与 fp64 一致，权重误差 5.1e-5，在容差以内。这个算子本身不走 cuBLAS，两种设置下误差相同。
- 诊断输出见 `diag_fused_moe_router_cudacore.txt`。

| 指标（误差均相对 fp64 计算） | TF32 开（容器默认） | TF32 关 |
|---|---|---|
| 参考实现 logits 最大误差 | 3.75e-3 | 3.2e-4 |
| 参考实现权重最大误差 | 3.7e-4（超出容差 1e-4） | 5.1e-5 |
| 算子权重最大误差 | 7.0e-6 | 7.0e-6 |
| 专家 ID 不一致的行 | 1 / 4096 | 0 / 4096 |
| 该算子的 8 个用例 | 7 个通过，1 个失败 | 全部通过 |

## 性能数据的解读

- 基线是仓库自带的 PyTorch 参考实现，不是 SGLang 线上使用的 CUDA kernel（本机没有安装 sgl_kernel / flashinfer）。
- 标 † 的 21 个算子，参考实现带 Python 循环或 `.item()` 等主机同步，加速比被大幅放大（例如 `chunk_state_varlen` 约 520 倍、`w8a8_block_int8_matmul` 约 310 倍），不代表相对生产实现的收益。
- kernel 模式下，只有 `fused_moe_router_cudacore` 在 bs=512 和 bs=4096（256 个专家，hidden 4096）两个形状上比参考实现慢（0.82、0.67）。这个 kernel 只用 CUDA core 做乘加，参考实现是 cuBLAS 的 FP32 矩阵乘，批量大时更快；大批量场景对应的是 `fused_moe_router_tensorcore`。
- operator 模式下，484 个点中有 17 个比参考实现慢，集中在小形状：`draft_topk1` 6 个、`fused_moe_router_tensorcore` 6 个、`chunk_local_cumsum_scalar` 3 个、`fused_moe_router_cudacore` 2 个。除 `fused_moe_router_cudacore` 的两个大形状外，这些形状在 kernel 模式下算子都比参考实现快，端到端变慢来自 Python 封装和 Triton kernel 启动的 CPU 开销。例如 `draft_topk1` 在大词表下每次调用要启动 2 个 kernel、分配 4–5 个张量，墙钟约 45 µs，参考实现约 28 µs。
- kernel 模式每次测量前清空 L2，operator 模式连续调用、L2 是热的，所以同一形状 kernel 模式的 GPU 时间可能比 operator 模式的墙钟时间还长（例如 `draft_topk1` 的 [[64,151936]] 形状，63.3 µs 对 45.0 µs）。
- operator 模式只跑了一遍，单个形状的波动没有量化，几十 µs 量级、以 CPU 开销为主的点可能有较大的单次偏差。
- 仓库的通用实现兼顾多个厂商平台的竞赛评分，4090 上的数字不等同于竞赛成绩。

## 稳定性说明

kernel 模式跑了两遍。只看第 1 遍延迟 ≥10 µs 的点：第 1、2 遍的算子延迟或参考实现延迟相差超过 5%、且超过 2 个计时刻度的，记为不稳定。10 µs 以下的点不做判定。

第 1、2 遍共有 35 个不稳定的点，分布在 14 个算子上：参考实现延迟不稳定的 34 个，算子延迟不稳定的 2 个（其中 1 个点两项都不稳定，下表中占两行）。

这 14 个算子用相同的参数和绑核又跑了一遍 kernel 模式（第 3 遍，`perf_kernel_run3/`）。对每个不稳定的点，看第 3 遍的不稳定项更接近第 1 遍还是第 2 遍，保留接近的一遍，另一遍整条换成第 3 遍的结果（参考实现延迟、算子延迟和加速比一起换）。如果第 3 遍离两遍的距离都比两遍之间的距离还大，就算作与哪一遍都不接近，这个点不替换。`RTX4090_report.md` 只用替换后的第 1 遍；三遍的原始数据见 `perf_detail.csv`，替换记录见 `replacements.csv`。

结果：32 个点做了替换（替换第 1 遍 18 个、第 2 遍 14 个），其中 28 个点的第 3 遍与保留的一遍相差在 5% 以内，4 个点替换后两遍仍相差超过 5%；3 个点的第 3 遍与两遍都不接近，没有替换。仍不稳定的 7 个点在下表「处理」一列注明。

| 算子 | dtype | 形状参数 | 不稳定项 | 第 1 遍 (µs) | 第 2 遍 (µs) | 第 3 遍 (µs) | 处理 |
|---|---|---|---|---:|---:|---:|---|
| `chunked_embedding_lora_a` | bf16 | `[[[4096],[4,32,32000]],{"batch_info":null,"vocab_size":32000}]` | 参考实现延迟 | 452.6 | 687.5 | 798.7 | 替换第 1 遍（替换后仍相差超过 5%） |
| `chunked_embedding_lora_a` | bf16 | `[[[8192],[2,64,128256]],{"batch_info":null,"vocab_size":128256}]` | 参考实现延迟 | 367.6 | 289.8 | 287.6 | 替换第 1 遍 |
| `chunked_sgmv_expand` | bf16 | `[[[512,64],[4,8192,32]],{"batch_info":null,"slice_offsets":[3],"max_slice_size":4096,"base_output":[512,8192]}]` | 参考实现延迟 | 1393.7 | 2207.7 | 2185.4 | 替换第 1 遍 |
| `chunked_sgmv_expand` | bf16 | `[[[1024,192],[2,6144,64]],{"batch_info":null,"slice_offsets":[4],"max_slice_size":4096,"base_output":[1024,6144]}]` | 参考实现延迟 | 978.9 | 1729.5 | 1522.8 | 替换第 1 遍（替换后仍相差超过 5%） |
| `chunked_sgmv_shrink` | bf16 | `[[[512,4096],[4,32,4096]],{"batch_info":null,"num_slices":1}]` | 参考实现延迟 | 551.9 | 1011.7 | 567.8 | 替换第 2 遍 |
| `context_attention` | fp32 | `[[128,32,128],[128,32,128],[128,32,128],[1],[1],128,true]` | 算子延迟 | 50.18 | 62.46 | 49.15 | 替换第 2 遍 |
| `context_attention` | fp32 | `[[128,32,128],[128,32,128],[128,32,128],[1],[1],128,true]` | 参考实现延迟 | 118.8 | 193.5 | 117.9 | 替换第 2 遍 |
| `context_attention` | fp32 | `[[1024,32,128],[1024,32,128],[1024,32,128],[8],[8],128,true]` | 算子延迟 | 132.3 | 124.9 | 122.9 | 替换第 1 遍 |
| `decode_grouped_attention` | bf16 | `[[4096,128,128],[524288,1,128],[524288,1,128],[4097],[524288],0.08838834764831843]` | 参考实现延迟 | 632749 | 525718 | 519581 | 替换第 1 遍 |
| `decode_grouped_attention` | bf16 | `[[2,32,64],[20,1,64],[20,1,64],[3],[20],0.125]` | 参考实现延迟 | 291.9 | 266.2 | 271.3 | 替换第 1 遍 |
| `decode_grouped_attention` | bf16 | `[[2,128,64],[20,1,64],[20,1,64],[3],[20],0.125]` | 参考实现延迟 | 286.2 | 266.4 | 271.4 | 替换第 1 遍 |
| `embedding_lora_a` | bf16 | `[[[4096],[4,32,32000]],{"batch_info":null,"vocab_size":32000,"extra_embeddings":null}]` | 参考实现延迟 | 448.7 | 687.1 | 449.6 | 替换第 2 遍 |
| `embedding_lora_a` | bf16 | `[[[22],[2,32,512]],{"batch_info":null,"vocab_size":512,"extra_embeddings":null}]` | 参考实现延迟 | 269.5 | 187.4 | 192.5 | 替换第 1 遍 |
| `extend_attention` | bf16 | `[[362,32,128],[362,8,128],[362,8,128],[670,8,128],[670,8,128],[2],[2],[308],362]` | 参考实现延迟 | 419.8 | 518.2 | 422.9 | 替换第 2 遍 |
| `extend_attention` | bf16 | `[[8462,32,128],[8462,8,128],[8462,8,128],[16660,8,128],[16660,8,128],[65],[65],[8198],255]` | 参考实现延迟 | 14468 | 15233 | 14479 | 替换第 2 遍 |
| `extend_attention` | bf16 | `[[15972,32,128],[15972,8,128],[15972,8,128],[32612,8,128],[32612,8,128],[257],[257],[16640],127]` | 参考实现延迟 | 75894 | 54636 | 53344 | 替换第 1 遍 |
| `gate_up_lora_b` | bf16 | `[[[1024,128],[2,8192,64]],{"batch_info":null,"output_dim":4096,"base_output":[1024,8192]}]` | 参考实现延迟 | 677.9 | 644.1 | 642.0 | 替换第 1 遍 |
| `gate_up_lora_b` | bf16 | `[[[5,32],[1,128,16]],{"batch_info":null,"output_dim":64,"base_output":[5,128]}]` | 参考实现延迟 | 174.1 | 162.8 | 159.5 | 替换第 1 遍 |
| `gate_up_lora_b` | bf16 | `[[[22,32],[2,256,16]],{"batch_info":null,"output_dim":128,"base_output":[22,256]}]` | 参考实现延迟 | 481.3 | 455.7 | 445.6 | 替换第 1 遍 |
| `gate_up_lora_b` | bf16 | `[[[13,64],[2,512,32]],{"batch_info":null,"output_dim":256,"base_output":[13,512]}]` | 参考实现延迟 | 321.5 | 300.0 | 294.5 | 替换第 1 遍 |
| `mrope_fused` | bf16 | `[[1,1024],[1,256],[4096,128],[3,1],[16,24,24],128,128]` | 参考实现延迟 | 263.2 | 86.16 | 84.99 | 替换第 1 遍 |
| `per_group_transpose` | bf16 | `[[128,128],[9]]` | 参考实现延迟 | 405.0 | 229.4 | 399.4 | 替换第 2 遍 |
| `per_group_transpose` | bf16 | `[[1024,128],[9]]` | 参考实现延迟 | 405.5 | 230.0 | 354.3 | 替换第 2 遍（替换后仍相差超过 5%） |
| `sgemm_lora_a` | bf16 | `[[[512,4096],[4,32,4096]],{"batch_info":null,"stack_num":1}]` | 参考实现延迟 | 878.7 | 540.7 | 873.5 | 替换第 2 遍 |
| `sgemm_lora_a` | bf16 | `[[[1024,4096],[2,64,4096]],{"batch_info":null,"stack_num":1}]` | 参考实现延迟 | 303.1 | 282.6 | 450.6 | 未替换（第 3 遍与两遍都不接近） |
| `sgemm_lora_a` | bf16 | `[[[5,64],[1,16,64]],{"batch_info":null,"stack_num":1}]` | 参考实现延迟 | 80.90 | 75.86 | 120.8 | 未替换（第 3 遍与两遍都不接近） |
| `sgemm_lora_a` | bf16 | `[[[22,128],[2,16,128]],{"batch_info":null,"stack_num":1}]` | 参考实现延迟 | 238.1 | 222.4 | 344.2 | 未替换（第 3 遍与两遍都不接近） |
| `sgemm_lora_b` | bf16 | `[[[512,32],[4,4096,32]],{"batch_info":null,"base_output":[512,4096]}]` | 参考实现延迟 | 1154.0 | 750.6 | 1166.3 | 替换第 2 遍 |
| `sgemm_lora_b` | bf16 | `[[[1024,64],[2,4096,64]],{"batch_info":null,"base_output":[1024,4096]}]` | 参考实现延迟 | 698.4 | 434.2 | 430.1 | 替换第 1 遍 |
| `sgemm_lora_b` | bf16 | `[[[5,16],[1,64,16]],{"batch_info":null,"base_output":[5,64]}]` | 参考实现延迟 | 188.4 | 113.6 | 109.6 | 替换第 1 遍 |
| `sgemm_lora_b` | bf16 | `[[[13,32],[2,256,32]],{"batch_info":null,"base_output":[13,256]}]` | 参考实现延迟 | 195.6 | 207.9 | 198.9 | 替换第 2 遍 |
| `silu_and_mul_masked` | bf16 | `[[4,256,4096],[4]]` | 参考实现延迟 | 360.4 | 197.6 | 320.5 | 替换第 2 遍（替换后仍相差超过 5%） |
| `w8a8_block_int8_matmul` | bf16 | `[[1,4096],[4096,4096],[1,32],[32,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 29477 | 31222 | 29445 | 替换第 2 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[64,4096],[1024,4096],[64,32],[8,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 7979.0 | 8562.7 | 7987.7 | 替换第 2 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[64,4096],[4096,4096],[64,32],[32,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 34146 | 31901 | 32264 | 替换第 1 遍 |
| `w8a8_block_int8_matmul` | bf16 | `[[4096,4096],[1024,4096],[4096,32],[8,32],[128,128],"torch.bfloat16"]` | 参考实现延迟 | 7387.1 | 7881.8 | 7422.0 | 替换第 2 遍 |

作为对照，这 14 个算子中前两遍一致的点（≥10 µs），第 3 遍相对前两遍均值的偏差：算子延迟 63 个点，中位偏差 0.0%，其中 4 个点与两遍都相差超过 5%（-8% 到 +18%）；参考实现延迟 63 个点，中位偏差 0.0%，其中 6 个点与两遍都相差超过 5%（-44% 到 +76%）。这些点仍使用前两遍的数据。

替换后两遍的相对差异（|第 1 遍 − 第 2 遍| / 两遍均值），按第 1 遍的延迟分档，每格依次为点数 / 中位数 / P90 / 最大：

| 延迟档 | 算子延迟 | 参考实现延迟 |
|---|---:|---:|
| <10 µs | 282 / 0.0% / 21.6% / 28.6% | 16 / 0.0% / 11.4% / 11.8% |
| 10–50 µs | 102 / 0.0% / 3.8% / 10.2% | 214 / 0.0% / 2.7% / 7.7% |
| 50–500 µs | 77 / 0.0% / 0.8% / 2.1% | 148 / 0.3% / 2.1% / 13.5% |
| ≥0.5 ms | 23 / 0.1% / 0.1% / 1.0% | 106 / 0.2% / 2.4% / 15.0% |

替换前两遍几何平均加速比相差超过 10% 的算子，替换前后对比：

| 算子 | 替换前 第 1 遍 | 替换前 第 2 遍 | 替换后 第 1 遍 | 替换后 第 2 遍 |
|---|---:|---:|---:|---:|
| `chunked_sgmv_expand` | 30.94× | 38.32× | 37.26× | 38.32× |
| `chunked_sgmv_shrink` | 17.43× | 20.06× | 17.43× | 18.08× |
| `mrope_fused` | 15.52× | 11.78× | 11.70× | 11.78× |
| `sgemm_lora_a` | 21.90× | 19.11× | 21.90× | 21.04× |
| `sgemm_lora_b` | 43.94× | 34.04× | 35.51× | 35.72× |

替换后，两遍几何平均加速比相差超过 10% 的算子：没有。

补充说明：

- 不稳定的点几乎都在参考实现上（35 个点中 34 个），除 `mrope_fused` 外都是带 Python 循环的参考实现，耗时以 CPU 端开销为主。算子延迟不稳定的只有 `context_attention` 的 2 个点。
- 这类参考实现的耗时在不同进程之间会落在几个档位上，而不是围绕一个值小幅波动。例如 `per_group_transpose` 有 4 个形状前两遍一致（230–264 µs），第 3 遍都在 352–358 µs；`sgemm_lora_a` 后 3 个形状的第 3 遍比前两遍慢约 50%，所以这 3 个点没有替换。替换后仍不稳定的 7 个点都属于这种情况，影响的是参考实现延迟和对应的加速比。
