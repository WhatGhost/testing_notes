# FlagGems-sglang 算子测试说明（AMD Radeon AI PRO R9700，20261013_0146_pinned）

测试结果见 [R9700_report.md](R9700_report.md)，与 RTX 4090 的逐形状对比见 [AMD_vs_RTX4090.md](AMD_vs_RTX4090.md)。

## 测试方法

- 每个算子单独起一个 pytest 进程（`scripts/run_ops.py`），一个 kernel 出错不会影响其他算子。每个进程都经 `scripts/guard.py` 启动：先绑核，再确认看到的是被测的卡、且卡是空闲的，任何一项不满足就不运行。
- 正确性：`tests/test_<算子>.py`，与仓库自带的 PyTorch 参考实现（`src/flaggems_sglang/reference/`）对比。命令为 `python scripts/run_ops.py accuracy --out results/20261012_0602 --tag accuracy`。
- 性能：`benchmark/test_<算子>.py --level comprehensive`，共 484 个形状。完整流程见 `scripts/run_pinned_perf.sh`（三遍 kernel 模式）和 `scripts/run_operator_rtx4090.sh`（operator 模式）。
- kernel 模式：`triton.testing.do_bench`（warmup 1000 ms、rep 100 ms，每次测量前清空 L2，取中位数），只计 GPU 时间。56 个算子都跑了三遍，结果表取第 1 遍，第 2 遍用于检查稳定性。两遍不一致的点用第 3 遍判断、替换偏离的一遍，规则与 RTX 4090 相同，见「稳定性说明」。
- operator 模式：连续调用的墙钟时间，包含 Python 封装和 kernel 启动开销，不清空 L2。只跑一遍，warmup / iter 为调用次数，分组和次数与 RTX 4090 相同，见下方参数。
- 加速比 = 参考实现延迟 / 算子延迟，大于 1 表示算子更快；每个算子取所有形状的几何平均。† 表示参考实现带 Python 循环或 `.item()` 等主机同步，加速比会被放大。† 的标注沿用 RTX 4090 报告，两边的参考实现是同一份代码。
- HIP event 计时的刻度约为 0.04 µs：本次所有 kernel 延迟都是 0.04 µs 的整数倍，或两个整数倍的中点（样本数为偶数时的中位数）。RTX 4090 的 CUDA event 刻度约 1.024 µs，相比之下本机 10 µs 以下的点刻度误差可以忽略。FlagGems 在本机最短的 kernel 约 6.5 µs。
- **fp32 矩阵乘为完整 FP32**：gfx1201 不支持 TF32。本机没有设置任何 TF32 相关的环境变量，PyTorch 的 `torch.backends.cuda.matmul.allow_tf32` 默认就是 False，性能测试框架 `benchmark/performance_utils.py` 在导入时也会设为 False。Triton 的 AMD 后端只在 gfx942 上允许 tf32（`triton/backends/amd/compiler.py`），gfx1201 上 fp32 输入的 `tl.dot` 都按 `ieee` 计算。所以参考实现和 FlagGems 两侧都是完整 FP32，正确性测试不需要像 RTX 4090 那样另做 TF32 对照。`tf32_check.txt` 记录了这些设置（测试结束后在同一环境下记录，没有访问 GPU）。
- **需要绑核**：性能测试进程必须绑定到 GPU 所在 NUMA 节点的 CPU 上，否则 CPU 开销大的参考实现和 operator 模式的耗时会明显波动。10-12 不绑核测过一次（卡、门控和流程都相同），第 1、2 遍之间同一测量点差异的 P90 为 FlagGems 5.2%、参考实现 4.2%；绑核重测后降到 1.1%、1.3%（`compare_20261012_0602.md`）。本次绑在节点 3 的 CPU 24–31、88–95，方法见下方「绑核方法」。
- **负载门控**：本机由多人共用，测试期间其他用户的任务时有时无。性能测试的每个算子都要等主机安静后才开始，运行期间受到干扰就作废重跑，见下方「负载门控」。

### 绑核方法

由 `scripts/guard.py` 在每个测试进程里自动完成：根据 GPU 的 PCI 地址从 sysfs 读出它所在 NUMA 节点的 CPU 列表，在导入 torch 之前用 `os.sched_setaffinity` 把进程绑到这些 CPU 上，torch 和 Triton 之后创建的线程都继承这个绑定。核心逻辑：

```python
dev = f"/sys/bus/pci/devices/0000:{bus:02x}:00.0"     # bus = 0x03
cpulist = open(f"{dev}/local_cpulist").read().strip()  # "24-31,88-95"
node = open(f"{dev}/numa_node").read().strip()         # "3"
os.sched_setaffinity(0, parse_cpulist(cpulist))        # 在 import torch 之前
if os.sched_getaffinity(0) != parse_cpulist(cpulist):
    sys.exit(97)
```

- 本次用 rocm-smi 编号 0 的卡，PCI 地址 `0000:03:00.0`，所在 NUMA 节点为 3，对应 CPU 24–31、88–95，共 16 个逻辑 CPU（8 核 16 线程）。绑的是这一组 CPU 而不是某一个核，线程仍由系统在组内调度。
- 机器上没有 `numactl`，所以只绑了 CPU，没有显式绑内存。Linux 默认在首次访问时从进程所在节点分配内存，进程只在节点 3 的 CPU 上运行，内存一般也落在节点 3。
- 本机 torch/HIP 的设备编号与 rocm-smi 不一致，所以按 UUID 选卡（`ROCR_VISIBLE_DEVICES=GPU-29dcb03ca15afb63`，见 `scripts/env.sh`），并清空 `HIP_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。`guard.py` 再确认进程里只看到 1 张卡、PCI 总线号为 0x03、空闲显存不少于 90%，否则以退出码 97 停止。
- 每个算子的日志开头有一行 `[guard] OK: AMD Radeon Graphics (gfx1201) on PCI bus 0x03, … pinned to CPUs 24-31,88-95 (NUMA node 3)`，可以用来核对。本次 `perf_kernel_1/`、`perf_kernel_2/`、`perf_kernel_3/`、`perf_operator_rtx4090/`、`perf_operator/` 下各 56 个算子的日志，以及 `repeat/`、`count_ab/` 下的 44 个日志都有这一行。
- 正确性测试是在加入绑核之前跑的，没有绑核，日志里没有 `pinned to`；CPU 位置不影响数值结果。

### 负载门控

本机是 2 路 EPYC 9334（128 个逻辑 CPU）、8 张卡的共享机器。性能测试的每个算子都经过 `scripts/quiet.py` 门控（`run_ops.py --wait-quiet --max-busy-cores 24 --max-pinned-busy 2`）：

- 每 2 秒采样一次：其他用户占用的 CPU（整机 `/proc/stat` 的忙碌时间减去本容器各进程的 CPU 时间）、其中落在绑定的 16 个 CPU 上的部分，以及被测卡上其他进程占用的显存（KFD sysfs 能看到整机所有进程）。
- 开始条件：其他用户最多占 24 个核、在绑定的 CPU 上最多占 2 个，被测卡上没有其他进程，并且这种状态已经持续 30 秒。其他用户的常驻负载约 8–10 个核。
- 运行期间只要有一次采样超标，这次结果就作废，等主机重新安静后重跑，每个算子最多 6 次。
- 本次作废重跑的记录：第 2 遍 3 次（其他用户占了 28、29、61 个核），第 3 遍 2 次（绑定的 CPU 上被占了 2.1、2.3 个），按 4090 次数的 operator 模式 1 次（70 个核）。重跑都是第 2 次就通过，最终结果里没有受干扰的算子。
- 日志里每次启动 `run_ops.py` 时出现的「1 other process(es) on the GPU」和约 0.6 分钟的等待，是上一个进程退出后在 KFD 里短暂残留的记录（容器里看不到宿主机的进程号），加上 30 秒的安静窗口，不是其他用户在用卡。

### operator 模式参数

```
operator mode: warmup/iter are call counts; groups and counts taken from RTX4090_testing.md by scripts/run_operator_rtx4090.sh
fast: --warmup 100 --iter 1000: bmm_chunk causal_conv1d_update chunk_cumsum chunk_local_cumsum_scalar chunk_local_cumsum_vector chunk_scaled_dot_kkt chunked_embedding_lora_a chunked_sgmv_expand chunked_sgmv_shrink draft_topk1 embedding_lora_a ernie45_rope_fused fla_layernorm_gated fused_dual_residual_rmsnorm fused_gdn_gating fused_moe_router_cudacore fused_moe_router_tensorcore fused_rmsnorm gate_up_lora_b gelu_and_mul interleaved_rope l2norm log_scaling_tau mamba_layernorm_gated merge_state moe_fused_gate moe_fused_mul_sum moe_sum_reduce mrope_fused per_group_transpose per_token_group_quant_int8 per_token_quant_int8 qkv_lora_b rotary_embedding sgemm_lora_a sgemm_lora_b sigmoid_gate_topk_renorm silu_and_mul silu_and_mul_masked softcap_inplace_logits softcap_out state_passing
slow: --warmup 5 --iter 20: apply_token_bitmask causal_conv1d_fn chain_speculative_sampling chunk_state chunk_state_varlen context_attention decode_attention decode_grouped_attention extend_attention fused_moe_gemm fused_norm_rope_stacked hc_head selective_state_update w8a8_block_int8_matmul
```

- RTX 4090 按其第 1 遍参考实现的最大延迟以 5 ms 为界分组。这里直接沿用它的分组，两边每个算子的次数相同。如果按本机第 1 遍的延迟分组，只有 `rotary_embedding`（参考实现最长 8.25 ms）会从快组移到慢组。
- 本次运行里还有一遍较早的 operator 模式（`perf_operator/`）：50 个算子用 benchmark 默认的 1000/100，6 个参考实现很慢的算子用 10/30。报告只用按 4090 次数测的这一遍，两遍的差别见「性能数据的解读」。

## 运行情况

| 项目 | 版本 / 配置 |
|---|---|
| 被测仓库 | flagos-ai/FlagGems-sglang @ b7a575a |
| GPU | AMD Radeon AI PRO R9700（rocm-smi GPU 0，PCI 0000:03:00.0，32 GB，gfx1201，功耗上限 300 W） |
| 驱动 | amdgpu（内核 7.0.0-31-generic 自带） |
| ROCm / HIP | ROCm 7.2.3，HIP 7.2.53211 |
| PyTorch | 2.10.0+rocm7.2.3.git1a270074 |
| Triton | 3.6.0+rocm7.2.3.git4ed88892 |
| Python | 3.12.3（`/opt/venv`） |
| pytest | 9.0.3 |
| fp32 矩阵乘 | 完整 FP32：gfx1201 不支持 TF32；PyTorch 的 `allow_tf32` 为默认的 False，性能测试框架导入时也设为 False；Triton 在 gfx1201 上只有 `ieee` 精度 |
| CPU | AMD EPYC 9334（2 路 × 32 核 × 2 线程，8 个 NUMA 节点） |
| 系统 | Ubuntu 24.04.4 LTS，内核 7.0.0-31-generic |

- 性能测试全部在 rocm-smi 编号 0 的卡上运行（PCI 03:00.0）。本机共 8 张卡，3 号卡属于其他用户，没有使用。
- 正确性测试在 2026-10-12 06:02–06:58（UTC）于 0 号卡上运行。`fused_moe_router_tensorcore`、`silu_and_mul_masked` 第一次运行时 autotune 编译极慢（单个算子要 1–2 小时，见 `correctness_issues.md` 的 P1），先跳过。08:14–08:29 离线预编译了这两个算子的 685 个 kernel。08:22 尝试在 0 号卡上补跑时，`guard.py` 发现卡又被其他用户的任务占用（空闲显存只有 23.0 / 31.9 GiB）而停止，于是 08:28–08:30 改在同型号、空闲的 1 号卡（PCI 23:00.0）上补跑，分别用时 90 秒和 53 秒。正确性只比较数值，换卡不影响结论。
- 性能测试在 2026-10-13（UTC）运行：kernel 模式第 1 遍 01:46–02:03，第 2 遍 02:03–02:22，较早的 1000/100 operator 模式 02:22–02:55，第 3 遍 02:55–03:14；`causal_conv1d_fn`、`causal_conv1d_update` 的单独复测（各 10 遍）03:25–03:34；调用次数的交替对照 05:53–06:05；按 4090 次数的 operator 模式 06:31–06:52。
- GPU 状态：没有连续监控，只在开始和结束时各记录了一次 `rocm-smi` 快照。三遍 kernel 模式开始时 31–32°C、13 W、空闲，结束时 53–57°C、15 W、空闲；按 4090 次数的 operator 模式开始时 32–33°C、13 W，结束时 58–59°C、15 W。测试中没有改过频率和功耗设置（性能档位为 auto，功耗上限 300 W）。没有频率记录，不能直接排除降频，但替换后第 1、2 遍之间同一测量点差异的 P90 在 1.3% 以内（见「稳定性说明」），看不到系统性的漂移。门控保证了每个算子运行期间被测卡上没有其他进程。
- 原始数据都在 `results/20261013_0146_pinned/` 下：
  - `perf_kernel_1/`、`perf_kernel_2/`、`perf_kernel_3/`：三遍 kernel 模式每个算子的日志和 benchmark JSON（延迟单位 ms）。`perf_kernel_1/replaced/`、`perf_kernel_3/replaced/` 保存了被复测结果替换掉的原始文件。
  - `perf_operator_rtx4090/`：按 4090 次数的 operator 模式（报告使用）；`perf_operator/`：较早的 1000/100 一遍。
  - `repeat/r01/`–`repeat/r10/`：两个算子的单独复测，汇总见 `repeat/summary.md`；`count_ab/`：调用次数的交替对照，汇总见 `count_ab/summary.md`。
  - `compare_rtx4090.csv`：每个形状按 4090 规则选定的 kernel 值（`amd_kernel_source` 列注明取自哪一遍）、两遍 operator 模式和 4090 报告的数据（µs）；`perf_points.csv`：三遍 kernel 模式和 1000/100 operator 模式每个形状的原始延迟（ms）。
  - `accuracy/`（链接到 `results/20261012_0602/accuracy/`）：正确性测试每个算子的日志、junit xml 和 JSON，汇总在 `status.json`。
  - `env_perf_start.txt`、`env_perf_end.txt`、`env_operator_rtx4090_start.txt`、`env_operator_rtx4090_end.txt`：软件版本和 `rocm-smi` 快照；`perf.log`、`perf_operator_rtx4090.log`：运行日志；`tf32_check.txt`：TF32 相关设置；`gpu_hw.txt`：被测卡的硬件参数和型号依据（`scripts/record_gpu_hw.sh` 在测试结束后读取 KFD 拓扑、sysfs 和 rocm-smi，没有在 GPU 上运行程序），`R9700_report.md` 的「GPU 硬件参数」一节由它生成。
  - `R9700_report.md` 由 `scripts/r9700_report.py` 生成。

## 正确性：失败用例

共 458 个用例，通过 456 个。失败的 2 个都是舍入级别的差异，不是算法错误；这两个用例在 RTX 4090 上（FP32）都通过。详细分析见 `correctness_issues.md` 的 C7、C8。

| 用例 | 参数 | 超差元素 | 最大差 | 容差 |
|---|---|---:|---:|---|
| `test_fused_dual_residual_rmsnorm[15]` | batch 512、hidden 8192、bf16 | 1 / 4194304 | 0.0215 | atol = rtol = 1.5e-2（该元素允许 0.0194） |
| `test_fused_moe_router_tensorcore[7]` | M 4096、hidden 4096、256 个专家、topk 2、bf16 输入 | 3 / 8192 个权重 | 1.52e-4 | atol = rtol = 1e-4 |

`fused_dual_residual_rmsnorm` 用例 15：

- 出错位置 (350, 3844)：参考值 −0.2949，算子 −0.3164。
- hidden ≥ 8192 时，算子把第一次 RMSNorm 的分母乘以经验常数 `RMS1_SCALE = 0.9999998807907104`（1 − 2⁻²³），代码里没有说明。在这个位置，第一次归一化的 fp32 中间值恰好在两个 bf16 值的中点附近，乘上常数后越过中点，转成 bf16 时舍入方向翻转，差了 1 个 bf16 ulp（0.0156）。之后加上 residual 时发生抵消，这个差异经第二次 RMSNorm 传到输出，超出容差。
- 用同一个 kernel、把常数换成 1.0 重跑，超差元素从 1 个降为 0 个（`results/20261012_0602/recheck/fused_dual_residual_rmsnorm_diag.json`）。这个常数应该是为了在 NVIDIA 上贴合参考实现的舍入而调出来的，换到 AMD 上反而造成了差异。

`fused_moe_router_tensorcore` 用例 7：

- 选出的专家 ID 完全一致，只有 3 个权重超差。正确性测试和单独的诊断脚本两次运行结果相同。
- 以 fp64 结果为真值（`results/20261012_0602/recheck/router_tensorcore.json`）：

| M | 参考实现相对 fp64 的最大误差 | 算子相对 fp64 的最大误差 | 算子与参考实现之差（最大 / 超差个数） |
|---|---:|---:|---:|
| 512 | 7.96e-5 | 8.99e-5 | 8.85e-5 / 0 |
| 2048 | 1.04e-4 | 1.23e-4 | 1.10e-4 / 0 |
| 4096 | 1.41e-4 | 1.11e-4 | 1.52e-4 / 3 |

- 用测试的容差衡量，算子和参考实现相对真值都没有超差，M=4096 时算子反而更接近真值。失败是因为测试拿两个各带约 1e-4 误差的 fp32 结果互相比较，容差却只有 1e-4；K=4096 时 fp32 逐项累加的误差本来就在这个量级。
- 在 NVIDIA 上，这个算子在 `allow_tf32=False` 时用 tf32x3，刻意贴合 cuBLAS 的计算路径，两边差得很小。gfx1201 不支持 tf32，算子改用 `ieee`（纯 fp32 FMA），和参考实现的 BLAS 矩阵乘累加顺序不同，差异就到了 1e-4 量级。

其他说明：

- `act_and_mul`：`tests/test_act_and_mul.py` 的 9 个用例全部跳过（`moe/act_and_mul ops module not found`），仓库里没有这个算子的实现（`correctness_issues.md` 的 C5）。RTX 4090 报告同样不含这个算子，两边都按 56 个算子、458 个用例统计。
- 本机不支持 TF32，正确性测试只跑了一次（FP32），没有 TF32 对照。

## 性能数据的解读

- 基线是仓库自带的 PyTorch 参考实现，不是 SGLang 线上使用的 kernel（本机没有安装 sgl_kernel、flashinfer、aiter、vllm）。
- 标 † 的 21 个算子，参考实现带 Python 循环或 `.item()` 等主机同步，加速比被大幅放大（例如 `chunk_state_varlen` 约 360 倍、`w8a8_block_int8_matmul` 约 190 倍），不代表相对生产实现的收益。
- kernel 模式下有 3 个点比参考实现慢：
  - `context_attention` 的 [128,32,128] fp32 形状（0.74）。fp32 输入时，这个 kernel 的 `tl.dot` 在 gfx1201 上按 `ieee` 计算，整个算子只有 1.64×；RTX 4090 上用 tf32 张量核心，为 4.69×。
  - `fused_moe_router_tensorcore` 的 [4096,4096] × 256 专家形状（0.93）。参考实现是 PyTorch 的 fp32 矩阵乘；算子在 gfx1201 上用 `ieee` 的 fp32 点积，RTX 4090 上用 tf32x3 张量核心。
  - `fused_moe_router_cudacore` 的 [83,512] × 32 专家形状（0.83）。算子三遍都是 67.9 µs，RTX 4090 上同一形状为 13.66 µs；这个形状在本机特别慢的原因没有细查。
- operator 模式下，484 个点中有 18 个比参考实现慢：`moe_sum_reduce` 5 个、`draft_topk1` 4 个、`chunk_local_cumsum_vector` 3 个、`chunk_local_cumsum_scalar` 2 个、`context_attention` 2 个、`fused_moe_router_tensorcore` 1 个、`log_scaling_tau` 1 个。除了 kernel 模式下也慢的 `context_attention` [128,32,128] 和 `fused_moe_router_tensorcore` [4096,4096]，这些形状在 kernel 模式下算子都比参考实现快（1.07–2.80），端到端变慢来自 Python 封装和 Triton kernel 启动的 CPU 开销。其中 `chunk_local_cumsum_*`、`log_scaling_tau`、`moe_sum_reduce` 的 11 个点，FlagGems 每次调用 12.8–19.1 µs，参考实现 9.6–17.2 µs；`draft_topk1` 在大词表（151936）下 FlagGems 约 57 µs，参考实现 26–46 µs。
- kernel 模式每次测量前清空 L2，operator 模式连续调用、L2 是热的，所以同一形状 kernel 模式的 GPU 时间可能比 operator 模式的墙钟时间还长（例如 `apply_token_bitmask` 的 [[64,152064],[64,4752]] 形状，239.6 µs 对 136.3 µs）。
- operator 模式的正式数据只跑了一遍。调用次数交替对照里（6 个算子，同一种次数各跑两轮），按 4090 的次数时两轮之间的差异：快组（100/1000）FlagGems 中位数 2.1%、P90 3.6%，参考实现 0.4%、2.7%；慢组（5/20）P90 为 2.3%–4.5%。几十 µs 量级、以 CPU 开销为主的点，单次偏差可能更大。
- 调用次数的影响：每个计时窗口有一份固定开销（首尾的同步和计时、最后一次调用的收尾），按计时次数摊薄。从交替对照反推，这份开销约 40–160 µs，多数在 60–110 µs，计时 1000 次时每次调用多出 0.04–0.16 µs，计时 20 次时多出 2–8 µs。和较早的 1000/100 那一遍相比（各测量点比值的中位数）：快组 42 个算子的 FlagGems 延迟低 2.9%、加速比高 1.4%；慢组中原来用 1000/100 的 8 个算子 FlagGems 高 7.5%、加速比低 3.8%（`chunk_state_varlen` 的 FlagGems 高 22%）。35 个非 † 算子的 operator 加速比由 3.57× 变为 3.60×。慢组只计时 20 次，FlagGems 单次调用较短的算子会多带几个百分点没摊薄的开销；RTX 4090 用的是相同的次数，但这份开销的大小取决于平台，不能完全抵消。逐算子的比值见 `compare_rtx4090.md` 第 5 节。
- 参考实现在本机的慢路径：PyTorch 的 fp32 矩阵乘在几个形状上明显偏慢，这些算子在本机的加速比因此偏高。`fused_moe_router_cudacore`、`fused_moe_router_tensorcore` 的大批量形状（[512 或 4096, 4096] × [256, 4096]），参考实现是 RTX 4090 的 6.7–6.9 倍；`hc_head` 小形状的 `F.linear`（M=1、N=4、K=28672）稳定要 0.81 ms，是 RTX 4090 的 22 倍。原因是按代码结构推断的，没有逐步 profiling。
- 仓库的通用实现兼顾多个厂商平台的竞赛评分，R9700 上的数字不等同于竞赛成绩。

## 稳定性说明

kernel 模式 56 个算子都跑了三遍，参数和绑核相同。判定规则与 RTX 4090 相同：只看第 1 遍延迟 ≥10 µs 的点，第 1、2 遍的算子延迟或参考实现延迟相差超过两遍均值的 5%、且超过 2 个计时刻度（本机为 0.08 µs）的，记为不稳定。10 µs 以下的点不做判定。

第 1、2 遍共有 11 个不稳定的点，分布在 8 个算子上：参考实现延迟不稳定的 10 个，算子延迟不稳定的 1 个，没有两项都不稳定的点。

对每个不稳定的点，看第 3 遍的不稳定项更接近第 1 遍还是第 2 遍，保留接近的一遍，另一遍整条换成第 3 遍的结果（参考实现延迟、算子延迟和加速比一起换）。如果第 3 遍离两遍的距离都比两遍之间的距离还大，就算作与哪一遍都不接近，这个点不替换。`R9700_report.md` 只用替换后的第 1 遍；三遍的原始数据见 `perf_points.csv`，每个形状最终取自哪一遍见 `compare_rtx4090.csv` 的 `amd_kernel_source` 列。

结果：11 个点都做了替换（替换第 1 遍 5 个、第 2 遍 6 个），每个点的第 3 遍与保留的一遍相差都在 5% 以内；没有与两遍都不接近的点。

| 算子 | dtype | 形状参数 | 不稳定项 | 第 1 遍 (µs) | 第 2 遍 (µs) | 第 3 遍 (µs) | 处理 |
|---|---|---|---|---:|---:|---:|---|
| `chunk_local_cumsum_vector` | bf16 | `[[32,256,8,128],64,false,null]` | 参考实现延迟 | 197.1 | 183.8 | 180.4 | 替换第 1 遍 |
| `chunk_local_cumsum_vector` | bf16 | `[[2,48,4,32],16,true,null]` | 参考实现延迟 | 19.40 | 21.00 | 19.40 | 替换第 2 遍 |
| `decode_grouped_attention` † | bf16 | `[[2,32,80],[20,4,80],[20,4,80],[3],[20],0.11180339887498948]` | 算子延迟 | 16.04 | 12.68 | 12.68 | 替换第 1 遍 |
| `ernie45_rope_fused` | bf16 | `[[1,1024],[1,256],[4096,128],[3,1],[16,16,32],128,128]` | 参考实现延迟 | 122.5 | 152.9 | 122.8 | 替换第 2 遍 |
| `fla_layernorm_gated` | bf16 | `[[[1,2048],[1,2048],[2048],[2048]],{"activation":"swish","eps":1e-05,"is_rms_norm":true}]` | 参考实现延迟 | 65.24 | 49.76 | 49.58 | 替换第 1 遍 |
| `fused_norm_rope_stacked` | bf16 | `[[1,8,2048],[8,128],[8],[4096,128],[1],8,128,128]` | 参考实现延迟 | 81.16 | 97.04 | 81.24 | 替换第 2 遍 |
| `fused_rmsnorm` | bf16 | `[[1,4096],[4096],1e-06]` | 参考实现延迟 | 32.60 | 44.60 | 32.80 | 替换第 2 遍 |
| `fused_rmsnorm` | bf16 | `[[1,8192],[8192],1e-06]` | 参考实现延迟 | 33.18 | 45.12 | 33.20 | 替换第 2 遍 |
| `mamba_layernorm_gated` | bf16 | `[[[83,512],[512],[512],1e-05],{"z":[83,512],"group_size":128}]` | 参考实现延迟 | 61.64 | 58.16 | 58.04 | 替换第 1 遍 |
| `silu_and_mul` | bf16 | `[[64,16384]]` | 参考实现延迟 | 56.20 | 66.12 | 65.88 | 替换第 1 遍 |
| `silu_and_mul` | bf16 | `[[512,2048]]` | 参考实现延迟 | 56.20 | 66.00 | 56.28 | 替换第 2 遍 |

作为对照，前两遍一致的点（≥10 µs），第 3 遍相对前两遍均值的偏差：算子延迟 256 个点，中位偏差 0.0%，其中 1 个点与两遍都相差超过 5%（`per_group_transpose` 的 [[8192,512],[9]]，−6%）；参考实现延迟 473 个点，中位偏差 0.0%，其中 3 个点与两遍都相差超过 5%（−6% 到 +36%，最大的是 `selective_state_update` 的一个小形状，93.56 / 93.28 / 127.2 µs）。这些点仍使用前两遍的数据。RTX 4090 只对 14 个算子跑了第 3 遍，这里的对照覆盖全部 56 个算子。

替换后两遍的相对差异（|第 1 遍 − 第 2 遍| / 两遍均值），按第 1 遍的延迟分档，每格依次为点数 / 中位数 / P90 / 最大：

| 延迟档 | 算子延迟 | 参考实现延迟 |
|---|---:|---:|
| <10 µs | 224 / 0.6% / 1.2% / 24.7% | 0 / - / - / - |
| 10–50 µs | 117 / 0.3% / 1.0% / 2.8% | 133 / 0.2% / 0.7% / 3.1% |
| 50–500 µs | 104 / 0.2% / 0.9% / 4.3% | 199 / 0.2% / 1.1% / 4.3% |
| ≥0.5 ms | 39 / 0.1% / 1.1% / 3.0% | 152 / 0.3% / 1.3% / 2.7% |

替换前，两遍几何平均加速比相差超过 10% 的算子：没有。相差最大的是 `fla_layernorm_gated`（7.35× 对 7.11×，替换后 7.10× 对 7.11×）。

补充说明：

- 不稳定的点几乎都在参考实现上（11 个中 10 个），延迟在 19–197 µs 之间，都不是带 † 的算子。算子延迟不稳定的只有 `decode_grouped_attention` 的 1 个点。
- 这类参考实现的耗时会落在几个固定的档位上，而不是围绕一个值小幅波动。例如 `silu_and_mul` 两个形状的参考实现第 1 遍都是 56.20 µs，第 2 遍都是约 66 µs，第 3 遍一个约 66 µs、一个约 56 µs；`fused_rmsnorm` 的 batch 1 形状第 1、3 遍约 33 µs，第 2 遍约 45 µs。
- 10 µs 以下的点不做判定。替换后这一档两遍差异最大的点是 `fused_rmsnorm` 的 [1,1024]（算子延迟 7.08 / 9.08 / 7.12 µs），同一遍里这个算子 batch 1 的其他形状也偏慢，属于上一条所说的档位变化。
- 离群值的替换（在套用上述规则之前完成）：最初检查三遍数据时，`causal_conv1d_update` 的第 1、3 遍和 `causal_conv1d_fn` 的第 3 遍有明显偏离。这两个算子另外各跑了 10 遍（每遍一个新进程，绑核和门控相同，`repeat/`），每个点的 10 次结果都集中在一个档位上（最大与最小相差不超过 4.3%），于是把偏离的几遍整条换成复测结果：`causal_conv1d_update` 第 1 遍换成 `repeat/r01`、第 3 遍换成 `repeat/r02`，`causal_conv1d_fn` 第 3 遍换成 `repeat/r01`，原始文件在 `replaced/`。例如 `causal_conv1d_update` 的 [64,4096] 形状，算子延迟原来三遍为 31.64 / 28.52 / 18.92 µs，10 次复测为 28.3–28.7 µs。
- 如果不做这一步替换，按上述规则还会多出 `causal_conv1d_update` 的 2 个不稳定点：[256,2048] 的参考实现延迟（298.35 / 274.31 / 275.59 µs）可以用第 3 遍替换；[64,4096] 的算子延迟（31.64 / 28.52 / 18.92 µs）第 3 遍与两遍都不接近，不能替换，报告里会留下偏离复测结果的 31.64 µs。
- 另一种更严的分类见 `stability.md`：每个点取三遍延迟的 (最大 − 最小) / 最小，所有点都不超过 10% 的算子算稳定。按替换后的数据，56 个算子中稳定 48 个，基本稳定 8 个（超过 10% 的只有 0.15 ms 以下的小点，且三遍的几何平均加速比相差不超过 10%），不稳定 0 个。
