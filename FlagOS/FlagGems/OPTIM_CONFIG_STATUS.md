# RDNA4 候选表覆盖现状

记录 gfx1201 上「哪些算子已经通过 `tune_configs.yaml` 优化、哪些没有、没有的原因是什么」。
调优过程本身见 `tuning/TUNING.md`，性能基线见 `RESULTS.md`。

- 统计范围：`RESULTS.md` 追踪的 10 个算子
- 数据来源：`tuning/ops/ab_full/`（`--level core`，fp16 + bf16，两种模式各三轮）
- 硬件：AMD Radeon `gfx1201`，32 CU，bus 0xf3；ROCm 7.2.3 / PyTorch 2.10.0 / Triton 3.6.0
- 更新时间：2026-08-03

## 结论

10 个算子里 4 个已通过 arch 候选表优化，6 个没有。**没优化的原因是同一类：这 6 个算子的
kernel 不从 `tune_configs.yaml` 取配置**，arch 表里写什么都影响不到它们，与「调不出收益」
无关。

`rdna4/tune_configs.yaml` 目前只有 4 个键：`mm`、`bmm`、`layer_norm_loop`、`rms_norm_loop`。

## 1. 已覆盖的 4 个

| 算子 | arch 表键 | PR | 效果 |
|---|---|---|---|
| `mm` | `mm` | [#4991](https://github.com/flagos-ai/FlagGems/pull/4991)（已合并） | 中大 shape 从约 0.70× 提到 0.88–1.03×，顺带修掉 3 个 fp32 失败 |
| `bmm` | `bmm` | [#4991](https://github.com/flagos-ai/FlagGems/pull/4991)（已合并） | +2–4%，最优已接近可配置空间边界 |
| `layer_norm` | `layer_norm_loop` | [#5100](https://github.com/flagos-ai/FlagGems/pull/5100)（待合并） | kernel +8.6–9.0%，operator +16.0–18.1% |
| `rms_norm` | `rms_norm_loop` | [#5100](https://github.com/flagos-ai/FlagGems/pull/5100)（待合并） | kernel +9.7–11.7%，operator +5.6–5.7% |

## 2. 未覆盖的 6 个及原因

| 算子 | benchmark 入口 | 实际执行的 kernel | config 管不到的原因 |
|---|---|---|---|
| `softmax` | `test_softmax::test_softmax` | `softmax_kernel_inner` / `_non_inner` | 走 `@triton.heuristics`，参数由 Python 函数算出 |
| `sum` | `test_sum::test_sum` | `sum_kernel_1` / `sum_kernel_2` | 无 autotune 装饰器，块大小 Python 里算好传入 |
| `mean` | `test_mean::test_mean` | `mean_kernel_1` / `mean_kernel_2` | 同上 |
| `mul` | `test_mul::test_mul` | `mul_kernel` | 有 `@libtuner`，但候选来自 Python 函数且只有 1 条 |
| `sub` | `test_sub::test_sub` | `sub_func` | `@pointwise_dynamic` 代码生成 |
| `div` | `test_div::test_div` | `div_*` | 同上 |

### 2.1 softmax：前向走启发式，YAML 键只喂反向

`ops/softmax.py` 里前向的两个 kernel 挂的是
`@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))`，`TILE_N` /
`ONE_TILE_PER_CTA` / `num_warps` 全部由 `_amd/heuristics_config_utils.py` 中的 Python
函数计算。同名的 YAML 键 `softmax_inner` / `softmax_non_inner` 确实存在，但被
`@triton.autotune` 消费的只有**反向** kernel（该文件 179 与 239 行）——两个注册表用了同一
批键名，容易看错。

更要紧的是层级：`backend/__init__.py` 的 `get_heuristic_config()` 只做
`importlib.import_module(f"_{vendor}.heuristics_config_utils")`，**没有 arch 层的覆盖机
制**。所以启发式无法只针对 RDNA4 特化，改了就是全体 AMD 设备一起改。顺带一提，
`sum_dim_kernel_inner` 复用了 `softmax_non_inner` 这个启发式键，说明该注册表的键是跨算子
共享的，动它的影响面比看起来大。

### 2.2 sum / mean：benchmark 测的是全量归约

`test_sum::test_sum` 与 `test_mean::test_mean` 的 `torch_op` 分别是 `torch.sum` /
`torch.mean`，都是不带 `dim` 的全量归约，落到 `sum_kernel_1/2`、`mean_kernel_1/2`。这几个
kernel 只有 `@libentry()` 和 `@triton.jit`，块大小是 Python 里
`next_power_of_2(ceil(sqrt(M)))` 算完当参数传进去的，压根不存在候选集。

注意这两个文件里**确实有**读 YAML 的路径：`sum_dim_kernel` 和 `mean_dim_kernel` 用的是
`runtime.get_tuned_config("naive_reduction")`。但它们服务的是 `sum_dim` / `mean_dim`，对应
`sum_dim_out`、`mean_dim` 两个独立 benchmark，不在这 10 个算子里。

### 2.3 mul：候选表硬编码在源码且只有一条

`mul_kernel` 挂了 `@libtuner`，但 `configs=mul_get_configs()` 返回的是 Python 里写死的单个
`triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3)`，不经过 YAML；即便接上
YAML，一条候选也没有可选余地。改它要动 `ops/mul.py`，影响所有后端。

### 2.4 sub / div：pointwise 代码生成

由 `@pointwise_dynamic` 生成，分块决策在 `utils/pointwise_dynamic.py` 的代码生成器里，没有
对应的 YAML 键。

## 3. 余量分布：缺口恰好都在不可调的算子上

改动后 kernel 模式相对 PyTorch 的位置（三轮中位）：

| 算子 | 中位 | 范围 | 最差 case |
|---|---|---|---|
| `layer_norm` | 1.281× | 1.134–2.776× | fp16 `[16,16,128]` |
| `sum` | 1.152× | 0.996–1.219× | fp16 `[1024,1024,1024]` |
| `mean` | 1.080× | **0.416**–1.216× | bf16 `[64,512,512]` |
| `rms_norm` | 1.073× | 0.981–2.359× | fp16 `[4096,4096]` |
| `softmax` | 1.062× | **0.826**–1.625× | bf16 `[1048576]` |
| `div_tensor` | 1.002× | 0.973–1.022× | bf16 `[1024,1024,1024]` |
| `mul` | 0.998× | 0.984–1.006× | bf16 `[4096,4096]` |
| `sub` | 0.998× | 0.974–1.013× | bf16 `[1024,1024,1024]` |
| `mm` | 0.994× | 0.865–1.061× | bf16 `[4096,4096]` |
| `bmm` | 0.954× | 0.906–1.074× | fp16 `[16,4096,4096]` |

`mul` / `sub` / `div` 三个 pointwise 算子的中位都在 0.998–1.002×，贴着带宽上限，即便能调也
没什么可捞。真正的缺口是两处，而且**都落在不可调的算子上**：`mean` 的 `[64,512,512]` 只有
0.416×（比 PyTorch 慢 2.4 倍，是全套件最大缺口），`softmax` 的 `[1048576]` 是 0.826×。这两
处要动就得改 kernel 或启发式，不是 config 能解决的。

## 4. 纯 config 路线还剩的入口

`_amd` 有、`rdna4` 尚未覆盖的键中，与 benchmark 有关的是这几个：

| 键 | 服务的 kernel | 对应 benchmark 入口 |
|---|---|---|
| `naive_reduction` | `sum_dim` / `mean_dim` / `max` / `min` / `amax` / `amin` / `all` / `any` / `prod` 等一批归约 | `sum_dim_out`、`mean_dim` |
| `softmax_inner` / `softmax_non_inner` | softmax 反向 | `softmax_backward_out` |
| `layer_norm_persistent` | `N ≤ 4096` 的 layer_norm | `test_layer_norm` 的小 shape（`TUNING.md` 第 12 节测过，默认已最优） |

`naive_reduction` 的现状是 `BLOCK_N` 固定 1024、`num_warps` 固定 4，只有
`BLOCK_M ∈ {1,2,4,8}` 可动，扩展空间明显。它一次覆盖十几个归约算子，是投入产出比最高的下
一步。

## 5. config 之外的其他手段

按影响面排序。改通用 kernel（`ops/*.py`）和改调优基础设施（`utils/libentry.py`、
`utils/models/sql.py`）会波及所有后端，暂不考虑，这里只记只影响 AMD 的三条。

### 5.1 退回 PyTorch：`CUSTOMIZED_UNUSED_OPS`

`_amd/__init__.py` 里的元组，列进去的算子放弃 FlagGems 实现、直接用 PyTorch。AMD 目前列了
`add`、`cos`、`cumsum`。

按第 3 节的数据，`bmm` 中位 0.954× 是稳定慢于 PyTorch 的，理论上是候选。但**粒度是整个算
子，无法按 shape 区分**——`bmm` 在 `[2,384,384]` 上有 1.074×，一刀切会连这个也丢掉。所以
这条只适合全区间都输的算子。

### 5.2 调优启发式函数：`_amd/heuristics_config_utils.py`

和候选表的本质区别是**没有实测兜底**。autotuner 会把候选逐个跑再挑最优；启发式是纯函数，
算出什么就用什么，改错了直接变慢，且对所有 shape 生效。所以流程要反过来：先实测出正确答
案，再拟合函数。

三步：

1. 把现有函数的输出表格化——对每个 core shape 算出它给的参数
2. 绕开启发式，取 `.jit_function` 手动传参扫参数空间（同 `tuning/ops/sweep_norm.py` 的做法）
3. 对比后改阈值，然后在**未参与拟合的 shape** 上验证不回退

**arch 层有独立入口，但目前是断开的。** `configs_loader.py:576` 的查找顺序是
arch → vendor → default，和候选表一样按键回退：

```python
def get_heuristics_config(self, op_name):
    if self.arch_heuristics_config and op_name in self.arch_heuristics_config:
        return self.arch_heuristics_config[op_name]
    elif op_name in self.vendor_heuristics_config:
        ...
```

`arch_heuristics_config` 来自 `getattr(arch_module, "HEURISTICS_CONFIGS", None)`
（`backend/__init__.py:161`），而 `_amd/rdna4/__init__.py` 现在只有 license 头、没有这个属性，
所以取到 None，全部回退到 vendor。要启用只需在 `_amd/rdna4/` 下建
`heuristics_config_utils.py`，再在 `_amd/rdna4/__init__.py` 里导出：

```python
from .heuristics_config_utils import HEURISTICS_CONFIGS  # noqa: F401
```

这样改动只对 gfx1200/1201 生效，CDNA 完全不受影响，比在函数体里查 device properties 分支干
净。`_nvidia/hopper/heuristics_config_utils.py` 是个 13 行的空占位文件，说明这条路设计上是留
好的，只是还没有厂商用。注意覆盖粒度是**整个键**，不是单个参数——接管 `softmax_inner` 就要
把它下面所有参数都写全。

**一个系统性怀疑：`num_warps` 上限。** 这个文件里 7 个 `*_num_warps` 启发式（dropout、
exponential、rand、randn、uniform、softmax_inner、softmax_non_inner）上限全是 16，没有一个
返回 32。而 16 warps 恰好是 wave64 上 1024 线程的 workgroup 上限，是 CDNA 的最大值；gfx1201
是 wave32，1024 线程等于 32 warps。也就是说这个文件只用到了 RDNA 一半的 workgroup 容量。

这和 layer_norm 踩的是同一个天花板（回退表卡在 `w16`，热态实测 `w32` 快 25%，见
`TUNING.md` 第 13 节）。值得按上面三步法单独验一次——判断依据是结构性的，且一次覆盖 7 个
函数。注意这是假设不是结论，softmax 的访存模式与 layer_norm 不同。

**softmax 的上限在哪。** 五个 core shape 里只有 `[1048576]` 低于 PyTorch（0.826×），而它的
瓶颈不在参数：`softmax_kernel_inner` 的 grid 是 `(M, 1, 1)`，一维张量做 softmax 必然
`M=1`，所以整个 100 万元素只跑在 1 个 workgroup 上，32 个 CU 闲 31 个，实测带宽约 20 GB/s
（对比 layer_norm 热态的 550 GB/s）。启发式最多把这个 workgroup 从 512 线程加到 1024，上限
2×，改不了并行度的本质。要修得拆行做两趟归约，属于 kernel 层。

**改之前必须查的坑。** 启发式的键跨算子共享，键名和算子名不是一回事。
`softmax_inner` / `softmax_non_inner` 这两个键被 8 个文件消费：`softmax`、`log_softmax`、
`sum`、`mean`、`std`、`prod`、`logsumexp`、`special_logsumexp`。为 softmax 调一次参，这 8 个
算子一起变。动手前先 `rg -n 'get_heuristic_config\("<键名>"\)' src/`，回归也要按这个清单测。

### 5.3 AMD 专属实现：`_amd/ops/` 与 `_amd/rdna4/ops/`

#### 覆盖机制

`flag_gems/__init__.py` 的顺序是关键：第 28 行 `from flag_gems.ops import *` 把通用算子拉进模
块 globals，第 51 行 `SpecOpRegistrar` 用厂商实现覆盖同名项，第 62 行才构造 `_FULL_CONFIG`
（aten 注册表引用的就是 globals 里那些名字）。所以只要写一个同名函数并导出，aten 就走新实
现，框架一行都不用改。

```python
# src/flag_gems/runtime/backend/__init__.py:254
    def apply(self, vendor=None):
        vendor = vendor or self.vendor
        spec_events = self._get_specific_events()
        for event in spec_events:
            if not event.is_available():
                continue
            operators = event.get_ops(vendor)
            for fn_name, fn in operators:
                self._globals[fn_name] = fn

    def _get_specific_events(self):
        return (_state, BackendArchEvent(), TritonVersionEvent())
```

事件顺序决定了 arch 层后写后赢，于是有两个作用域：

| 目录 | 生效范围 |
|---|---|
| `_amd/ops/`、`_amd/fused/` | 所有 AMD 卡，含 CDNA gfx942 |
| `_amd/rdna4/ops/` | 只有 gfx1200/1201 |

现状是 `_amd/ops/__init__.py` 为 `__all__ = []` 空壳，`_amd/fused/` 还不存在，一个自定义算子都
没有。对比其他厂商的算子文件数：kunlunxin 257、cambricon 140、sunrise 115、ascend 80、
tsingmicro 48、mthreads 46、metax 38。`_nvidia/ops/` 下的 add / gelu 是被整段注释掉的示例，不
是活代码；真正能照抄的范例是 arch 层的 `_nvidia/hopper/ops/`（fill、mm、sqrt 等 6 个文件）。

#### fused 的分派方式和 ops 不同

`fused/` 下 59 个文件、`__init__.py` 导出 79 个函数，做的是把多个算子合进一个 kernel 省掉中间
张量的往返，内容基本都是 LLM 推理里的固定组合：`fused_add_rms_norm`、`silu_and_mul`、
`rotary_embedding`、`reshape_and_cache`、`fused_moe`、`flash_mla`、`cross_entropy_loss`。

**它不走 aten 分派。** `ops/` 的算子全部进 `_FULL_CONFIG` 注册到 aten，写 `torch.mean(x)` 就自
动走 FlagGems；fused 的 79 个导出里只有 5 个进了 `_FULL_CONFIG`（`add_rms_norm`、
`beam_search_score`、`beam_search_score_`、`bincount`、`matmuladd`），其余 74 个在 aten 里没有
对应算子——PyTorch 不存在 `silu_and_mul` 这种 op，没有名字可以 impl。所以 fused 只能被显式调
用，三条入口：

1. 直接调 `flag_gems.silu_and_mul(x1, x2)`
2. `flag_gems/modules/` 的 `GemsRMSNorm` 等 nn.Module，以及 `gems_rms_forward` /
   `gems_silu_and_mul` / `gems_rope_forward` 三个函数式接口
3. `patches/patch_vllm_all.py` 猴补丁替换 vLLM 的 forward，这是生产里的主路径

第 3 条的完整链路是 `RMSNorm.forward_cuda`（`patch_vllm_all.py:662` 挂钩）→
`gems_rms_forward` → 顶层名，有 residual 才走 fused：

```python
# src/flag_gems/modules/normalization.py:50
    add_residual = residual is not None
    if add_residual:
        if use_c_extension:
            torch.ops.flag_gems.fused_add_rms_norm(x, residual, weight, eps)
            return x, residual
        else:
            return flag_gems.fused_add_rms_norm(
                x, residual, list(weight.size()), weight, eps
            )
```

**所以第 3 节那 10 个算子的 benchmark 和整轮 tune 完全没有碰到任何 fused 实现。** 要测得单独跑
`benchmark/test_fused_add_rms_norm.py`、`test_silu_and_mul.py` 这些文件，或跑真实 vLLM 负载。

**覆盖有效，但有两个例外。** 上面那段用的是 `flag_gems.fused_add_rms_norm` 顶层属性，正是
`SpecOpRegistrar` 覆盖的目标，所以在 `_amd/fused/` 放一份实现，vLLM 那条链路会自动切过去，
patches 一行都不用改。看不到覆盖的写法有两个：

- `from flag_gems.fused import flash_mla` 这种子模块路径（`patch_vllm_all.py:124` 就是），走的
  是 `flag_gems.fused` 包命名空间，而覆盖只写进 `flag_gems` 的 globals
- `use_c_extension = True` 时走 `torch.ops.flag_gems.fused_add_rms_norm`，绕过整个 Python 层。
  默认为 False（`config.py:29`），编了 C++ 扩展才需要留意

现状和 `ops/` 一样空：`_nvidia/fused/__init__.py` 同为整段注释的示例，`fused_add_rms_norm.py`
文件在那儿但没导出、是死代码。真正在用的是 kunlunxin 22 个、cambricon 9 个、ascend 9 个、
metax 3 个、arm 2 个。cambricon 那 9 个（`cross_entropy_loss`、`fused_add_rms_norm`、
`silu_and_mul`、`skip_layernorm`、`outer` 等）可以当模板。

#### 相比 5.1 / 5.2 的两个优势

**能按 shape 分派。** 5.1 的粒度是整个算子，`bmm` 一刀切会连 `[2,384,384]` 的 1.074× 一起丢
掉；专属实现的 wrapper 里可以按形状判断，赢的走 Triton、输的转 torch。

**能保留实测兜底。** 专属 kernel 照样挂 `@libtuner`，而且 arch 目录能带自己的 YAML 键。
hopper 的 `sqrt` 就只存在于 `_nvidia/hopper/tune_configs.yaml`（3 条候选），vendor 层 90 个键
里没有，整个文件 70 行就替掉了通用 `pointwise_dynamic` 版本：

```python
# src/flag_gems/runtime/backend/_nvidia/hopper/ops/sqrt.py:27
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sqrt"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def sqrt_kernel(
```

即这条路把 config 与 kernel 两条线接上了：自己定 kernel 形状，再配候选表让 autotuner 挑参
数，不必手工拟合。

#### 候选

前三个来自第 3 节的 benchmark 缺口，按性价比排；第四个不在那 10 个算子里，优先级取决于目标场
景是 LLM 推理还是通用算子库。

**softmax `[1048576]`，0.826×，止血成本最低。** 一维输入必然 `M=1`，而 grid 是
`(M, 1, 1)`（`ops/softmax.py:333`），100 万元素跑在 1 个 workgroup 上，31 个 CU 全闲。wrapper
里判断 `M == 1 and N > 阈值` 转 `torch.softmax` 几行就能回到 1.0×；要超越得拆 split-N 两趟
（partial max/sum 再合并）。

**mean 全量归约，0.416×，收益最大。** 瓶颈不在两个 kernel 而在启动前这几行：

```python
# src/flag_gems/ops/mean.py:84
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
```

`[64,512,512]` 是 1677 万元素，`sqrt` 一开就是 4096 个 workgroup × 每个 4096 元素；`num_warps`
没传、取 Triton 默认 4，wave32 下 128 条 lane 扛 4096 个元素，每 lane 32 个全压寄存器。`mid`
用输入 dtype，bf16 存中间和。分块规则里没有 CU 数也没有 wave 宽度。可试的方向：按 CU 数定
grid、循环累加代替一次性大块、中间量固定 fp32，再挂 libtuner。**这些是假设，先扫参数确认瓶
颈是寄存器还是访存并行度再动手。**

**bmm，纯 shape 回退。** 不写 kernel，wrapper 里判形状即可，是 5.1 做不到的那部分。

**fused_add_rms_norm，若目标是 LLM 推理则优先级可能最高。** 调参水平和 `mean` 同一档，零调
优：

```python
# src/flag_gems/fused/fused_add_rms_norm.py:87
    BLOCK_SIZE = triton.next_power_of_2(N)
    ...
        fused_add_rms_norm_kernel[M,](
            x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
        )
```

只有 `@libentry() @triton.jit`，没有 autotune 也没有启发式，`BLOCK_SIZE` 直接取
`next_power_of_2(N)`，`num_warps` 压根没传（Triton 默认 4，wave32 下只有 128 条 lane）——和
layer_norm 踩的是同一个坑，而 layer_norm 那边实测 `w32` 比 `w16` 快 25%。它只有一个 kernel、
逻辑简单、是 decode 阶段每层都要调的热点。**但目前连 baseline 都没有**，连是否慢于 PyTorch 都
不知道，动手前先跑 `benchmark/test_fused_add_rms_norm.py`。

#### 坑

- 覆盖按**函数名**匹配，必须和 `_FULL_CONFIG` 引用的名字一致：`mean`、`bmm`、`softmax`
  （aten 键是 `_softmax`，函数名是 `softmax`）
- `get_customized_ops` 用 `inspect.getmembers(module, inspect.isfunction)`，会把
  `__init__.py` 命名空间里**所有**函数注入 globals。`from .mean import mean, _helper` 会连
  `_helper` 一起注入，撞名即出事——只导出要覆盖的算子
- 算子间的直接 import 绕过覆盖：`ops/subtract_.py` 走
  `from flag_gems.ops.sub import sub_`，覆盖 `sub_` 它看不见。mean/softmax/bmm 无此问题，加
  新算子前先 `rg 'from flag_gems.ops.<模块>' src/`
- 专属实现等于复制一份代码，上游改通用版不会同步。只对有结构性差异的算子做，纯参数问题优
  先用 5.2 或候选表解决

## 6. 复核命令

```bash
cd /target/FlagGems

# 各层候选表分别有哪些键
python - <<'PY'
import yaml
for name, f in (("rdna4", "src/flag_gems/runtime/backend/_amd/rdna4/tune_configs.yaml"),
                ("_amd",  "src/flag_gems/runtime/backend/_amd/tune_configs.yaml")):
    print(name, ":", ", ".join(sorted(yaml.safe_load(open(f)) or {})))
PY

# 某个键到底被哪些 kernel 消费（tuned 与 heuristic 是两个注册表）
rg -n 'get_tuned_config\("naive_reduction"\)' src/
rg -n 'get_heuristic_config\("softmax_inner"\)' src/

# 某个算子的 benchmark 入口走的是哪条路径
rg -n 'op_name|torch_op' benchmark/test_sum.py
rg -n -B3 'def sum_kernel_1|def sum_dim_kernel' src/flag_gems/ops/sum.py

# 各厂商自定义了多少算子（第 5.3 节）
cd src/flag_gems/runtime/backend
for d in _*/; do v=${d%/}
  echo "$v ops=$(ls $v/ops/*.py 2>/dev/null | grep -vc __init__)" \
       "fused=$(ls $v/fused/*.py 2>/dev/null | grep -vc __init__)"
done

# 某个启发式键被哪些算子共享（改之前必查）
rg -n 'get_heuristic_config\("softmax_inner"\)' src/flag_gems/ops/

# 某个 fused 算子是怎么被调到的（三条入口都查一遍）
rg -n 'fused_add_rms_norm' src/flag_gems/__init__.py src/flag_gems/modules/ \
       src/flag_gems/patches/

# fused 里哪些进了 aten 注册表（只有 5 个）
python - <<'PY'
import ast, pathlib
root = pathlib.Path("src/flag_gems")
reg = {n.id for x in ast.walk(ast.parse((root / "__init__.py").read_text()))
       if isinstance(x, ast.Tuple) for n in x.elts if isinstance(n, ast.Name)}
fused = {a.asname or a.name
         for x in ast.walk(ast.parse((root / "fused/__init__.py").read_text()))
         if isinstance(x, ast.ImportFrom) for a in x.names}
print(f"fused 导出 {len(fused)} 个，其中走 aten 的 {len(fused & reg)} 个:",
      ", ".join(sorted(fused & reg)))
PY
```
