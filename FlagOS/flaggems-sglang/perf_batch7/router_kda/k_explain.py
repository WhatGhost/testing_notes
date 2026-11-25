"""Characterize the two bitwise mismatches seen while tuning the rdna3 KDA op:
RV=16 tiles, and RV=2 on the large benchmark shapes."""

import importlib.util
import os
import sys

import torch
import triton
import triton.language as tl

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen", f"{W}/src/flaggems_sglang/ops/fused_recurrent_kda.py")
new = load("new", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_recurrent_kda.py")
bm = __import__("benchmark.test_fused_recurrent_kda", fromlist=["x"])
_sum_rows = new._sum_rows


def cl(kw):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}


def run(cfg, kw):
    new._config = lambda b, t, h, vd: cfg
    return new.fused_recurrent_kda(**cl(kw))


def ulps16(a, b):
    def o(x):
        i = x.view(torch.int16).to(torch.int32) & 0xFFFF
        return torch.where(i >= 0x8000, -(i & 0x7FFF), i)
    return (o(a) - o(b)).abs()


def ulps32(a, b):
    def o(x):
        i = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        return torch.where(i >= 0x80000000, -(i & 0x7FFFFFFF), i)
    return (o(a) - o(b)).abs()


def stable_case(b, t, h=8, kd=128, vd=128):
    g = torch.Generator(device="cuda").manual_seed(1)
    r = lambda *d: torch.randn(*d, device="cuda", generator=g)  # noqa: E731
    return dict(q=r(b, t, h, kd).bfloat16(), k=r(b, t, h, kd).bfloat16(), v=r(b, t, h, vd).bfloat16(),
                g=-0.05 * torch.rand(b, t, h, kd, device="cuda", generator=g),
                beta=torch.rand(b, t, h, device="cuda", generator=g).bfloat16(), scale=kd ** -0.5,
                initial_state=0.1 * r(b, h, vd, kd), use_qk_l2norm_in_kernel=True)


def compare(label, kw, cfg):
    o1, s1 = gen.fused_recurrent_kda(**cl(kw))
    o2, s2 = run(cfg, kw)
    fin = torch.isfinite(o1.float()) & torch.isfinite(o2.float())
    od = o1.view(torch.int16) != o2.view(torch.int16)
    sd = s1.view(torch.int32) != s2.view(torch.int32)
    msg = f"{label:34s} cfg {cfg}: out diffs {od.sum().item():6d}/{o1.numel()}"
    if od.any():
        u = ulps16(o1, o2)[od & fin]
        mag = o1.float().abs()[od & fin]
        msg += f" (max {u.max().item() if u.numel() else '-'} ulp, |value| {mag.min().item():.2e}..{mag.max().item():.2e})"
    msg += f" | state diffs {sd.sum().item()}"
    if sd.any():
        f2 = torch.isfinite(s1) & torch.isfinite(s2)
        u = ulps32(s1, s2)[sd & f2]
        msg += f" (finite: {u.numel()}, max {u.max().item() if u.numel() else '-'} ulp)"
    print(msg, flush=True)


print("== 1. benchmark inputs (raw randn k, no L2 norm): growth over tokens, generic op ==")
for shape in [(4, 64), (32, 128)]:
    kw = next(iter(bm._input_fn(shape, torch.bfloat16, "cuda")))
    o, s = gen.fused_recurrent_kda(**cl(kw))
    of = o.float()
    print(f"shape {shape}: |k|^2 mean {kw['k'].float().pow(2).sum(-1).mean().item():.0f}, beta mean {kw['beta'].float().mean().item():.2f}")
    row = []
    for ti in [0, 2, 4, 8, 12, 16, 20, 24, 32, 63]:
        x = of[:, ti]
        f = x[torch.isfinite(x)]
        row.append(f"t={ti}: {f.abs().max().item() if f.numel() else float('nan'):.1e} ({(~torch.isfinite(x)).float().mean().item() * 100:.0f}% inf/nan)")
    print("  max |output| per token: " + ", ".join(row))
    print(f"  final state: {torch.isnan(s).float().mean().item() * 100:.1f}% NaN, {torch.isinf(s).float().mean().item() * 100:.1f}% inf")

print("\n== 2. RV=16 vs generic ==")
compare("bench (4,64) [blows up]", next(iter(bm._input_fn((4, 64), torch.bfloat16, "cuda"))), (16, 4))
compare("stable (16,64) l2norm", stable_case(16, 64), (16, 4))
compare("stable (4,1) single step", stable_case(4, 1), (16, 4))
for cfg in [(1, 1), (2, 1), (4, 1), (8, 4), (32, 8)]:
    compare("stable (16,64) l2norm", stable_case(16, 64), cfg)


@triton.jit
def _rows_kernel(X, Y, RV: tl.constexpr, BK: tl.constexpr):
    r = tl.program_id(0) * RV + tl.arange(0, RV)
    c = tl.arange(0, BK)
    x = tl.load(X + r[:, None] * BK + c[None, :])
    tl.store(Y + r, _sum_rows(x, RV, BK))


print("\n== 3. _sum_rows alone vs the exact adjacent-pair tree (4096 rows x 128) ==")
x = torch.randn(4096, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
x = x * torch.exp(3 * torch.randn_like(x))
ref = x.clone()
while ref.shape[1] > 1:
    ref = ref[:, 0::2] + ref[:, 1::2]
ref = ref[:, 0]
for rv, nw in [(1, 1), (2, 1), (4, 1), (8, 4), (16, 1), (16, 2), (16, 4), (16, 8), (32, 8), (64, 8)]:
    y = torch.empty(4096, device="cuda")
    _rows_kernel[(4096 // rv,)](x, y, rv, 128, num_warps=nw, enable_fp_fusion=False)
    bad = (y.view(torch.int32) != ref.view(torch.int32)).sum().item()
    print(f"RV={rv:2d} warps={nw}: {bad} rows differ from the pair tree")

print("\n== 4. RV=2 vs generic (4,1) on bench (32,128): what differs in the final state ==")
kw = next(iter(bm._input_fn((32, 128), torch.bfloat16, "cuda")))
o1, s1 = gen.fused_recurrent_kda(**cl(kw))
o2, s2 = run((2, 1), kw)
b1, b2 = s1.view(torch.int32), s2.view(torch.int32)
d = b1 != b2
print(f"output bitwise equal: {torch.equal(o1.view(torch.int16), o2.view(torch.int16))}")
print(f"state: {d.sum().item()} words differ; NaN in generic {torch.isnan(s1).sum().item()}, in RV=2 {torch.isnan(s2).sum().item()}, "
      f"NaN positions differ at {(torch.isnan(s1) != torch.isnan(s2)).sum().item()}")
xor = (b1 ^ b2)[d]
print(f"differing words whose XOR is only the sign bit: {(xor == -2147483648).sum().item()} / {xor.numel()}")
vals, cnt = torch.unique(torch.stack([b1[d], b2[d]], 1), dim=0, return_counts=True)
for v, c in zip(vals.tolist(), cnt.tolist()):
    print(f"  generic {v[0] & 0xFFFFFFFF:#010x} vs RV=2 {v[1] & 0xFFFFFFFF:#010x}: {c}")
