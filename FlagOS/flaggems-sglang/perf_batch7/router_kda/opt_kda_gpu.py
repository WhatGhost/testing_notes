"""KDA headroom on GPU: decode copy roofline, per-kernel times, the a * b
variant (speed + bitwise vs generic) and a decode tile sweep."""

import importlib.util
import os
import sys

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen", f"{W}/src/flaggems_sglang/ops/fused_recurrent_kda.py")
new = load("new", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_recurrent_kda.py")
mul = load("kda_mul", "/tmp/kda_mul.py")
bm = __import__("benchmark.test_fused_recurrent_kda", fromlist=["x"])


def bench(fn):
    return do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3


def cl(kw):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}


def decode_case(b, h=8, kd=128, vd=128):
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *d: torch.randn(*d, device="cuda", generator=g)  # noqa: E731
    return dict(q=r(b, 1, h, kd).bfloat16(), k=r(b, 1, h, kd).bfloat16(), v=r(b, 1, h, vd).bfloat16(),
                g=-0.05 * torch.rand(b, 1, h, kd, device="cuda", generator=g),
                beta=torch.rand(b, 1, h, device="cuda", generator=g).bfloat16(), scale=kd ** -0.5,
                initial_state=0.1 * r(b, h, vd, kd), use_qk_l2norm_in_kernel=True)


def pieces(mod, kw, cfg=None):
    q, k, v, g, beta, s0 = kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"], kw["initial_state"].contiguous()
    b, t, h, kd = k.shape
    vd = v.shape[-1]
    qp = torch.empty((b, t, h, kd), dtype=torch.float32, device="cuda")
    kp, dp = torch.empty_like(qp), torch.empty_like(qp)
    out = torch.empty((b, t, h, vd), dtype=q.dtype, device="cuda")
    fs = torch.empty_like(s0)
    rv, nw = cfg or mod._config(b, t, h, vd)

    def prep():
        mod._prepare[(b * t * h,)](q, k, g, qp, kp, dp, h, t, kd, q.stride(), k.stride(), g.stride(), kw["scale"],
                                   kw["use_qk_l2norm_in_kernel"], 128, num_warps=4, enable_fp_fusion=False)

    def rec():
        mod._recurrent[(b * h * triton.cdiv(vd, rv),)](qp, kp, dp, v, beta, s0, out, fs, t, h, kd, vd, v.stride(),
                                                      beta.stride(), beta.ndim == v.ndim, rv, 128, num_warps=nw,
                                                      enable_fp_fusion=False)
    prep()
    rec()
    return prep, rec


@triton.jit
def _copy(X, Y, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + offs, tl.load(X + offs, offs < N), offs < N)


print("== 1. decode roofline: read + write the fp32 state once ==")
for b in (16, 64, 256):
    s = torch.randn(b, 8, 128, 128, device="cuda")
    y = torch.empty_like(s)
    n = s.numel()
    t_clone = bench(lambda: y.copy_(s))
    t_tri = bench(lambda: _copy[(triton.cdiv(n, 4096),)](s, y, n, 4096, num_warps=8))
    mb = 2 * s.numel() * 4 / 1e6
    print(f"b={b:3d}: {mb:6.1f} MB moved | copy_ {t_clone:7.1f} us ({mb / t_clone * 1e3:5.0f} GB/s)  triton copy {t_tri:7.1f} us ({mb / t_tri * 1e3:5.0f} GB/s)")

print("\n== 2. per-kernel times (shipped config) ==")
cases = [(f"bench {s}", next(iter(bm._input_fn(s, torch.bfloat16, "cuda")))) for s in list(bm.SHAPES) + list(bm.MORE_SHAPES)]
cases += [(f"decode b={b}", decode_case(b)) for b in (16, 64, 256)]
for name, kw in cases:
    prep, rec = pieces(new, kw)
    t_full = bench(lambda: new.fused_recurrent_kda(**kw))
    t_gen = bench(lambda: gen.fused_recurrent_kda(**kw))
    t_mul = bench(lambda: mul.fused_recurrent_kda(**kw))
    print(f"{name:18s} generic {t_gen:8.1f} | shipped {t_full:8.1f} (prepare {bench(prep):6.1f}, recurrent {bench(rec):8.1f}) | a*b variant {t_mul:8.1f} ({t_full / t_mul:4.2f}x vs shipped)")

print("\n== 3. a*b variant vs generic: bitwise ==")
for name, kw in cases:
    o1, s1 = gen.fused_recurrent_kda(**cl(kw))
    o2, s2 = mul.fused_recurrent_kda(**cl(kw))
    strict = torch.equal(o1.view(torch.int16), o2.view(torch.int16)) and torch.equal(s1.view(torch.int32), s2.view(torch.int32))
    nan_ok = (torch.equal(torch.isnan(o1), torch.isnan(o2)) and torch.equal(torch.isnan(s1), torch.isnan(s2))
              and torch.equal(o1.view(torch.int16)[~torch.isnan(o1)], o2.view(torch.int16)[~torch.isnan(o2)])
              and torch.equal(s1.view(torch.int32)[~torch.isnan(s1)], s2.view(torch.int32)[~torch.isnan(s2)]))
    print(f"{name:18s} strict {'ok' if strict else 'DIFF'} | ignoring NaN sign {'ok' if nan_ok else 'DIFF'}")

print("\n== 4. decode tile sweep (recurrent kernel only) ==")
for b in (64, 256):
    kw = decode_case(b)
    res = []
    for cfg in [(1, 1), (2, 1), (4, 1), (8, 1), (2, 2), (4, 2), (8, 2), (16, 2), (4, 4), (8, 4), (32, 4), (4, 8), (8, 8), (16, 8), (32, 8)]:
        _, rec = pieces(new, kw, cfg)
        res.append((bench(rec), cfg))
    res.sort()
    print(f"b={b}: " + "  ".join(f"{c}:{t:.1f}" for t, c in res[:8]))
