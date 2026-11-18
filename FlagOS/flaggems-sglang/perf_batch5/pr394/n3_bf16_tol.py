"""Would a 16-bit intermediate pass the fused_moe unit-test check?

Unit-test semantics: reference in float32, rounded to the input dtype,
compared with |out - ref| <= atol + rtol * |ref| (bf16 2e-2, fp16 1e-2).
'margin' is the worst element's share of that budget (> 1 fails).

Variants (grouped path):
  current  hi/lo intermediate, float32 c3 (this PR)
  a        16-bit intermediate, 16-bit c3
  b        16-bit intermediate, float32 c3
plus master's generic op and SGLang's Triton path (as the plugin runs it).
"""

import importlib
import importlib.util
import os
import re
import sys

import torch
import torch.nn.functional as F
from triton.testing import do_bench

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401
from flaggems_sglang.reference import get_reference  # noqa: E402

assert flaggems_sglang.__file__.startswith(W)
r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
g = importlib.import_module("flaggems_sglang.ops.fused_moe")
ref_op = get_reference("fused_moe")

src = open("/tmp/n3_fused_moe.before_cleanup.py").read()
src, n1 = re.subn(r"cfg=None, split_a=True\n", "cfg=None, split_a=True, c3_fp32=False\n", src)
src, n2 = re.subn(r"dtype=torch\.float32 if split_a else x\.dtype",
                  "dtype=torch.float32 if (split_a or c3_fp32) else x.dtype", src)
src, n3 = re.subn(r"SPLIT_STORE=not split_a,", "SPLIT_STORE=not (split_a or c3_fp32),", src)
assert (n1, n2, n3) == (1, 1, 1), (n1, n2, n3)
open("/tmp/n3_moe_variants.py", "w").write(src)
spec = importlib.util.spec_from_file_location("moe_variants", "/tmp/n3_moe_variants.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

spec = importlib.util.spec_from_file_location("sgl_moe", "/tmp/sglang_new/standalone/sgl_moe.py")
sgl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sgl)

TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}


def per_expert_reference(x, w1, w2, tw, ti, act, rsf):
    T, H = x.shape
    inter = w1.shape[1] // 2
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in range(w1.shape[0]):
        tok, slot = (ti == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = xf[tok] @ w1[e].float().t()
        a = (F.silu(h[:, :inter]) if act == "silu" else F.gelu(h[:, :inter])) * h[:, inter:]
        out.index_add_(0, tok, (a @ w2[e].float().t()) * tw[tok, slot].float()[:, None])
    return (out * rsf).to(x.dtype)


def margin(out, want, tol):
    o, w = out.float(), want.float()
    return ((o - w).abs() / (tol + tol * w.abs())).max().item()


def variants(x, w1, w2, to, act, rsf):
    tw, ti, _ = to
    T, H = x.shape
    E, inter, top_k = w1.shape[0], w2.shape[2], ti.shape[1]
    P = T * top_k
    ids = ti.reshape(-1).contiguous()
    wts = tw.reshape(-1).contiguous()
    a_code = 0 if act == "silu" else 1
    cfg = r._grouped_config(P, E, inter)

    def grouped(**kw):
        out = torch.empty_like(x)
        return v._fused_moe_grouped(x, w1, w2, ids, wts, top_k, a_code, rsf, out, cfg=cfg, **kw)

    return {
        "current": lambda: grouped(split_a=True),
        "a": lambda: grouped(split_a=False),
        "b": lambda: grouped(split_a=False, c3_fp32=True),
        "master": lambda: g.fused_moe(x, w1, w2, to, act, rsf),
        "sglang": lambda: sgl.fused_experts(x, w1, w2, tw, ti, act, rsf),
    }


def unit_case(T, H, I, E, k, act="silu", rsf=1.0, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    logits = torch.randn(T, E, device="cuda", generator=gen)
    tw, ti = torch.topk(torch.softmax(logits, dim=-1), k, dim=-1)
    return x, w1, w2, (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None), act, rsf


names = ("current", "a", "b", "master", "sglang")
print("case | " + " | ".join(f"{n:>14s}" for n in names))
fails = {n: 0 for n in names}
total = 0

# the 4 unit-test cases, all variants forced onto the grouped path
for i, args in enumerate([(1, 128, 256, 8, 2), (37, 256, 512, 8, 2, "silu", 1.0, 1),
                          (64, 512, 256, 16, 4, "silu", 2.5, 2), (128, 256, 512, 8, 2, "gelu", 1.0, 3)]):
    x, w1, w2, to, act, rsf = unit_case(*args)
    want = ref_op(x, w1, w2, to, act, rsf)
    res = []
    total += 1
    for n, fn in variants(x, w1, w2, to, act, rsf).items():
        m = margin(fn(), want, TOL[torch.bfloat16])
        fails[n] += m > 1
        res.append(f"{m:6.3f} {'ok  ' if m <= 1 else 'FAIL'}  ")
    print(f"unit test case {i} {args[:5]} bf16 | " + " | ".join(res), flush=True)

# large-T shapes: official benchmark layer and a 256-small-expert layer, plus two more
shapes = [(1024, 2048, 1408, 64, 8), (2048, 2048, 1408, 64, 8), (4096, 2048, 1408, 64, 8),
          (1024, 2048, 128, 256, 8), (2048, 2048, 128, 256, 8), (4096, 2048, 128, 256, 8),
          (4096, 1024, 512, 16, 2), (8192, 1024, 512, 8, 2)]
times = {}
for (T, H, I, E, k) in shapes:
    for dtype in (torch.bfloat16, torch.float16):
        for act, rsf in (("silu", 1.0), ("gelu", 2.5)):
            gen = torch.Generator(device="cuda").manual_seed(T + E + I)
            x = torch.randn(T, H, device="cuda", generator=gen).to(dtype)
            w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(dtype)
            w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(dtype)
            tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), k, dim=-1)
            to = (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None)
            want = per_expert_reference(x, w1, w2, to[0], to[1], act, rsf)
            fns = variants(x, w1, w2, to, act, rsf)
            res = []
            total += 1
            for n, fn in fns.items():
                m = margin(fn(), want, TOL[dtype])
                fails[n] += m > 1
                res.append(f"{m:6.3f} {'ok  ' if m <= 1 else 'FAIL'}  ")
            print(f"T={T} H={H} I={I} E={E} k={k} {str(dtype)[6:]} {act} rsf={rsf} | " + " | ".join(res), flush=True)
            if dtype == torch.bfloat16 and act == "silu" and k == 8:
                times[(T, I, E)] = {n: do_bench(fns[n], warmup=5, rep=40, return_mode="median") * 1e3
                                    for n in ("current", "a", "b", "sglang")}
print("fails out of", total, fails)
print("\ntimes (us), bf16 silu:")
for key, t in times.items():
    print(key, " ".join(f"{n} {val:8.1f}" for n, val in t.items()),
          "| a/current %.2fx b/current %.2fx" % (t["current"] / t["a"], t["current"] / t["b"]),
          "| vs sglang: current %.2fx a %.2fx b %.2fx" % (t["sglang"] / t["current"], t["sglang"] / t["a"],
                                                           t["sglang"] / t["b"]))
