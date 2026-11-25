"""Compile-only (no GPU): instruction mix of the rdna3 KDA _recurrent kernel,
as shipped and with tl.fma(a, b, -0.0) written as a * b."""

import importlib.util
import os
import re
import sys
from collections import Counter

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

W = open("/tmp/fgs-b.path").read().strip()
SRC = f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_recurrent_kda.py"
SUBS = [
    ("state = tl.fma(state, d[None, :], -0.0)", "state = state * d[None, :]"),
    ("_sum_rows(tl.fma(state, k[None, :], -0.0), RV, BK)", "_sum_rows(state * k[None, :], RV, BK)"),
    ("delta = tl.fma(value - pred, beta, -0.0)", "delta = (value - pred) * beta"),
    ("state + tl.fma(delta[:, None], k[None, :], -0.0)", "state + delta[:, None] * k[None, :]"),
    ("_sum_rows(tl.fma(state, q[None, :], -0.0), RV, BK)", "_sum_rows(state * q[None, :], RV, BK)"),
]


def variant(name):
    text = open(SRC).read()
    if name == "mul":
        for a, b in SUBS:
            assert a in text, a
            text = text.replace(a, b)
    path = f"/tmp/kda_{name}.py"
    open(path, "w").write(text)
    spec = importlib.util.spec_from_file_location(f"kda_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(mod, rv, nw, t=64, h=8, kd=128, vd=128):
    ptrs = ["Q", "K", "D", "V", "Beta", "S0", "O", "S1"]
    types = ["*fp32", "*fp32", "*fp32", "*bf16", "*bf16", "*fp32", "*bf16", "*fp32"]
    sig = dict(zip(ptrs, types))
    consts = dict(T=t, H=h, KD=kd, VD=vd, VS=(t * h * vd, h * vd, vd, 1), BS=(t * h, h, 1), PER_V=False, RV=rv, BK=128)
    sig.update({k: "constexpr" for k in consts})
    src = ASTSource(mod._recurrent, sig, consts, {(i,): [["tt.divisibility", 16]] for i in range(8)})
    k = triton.compile(src, target=GPUTarget("hip", "gfx1100", 32), options=dict(num_warps=nw, enable_fp_fusion=False))
    asm = k.asm["amdgcn"]
    body = [ln.strip() for ln in asm.splitlines() if ln.startswith("\t") and not ln.strip().startswith((".", ";"))]
    ops = Counter(ln.split()[0] for ln in body if ln)
    vgpr = re.search(r"\.vgpr_count:\s+(\d+)", asm)
    valu = sum(v for o, v in ops.items() if o.startswith("v_"))
    dual = sum(v for o, v in ops.items() if o.startswith("v_dual"))
    return dict(instrs=len(body), valu=valu, dual=dual, fma=ops.get("v_fma_f32", 0), mul=ops.get("v_mul_f32_e32", 0),
                dpp=ops.get("v_mov_b32_dpp", 0), delay=ops.get("s_delay_alu", 0), vgpr=int(vgpr.group(1)) if vgpr else None)


mods = {n: variant(n) for n in ("base", "mul")}
print("variant  RV warps   T | instrs  VALU  dual  v_fma  v_mul  mov_dpp  s_delay  VGPRs  (VALU per row-token)")
for rv, nw, t in [(1, 1, 64), (2, 1, 64), (4, 1, 64), (4, 8, 1)]:
    for n, m in mods.items():
        s = build(m, rv, nw, t)
        steps = min(t, 8) * rv
        print(f"{n:7s} {rv:3d} {nw:5d} {t:3d} | {s['instrs']:6d} {s['valu']:5d} {s['dual']:5d} {s['fma']:6d} {s['mul']:6d} {s['dpp']:8d} {s['delay']:8d} {str(s['vgpr']):>6}  ({s['valu'] / steps:.1f})")
