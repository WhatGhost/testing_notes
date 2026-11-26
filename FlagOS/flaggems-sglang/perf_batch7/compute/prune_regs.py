"""Compile-only (no GPU) register use of the rdna3 grouped_gemm and absorbed
kernels over their sweep spaces; writes the spill-free configs to json."""

import importlib.util
import itertools
import json
import re
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, f"{W}/src"]
TARGET = GPUTarget("hip", "gfx1100", 32)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def regs(fn, sig, consts, nptr, nw, ns=None):
    sig = dict(sig, **{k: "constexpr" for k in consts})
    opts = dict(num_warps=nw)
    if ns is not None:
        opts["num_stages"] = ns
    k = triton.compile(ASTSource(fn, sig, consts, {(i,): [["tt.divisibility", 16]] for i in range(nptr)}),
                       target=TARGET, options=opts)
    asm = k.asm["amdgcn"]
    g = lambda key: int(re.search(rf"\.{key}:\s+(\d+)", asm).group(1))  # noqa: E731
    return g("vgpr_count"), g("vgpr_spill_count")


out = {}
gg = load("gg_r", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/grouped_gemm.py")
ok = []
for (n, k) in [(512, 2048), (1536, 2048), (2048, 768)]:
    for bm, bn, bk, nw, ns in itertools.product((16, 32, 64, 128), (32, 64, 128, 256), (32, 64, 128, 256), (2, 4, 8), (1, 2, 3)):
        if bm * bn // (32 * nw) > 256 or (bm + bn) * bk * 2 * max(ns - 1, 1) > 64 * 1024 or bk > k:
            continue
        try:
            v, s = regs(gg._grouped_kernel, dict(A="*bf16", B="*bf16", C="*bf16", OFFS="*i32", M="i32", E="i32"),
                        dict(N=n, K=k, EP=8, BM=bm, BN=bn, BK=bk, SPLIT_STORE=k <= 1024), 4, nw, ns)
        except Exception as e:  # noqa: BLE001
            print("gg fail", (n, k, bm, bn, bk, nw, ns), str(e)[:80], flush=True)
            continue
        if s == 0:
            ok.append([n, k, bm, bn, bk, nw, ns, v])
    print(f"grouped_gemm N={n} K={k}: {sum(1 for o in ok if o[0] == n and o[1] == k)} spill-free configs", flush=True)
out["grouped_gemm"] = ok

ab = load("ab_r", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/absorbed_bmm_concat_cast_q_fp8.py")
ok = []
for bm, bn, bk, nw, pack in itertools.product((32, 64, 128), (64, 128, 256), (64, 128), (2, 4, 8), (True, False)):
    if bm * bn // (32 * nw) > 256:
        continue
    try:
        v, s = regs(ab._absorbed_kernel, dict(X="*bf16", W="*bf16", R="*bf16", O="*i32" if pack else "*u8", T="i32", O0="i32", O1="i32"),
                    dict(H=16, K=128, N=512, D=64, BM=bm, BN=bn, BK=bk, BR=64, PACK=pack), 4, nw)
    except Exception as e:  # noqa: BLE001
        print("ab fail", (bm, bn, bk, nw, pack), str(e)[:80], flush=True)
        continue
    if s == 0:
        ok.append([bm, bn, bk, nw, pack, v])
print(f"absorbed: {len(ok)} spill-free configs", flush=True)
out["absorbed"] = ok
json.dump(out, open("/tmp/prune_regs.json", "w"))
