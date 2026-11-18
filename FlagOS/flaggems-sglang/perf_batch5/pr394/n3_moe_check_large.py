"""Large-T fused_moe cases (BLOCK_M=64 grouped path) vs a per-expert fp32
reference, unit-test tolerance; generic op error alongside."""

import importlib
import os
import sys

import torch
import torch.nn.functional as F

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401

r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
g = importlib.import_module("flaggems_sglang.ops.fused_moe")
TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}


def reference(x, w1, w2, tw, ti, act, rsf):
    T, H = x.shape
    E, N, _ = w1.shape
    inter = N // 2
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in range(E):
        tok, slot = (ti == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = xf[tok] @ w1[e].float().t()
        gate, up = h[:, :inter], h[:, inter:]
        a = (F.silu(gate) if act == "silu" else F.gelu(gate)) * up
        y = a @ w2[e].float().t()
        out.index_add_(0, tok, y * tw[tok, slot].float()[:, None])
    return out * rsf


def ulp_err(out, want, dtype):
    mant = 7 if dtype == torch.bfloat16 else 10
    tiny = torch.finfo(dtype).tiny
    rw = want.to(dtype).float()
    ulp = torch.exp2(torch.floor(torch.log2(rw.abs().clamp_min(tiny))) - mant)
    return ((out.float() - want.float()).abs() / ulp).max().item()


fails = cases = 0
for (T, H, I, E, topk) in ((2048, 2048, 1408, 64, 8), (4096, 1024, 512, 16, 2), (8192, 1024, 512, 8, 2)):
    for dtype in (torch.bfloat16, torch.float16):
        for act, rsf in (("silu", 1.0), ("gelu", 2.5)):
            gen = torch.Generator(device="cuda").manual_seed(T + E)
            x = torch.randn(T, H, device="cuda", generator=gen).to(dtype)
            w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(dtype)
            w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(dtype)
            tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
            to = (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None)
            P = T * topk
            assert not r._use_pairs(P, E) and r._grouped_config(P, E, I)[0] == 64
            want = reference(x, w1, w2, to[0], to[1], act, rsf)
            out = r.fused_moe(x, w1, w2, to, act, rsf).float()
            gout = g.fused_moe(x, w1, w2, to, act, rsf).float()
            tol = TOL[dtype]
            ok = torch.allclose(out, want, atol=tol, rtol=tol)
            cases += 1
            fails += not ok
            print(f"{'ok  ' if ok else 'FAIL'} T={T} H={H} I={I} E={E} k={topk} {str(dtype)[6:]} {act} rsf={rsf} "
                  f"err {(out - want).abs().max().item():.3e} generic {(gout - want).abs().max().item():.3e} "
                  f"mean {(out - want).abs().mean().item():.3e} generic {(gout - want).abs().mean().item():.3e}",
                  flush=True)
print("cases", cases, "fails", fails)
