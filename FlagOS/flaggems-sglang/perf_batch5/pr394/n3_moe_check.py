"""rdna3 fused_moe vs the fp32 reference (unit-test tolerance) and vs the
generic op, over both paths."""

import importlib
import os
import sys

import torch

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401
from flaggems_sglang.reference import get_reference  # noqa: E402

ref = get_reference("fused_moe")
r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
g = importlib.import_module("flaggems_sglang.ops.fused_moe")
TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
def ulp_err(out, want, dtype):
    mant = 7 if dtype == torch.bfloat16 else 10
    tiny = torch.finfo(dtype).tiny
    rw = want.to(dtype).float()
    ulp = torch.exp2(torch.floor(torch.log2(rw.abs().clamp_min(tiny))) - mant)
    return ((out.float() - want.float()).abs() / ulp).max().item()


ulps = []
means = []
odd = []
fails = cases = 0
ratios = []
worse = []


def make(T, H, I, E, topk, dtype, seed, skewed=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, H, device="cuda", generator=gen).to(dtype)
    w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(dtype)
    w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(dtype)
    logits = torch.randn(T, E, device="cuda", generator=gen)
    if skewed:
        logits[:, E // 2:] -= 100.0  # half the experts get no token
    tw, ti = torch.topk(torch.softmax(logits, -1), topk, dim=-1)
    return x, w1, w2, (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None)


for dtype in (torch.bfloat16, torch.float16):
    for (T, H, I, E, topk) in ((1, 128, 256, 8, 2), (37, 256, 512, 8, 2), (64, 512, 256, 16, 4), (128, 256, 512, 8, 2),
                               (1, 2048, 1408, 64, 8), (8, 2048, 1408, 64, 8), (64, 2048, 1408, 64, 8),
                               (300, 2048, 1408, 64, 8), (1, 2048, 128, 256, 8), (4, 2048, 128, 256, 8),
                               (16, 2048, 128, 256, 8), (200, 2048, 128, 256, 8), (513, 1024, 96, 32, 6),
                               (9, 576, 80, 12, 3), (2, 1536, 128, 64, 4)):
        for act in ("silu", "gelu"):
            for rsf in (1.0, 2.5):
                for skewed in (False, True):
                    if act == "gelu" and (rsf != 1.0 or skewed) or dtype == torch.float16 and (rsf != 1.0 or skewed):
                        continue
                    x, w1, w2, topk_output = make(T, H, I, E, topk, dtype, T + E, skewed)
                    out = r.fused_moe(x, w1, w2, topk_output, act, rsf)
                    want = ref(x, w1, w2, topk_output, act, rsf)
                    gen_out = g.fused_moe(x, w1, w2, topk_output, act, rsf)
                    cases += 1
                    tol = TOL[dtype]
                    ok = torch.allclose(out.float(), want.float(), atol=tol, rtol=tol)
                    err = (out.float() - want.float()).abs().max().item()
                    gerr = (gen_out.float() - want.float()).abs().max().item()
                    ratios.append(err / max(gerr, 1e-12))
                    ulps.append((ulp_err(out, want, dtype), ulp_err(gen_out, want, dtype)))
                    means.append(((out.float() - want.float()).abs().mean().item(),
                                  (gen_out.float() - want.float()).abs().mean().item()))
                    rr = means[-1][0] / max(means[-1][1], 1e-30)
                    if rr > 1.1 or rr < 0.9:
                        odd.append((T, H, I, E, topk, str(dtype)[6:], act, rsf, skewed, 'pairs' if r._use_pairs(T * topk, E) else 'grouped', '%.3e' % means[-1][0], '%.3e' % means[-1][1], round(rr, 3)))
                    if err > gerr:
                        worse.append((T, H, I, E, topk, str(dtype)[6:], act, rsf, skewed, err, gerr))
                    if not ok:
                        fails += 1
                        print(f"FAIL {dtype} T={T} H={H} I={I} E={E} k={topk} {act} rsf={rsf} skewed={skewed} "
                              f"err {err:.3e} (generic {gerr:.3e})", flush=True)
print("cases", cases, "fails", fails)
print("err/generic: max %.3f median %.3f; worse than generic in %d cases" % (max(ratios), sorted(ratios)[len(ratios) // 2], len(worse)))
print("max error in output ulps: new %.3f generic %.3f" % (max(u[0] for u in ulps), max(u[1] for u in ulps)))
print("cases over 1 ulp: new %d generic %d" % (sum(u[0] > 1 for u in ulps), sum(u[1] > 1 for u in ulps)))
mr = sorted(m[0] / max(m[1], 1e-30) for m in means)
print("mean abs error new/generic: min %.3f median %.3f max %.3f" % (mr[0], mr[len(mr) // 2], mr[-1]))
for o in odd:
    print("  odd", o)
