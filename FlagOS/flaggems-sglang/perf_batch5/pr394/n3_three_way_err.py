"""Max error of SGLang / master / new fused_moe for every T of the three-way
comparison: same inputs (same generator and order as n3_three_way.py), a
per-expert float32 reference rounded to bf16 (the unit-test reference's
output dtype), so large T fits in memory."""

import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F

W = open("/tmp/fgs-n2.path").read().strip()
M = open("/tmp/fgs-master.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401
from flaggems_sglang.reference import get_reference  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sgl = load("sgl_moe", "/tmp/sglang_new/standalone/sgl_moe.py")
master = load("master_fused_moe", os.path.join(M, "src/flaggems_sglang/ops/fused_moe.py"))
new = load("new_fused_moe", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_moe.py"))
ref_op = get_reference("fused_moe")
only = sys.argv[1]


def per_expert_reference(x, w1, w2, tw, ti):
    T, H = x.shape
    inter = w1.shape[1] // 2
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in range(w1.shape[0]):
        tok, slot = (ti == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = xf[tok] @ w1[e].float().t()
        a = F.silu(h[:, :inter]) * h[:, inter:]
        out.index_add_(0, tok, (a @ w2[e].float().t()) * tw[tok, slot].float()[:, None])
    return out.to(x.dtype)


rows = []
H, I, E, topk = {"official": (2048, 1408, 64, 8), "35B": (2048, 128, 256, 8)}[only]
gen = torch.Generator(device="cuda").manual_seed(0)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
for T in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
    tw, ti = tw.float().contiguous(), ti.to(torch.int32).contiguous()
    to = (tw, ti, None)
    want = per_expert_reference(x, w1, w2, tw, ti).float()
    err = {}
    for name, out in (("sglang", sgl.fused_experts(x, w1, w2, tw, ti)),
                      ("master", master.fused_moe(x, w1, w2, to)),
                      ("new", new.fused_moe(x, w1, w2, to))):
        err[name] = (out.float() - want).abs().max().item()
    same = None
    if T <= 512:  # cross-check against the unit-test reference itself
        same = torch.equal(ref_op(x, w1, w2, to, "silu", 1.0).float(), want)
    rows.append(dict(model=only, T=T, err=err, same_as_unit_ref=same))
    print(f"{only:8s} T={T:5d} " + " ".join(f"{k}={v:.2e}" for k, v in err.items())
          + f" | sglang/new {err['sglang'] / max(err['new'], 1e-30):7.1f}x | same_as_unit_ref={same}", flush=True)
json.dump(rows, open(f"/tmp/n3_three_way_err_{only}.json", "w"), indent=1)
