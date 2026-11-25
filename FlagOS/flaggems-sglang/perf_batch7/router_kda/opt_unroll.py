"""KDA _recurrent with other loop unroll factors (scheduling only): recurrent
kernel time per shape, 3 rounds, medians, and bitwise vs the shipped factor."""

import importlib.util
import os
import sys

import torch
import triton

sys.path.insert(0, "/tmp")
import prep_lib as P  # noqa: E402

src = open(f"{P.W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_recurrent_kda.py").read()
assert "loop_unroll_factor=8" in src
mods = {}
for uf in (1, 2, 4, 8, 16):
    path = f"/tmp/kda_uf{uf}.py"
    open(path, "w").write(src.replace("loop_unroll_factor=8", f"loop_unroll_factor={uf}"))
    mods[uf] = P.load(f"kda_uf{uf}", path)


def recurrent(mod, kw):
    q, k, v, g, beta, s0 = kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"], kw["initial_state"].contiguous()
    b, t, h, kd = k.shape
    vd = v.shape[-1]
    qp = torch.empty((b, t, h, kd), dtype=torch.float32, device="cuda")
    kp, dp = torch.empty_like(qp), torch.empty_like(qp)
    out = torch.empty((b, t, h, vd), dtype=q.dtype, device="cuda")
    fs = torch.empty_like(s0)
    rv, nw = mod._config(b, t, h, vd)
    mod._prepare[(b * t * h,)](q, k, g, qp, kp, dp, h, t, kd, q.stride(), k.stride(), g.stride(), kw["scale"],
                               kw["use_qk_l2norm_in_kernel"], 128, num_warps=4, enable_fp_fusion=False)

    def rec():
        mod._recurrent[(b * h * triton.cdiv(vd, rv),)](qp, kp, dp, v, beta, s0, out, fs, t, h, kd, vd, v.stride(),
                                                      beta.stride(), beta.ndim == v.ndim, rv, 128, num_warps=nw,
                                                      enable_fp_fusion=False)
    rec()
    return rec, out, fs


shapes = [(1, 64), (4, 64), (16, 64), (64, 64), (2, 64), (8, 32), (32, 128)]
print(f"{'case':16s} " + " ".join(f"{'uf=' + str(u):>9s}" for u in mods) + " | bitwise vs uf=8")
for s in shapes:
    kw = next(iter(P.bm._input_fn(s, torch.bfloat16, "cuda")))
    res, outs = {}, {}
    for uf, mod in mods.items():
        rec, o, f = recurrent(mod, kw)
        outs[uf] = (o.clone(), f.clone())
        res[uf] = sorted(P.bench(rec) for _ in range(3))[1]
    same = [uf for uf in mods if torch.equal(outs[uf][0].view(torch.int16), outs[8][0].view(torch.int16))
            and torch.equal(outs[uf][1].view(torch.int32), outs[8][1].view(torch.int32))]
    print(f"bench {str(s):10s} " + " ".join(f"{res[u]:9.1f}" for u in mods) + f" | equal: {same}", flush=True)
