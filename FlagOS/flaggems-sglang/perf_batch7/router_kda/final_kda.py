"""Final KDA check: bitwise vs generic (output and final state) with and
without L2 norm, then timings vs generic and vs the previous prepare launch
(one row per program, 4 warps), 3 rounds alternated, medians."""

import sys

import torch
import triton

sys.path.insert(0, "/tmp")
import prep_lib as P  # noqa: E402

gen = P.load("gen_final", f"{P.W}/src/flaggems_sglang/ops/fused_recurrent_kda.py")
new = P.new


def cl(kw):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}


def previous(kw):
    q, k, v, g, beta, s0 = kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"], kw["initial_state"].contiguous()
    b, t, h, kd = k.shape
    vd = v.shape[-1]
    qp = torch.empty((b, t, h, kd), dtype=torch.float32, device="cuda")
    kp, dp = torch.empty_like(qp), torch.empty_like(qp)
    out = torch.empty((b, t, h, vd), dtype=q.dtype, device="cuda")
    fs = torch.empty_like(s0)
    rv, nw = new._config(b, t, h, vd)
    new._prepare[(b * t * h,)](q, k, g, qp, kp, dp, b * t * h, h, t, kd, q.stride(), k.stride(), g.stride(),
                               kw["scale"], kw["use_qk_l2norm_in_kernel"], 128, 1, num_warps=4, enable_fp_fusion=False)
    new._recurrent[(b * h * triton.cdiv(vd, rv),)](qp, kp, dp, v, beta, s0, out, fs, t, h, kd, vd, v.stride(),
                                                  beta.stride(), beta.ndim == v.ndim, rv, 128, num_warps=nw,
                                                  enable_fp_fusion=False)
    return out, fs


cases = []
for norm in (False, True):
    for s in list(P.bm.SHAPES) + list(P.bm.MORE_SHAPES):
        cases.append((f"bench {s}{' +norm' if norm else ''}", dict(next(iter(P.bm._input_fn(s, torch.bfloat16, "cuda"))), use_qk_l2norm_in_kernel=norm)))
    for b in (1, 16, 64, 256):
        cases.append((f"decode b={b}{' +norm' if norm else ''}", P.decode_case(b, norm=norm)))

print("== bitwise vs generic ==")
bad = 0
for name, kw in cases:
    o1, s1 = gen.fused_recurrent_kda(**cl(kw))
    o2, s2 = new.fused_recurrent_kda(**cl(kw))
    ok = torch.equal(o1.view(torch.int16), o2.view(torch.int16)) and torch.equal(s1.view(torch.int32), s2.view(torch.int32))
    bad += not ok
    if not ok:
        print(f"DIFF {name}")
print(f"{len(cases) - bad}/{len(cases)} bitwise identical")

print("\n== timings (us) ==")
print(f"{'case':22s} {'generic':>8s} {'previous':>9s} {'now':>8s}  now vs generic  now vs previous")
for name, kw in cases:
    tg, tp, tn = [], [], []
    for _ in range(3):
        tg.append(P.bench(lambda: gen.fused_recurrent_kda(**kw)))
        tp.append(P.bench(lambda: previous(kw)))
        tn.append(P.bench(lambda: new.fused_recurrent_kda(**kw)))
    tg, tp, tn = (sorted(x)[1] for x in (tg, tp, tn))
    print(f"{name:22s} {tg:8.1f} {tp:9.1f} {tn:8.1f}  {tg / tn:13.2f}x  {tp / tn:14.2f}x", flush=True)
