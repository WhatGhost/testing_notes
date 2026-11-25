"""Is the RV=16 / 4-warp output corruption run-to-run nondeterministic, and
what do the wrong values look like?"""

import sys

sys.argv = ["x"]
exec(open("/tmp/k_explain.py").read().split('print("== 1.')[0])

kw = stable_case(16, 64)
o_ref, s_ref = gen.fused_recurrent_kda(**cl(kw))
outs = []
for i in range(5):
    o, s = run((16, 4), kw)
    d = o.view(torch.int16) != o_ref.view(torch.int16)
    outs.append(o)
    tok = d.nonzero()[:, 1]
    print(f"run {i}: {d.sum().item()} wrong outputs, state identical: {torch.equal(s.view(torch.int32), s_ref.view(torch.int32))}, "
          f"tokens hit {tok.unique().numel()}/64, token 0 hit: {(tok == 0).any().item()}")
print("runs identical to each other:", [torch.equal(outs[0].view(torch.int16), o.view(torch.int16)) for o in outs[1:]])
d = (outs[0].view(torch.int16) != o_ref.view(torch.int16)).nonzero()[:6]
for b, t, h, v in d.tolist():
    print(f"  b={b} t={t} h={h} v={v}: generic {o_ref[b, t, h, v].item():+.4e}  RV=16 {outs[0][b, t, h, v].item():+.4e}")
