"""Full (rows per program, warps) table for the KDA _prepare kernel with the
shipped arithmetic, with and without L2 norm, 3 rounds, medians, plus a
bitwise check of every config against the shipped launch."""

import sys

import torch

sys.path.insert(0, "/tmp")
import prep_lib as P  # noqa: E402

shapes = [(1, 64), (4, 64), (16, 64), (64, 64), (8, 32), (32, 128)]
cases = []
for norm in (False, True):
    for s in shapes:
        cases.append((f"bench {s}{' +norm' if norm else ''}", dict(next(iter(P.bm._input_fn(s, torch.bfloat16, "cuda"))), use_qk_l2norm_in_kernel=norm)))
    for b in (1, 16, 64, 256):
        cases.append((f"decode b={b}{' +norm' if norm else ''}", P.decode_case(b, norm=norm)))

cfgs = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4), (8, 2), (8, 4)]
print(f"{'case':24s} shipped | " + " ".join(f"{str(c):>7s}" for c in cfgs) + " | bitwise")
for name, kw in cases:
    bufs, shipped, variant = P.setup(kw)
    shipped()
    ref = [x.clone() for x in bufs]
    t0 = sorted(P.bench(shipped) for _ in range(3))[1]
    ts, same = [], True
    for r, nw in cfgs:
        fn = variant(r, nw)
        fn()
        same &= all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(ref, bufs))
        ts.append(sorted(P.bench(fn) for _ in range(3))[1])
    print(f"{name:24s} {t0:7.1f} | " + " ".join(f"{t:7.1f}" for t in ts) + f" | {'ok' if same else 'DIFF'}", flush=True)
