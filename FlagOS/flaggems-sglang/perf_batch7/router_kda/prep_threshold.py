"""Where (2 rows, 2 warps) starts beating (1 row, 4 warps) for the L2-norm
prepare: small decode batches, 5 rounds alternated, medians."""

import sys

sys.path.insert(0, "/tmp")
import prep_lib as P  # noqa: E402

print(" b  rows | (1,4)   (2,2)")
for b in (1, 2, 4, 8, 16, 32):
    kw = P.decode_case(b, norm=True)
    _, _, variant = P.setup(kw)
    a, c = variant(1, 4), variant(2, 2)
    ta, tc = [], []
    for _ in range(5):
        ta.append(P.bench(a))
        tc.append(P.bench(c))
    print(f"{b:2d} {b * 8:5d} | {sorted(ta)[2]:5.1f}  {sorted(tc)[2]:5.1f}", flush=True)
