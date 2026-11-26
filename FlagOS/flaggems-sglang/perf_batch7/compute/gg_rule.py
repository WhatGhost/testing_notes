"""Fit a grouped_gemm config rule on the grid json: for candidate rules, the
speedup over the generic op and the loss against each shape's best config."""

import json

rows = []
for nk in ("512_2048", "1536_2048", "2048_768"):
    rows += json.load(open(f"/tmp/gg_grid_{nk}.json"))

A = "(16, 32, 128, 2, 3, True)"
A0 = "(16, 32, 128, 2, 3, False)"
B = "(32, 64, 256, 8, 1, True)"
C = "(64, 32, 64, 2, 2, True)"
D = "(64, 128, 64, 4, 2, True)"


def rule(r):
    per = r["m"] / r["e"]
    if r["m"] <= 512 and per <= 8:
        return "generic"
    if per <= 16:
        return A
    if per <= 64:
        return B
    return D if r["k"] < 1536 else C


worst_gain, worst_loss = 9.0, 0.0
for r in rows:
    choice = rule(r)
    t = r["generic"] if choice == "generic" else r["cand"][choice]
    best = min(r["cand"].values())
    gain = r["generic"] / t
    loss = t / min(best, r["generic"])
    worst_gain, worst_loss = min(worst_gain, gain), max(worst_loss, loss)
    print(f"m={r['m']:5d} E={r['e']:3d} N={r['n']:4d} K={r['k']:4d} per={r['m'] / r['e']:7.1f} -> {choice:28s} "
          f"{gain:4.2f}x vs generic, {loss:4.2f}x of the best")
print(f"worst gain {worst_gain:.2f}x, worst loss vs best {worst_loss:.2f}x")
