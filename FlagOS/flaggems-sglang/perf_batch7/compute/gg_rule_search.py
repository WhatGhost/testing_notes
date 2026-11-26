"""Search small grouped_gemm config rules on the grid json: generic when m <=
512 and <= 8 rows per expert, then thresholds on rows per expert and tokens;
score = (worst gain over generic, geometric-mean and worst loss vs best)."""

import itertools
import json
import math

rows = []
for nk in ("512_2048", "1536_2048", "2048_768"):
    rows += json.load(open(f"/tmp/gg_grid_{nk}.json"))
names = sorted(rows[0]["cand"])


def score(rule):
    gains, losses = [], []
    for r in rows:
        c = rule(r)
        t = r["generic"] if c == "generic" else r["cand"][c]
        gains.append(r["generic"] / t)
        losses.append(t / min(min(r["cand"].values()), r["generic"]))
    return min(gains), math.exp(sum(map(math.log, losses)) / len(losses)), max(losses)


best = None
small = ["(16, 32, 128, 2, 3, True)", "(16, 32, 128, 2, 3, False)"]
mid = ["(32, 64, 256, 8, 1, True)"]
big = ["(64, 32, 64, 2, 2, True)", "(64, 128, 64, 4, 2, True)", "(128, 128, 64, 8, 2, True)", "(32, 64, 256, 8, 1, True)",
       "(64, 64, 64, 8, 2, True)"]
for s, mcfg, p1, p2, mt, bk1, bk2 in itertools.product(small, mid, (8, 16, 32), (32, 64, 128, 256), (1024, 2048, 4096),
                                                     big, big):
    if p2 <= p1:
        continue

    def rule(r, s=s, mcfg=mcfg, p1=p1, p2=p2, mt=mt, bk1=bk1, bk2=bk2):
        per = r["m"] / r["e"]
        if r["m"] <= 512 and per <= 8:
            return "generic"
        if per <= p1:
            return s
        if per <= p2 or r["m"] <= mt:
            return mcfg
        return bk1 if r["k"] >= 1536 else bk2

    g, gm, worst = score(rule)
    key = (g >= 0.98, -gm, -worst)
    if best is None or key > best[0]:
        best = (key, dict(small=s, mid=mcfg, p1=p1, p2=p2, mt=mt, big_k_ge_1536=bk1, big_k_lt_1536=bk2), (g, gm, worst))
print("best rule:", best[1])
print("worst gain %.2fx, geomean loss vs best %.3fx, worst loss %.2fx" % best[2])
