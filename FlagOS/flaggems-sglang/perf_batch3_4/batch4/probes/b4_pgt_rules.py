"""Evaluate per_group_transpose launch rules on /tmp/b4_pgt_sweep.json."""

import json
import math

rows = json.load(open("/tmp/b4_pgt_sweep.json"))


def t_of(r, cfg):
    if cfg is None:
        return r["cur"]
    return r["times"].get(str(cfg))


def r1(dt, e, k, avg, total):
    if dt != "bfloat16":
        return None
    if avg >= 512 and k <= 256:
        return (8, 32, 32, 8)
    if avg >= 512 and k >= 2048:
        return (4, 32, 32, 8)
    return None


def r2(dt, e, k, avg, total):
    # 16-bit: split rows when the groups are tall
    if dt != "bfloat16" or avg < 128:
        return None
    if k <= 256:
        return (8, 32, 32, 8) if avg >= 256 else (4, 32, 32, 8)
    if k >= 2048:
        return (4, 32, 32, 8) if avg >= 256 else None
    return None


def r3(dt, e, k, avg, total):
    if dt != "bfloat16" or avg < 128:
        return None
    if k <= 256:
        return (8, 32, 64, 8)
    if k >= 2048:
        return (4, 32, 128, 8)
    return None


def r4(dt, e, k, avg, total):
    if avg < 128:
        return None
    if dt != "bfloat16":
        return (4, 32, 32, 8) if k <= 512 else None
    if k <= 256:
        return (8, 32, 32, 8)
    if k >= 2048:
        return (4, 32, 128, 8) if k * avg >= 1 << 20 or e >= 32 else None
    return None


def r5(dt, e, k, avg, total):
    if avg < 128:
        return None
    if dt != "bfloat16":
        return (4, 32, 32, 8) if k <= 512 else None
    if k <= 256:
        return (8, 32, 32, 8)
    if k >= 2048:
        return (4, 32, 32, 8)
    return None


for name, rule in (("r1", r1), ("r2", r2), ("r3", r3), ("r4", r4), ("r5", r5)):
    ratios = []
    worst = []
    for r in rows:
        avg = sum(r["sizes"]) // r["e"]
        cfg = rule(r["dtype"], r["e"], r["k"], avg, sum(r["sizes"]))
        t = t_of(r, cfg)
        ratio = r["cur"] / t
        ratios.append(ratio)
        worst.append((ratio, r["dtype"][:4], r["e"], r["k"], avg, cfg))
    geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    worst.sort()
    print(f"{name}: geomean {geo:.3f} | worst " + "; ".join(f"{w[0]:.2f} {w[1:]}" for w in worst[:4]) +
          " | best " + "; ".join(f"{w[0]:.2f} {w[1:5]}" for w in worst[-3:]))
