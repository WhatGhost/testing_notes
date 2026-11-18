import ast
import json
import sys

d = json.load(open("/tmp/tng_eval.json"))
SHAPES = {
    "27B qkv": (3584, 5120), "27B gdn_qkvz": (4096, 5120), "27B gate_up": (8704, 5120),
    "27B down": (5120, 4352), "27B o_proj": (5120, 1536), "27B lm_head": (62080, 5120),
    "35B qkv": (2560, 2048), "35B gdn_qkvz": (3072, 2048), "35B o_proj": (2048, 1024),
    "35B lm_head": (62080, 2048), "35B shared_down": (2048, 128), "35B router": (256, 2048),
    "repo n512": (512, 7168), "repo n1536": (1536, 7168),
}


def cdiv(a, b):
    return -(-a // b)


def warps(bm, bn):
    return {(16, 16): 2, (16, 32): 2, (16, 64): 4, (32, 16): 4, (32, 32): 4,
            (32, 64): 8, (64, 16): 4, (64, 32): 8, (64, 64): 8}[(bm, bn)]


def evaluate(rule, show=False, label=""):
    regrets, vs_cur, missing = [], [], 0
    worse_than_cur = []
    for key, row in d.items():
        name, M = key.split("|")
        M = int(M)
        N, K = SHAPES[name]
        if M == 1 or K <= 256:
            continue
        c = {ast.literal_eval(k): v for k, v in row.items() if k not in ("hip", "cur")}
        cfg = rule(M, N, K)
        if cfg not in c:
            missing += 1
            if show:
                print("   missing", name, M, cfg)
            continue
        t = c[cfg]
        b = min(c.values())
        regrets.append((t / b, name, M, cfg))
        vs_cur.append(row["cur"] / t)
        if t > row["cur"] * 1.02:
            worse_than_cur.append((name, M, cfg, round(t / row["cur"], 3)))
    regrets.sort(reverse=True)
    avg = sum(r[0] for r in regrets) / len(regrets)
    print(f"{label:8s} avg regret {avg:.3f} worst {regrets[0][0]:.2f} "
          f">1.10: {sum(r[0] > 1.10 for r in regrets)} missing {missing} "
          f"worse-than-cur(>2%): {len(worse_than_cur)} "
          f"min speedup vs cur {min(vs_cur):.2f}")
    if show:
        for r in regrets[:10]:
            print("   ", round(r[0], 3), r[1:])
        for w in worse_than_cur:
            print("    slower than cur:", w)


def rule_v1(M, N, K):
    kb = cdiv(K, 256)
    big = N * K >= 1 << 26
    if M <= 16:
        if big:
            return (16, 64, 256, 3, 4) if kb > 8 else (16, 64, 256, 1, 4)
        if N * K < 1 << 23:
            return (16, 16, 256, 1, 2) if kb <= 8 else (16, 32, 256, 3, 2)
        return (16, 64, 256, 1, 4)
    bm = min(64, M)
    if big:
        return (bm, 64, 256, 6, 8) if kb > 8 else (bm, 16, 256, 8, 4)
    if N * K < 1 << 23:
        return (16, 16, 256, 1, 2) if kb <= 8 else (32, 32, 256, 3, 4)
    if M <= 32:
        return (16, 32, 256, 1, 2)
    return (32, 32, 256, 3, 4)


def rule_v2(M, N, K):
    kb = cdiv(K, 256)
    big = N * K >= 1 << 26
    small = N * K < 1 << 23
    if big:
        if M <= 16:
            return (16, 64, 256, 3, 4) if kb > 8 else (16, 64, 256, 1, 4)
        bm = min(64, M)
        return (bm, 64, 256, 6, 8) if kb > 8 else (bm, 16, 256, 8, 4)
    if small and kb <= 8:
        return (16, 16, 256, 1, 2)
    if small:
        return (16, 32, 256, 3, 2) if M <= 16 else (32, 32, 256, 3, 4)
    if M <= 16:
        return (16, 64, 256, 1, 4)
    if M <= 32:
        return (16, 32, 256, 1, 2)
    return (32, 32, 256, 3, 4)


def rule_v3(M, N, K):
    kb = cdiv(K, 256)
    big = N * K >= 1 << 26
    small = N * K < 1 << 23
    if big:
        if M <= 16:
            return (16, 64, 256, 3, 4) if kb > 8 else (16, 64, 256, 1, 4)
        bm = min(64, M)
        return (bm, 64, 256, 6, 8) if kb > 8 else (bm, 16, 256, 8, 4)
    if small and kb <= 8:
        return (16, 16, 256, 1, 2)
    if small:
        return (16, 32, 256, 3, 2) if M <= 16 else (32, 32, 256, 3, 4)
    if M <= 16:
        return (16, 64, 256, 1, 4) if kb <= 20 else (16, 32, 256, 3, 2)
    if M <= 32:
        return (16, 32, 256, 1, 2)
    return (32, 32, 256, 3, 4)


for name, rule in (("v1", rule_v1), ("v2", rule_v2), ("v3", rule_v3)):
    evaluate(rule, show="-v" in sys.argv, label=name)


def rule_v6(M, N, K, PMIN=800):
    kb = cdiv(K, 256)
    wbytes = N * K * 2
    if wbytes >= 128 << 20:
        if M <= 16:
            return (16, 64, 256, 3 if kb > 8 else 1, 4)
        bm = 32 if M <= 32 else 64
        return (bm, 64, 256, 6, 8) if kb > 8 else (bm, 16, 256, 8, 4)
    if wbytes >= 64 << 20 and M > 16:
        return (32, 64, 256, 3, 8) if M <= 32 else (32, 64, 256, 6, 8)
    if M <= 16 or wbytes <= 12 << 20:
        ks = 1 if kb <= 20 else 3
        split = cdiv(kb, ks)
        for bn in (64, 32, 16):
            if cdiv(N, bn) * split >= PMIN:
                break
        return (16, bn, 256, ks, warps(16, bn))
    if M <= 32:
        return (16, 32, 256, 1, 2)
    return (32, 32, 256, 3, 4)


print("--- v6")
evaluate(rule_v6, show=True, label="v6")
for p in (400, 600, 1000, 1200):
    evaluate(lambda M, N, K, p=p: rule_v6(M, N, K, PMIN=p), label=f"v6 p{p}")


def rule_v7(M, N, K, WMIN=800):
    kb = cdiv(K, 256)
    wbytes = N * K * 2
    if wbytes >= 128 << 20:
        if M <= 16:
            return (16, 64, 256, 3 if kb > 8 else 1, 4)
        bm = 32 if M <= 32 else 64
        return (bm, 64, 256, 6, 8) if kb > 8 else (bm, 16, 256, 8, 4)
    if wbytes >= 64 << 20 and M > 16:
        return (32, 64, 256, 3, 8) if M <= 32 else (32, 64, 256, 6, 8)
    if M <= 16 or wbytes <= 12 << 20:
        ks = 1 if kb <= 8 else 3
        for bn in (64, 32, 16):
            if cdiv(N, bn) * kb >= WMIN:
                break
        return (16, bn, 256, ks, warps(16, bn))
    if M <= 32:
        return (16, 64, 256, 3, 4)
    return (32, 32, 256, 3, 4)


print("--- v7")
evaluate(rule_v7, show=True, label="v7")
