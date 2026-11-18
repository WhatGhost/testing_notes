import json
import re
import statistics
from collections import Counter, defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def rows(path):
    res, seen = [], Counter()
    for line in open(path):
        line = ANSI.sub("", line)
        if not line.startswith("SUCCESS"):
            continue
        parts = line.split(None, 4)
        detail = parts[4].strip()
        seen[detail] += 1
        res.append(((detail, seen[detail]), float(parts[2]) * 1e3))
    return res


def fmt(m, p):
    s = m / p
    sp = f"**{s:.2f}×**" if s >= 1.05 else f"{s:.2f}×"
    return f"{m:.1f} µs | {p:.1f} µs | {sp}"


data = defaultdict(lambda: defaultdict(list))
order = []
for tag in ("master", "patched"):
    for r in (1, 2, 3):
        for key, lat in rows(f"/tmp/bench_n2/fused_moe_{tag}_{r}.log"):
            if key not in order:
                order.append(key)
            data[key][tag].append(lat)
out = ["### Official benchmark (64 experts, hidden 2048, inter 1408, top-8)", "",
       "| tokens | master | this PR | speedup |", "|---:|---:|---:|---:|"]
ratios = []
official_T = []
for key in order:
    T = int(re.search(r"torch\.Size\(\[(\d+), 2048\]\)", key[0]).group(1))
    official_T.append(T)
    m = statistics.median(data[key]["master"])
    p = statistics.median(data[key]["patched"])
    ratios.append(m / p)
    out.append(f"| {T} | {fmt(m, p)} |")
out.append("")
print("official geomean %.2f" % statistics.geometric_mean(ratios))

for model, title in (("official", "64 experts, hidden 2048, inter 1408, top-8"),
                     ("35B", "256 experts, hidden 2048, inter 128, top-8")):
    js = json.load(open(f"/tmp/n3_three_way_{model}.json"))
    out += [f"### Additional shapes ({title})", "", "| tokens | master | this PR | speedup |", "|---:|---:|---:|---:|"]
    rs = []
    for r in js:
        if model == "official" and r["T"] in official_T:
            continue
        rs.append(r["master"] / r["new"])
        out.append(f"| {r['T']} | {fmt(r['master'], r['new'])} |")
    out.append("")
    print(model, "range %.2f-%.2f" % (min(rs), max(rs)))
open("/tmp/pr_tables_n3.md", "w").write("\n".join(out))
print("\n".join(out))
