import re
import statistics
import sys
from collections import Counter, defaultdict

OUT = "/tmp/bench_n1"
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


def short(detail):
    sizes = re.findall(r"torch\.Size\(\[([^\]]*)\]\)", detail)
    tail = detail.rsplit(",", 1)[-1].strip(" ]")
    return " ".join("[" + s.replace(" ", "") + "]" for s in sizes[:3]) + (
        f" {tail}" if not tail.startswith("torch.Size") else ""
    )


for op in sys.argv[1:]:
    data = defaultdict(lambda: defaultdict(list))
    order = []
    for tag in ("master", "patched"):
        for r in (1, 2, 3):
            for key, lat in rows(f"{OUT}/{op}_{tag}_{r}.log"):
                if key not in order:
                    order.append(key)
                data[key][tag].append(lat)
    ratios = []
    print(f"\n### {op}\n| # | shape | master µs | patched µs | speedup |\n|---|---|---:|---:|---:|")
    for i, key in enumerate(order):
        m = statistics.median(data[key]["master"])
        p = statistics.median(data[key]["patched"])
        ratios.append(m / p)
        print(f"| {i} | {short(key[0])} (#{key[1]}) | {m:.1f} | {p:.1f} | {m / p:.2f}× |")
    g = statistics.geometric_mean(ratios)
    print(f"geomean {g:.3f}  min {min(ratios):.2f}  max {max(ratios):.2f}  n={len(ratios)}")
