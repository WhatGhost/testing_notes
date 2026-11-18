"""Official benchmark harness, master vs perf tree, median of 3 rounds ->
/tmp/official_rows.json with readable shape labels."""

import json
import re
import statistics
from collections import Counter, defaultdict

OUT = "/tmp/bench_perf"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
OPS = ["silu_and_mul", "tiny_n_gemm", "fused_moe_gemm", "extend_attention"]


def rows(path):
    res, seen, dtype = [], Counter(), "?"
    for line in open(path):
        line = ANSI.sub("", line)
        m = re.search(r"dtype=torch\.(\w+)", line)
        if m:
            dtype = m.group(1)
        if not line.startswith("SUCCESS"):
            continue
        detail = line.split(None, 4)[4].strip()
        seen[(dtype, detail)] += 1
        res.append(((dtype, detail, seen[(dtype, detail)]), float(line.split()[2]) * 1e3))
    return res


def dims(detail):
    return [[int(v) for v in s.split(",") if v.strip()]
            for s in re.findall(r"torch\.Size\(\[([^\]]*)\]\)", detail)]


def label(op, detail):
    d = dims(detail)
    tail = detail.rstrip("]").rsplit(",", 1)[-1].strip()
    if op == "silu_and_mul":
        return f"x [{d[0][0]}, {d[0][1]}]"
    if op == "tiny_n_gemm":
        return f"M={d[0][0]} N={d[1][0]} K={d[0][1]}"
    if op == "fused_moe_gemm":
        return f"T={d[0][0]} E={d[1][0]} N={d[1][1]} K={d[1][2]} top_k={tail}"
    if op == "extend_attention":
        q, kb, qo, kvi = d[0], d[3], d[5], d[7]
        return (f"batch={qo[0] - 1} heads={q[1]}/{d[1][1]} D={q[2]} "
                f"extend tokens={q[0]} prefix tokens={kvi[0]} max_len={tail}")
    return detail


result = []
for op in OPS:
    data = defaultdict(lambda: defaultdict(list))
    order = []
    for tag in ("master", "patched"):
        for r in (1, 2, 3):
            for key, lat in rows(f"{OUT}/{op}_{tag}_{r}.log"):
                if key not in order:
                    order.append(key)
                data[key][tag].append(lat)
    for key in order:
        m = statistics.median(data[key]["master"])
        p = statistics.median(data[key]["patched"])
        result.append(dict(op=op, dtype=key[0], shape=label(op, key[1]),
                           master=round(m, 2), patched=round(p, 2)))
json.dump(result, open("/tmp/official_rows.json", "w"), indent=1)
for r in result:
    print(f"{r['op']:18s} {r['dtype']:9s} {r['shape']:70s} {r['master']:9.1f} {r['patched']:9.1f} {r['master'] / r['patched']:.2f}x")
