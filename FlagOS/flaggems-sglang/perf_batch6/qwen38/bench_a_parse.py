"""Medians over the 3 rounds of the official benchmark logs in /tmp/bench_a:
master vs patched per op and shape, as markdown tables."""

import glob
import os
import re
import statistics
import sys
from collections import OrderedDict

ops = sys.argv[1:] or sorted({re.sub(r"_(master|patched)_\d\.log$", "", p.split("/")[-1]) for p in glob.glob(os.environ.get("BENCH_DIR", "/tmp/bench_a") + "/*.log")})
for op in ops:
    rows = OrderedDict()
    for tag in ("master", "patched"):
        for rnd in (1, 2, 3):
            try:
                lines = open(f"{os.environ.get('BENCH_DIR', '/tmp/bench_a')}/{op}_{tag}_{rnd}.log").read().splitlines()
            except FileNotFoundError:
                continue
            for line in lines:
                parts = line.split()
                if not parts or parts[0] != "SUCCESS":
                    continue
                shape = " ".join(parts[4:])
                rows.setdefault(shape, {"master": [], "patched": []})[tag].append(float(parts[2]) * 1e3)
    print(f"\n### {op}\n\n| shape | master | this PR | speedup |\n|---|---:|---:|---:|")
    for shape, d in rows.items():
        if not d["master"] or not d["patched"]:
            continue
        m, p = statistics.median(d["master"]), statistics.median(d["patched"])
        short = re.sub(r"torch\.Size\(\[([^\]]*)\]\)", r"[\1]", shape)[:90]
        print(f"| {short} | {m:.1f} µs | {p:.1f} µs | {m / p:.2f}× | ({len(d['master'])}/{len(d['patched'])} rounds)")
