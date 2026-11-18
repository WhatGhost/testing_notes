"""Per fixed config: min / geomean ratio vs current over case buckets."""

import json
import math

rows = json.load(open("/tmp/b4_pgt_sweep.json"))
buckets = {
    "bf16 k<=256 avg>=128": lambda r, avg: r["dtype"] == "bfloat16" and r["k"] <= 256 and avg >= 128,
    "bf16 k=512 avg>=128": lambda r, avg: r["dtype"] == "bfloat16" and r["k"] == 512 and avg >= 128,
    "bf16 k=1024 avg>=128": lambda r, avg: r["dtype"] == "bfloat16" and r["k"] == 1024 and avg >= 128,
    "bf16 k>=2048 avg>=128": lambda r, avg: r["dtype"] == "bfloat16" and r["k"] >= 2048 and avg >= 128,
    "fp32 k<=512 avg>=128": lambda r, avg: r["dtype"] == "float32" and r["k"] <= 512 and avg >= 128,
    "fp32 k>=4096 avg>=128": lambda r, avg: r["dtype"] == "float32" and r["k"] >= 4096 and avg >= 128,
}
for bname, pred in buckets.items():
    sel = [r for r in rows if pred(r, sum(r["sizes"]) // r["e"])]
    cfgs = sel[0]["times"].keys()
    stats = []
    for c in cfgs:
        rs = [r["cur"] / r["times"][c] for r in sel if c in r["times"]]
        if len(rs) != len(sel):
            continue
        geo = math.exp(sum(math.log(x) for x in rs) / len(rs))
        stats.append((min(rs), geo, c))
    stats.sort(reverse=True)
    print(f"== {bname} ({len(sel)} cases)")
    for mn, geo, c in stats[:5]:
        print(f"   {c:20s} min {mn:.2f} geo {geo:.2f}")
