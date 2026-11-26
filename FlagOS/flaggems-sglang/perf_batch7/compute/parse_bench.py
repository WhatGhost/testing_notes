"""Median kernel latency per shape from run_bench.sh logs, master vs branch.
Usage: python parse_bench.py <outdir> <op>..."""

import glob
import re
import statistics
import sys

outdir, ops = sys.argv[1], sys.argv[2:]
line_re = re.compile(
    r"^SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\[.*\])\s*$")
for op in ops:
    lat = {}
    for side in ("master", "branch"):
        for path in sorted(glob.glob(f"{outdir}/{op}_{side}_r*.log")):
            for line in open(path):
                m = line_re.match(line.strip())
                if m:
                    lat.setdefault(m.group(4), {}).setdefault(side, []).append(
                        float(m.group(2)) * 1000)
    print(f"== {op}")
    ratios = []
    for shape, sides in lat.items():
        if not ("master" in sides and "branch" in sides):
            print(f"  {shape}: missing {set(('master', 'branch')) - set(sides)}")
            continue
        m, b = statistics.median(sides["master"]), statistics.median(sides["branch"])
        ratios.append(m / b)
        runs = f"master {[round(v, 2) for v in sides['master']]} branch " \
               f"{[round(v, 2) for v in sides['branch']]}"
        print(f"  {shape:<70} master {m:9.2f} us  branch {b:9.2f} us  "
              f"{m / b:5.2f}x   ({runs})")
    if ratios:
        print(f"  geomean speedup {statistics.geometric_mean(ratios):.3f}x, "
              f"range {min(ratios):.2f}-{max(ratios):.2f}x")
