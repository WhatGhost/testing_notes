"""Same-session comparison: master vs patched, 3 alternating rounds each, with
the user's SGLang bench scripts. Medians per shape -> /tmp/cmp2_rows.json and
compact canvas data on stdout."""

import glob
import json
import re
import statistics
from collections import defaultdict

runs = defaultdict(lambda: defaultdict(list))  # key -> tree -> [row]
for path in sorted(glob.glob("/tmp/cmp/*.json")):
    m = re.match(r".*/(\w+)_(master|patched)_(\d)\.json$", path)
    if not m:
        continue
    for r in json.load(open(path))["rows"]:
        if "error" in r:
            print("ERROR", path, r.get("op"), r.get("case"), r["error"][-200:])
            continue
        runs[(r["op"], r["model"], r["case"])][m.group(2)].append(r)


def med(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return statistics.median(vals) if vals else None


out = []
spread = []
for key, trees in runs.items():
    allr = trees["master"] + trees["patched"]
    ref = med(allr, "t_ref_us")
    g_ref = med(allr, "g_ref_us")
    refs = [r["t_ref_us"] for r in allr]
    spread.append(max(refs) / min(refs))
    out.append(dict(
        op=key[0], model=key[1], case=key[2], n_master=len(trees["master"]),
        n_patched=len(trees["patched"]), ref=ref, g_ref=g_ref,
        old=med(trees["master"], "t_new_us"), new=med(trees["patched"], "t_new_us"),
        g_old=med(trees["master"], "g_new_us"), g_new=med(trees["patched"], "g_new_us"),
        err_old=max((r["max_abs"] for r in trees["master"]), default=None),
        err_new=max((r["max_abs"] for r in trees["patched"]), default=None),
    ))
json.dump(out, open("/tmp/cmp2_rows.json", "w"), ensure_ascii=False, indent=1)
spread.sort()
print(f"{len(out)} shapes; SGLang reference max/min over 6 runs: median {statistics.median(spread):.3f}, "
      f"p90 {spread[len(spread) * 9 // 10]:.3f}, max {spread[-1]:.3f}")
print("incomplete:", [(r["op"], r["case"], r["n_master"], r["n_patched"]) for r in out
                      if r["n_master"] != 3 or r["n_patched"] != 3])
