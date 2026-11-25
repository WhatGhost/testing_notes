"""Inspect cached rdna3/generic KDA _recurrent builds: RV, warps, T, and the
instruction mix (LDS, barriers, cross-lane ops, fp32 ALU) in the asm."""

import glob
import json
import os
import re
import sys
from collections import Counter

rows = []
for asm in glob.glob(os.path.expanduser("~/.triton/cache/*/_recurrent.amdgcn")):
    d = os.path.dirname(asm)
    ttgir = open(os.path.join(d, "_recurrent.ttgir")).read()
    meta = json.load(open(os.path.join(d, "_recurrent.json")))
    m = re.search(r"tensor<(\d+)x(\d+)xf32", ttgir)
    rv, bk = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    text = open(asm).read()
    body = [ln.strip() for ln in text.splitlines() if ln.startswith("\t") and not ln.strip().startswith((".", ";"))]
    ops = Counter(ln.split()[0] for ln in body if ln)
    ds = sum(v for k, v in ops.items() if k.startswith("ds_"))
    dpp = sum(1 for ln in body if "row_" in ln or "quad_perm" in ln or "dpp" in ln)
    perm = sum(v for k, v in ops.items() if "permlane" in k or "bpermute" in k or "swizzle" in k)
    f32 = sum(v for k, v in ops.items() if k.startswith(("v_add_f32", "v_mul_f32", "v_fma_f32", "v_sub_f32", "v_dual_")))
    cvt = len(re.findall(r"ttg\.convert_layout", ttgir))
    lds = meta.get("shared", meta.get("shared_mem"))
    rows.append((rv, meta.get("num_warps"), len(body), ds, ops.get("s_barrier", 0), dpp, perm, f32, cvt, lds, os.path.basename(d)[:8], int(os.path.getmtime(asm))))

rows.sort(key=lambda r: (r[0] or 0, r[1] or 0, -r[-1]))
print("RV warps  instrs   ds_*  s_barrier  dpp  perm/swz  f32-alu  convert_layout(ttgir)  LDS bytes  dir")
for r in rows:
    print(f"{str(r[0]):>2} {r[1]:>5} {r[2]:>7} {r[3]:>6} {r[4]:>10} {r[5]:>4} {r[6]:>9} {r[7]:>8} {r[8]:>22} {str(r[9]):>10}  {r[10]}")

if len(sys.argv) > 1:
    d = glob.glob(os.path.expanduser(f"~/.triton/cache/{sys.argv[1]}*"))[0]
    tt = open(os.path.join(d, "_recurrent.ttgir")).read()
    print("\n--- ttgir ops inside the token loop (first scf.for) ---")
    start = tt.find("scf.for")
    seg = tt[start:start + 60000]
    end = seg.find("scf.yield")
    for ln in seg[:end].splitlines():
        s = ln.strip()
        if any(x in s for x in ("convert_layout", "tt.reshape", "tt.reduce", "\"tt.reduce\"", "ttg.local", "tt.split", "tt.join", "tt.trans")):
            print("  " + s[:200])
