"""Compact canvas data from the same-session runs (/tmp/cmp2_rows.json)."""

import json

rows = json.load(open("/tmp/cmp2_rows.json"))
official = json.load(open("/tmp/official_rows.json"))


def r1(x):
    return None if x is None else round(x, 1)


def M(r):
    return int(r["case"].split("M=")[1].split()[0])


out = {}
moe = sorted((r for r in rows if r["op"].startswith("fused_moe_gemm")), key=M)
out["moe"] = [[M(r), r1(r["ref"]), r1(r["old"]), r1(r["new"]), r1(r["g_ref"]), r1(r["g_old"]), r1(r["g_new"])]
              for r in moe]
fm = sorted((r for r in rows if r["op"] == "fused_moe"), key=M)
out["fused_moe"] = [[M(r), round(r["ref"] / r["new"], 2), round(r["old"] / r["new"], 2)] for r in fm]
out["extend"] = [[r["model"], r["case"], r1(r["ref"]), r1(r["old"]), r1(r["new"])]
                 for r in rows if r["op"] == "extend_attention"]
out["silu"] = [[r["model"], M(r), r1(r["ref"]), r1(r["old"]), r1(r["new"]), r1(r["g_ref"]), r1(r["g_old"]),
                r1(r["g_new"])] for r in rows if r["op"] == "silu_and_mul"]
gemm = []
for r in rows:
    if r["op"] != "tiny_n_gemm":
        continue
    p = r["case"].split()
    gemm.append([r["model"], p[0], int(p[2].split("=")[1]), int(p[3].split("=")[1]), M(r),
                 r1(r["g_ref"]), r1(r["g_old"]), r1(r["g_new"]), r1(r["ref"]), r1(r["old"]), r1(r["new"])])
out["gemm"] = gemm
out["lmhead"] = [[r["model"], M(r), int(r["case"].split()[2].split("=")[1]), r1(r["g_ref"]), r1(r["g_old"]),
                  r1(r["g_new"]), r1(r["ref"]), r1(r["old"]), r1(r["new"])]
                 for r in rows if r["op"].startswith("tiny_n_gemm（")]
out["official"] = [[o["op"], o["shape"], o["master"], o["patched"]] for o in official]
for k, v in out.items():
    print(f"const {k.upper()} = {json.dumps(v, ensure_ascii=False, separators=(',', ':'))};")
