"""Markdown tables for the decode-attention / GDN-decode PR."""

import json
import statistics
import sys

sys.argv = ["x"]
sys.path.insert(0, "/tmp")
import parse_n1 as P  # noqa: E402


def official(op):
    data, order = {}, []
    for tag in ("master", "patched"):
        for r in (1, 2, 3):
            for key, lat in P.rows(f"{P.OUT}/{op}_{tag}_{r}.log"):
                if key not in order:
                    order.append(key)
                data.setdefault(key, {}).setdefault(tag, []).append(lat)
    return [(k[0], statistics.median(data[k]["master"]), statistics.median(data[k]["patched"])) for k in order]


def fx(m, p):
    s = m / p
    return f"**{s:.2f}×**" if s >= 1.05 else f"{s:.2f}×"


def us(t):
    return f"{t:.1f} µs"


out = []
rows = json.load(open("/tmp/n1_strided_table.json"))
out.append("### Strided `q` (a view into a fused qkv projection), bf16, 200K-token KV pool\n")
out.append("| op | batch × seq | q / kv heads | head_dim | master | max error | this PR | max error | speedup |")
out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    me = "nan" if r["master_err"] != r["master_err"] else f"{r['master_err']:.1e}"
    out.append(f"| `{r['op']}` | {r['bs']} × {r['seq']} | {r['hq']} / {r['hkv']} | {r['d']} | {us(r['master_us'])} | {me} | "
               f"{us(r['new_us'])} | {r['new_err']:.1e} | {fx(r['master_us'], r['new_us'])} |")

for op, title in (("decode_attention", "`decode_attention`, official benchmark (contiguous inputs)"),
                  ("decode_grouped_attention", "`decode_grouped_attention`, official benchmark (contiguous inputs)")):
    out.append(f"\n### {title}\n")
    out.append("| batch | q heads | kv heads | head_dim | KV tokens | master | this PR | speedup |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for detail, m, p in official(op):
        sizes = [list(map(int, s.split(","))) for s in __import__("re").findall(r"torch\.Size\(\[([^\]]*)\]\)", detail)]
        q, k = sizes[0], sizes[1]
        out.append(f"| {q[0]} | {q[1]} | {k[1]} | {q[2]} | {k[0]} | {us(m)} | {us(p)} | {fx(m, p)} |")

out.append("\n### `fused_recurrent_gated_delta_rule_packed_decode`, official benchmark (32 value heads, 16 key heads, "
           "K = V = 128)\n")
out.append("| batch | master | this PR | speedup |")
out.append("|---:|---:|---:|---:|")
for (detail, m, p), bs in zip(official("fused_recurrent_gated_delta_rule_packed_decode"), (1, 8, 64, 256, 512)):
    out.append(f"| {bs} | {us(m)} | {us(p)} | {fx(m, p)} |")

sweep = json.load(open("/tmp/n2_gdn_rule.json"))
pick = {(4, 12): (1, 8, 32, 128), (4, 8): (1, 16, 64, 256), (16, 48): (4, 32)}
out.append("\n### `fused_recurrent_gated_delta_rule_packed_decode`, additional shapes (K = V = 128)\n")
out.append("| key heads | value heads | batch | master | this PR | speedup |")
out.append("|---:|---:|---:|---:|---:|---:|")
for r in sweep:
    if r["bs"] not in pick.get((r["h"], r["hv"]), ()):
        continue
    m = r["times"]["(64, 8)"]
    p = r["times"]["(32, 16)" if r["bs"] * r["hv"] >= 128 else "(32, 8)"]
    out.append(f"| {r['h']} | {r['hv']} | {r['bs']} | {us(m)} | {us(p)} | {fx(m, p)} |")

open("/tmp/pr_tables_n1.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
