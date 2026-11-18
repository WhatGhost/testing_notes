"""Markdown tables for the batch-4 PR from /tmp/bench_b4 logs and b4_extra JSON."""

import json
import statistics
import sys

sys.argv = ["x"]
sys.path.insert(0, "/tmp")
import parse_b4 as P  # noqa: E402


def official(op):
    data, order = {}, []
    for tag in ("master", "patched"):
        for r in (1, 2, 3):
            for key, lat in P.rows(f"{P.OUT}/{op}_{tag}_{r}.log"):
                if key not in order:
                    order.append(key)
                data.setdefault(key, {}).setdefault(tag, []).append(lat)
    return [(statistics.median(data[k]["master"]), statistics.median(data[k]["patched"])) for k in order]


def fmt(t):
    return f"{t:.1f} µs"


def sp(m, p):
    s = m / p
    return f"**{s:.2f}×**" if s >= 1.05 else f"{s:.2f}×"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" for _ in header) + "|"]
    for cells, (m, p) in rows:
        lines.append("| " + " | ".join(str(c) for c in cells) + f" | {fmt(m)} | {fmt(p)} | {sp(m, p)} |")
    return "\n".join(lines)


extra = {}
for f in ("fused_softmax_pool_per_group_transpose", "ltx2_split_rotary_apply_token_bitmask", "dsv3_fused_a_gemm",
          "embedding_lora_a"):
    for row in json.load(open(f"/tmp/b4_extra_{f}.json")):
        extra.setdefault(row["op"], []).append(row)
extra["apply_token_bitmask"] = json.load(open("/tmp/b4_extra_apply_token_bitmask.json"))
extra["per_group_transpose"] = json.load(open("/tmp/b4_extra_per_group_transpose.json"))
extra["ltx2_split_rotary"] = json.load(open("/tmp/b4_extra_ltx2_split_rotary.json"))

out = []
# fused_softmax_pool
o = official("fused_softmax_pool")
shapes = [(bs, k) for bs in (1, 64, 1024) for k in (4, 16, 64)] + [(4096, k) for k in (4, 16, 64)]
out.append("### `fused_softmax_pool`, official benchmark (head_dim 128)\n")
out.append(table(["rows", "k", "master", "this PR", "speedup"], [((bs, k), t) for (bs, k), t in zip(shapes, o)]))
out.append("\n### `fused_softmax_pool`, additional shapes (head_dim 128)\n")
rows = []
for r in extra["fused_softmax_pool"]:
    bs, k = r["shape"].split()[0][3:], r["shape"].split()[1][2:]
    rows.append(((bs, k), (r["master"], r["new"])))
out.append(table(["rows", "k", "master", "this PR", "speedup"], rows))

# per_group_transpose
o = official("per_group_transpose")
shapes = [(k, n) for k in (128, 512, 4096) for n in (16, 128, 1024)]
out.append("\n### `per_group_transpose`, official benchmark (8 groups, bf16)\n")
out.append(table(["k", "rows per group", "master", "this PR", "speedup"], [((k, n), t) for (k, n), t in zip(shapes, o)]))
out.append("\n### `per_group_transpose`, additional shapes\n")
rows = []
for r in extra["per_group_transpose"]:
    parts = r["shape"].split()
    dt, e, k = parts[0], parts[1][2:], parts[2][2:]
    rows_g = parts[3].split("~")[1] + (" (uneven)" if "uneven" in r["shape"] else "")
    rows.append(((dt, e, k, rows_g), (r["master"], r["new"])))
out.append(table(["dtype", "groups", "k", "rows per group", "master", "this PR", "speedup"], rows))

# ltx2_split_rotary
o = official("ltx2_split_rotary")
out.append("\n### `ltx2_split_rotary`, official benchmark (batch 1, 16 heads, head_dim 128)\n")
out.append(table(["tokens", "master", "this PR", "speedup"], [((t,), x) for t, x in zip((1, 64, 1024, 8192), o)]))
out.append("\n### `ltx2_split_rotary`, additional shapes\n")
rows = []
for r in extra["ltx2_split_rotary"]:
    d = dict(kv.split("=") for kv in r["shape"].split())
    rows.append(((d["b"], d["t"], d["heads"], d["head_dim"]), (r["master"], r["new"])))
out.append(table(["batch", "tokens", "heads", "head_dim", "master", "this PR", "speedup"], rows))

# dsv3_fused_a_gemm
o = official("dsv3_fused_a_gemm")
out.append("\n### `dsv3_fused_a_gemm`, official benchmark (K 7168, N 2112)\n")
out.append(table(["M", "master", "this PR", "speedup"], [((m,), x) for m, x in zip((1, 2, 4, 8, 16), o)]))
out.append("\n### `dsv3_fused_a_gemm`, additional shapes\n")
rows = []
for r in extra["dsv3_fused_a_gemm"]:
    d = dict(kv.split("=") for kv in r["shape"].split())
    rows.append(((d["M"], d["K"], d["N"]), (r["master"], r["new"])))
out.append(table(["M", "K", "N", "master", "this PR", "speedup"], rows))

# embedding_lora_a
o = official("embedding_lora_a")
shapes = [("8 × 512", 4, 32, 32000), ("4 × 2048", 2, 64, 128256), ("1 × 5", 1, 16, 128), ("4 segments, 22 tokens", 2, 32, 512),
          ("2 segments, 13 tokens, 8 extra", 2, 16, 256)]
out.append("\n### `embedding_lora_a`, official benchmark\n")
out.append(table(["segments × length", "adapters", "rank", "vocab", "master", "this PR", "speedup"],
                 list(zip(shapes, o))))
out.append("\n### `embedding_lora_a`, additional shapes\n")
rows = []
for r in extra["embedding_lora_a"]:
    d = dict(kv.split("=") for kv in r["shape"].split())
    segs, length = d["tokens"].split("x")
    rows.append(((f"{segs} × {length}", d["adapters"], d["rank"], d["vocab"]), (r["master"], r["new"])))
out.append(table(["segments × length", "adapters", "rank", "vocab", "master", "this PR", "speedup"], rows))

# apply_token_bitmask
o = official("apply_token_bitmask")
shapes = [(1, 152064), (8, 152064), (64, 152064), (512, 152064), (4096, 152064), (3, 100), (7, 1000), (1, 32000)]
out.append("\n### `apply_token_bitmask`, official benchmark (fp32 logits)\n")
out.append(table(["batch", "vocab", "master", "this PR", "speedup"], list(zip(shapes, o))))
out.append("\n### `apply_token_bitmask`, additional shapes (fp32 logits)\n")
rows = []
for r in extra["apply_token_bitmask"]:
    d = dict(kv.split("=") for kv in r["shape"].split()[:2])
    rows.append(((d["B"], d["V"]), (r["master"], r["new"])))
out.append(table(["batch", "vocab", "master", "this PR", "speedup"], rows))

open("/tmp/pr_tables_b4.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
