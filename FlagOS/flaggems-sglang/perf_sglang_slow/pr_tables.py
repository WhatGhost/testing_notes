"""Markdown perf tables for the PR: this PR vs upstream master only."""

import json
import math

rows = json.load(open("/tmp/cmp2_rows.json"))
official = json.load(open("/tmp/official_rows.json"))


def M(r):
    return int(r["case"].split("M=")[1].split()[0])


def us(x):
    return f"{x:.0f} µs" if x >= 100 else f"{x:.1f} µs"


def x(a, b, bold=True):
    if a is None or b is None:
        return "—"
    v = a / b
    s = f"{v:.2f}×"
    return f"**{s}**" if bold and v >= 1.05 else s


out = []
# MoE experts
moe = sorted((r for r in rows if r["op"].startswith("fused_moe_gemm")), key=M)
out.append("### MoE experts: `fused_moe_gemm` (up) → `silu_and_mul` → `fused_moe_gemm` (down)")
out.append("E=256, top-8, H=2048, inter=128 (Qwen3.6-35B-A3B per GPU).\n")
out.append("| M | master | this PR | speedup (eager) | speedup (graph) |")
out.append("|---:|---:|---:|---:|---:|")
for r in moe:
    out.append(f"| {M(r)} | {us(r['old'])} | {us(r['new'])} | {x(r['old'], r['new'])} | {x(r['g_old'], r['g_new'])} |")

# extend
out.append("\n### `extend_attention`, head_dim=256, bf16")
out.append("Per GPU: Qwen3.6-27B has 6 query / 1 KV heads, Qwen3.6-35B-A3B 4 / 1. Eager timing.\n")
out.append("| model | batch | extend | prefix | master | this PR | speedup |")
out.append("|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    if r["op"] != "extend_attention":
        continue
    p = dict(kv.split("=") for kv in r["case"].split())
    model = "27B" if r["model"] == "27B" else "35B-A3B"
    out.append(f"| {model} | {p['bs']} | {p['ext']} | {p['prefix']} | {us(r['old'])} | {us(r['new'])} | {x(r['old'], r['new'])} |")

# tiny_n_gemm matrix
LAYERS = [("27B", "qkv"), ("27B", "o_proj"), ("27B", "gdn_qkvz"), ("27B", "gdn_ba"), ("27B", "gate_up"),
          ("27B", "down"), ("35B-A3B", "qkv"), ("35B-A3B", "o_proj"), ("35B-A3B", "gdn_qkvz"),
          ("35B-A3B", "gdn_ba"), ("35B-A3B", "router"), ("35B-A3B", "shared_down"),
          ("35B-A3B", "shared_expert_gate")]
LABEL = {"router": "router / shared gate_up", "shared_expert_gate": "shared expert gate"}
MS = [2, 4, 8, 16, 32, 64]
out.append("\n### `tiny_n_gemm`, decode GEMMs")
out.append("Each cell: speedup with CUDA graph / eager timing. M=1 keeps the unchanged GEMV path.\n")
out.append("| model | layer | N × K | " + " | ".join(f"M={m}" for m in MS) + " |")
out.append("|---|---|---:|" + "---:|" * len(MS))
for model, layer in LAYERS:
    cells = []
    nk = ""
    for m in MS:
        r = next(r for r in rows if r["op"] == "tiny_n_gemm" and r["model"] == model
                 and r["case"].split()[0] == layer and M(r) == m)
        p = r["case"].split()
        nk = f"{p[2].split('=')[1]} × {p[3].split('=')[1]}"
        cells.append(f"{r['g_old'] / r['g_new']:.2f} / {r['old'] / r['new']:.2f}")
    out.append(f"| {'27B' if model == '27B' else '35B-A3B'} | {LABEL.get(layer, layer)} | {nk} | " + " | ".join(cells) + " |")

# LM head
out.append("\n### `tiny_n_gemm`, LM head (N=62080)\n")
out.append("| model | K | M | master (graph) | this PR (graph) | speedup (graph) | speedup (eager) |")
out.append("|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    if not r["op"].startswith("tiny_n_gemm（"):
        continue
    K = r["case"].split()[2].split("=")[1]
    out.append(f"| {'27B' if r['model'] == '27B' else '35B-A3B'} | {K} | {M(r)} | {us(r['g_old'])} | {us(r['g_new'])} | {x(r['g_old'], r['g_new'])} | {x(r['old'], r['new'])} |")

# silu
out.append("\n### `silu_and_mul`, bf16\n")
out.append("| input | M | master | this PR | speedup (eager) | speedup (graph) |")
out.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    if r["op"] != "silu_and_mul":
        continue
    shape = "27B MLP, d=4352" if r["model"].startswith("27B") else "35B-A3B experts, d=128"
    out.append(f"| {shape} | {M(r)} | {us(r['old'])} | {us(r['new'])} | {x(r['old'], r['new'])} | {x(r['g_old'], r['g_new'])} |")

# official
out.append("\n### Official benchmark (`benchmark/`, comprehensive, `--mode kernel`, bf16)\n")
out.append("| op | shapes | geomean | range |")
out.append("|---|---:|---:|---:|")
for op in ("silu_and_mul", "tiny_n_gemm", "fused_moe_gemm", "extend_attention"):
    s = [o["master"] / o["patched"] for o in official if o["op"] == op]
    g = math.exp(sum(math.log(v) for v in s) / len(s))
    out.append(f"| `{op}` | {len(s)} | {g:.3f}× | {min(s):.2f}–{max(s):.2f}× |")
print("\n".join(out))
