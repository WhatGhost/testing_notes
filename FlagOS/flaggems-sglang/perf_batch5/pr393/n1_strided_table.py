"""Decode attention with a strided q (view into qkv): master vs this change in
one process, time and max error against a float64 reference."""

import importlib.util
import json
import os
import sys
import types

import torch
from triton.testing import do_bench

N = open("/tmp/fgs-n1.path").read().strip()
M = open("/tmp/fgs-master.path").read().strip()
sys.path.insert(0, os.path.join(N, "src"))
import flaggems_sglang  # noqa: E402,F401


def load(root, name, alias):
    pkg = f"{root}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops"
    if alias not in sys.modules:
        p = types.ModuleType(alias)
        p.__path__ = [pkg]
        sys.modules[alias] = p
    spec = importlib.util.spec_from_file_location(f"{alias}.{name}", f"{pkg}/{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"{alias}.{name}"] = m
    spec.loader.exec_module(m)
    return m


mods = {}
for tag, alias, root in (("master", "m_ops", M), ("new", "n_ops", N)):
    load(root, "_decode_splitkv", alias)
    mods[tag] = {n: getattr(load(root, n, alias), n) for n in ("decode_attention", "decode_grouped_attention")}


def reference(q, k_buf, v_buf, kv_indptr, kv_indices, sm_scale):
    out = []
    for b in range(q.shape[0]):
        idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].long()
        g = q.shape[1] // k_buf.shape[1]
        k = k_buf[idx].double().repeat_interleave(g, dim=1)
        v = v_buf[idx].double().repeat_interleave(g, dim=1)
        s = torch.einsum("hd,lhd->hl", q[b].double(), k) * sm_scale
        out.append(torch.einsum("hl,lhd->hd", torch.softmax(s, -1), v))
    return torch.stack(out)


torch.manual_seed(0)
pool = 200_000
rows = []
for (bs, seq, hq, hkv, d) in ((1, 4096, 16, 4, 128), (8, 2048, 16, 4, 128), (32, 1024, 16, 4, 128),
                              (4, 8192, 6, 1, 256), (16, 2048, 24, 4, 256)):
    k_buf = torch.randn(pool, hkv, d, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.randn(pool, hkv, d, device="cuda", dtype=torch.bfloat16)
    kv_indptr = torch.arange(0, (bs + 1) * seq, seq, device="cuda", dtype=torch.int32)
    kv_indices = torch.randperm(pool, device="cuda")[: bs * seq].to(torch.int32)
    qkv = torch.randn(bs, hq * d + 2 * hkv * d, device="cuda", dtype=torch.bfloat16)
    q = qkv[:, : hq * d].view(bs, hq, d)
    ref = reference(q, k_buf, v_buf, kv_indptr, kv_indices, d ** -0.5)
    for name in ("decode_attention", "decode_grouped_attention"):
        r = dict(op=name, bs=bs, seq=seq, hq=hq, hkv=hkv, d=d)
        for tag in ("master", "new"):
            fn = mods[tag][name]
            out = fn(q, k_buf, v_buf, kv_indptr, kv_indices, d ** -0.5)
            r[f"{tag}_err"] = (out.double() - ref).abs().max().item()
            r[f"{tag}_us"] = min(do_bench(lambda: fn(q, k_buf, v_buf, kv_indptr, kv_indices, d ** -0.5), warmup=5,
                                          rep=50, return_mode="median") for _ in range(2)) * 1e3
        rows.append(r)
        print(r, flush=True)
json.dump(rows, open("/tmp/n1_strided_table.json", "w"), indent=1)
