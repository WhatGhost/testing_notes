"""rdna3 decode_attention / decode_grouped_attention with strided inputs:
fp64 reference, and bitwise equality with the same data made contiguous."""

import importlib
import os
import sys

import torch

N = open("/tmp/fgs-n1.path").read().strip()
sys.path.insert(0, os.path.join(N, "src"))
import flaggems_sglang  # noqa: E402,F401

R = "flaggems_sglang.runtime.backend._amd.rdna3.ops."
OPS = {"decode_attention": importlib.import_module(R + "decode_attention").decode_attention,
       "decode_grouped_attention": importlib.import_module(R + "decode_grouped_attention").decode_grouped_attention}
fails = 0
cases = 0


def reference(q, k_buf, v_buf, kv_indptr, kv_indices, sm_scale):
    out = []
    for b in range(q.shape[0]):
        idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].long()
        g = q.shape[1] // k_buf.shape[1]
        if idx.numel() == 0:
            out.append(torch.full((q.shape[1], v_buf.shape[-1]), float("nan"), dtype=torch.float64, device=q.device))
            continue
        k = k_buf[idx].double().repeat_interleave(g, dim=1)
        v = v_buf[idx].double().repeat_interleave(g, dim=1)
        s = torch.einsum("hd,lhd->hl", q[b].double(), k) * sm_scale
        out.append(torch.einsum("hl,lhd->hd", torch.softmax(s, -1), v))
    return torch.stack(out)


def make_q(kind, bs, hq, d, dtype):
    if kind == "contiguous":
        return torch.randn(bs, hq, d, device="cuda", dtype=dtype)
    if kind == "qkv_view":
        qkv = torch.randn(bs, hq * d * 3, device="cuda", dtype=dtype)
        return qkv[:, : hq * d].view(bs, hq, d)
    if kind == "head_major":
        return torch.randn(hq, bs, d, device="cuda", dtype=dtype).transpose(0, 1)
    if kind == "batch_step":
        return torch.randn(2 * bs, hq, d, device="cuda", dtype=dtype)[::2]
    if kind == "dim_strided":
        return torch.randn(bs, hq, d, 2, device="cuda", dtype=dtype)[..., 0]
    raise ValueError(kind)


def make_kv(kind, pool, hkv, d, dv, dtype):
    if kind == "contiguous":
        return (torch.randn(pool, hkv, d, device="cuda", dtype=dtype),
                torch.randn(pool, hkv, dv, device="cuda", dtype=dtype))
    if kind == "fused_pool" and d == dv:
        kv = torch.randn(pool, 2, hkv, d, device="cuda", dtype=dtype)
        return kv[:, 0], kv[:, 1]
    if kind == "wide_rows":
        k = torch.randn(pool, hkv, d + 16, device="cuda", dtype=dtype)[..., :d]
        v = torch.randn(pool, hkv, dv + 16, device="cuda", dtype=dtype)[..., :dv]
        return k, v
    return None


def check(name, q, k, v, indptr, indices, scale, tol):
    global fails, cases
    cases += 1
    fn = OPS[name]
    out = fn(q, k, v, indptr, indices, scale)
    ref = reference(q, k, v, indptr, indices, scale)
    valid = ~torch.isnan(ref)
    err = (out.double()[valid] - ref[valid]).abs().max().item() if valid.any() else 0.0
    bitwise = True
    if q.stride(-1) == 1:  # same code path as the contiguous copy
        same = fn(q.contiguous(), k.contiguous(), v.contiguous(), indptr, indices, scale)
        bitwise = torch.equal(torch.nan_to_num(out, nan=7.0), torch.nan_to_num(same, nan=7.0))
    if not (err <= tol and bitwise):
        fails += 1
        print(f"FAIL {name} q{tuple(q.shape)} strides q{q.stride()} k{k.stride()} {q.dtype} err {err:.2e} "
              f"bitwise {bitwise}", flush=True)


torch.manual_seed(0)
for dtype, tol in ((torch.bfloat16, 3e-2), (torch.float16, 1e-2), (torch.float32, 1e-4)):
    for (hq, hkv, d, dv) in ((16, 4, 128, 128), (32, 8, 64, 64), (6, 1, 256, 256), (24, 4, 80, 80),
                              (16, 1, 128, 128), (8, 8, 128, 128), (16, 2, 512, 512), (32, 2, 128, 128)):
        for seq_lens in ((300,), (64, 1, 0, 513, 200), (1024, 2048, 17)):
            bs = len(seq_lens)
            pool = 4096
            indptr = torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
            indices = torch.randperm(pool, device="cuda")[: sum(seq_lens)].to(torch.int32)
            for kv_kind in ("contiguous", "fused_pool", "wide_rows"):
                kv = make_kv(kv_kind, pool, hkv, d, dv, dtype)
                if kv is None:
                    continue
                for q_kind in ("contiguous", "qkv_view", "head_major", "batch_step", "dim_strided"):
                    q = make_q(q_kind, bs, hq, d, dtype)
                    for name in OPS:
                        check(name, q, kv[0], kv[1], indptr, indices, d ** -0.5, tol)
print("regular cases", cases, "fails", fails, flush=True)

# offsets past 2^31 elements in the KV pool
hkv, d = 8, 128
pool = (1 << 31) // (hkv * d) + 4096
k = torch.empty(pool, hkv, d, device="cuda", dtype=torch.bfloat16)
v = torch.empty(pool, hkv, d, device="cuda", dtype=torch.bfloat16)
indices = torch.arange(pool - 3000, pool, device="cuda", dtype=torch.int32)[torch.randperm(3000, device="cuda")]
k[indices.long()] = torch.randn(3000, hkv, d, device="cuda", dtype=torch.bfloat16)
v[indices.long()] = torch.randn(3000, hkv, d, device="cuda", dtype=torch.bfloat16)
indptr = torch.tensor([0, 1000, 2000, 3000], device="cuda", dtype=torch.int32)
for q_kind in ("contiguous", "qkv_view"):
    q = make_q(q_kind, 3, 32, d, torch.bfloat16)
    for name in OPS:
        cases += 1
        out = OPS[name](q, k, v, indptr, indices, d ** -0.5)
        ref = reference(q, k, v, indptr, indices, d ** -0.5)
        err = (out.double() - ref).abs().max().item()
        if err > 3e-2:
            fails += 1
            print(f"FAIL int64 {name} {q_kind} err {err:.2e}", flush=True)
print("total cases", cases, "fails", fails)
