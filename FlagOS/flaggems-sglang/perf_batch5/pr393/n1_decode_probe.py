"""rdna3 decode attention with a strided q (a view into qkv, as SGLang passes it)
and a large KV pool: correctness of the fallback path and its cost."""

import importlib
import os
import sys

import torch
from triton.testing import do_bench

N = open("/tmp/fgs-n1.path").read().strip()
sys.path.insert(0, os.path.join(N, "src"))
import flaggems_sglang  # noqa: E402,F401

R = "flaggems_sglang.runtime.backend._amd.rdna3.ops."
da = importlib.import_module(R + "decode_attention")
dga = importlib.import_module(R + "decode_grouped_attention")


def reference(q, k_buf, v_buf, kv_indptr, kv_indices, sm_scale):
    out = []
    for b in range(q.shape[0]):
        idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].long()
        k = k_buf[idx].double()  # [L, Hkv, D]
        v = v_buf[idx].double()
        g = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
        s = torch.einsum("hd,lhd->hl", q[b].double(), k) * sm_scale
        p = torch.softmax(s, dim=-1)
        out.append(torch.einsum("hl,lhd->hd", p, v))
    return torch.stack(out)


torch.manual_seed(0)
pool = 200_000
for (bs, seq, hq, hkv, d) in ((1, 4096, 16, 4, 128), (8, 2048, 16, 4, 128), (32, 1024, 16, 4, 128),
                              (4, 8192, 6, 1, 256), (16, 2048, 24, 4, 256)):
    k_buf = torch.randn(pool, hkv, d, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.randn(pool, hkv, d, device="cuda", dtype=torch.bfloat16)
    kv_indptr = torch.arange(0, (bs + 1) * seq, seq, device="cuda", dtype=torch.int32)
    kv_indices = torch.randperm(pool, device="cuda")[: bs * seq].to(torch.int32)
    qkv = torch.randn(bs, hq * d + 2 * hkv * d, device="cuda", dtype=torch.bfloat16)
    q_view = qkv[:, : hq * d].view(bs, hq, d)  # strided: row stride = qkv width
    q_cont = q_view.contiguous()
    scale = d ** -0.5
    ref = reference(q_cont, k_buf, v_buf, kv_indptr, kv_indices, scale)
    for name, mod, fn in (("decode_attention", da, da.decode_attention),
                          ("decode_grouped_attention", dga, dga.decode_grouped_attention)):
        res = {}
        for tag, q in (("strided", q_view), ("contiguous", q_cont)):
            out = fn(q, k_buf, v_buf, kv_indptr, kv_indices, scale)
            err = (out.double() - ref).abs().max().item()
            t = do_bench(lambda: fn(q, k_buf, v_buf, kv_indptr, kv_indices, scale), warmup=5, rep=50,
                         return_mode="median") * 1e3
            res[tag] = (t, err)
        print(f"{name:25s} bs={bs:2d} seq={seq:5d} hq={hq} hkv={hkv} d={d} | strided {res['strided'][0]:9.1f} us "
              f"err {res['strided'][1]:.2e} | contiguous {res['contiguous'][0]:7.1f} us err {res['contiguous'][1]:.2e}",
              flush=True)
