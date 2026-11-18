import importlib, os, sys, torch
N = open("/tmp/fgs-n1.path").read().strip()
sys.path.insert(0, os.path.join(N, "src"))
import flaggems_sglang
g_da = importlib.import_module("flaggems_sglang.ops.decode_attention")
g_dga = importlib.import_module("flaggems_sglang.ops.decode_grouped_attention")

def reference(q, k_buf, v_buf, kv_indptr, kv_indices, sm_scale):
    out = []
    for b in range(q.shape[0]):
        idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].long()
        k = k_buf[idx].double(); v = v_buf[idx].double()
        g = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(g, dim=1); v = v.repeat_interleave(g, dim=1)
        s = torch.einsum("hd,lhd->hl", q[b].double(), k) * sm_scale
        out.append(torch.einsum("hl,lhd->hd", torch.softmax(s, -1), v))
    return torch.stack(out)

torch.manual_seed(0)
pool, bs, seq, hq, hkv, d = 20000, 8, 512, 16, 4, 128
k_buf = torch.randn(pool, hkv, d, device="cuda", dtype=torch.bfloat16)
v_buf = torch.randn(pool, hkv, d, device="cuda", dtype=torch.bfloat16)
kv_indptr = torch.arange(0, (bs + 1) * seq, seq, device="cuda", dtype=torch.int32)
kv_indices = torch.randperm(pool, device="cuda")[: bs * seq].to(torch.int32)
qkv = torch.randn(bs, hq * d + 2 * hkv * d, device="cuda", dtype=torch.bfloat16)
q_view = qkv[:, : hq * d].view(bs, hq, d)
ref = reference(q_view.contiguous(), k_buf, v_buf, kv_indptr, kv_indices, d ** -0.5)
for name, fn in (("generic decode_attention", g_da.decode_attention), ("generic decode_grouped_attention", g_dga.decode_grouped_attention)):
    for tag, q in (("strided", q_view), ("contiguous", q_view.contiguous())):
        out = fn(q, k_buf, v_buf, kv_indptr, kv_indices, d ** -0.5)
        print(f"{name:34s} {tag:10s} max err {(out.double() - ref).abs().max().item():.2e}")
