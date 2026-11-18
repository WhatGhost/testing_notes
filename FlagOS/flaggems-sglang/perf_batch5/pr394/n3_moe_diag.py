"""Where the rdna3 fused_moe error comes from: c1 (activation), c3 (weighted
down rows) and the output against float64."""

import importlib
import os
import sys

import torch
import triton

W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401

r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
g = importlib.import_module("flaggems_sglang.ops.fused_moe")

T, H, I, E, topk, rsf = 300, 2048, 1408, 64, 8, 2.5
gen = torch.Generator(device="cuda").manual_seed(T + E)
x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
ids = ti.to(torch.int32).reshape(-1).contiguous()
wts = tw.float().reshape(-1).contiguous()
P = T * topk

# float64 reference intermediates, per pair
tok = torch.arange(P, device="cuda") // topk
hid = torch.empty((P, I), dtype=torch.float64, device="cuda")
down = torch.empty((P, H), dtype=torch.float64, device="cuda")
for e in range(E):
    sel = (ids == e).nonzero().flatten()
    if sel.numel() == 0:
        continue
    gu = x.double()[tok[sel]] @ w1[e].double().t()
    h = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
    hid[sel] = h
    down[sel] = (h @ w2[e].double().t()) * wts.double()[sel][:, None]
out_ref = down.view(T, topk, H).sum(1) * rsf

print("reference done", flush=True)
for split_a in (False, True):
    block_m, up_cfg, down_cfg = r._grouped_config(P, E, H, I)
    sorted_ids, block_expert, num_blocks = r._moe_align(ids, E, block_m); torch.cuda.synchronize(); print("align", flush=True)
    c1 = torch.empty((num_blocks * block_m, I), dtype=torch.float32 if split_a else x.dtype, device="cuda")
    r._grouped_gate_up_kernel[(num_blocks, triton.cdiv(I, up_cfg[0]))](
        x, w1, sorted_ids, block_expert, c1, P, E, H, I, topk, x.stride(0), w1.stride(0), w1.stride(1),
        BLOCK_M=block_m, BLOCK_N=up_cfg[0], BLOCK_K=up_cfg[1], ACT=0, SPLIT_STORE=not split_a, num_warps=up_cfg[2],
        num_stages=up_cfg[3])
    c3 = torch.empty((P, H), dtype=torch.float32, device="cuda")
    r._grouped_down_kernel[(num_blocks, triton.cdiv(H, down_cfg[0]))](
        c1, w2, sorted_ids, block_expert, wts, c3, P, E, H, I, w2.stride(0), w2.stride(1), BLOCK_M=block_m,
        BLOCK_N=down_cfg[0], BLOCK_K=down_cfg[1], SPLIT_A=split_a, SPLIT_STORE=False, num_warps=down_cfg[2],
        num_stages=down_cfg[3])
    in_expert = (block_expert < E).repeat_interleave(block_m)
    valid = in_expert & (sorted_ids >= 0) & (sorted_ids < P)
    rows = sorted_ids[valid].long()
    c1_pairs = torch.empty((P, I), dtype=torch.float64, device="cuda")
    c1_pairs[rows] = c1[valid].double()
    e1 = (c1_pairs - hid).abs()
    e3 = (c3.double() - down).abs()
    out = (c3.double().view(T, topk, H).sum(1) * rsf)
    print(f"split_a={split_a} | c1 max abs err {e1.max().item():.3e} (|hid| max {hid.abs().max().item():.2f}) | "
          f"c3 (fp32 store) max abs err {e3.max().item():.3e} (|down| max {down.abs().max().item():.2f}) | "
          f"out before final rounding max err {(out - out_ref).abs().max().item():.3e}", flush=True)
gen_out = g.fused_moe(x, w1, w2, (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None), "silu", rsf)
print(f"generic out (bf16) max err {(gen_out.double() - out_ref).abs().max().item():.3e}; "
      f"exact rounded to bf16 max err {(out_ref.to(torch.bfloat16).double() - out_ref).abs().max().item():.3e}; "
      f"|out| max {out_ref.abs().max().item():.2f}")
