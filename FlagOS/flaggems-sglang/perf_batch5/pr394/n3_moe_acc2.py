import importlib, os, sys, torch
W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang
from flaggems_sglang.reference import get_reference
ref = get_reference("fused_moe")
r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
g = importlib.import_module("flaggems_sglang.ops.fused_moe")
for (T, H, I, E, topk) in ((64, 2048, 1408, 64, 8), (300, 2048, 1408, 64, 8), (200, 2048, 128, 256, 8), (1024, 2048, 1408, 64, 8)):
    for rsf in (1.0, 2.5):
        gen = torch.Generator(device="cuda").manual_seed(T + E)
        x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
        w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
        to = (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None)
        want = ref(x, w1, w2, to, "silu", rsf).float()
        ids = to[1].reshape(-1).contiguous(); wts = to[0].reshape(-1).contiguous()
        cells = []
        for name, fn in (("generic", lambda: g.fused_moe(x, w1, w2, to, "silu", rsf)),
                         ("bf16 c1", lambda: r._fused_moe_grouped(x, w1, w2, ids, wts, topk, 0, rsf, torch.empty_like(x), split_a=False)),
                         ("split c1", lambda: r._fused_moe_grouped(x, w1, w2, ids, wts, topk, 0, rsf, torch.empty_like(x), split_a=True))):
            d = (fn().float() - want).abs()
            excess = (d - (2e-2 + 2e-2 * want.abs())).max().item()
            cells.append(f"{name}: max {d.max().item():.3e} excess {excess:+.3e}")
        print(f"T={T} I={I} E={E} rsf={rsf} | " + " | ".join(cells), flush=True)
