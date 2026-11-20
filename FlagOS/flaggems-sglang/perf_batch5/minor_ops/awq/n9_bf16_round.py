import torch, triton, triton.language as tl

@triton.jit
def _cvt(X, Y0, Y1, Y2, P0, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(X + i)
    tl.store(Y0 + i, x)                                   # implicit at store
    tl.store(Y1 + i, x.to(tl.bfloat16))                   # default .to
    tl.store(Y2 + i, x.to(tl.bfloat16, fp_downcast_rounding="rtne"))
    s = tl.load(X + i).to(tl.bfloat16)
    tl.store(P0 + i, (s * s))                             # bf16 * bf16

x = torch.randn(4096, device="cuda", dtype=torch.float32)
ys = [torch.empty(4096, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
_cvt[(1,)](x, *ys, 4096)
ref = x.to(torch.bfloat16)
for name, y in zip(("store", ".to", ".to rtne"), ys[:3]):
    print(f"{name:9s} mismatches vs torch RTNE: {(y != ref).sum().item()} / 4096")
xs = x.to(torch.bfloat16)
print(f"bf16*bf16 mismatches vs torch: {(ys[3] != xs * xs).sum().item()} / 4096")
