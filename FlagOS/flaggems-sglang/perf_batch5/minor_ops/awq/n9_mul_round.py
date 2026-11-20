import torch, triton, triton.language as tl

@triton.jit
def _mul(X, S, Y_native, Y_f32, N: tl.constexpr):
    i = tl.arange(0, N)
    d = tl.load(X + i)
    s = tl.load(S + i)
    tl.store(Y_native + i, d.to(S.dtype.element_ty) * s)
    tl.store(Y_f32 + i, (d.to(tl.float32) * s.to(tl.float32)).to(S.dtype.element_ty))

for dt in (torch.float16, torch.bfloat16):
    d = torch.randint(-15, 16, (4096,), device="cuda", dtype=torch.int32)
    s = torch.rand(4096, device="cuda").to(dt)
    yn = torch.empty(4096, device="cuda", dtype=dt); yf = torch.empty_like(yn)
    _mul[(1,)](d, s, yn, yf, 4096)
    ref = d.to(torch.int8) * s   # as in the reference dequantize
    print(f"{str(dt):15s} native mul mismatches {(yn != ref).sum().item()}, fp32 mul + cast mismatches {(yf != ref).sum().item()} / 4096")
