import sys, importlib.util, torch, triton
sys.argv = ["x", "512", "1024", "1024"]
spec = importlib.util.spec_from_file_location("hv", "/tmp/n6_block_halves.py")
src = open("/tmp/n6_block_halves.py").read()
src = src[: src.index("def tflops")]  # kernel only
ns = {}
exec(compile(src, "/tmp/n6_block_halves.py", "exec"), ns)
spec2 = open("/tmp/n6_block_probe.py").read()
spec2 = spec2[: spec2.index("def tflops")]
ns2 = {}
exec(compile(spec2, "/tmp/n6_block_probe.py", "exec"), ns2)
for (M, N, K) in ((512, 1024, 1024), (300, 2048, 4096), (4096, 7168, 4096)):
    torch.manual_seed(1)
    a = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    b = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    As = torch.rand(M, K // 128, device="cuda") + 0.5
    Bs = torch.rand((N + 127) // 128, K // 128, device="cuda") + 0.5
    for name, kern, cfg in (("halves", ns["_halves"], dict(BM=128, BN=128, BK=32, num_warps=8, num_stages=2)),
                            ("single", ns2["_block"], dict(B_KN=0, BM=64, BN=64, BK=64, num_warps=8, num_stages=1))):
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)
        kern[(triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"]),)](
            a, b, As, Bs, c, M, N, K, a.stride(0), b.stride(0), As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1),
            GK=128, GN=128, GROUP_M=8, **cfg)
        rows = torch.arange(0, M, max(1, M // 128), device="cuda")
        af = a[rows].double().view(len(rows), K // 128, 128)
        bf = b.double().view(N, K // 128, 128)
        part = torch.einsum("mbk,nbk->mnb", af, bf)
        sc = As[rows].double()[:, None, :] * Bs.double().repeat_interleave(128, 0)[:N][None, :, :]
        ref = (part * sc).sum(-1)
        d = (c[rows].double() - ref).abs()
        print(f"M={M} N={N} K={K} {name}: max abs err {d.max().item():.3e}, max |ref| {ref.abs().max().item():.1f}, "
              f"max err / max|ref| {d.max().item() / ref.abs().max().item():.2e}, rms err / rms ref "
              f"{(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item():.2e}", flush=True)
