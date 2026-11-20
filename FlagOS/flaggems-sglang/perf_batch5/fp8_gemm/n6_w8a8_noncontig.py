import importlib.util, os, sys, torch
T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, os.path.join(T, "src"))
from flaggems_sglang.reference import get_reference
spec = importlib.util.spec_from_file_location("new_w8", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/w8a8_block_fp8_matmul.py"))
new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
ref = get_reference("w8a8_block_fp8_matmul")
FP8 = torch.float8_e4m3fn
for m, n, k, how in ((300, 1024, 1024, "A cols"), (8, 1024, 1024, "A cols"), (300, 1024, 1024, "B rows"), (4, 2048, 512, "A rows")):
    g = torch.Generator(device="cuda").manual_seed(m + n)
    A = torch.randn(m, k, device="cuda", generator=g).to(FP8)
    B = torch.randn(n, k, device="cuda", generator=g).to(FP8)
    if how == "A cols":
        w = torch.zeros(m, k + 128, device="cuda").to(FP8); w[:, :k] = A; A = w[:, :k]
    elif how == "B rows":
        w = torch.zeros(2 * n, k, device="cuda").to(FP8); w[::2] = B; B = w[::2]
    elif how == "A rows":
        w = torch.zeros(m + 1, k, device="cuda").to(FP8); w[1:] = A; A = w[1:]
    As = 0.5 + torch.rand(m, k // 128, device="cuda", generator=g)
    Bs = 0.5 + torch.rand(n // 128, k // 128, device="cuda", generator=g)
    want = ref(A, B, As, Bs, [128, 128], torch.bfloat16).float()
    out = new.w8a8_block_fp8_matmul(A, B, As, Bs, [128, 128], torch.bfloat16).float()
    err = ((out - want).abs().max() / want.abs().max()).item()
    print(f"{how:7s} M={m} N={n} K={k} contiguous A={A.is_contiguous()} B={B.is_contiguous()}: max err / max|ref| {err:.2e} {'ok' if err < 7.9e-3 else 'FAIL'}")
