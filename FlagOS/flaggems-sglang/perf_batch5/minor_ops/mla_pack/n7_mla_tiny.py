import importlib.util, os, statistics, torch
from triton.testing import do_bench
T = open("/tmp/fgs-f8.path").read().strip()
def load(n, p):
    spec = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
gen = load("gen_mla", os.path.join(T, "src/flaggems_sglang/ops/mla_kv_pack_quantize_fp8.py"))
new = load("new_mla", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/mla_kv_pack_quantize_fp8.py"))
for s, h in ((1, 1), (1, 16), (8, 1), (8, 16), (7, 8), (64, 1), (64, 16), (333, 1)):
    g = torch.Generator(device="cuda").manual_seed(0)
    k_nope = torch.randn(s, h, 512, device="cuda", generator=g).to(torch.bfloat16)
    k_pe = torch.randn(s, 64, device="cuda", generator=g).to(torch.bfloat16)
    v = torch.randn(s, h, 512, device="cuda", generator=g).to(torch.bfloat16)
    tg, tn = [], []
    for _ in range(7):
        tg.append(do_bench(lambda: gen.mla_kv_pack_quantize_fp8(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn), warmup=50, rep=200, return_mode="median") * 1e3)
        tn.append(do_bench(lambda: new.mla_kv_pack_quantize_fp8(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn), warmup=50, rep=200, return_mode="median") * 1e3)
    print(f"s={s:4d} h={h:2d}: master median {statistics.median(tg):5.2f} (min {min(tg):5.2f}) new median {statistics.median(tn):5.2f} (min {min(tn):5.2f}) -> {statistics.median(tg) / statistics.median(tn):.3f}x", flush=True)
