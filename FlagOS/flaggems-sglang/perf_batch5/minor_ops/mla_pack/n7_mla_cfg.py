import importlib.util, os, math, torch, itertools
from triton.testing import do_bench
T = open("/tmp/fgs-f8.path").read().strip()
spec = importlib.util.spec_from_file_location("new_mla", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/mla_kv_pack_quantize_fp8.py"))
new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
spec = importlib.util.spec_from_file_location("gen_mla", os.path.join(T, "src/flaggems_sglang/ops/mla_kv_pack_quantize_fp8.py"))
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
for s, h in ((4096, 16), (512, 16), (256, 8)):
    g = torch.Generator(device="cuda").manual_seed(0)
    k_nope = torch.randn(s, h, 512, device="cuda", generator=g).to(torch.bfloat16)
    k_pe = torch.randn(s, 64, device="cuda", generator=g).to(torch.bfloat16)
    v = torch.randn(s, h, 512, device="cuda", generator=g).to(torch.bfloat16)
    k_out = torch.empty((s, h, 576), device="cuda", dtype=torch.float8_e4m3fn)
    v_out = torch.empty((s, h, 512), device="cuda", dtype=torch.float8_e4m3fn)
    ref = gen.mla_kv_pack_quantize_fp8(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)
    tg = do_bench(lambda: gen.mla_kv_pack_quantize_fp8(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn), warmup=25, rep=100, return_mode="median") * 1e3
    res = []
    for br, nw in itertools.product((1, 2, 4, 8, 16), (2, 4, 8, 16)):
        if br > h:
            continue
        meta = (s * h, h, 512, 64, 512, None, 1.0, 1.0, True, br, 0, 0, True)
        groups = (h + br - 1) // br
        fn = lambda: new._address_shared[s, groups](k_nope, k_pe, v, k_out.data_ptr(), v_out.data_ptr(), meta, num_warps=nw, enable_fp_fusion=False)
        fn()
        ok = torch.equal(k_out.view(torch.uint8), ref[0].view(torch.uint8)) and torch.equal(v_out.view(torch.uint8), ref[1].view(torch.uint8))
        t = do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3
        res.append((t, br, nw, ok))
    res.sort()
    print(f"s={s} h={h}: master {tg:.1f} us | " + " ".join(f"(BR={br},w={nw}{'' if ok else ',BAD'}):{t:.1f}" for t, br, nw, ok in res[:6]), flush=True)
