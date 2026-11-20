import importlib.util, itertools, os, sys, torch, triton
from triton.testing import do_bench
T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, T)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
gen = load("gen_awq", os.path.join(T, "src/flaggems_sglang/ops/awq_gemm.py"))
new = load("new_awq", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/awq_gemm.py"))
deq = load("ref_deq", os.path.join(T, "src/flaggems_sglang/reference/awq_dequantize.py"))
bm = importlib.import_module("benchmark.test_awq_gemm")
quick = lambda fn: do_bench(fn, warmup=5, rep=25, return_mode="median") * 1e3
bench = lambda fn: do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3
dts = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
for job in sys.argv[1:]:
    name, n = job.split(":"); n = int(n)
    kw = next(iter(bm._input_fn((1, n), torch.float32, "cuda")))
    kw["input"] = kw["input"].to(dts[name]); kw["scales"] = kw["scales"].to(dts[name])
    exact = kw["input"].double() @ deq.reference(kw["qweight"], kw["scales"], kw["qzeros"]).double()
    res = []
    for bn, bk, nw, ns, sp in itertools.product((32, 64, 128, 256, 512), (16, 32, 64), (1, 2, 4, 8), (1, 2), (1, 2, 4, 8, 16, 32)):
        if bn > n or nw * 32 > bn * bk // 8 or triton.cdiv(n, bn) * sp > 2048:
            continue
        new._gemv_config = lambda n_, g_, c=(bn, bk, nw, ns, sp): c
        try:
            t = quick(lambda: new.awq_gemm(**kw))
        except Exception:
            continue
        res.append((t, (bn, bk, nw, ns, sp)))
    res.sort()
    top = []
    for _, c in res[:4]:
        new._gemv_config = lambda n_, g_, c=c: c
        top.append((sorted(bench(lambda: new.awq_gemm(**kw)) for _ in range(3))[1], c))
    top.sort()
    new._gemv_config = lambda n_, g_, c=top[0][1]: c
    err = ((new.awq_gemm(**kw).double() - exact).abs().max() / exact.abs().max()).item()
    tg = sorted(bench(lambda: gen.awq_gemm(**kw)) for _ in range(3))[1]
    print(f"{name} (1, {n}): generic {tg:6.1f} | best {top[0][0]:6.1f} {top[0][1]} {tg / top[0][0]:4.2f}x err {err:.1e} | "
          + " ".join(f"{c}:{t:.1f}" for t, c in top[1:]) + f" | {len(res)} configs", flush=True)
