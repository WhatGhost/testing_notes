import importlib.util, os, sys, torch
from torch.profiler import profile, ProfilerActivity
T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, T)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
gen = load("gen_smla", os.path.join(T, "src/flaggems_sglang/ops/triton_sparse_mla_fwd.py"))
new = load("new_smla", os.path.join(T, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/triton_sparse_mla_fwd.py"))
bm = importlib.import_module("benchmark.test_triton_sparse_mla_fwd")
shapes = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1:]] or [(128, 2048), (128, 512), (16, 2048), (1, 2048)]
for shape in shapes:
    kw = next(iter(bm._input_fn(shape, torch.float8_e4m3fn, "cuda")))
    for name, mod in (("gen", gen), ("new", new)):
        for _ in range(3):
            mod.triton_sparse_mla_fwd(**kw)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                mod.triton_sparse_mla_fwd(**kw)
            torch.cuda.synchronize()
        tot = {}
        for e in prof.events():
            if e.device_type.name == "CUDA":
                tot[e.name] = tot.get(e.name, 0) + e.device_time_total / 10
        print(f"{name} {shape}: " + ", ".join(f"{k[:20]} {v:.1f}us" for k, v in sorted(tot.items(), key=lambda x: -x[1])), flush=True)
